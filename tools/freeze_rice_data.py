"""Reproduce the exploratory rice-pest data freeze without model execution.

Requires Pillow and an audited source directory containing archive_index.json,
zip-directory.bin, and extracted/datasets metadata CSVs. Reconstruct these inputs
with rice_data_audit.py from the pinned Zenodo ZIP central-directory/metadata
ranges. The audited input hashes below refuse drift. Cached images are reused
only after ZIP CRC/size checks. Otherwise downloads use public HTTPS ZIP ranges.

Example (output is separate from committed manifests):
  python tools/freeze_rice_data.py --audit-dir outputs/rice-data-audit     --output-dir outputs/rice-refreeze

This writes rice_data.json and rice_dataset_audit.json in --output-dir, and
selection/download reports plus image caches in --audit-dir. Review outputs before
replacing committed manifests. No model inference or GPU access occurs.
"""

from __future__ import annotations

import argparse
import collections
import concurrent.futures
import csv
import hashlib
import io
import json
import pathlib
import struct
import time
import urllib.request
import zlib

from PIL import Image, ImageOps

INPUT_SHA256 = {
    "archive_index.json": "fd13d06ab4d34fd1a46c772443e87dca2d53a78902e577a9378dc9da22bad6c3",
    "zip-directory.bin": "6047989ee5e1b4fe244b2065c3cea1b2093d10d70d0066711f0e77db94c08804",
    ("extracted/datasets/04_metadata/acquisition_metadata.csv"): (
        "968072410d1c4608fcad3aa388ce7e4a4f5bff08e762b4c4a495b20f71298102"
    ),
    ("extracted/datasets/04_metadata/bounding_box_annotations.csv"): (
        "87e78d239cd5eb59097113cf409cb6d2efac9081e9d9f419eec510ab93ada084"
    ),
    ("extracted/datasets/04_metadata/class_dictionary.csv"): (
        "e1a239d7476fd963e17ed2a209e29db0b542ccf7529a7da33d64058a3b216b05"
    ),
    ("extracted/datasets/04_metadata/image_quality_flags.csv"): (
        "169ef17d9e076f65a9dc043f0bb0d211f55016c4ac77643d958986c338e4b175"
    ),
    ("extracted/datasets/04_metadata/split_manifest.csv"): (
        "a61eb3502131e1f8a609c21b1e784161a21575943e03696a8edc891e91633159"
    ),
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-dir", required=True, type=pathlib.Path)
    parser.add_argument("--output-dir", required=True, type=pathlib.Path)
    parser.add_argument("--offline", action="store_true", help="Require all candidate images in the cache")
    args = parser.parse_args()
    for name, expected in INPUT_SHA256.items():
        source = args.audit_dir / name
        payload = source.read_bytes()
        if name == "archive_index.json":
            payload = json.dumps(json.loads(payload), sort_keys=True, separators=(",", ":")).encode()
        if hashlib.sha256(payload).hexdigest() != expected:
            raise ValueError(f"Audited input changed: {name}")
    P = args.audit_dir.resolve()
    OUT = args.output_dir.resolve()
    OUT.mkdir(parents=True, exist_ok=True)
    D = P / "extracted/datasets"
    A = json.loads((P / "archive_index.json").read_text())
    INDEX = {x["name"]: x for x in A}
    URL = "https://zenodo.org/api/records/20066074/files/datasets.zip/content"

    def read(n):
        return list(csv.DictReader((D / n).open(encoding="utf-8-sig")))

    S = read("04_metadata/split_manifest.csv")
    B = read("04_metadata/bounding_box_annotations.csv")
    BOX = collections.defaultdict(list)
    for r in B:
        BOX[r["image_file"]].append(r)
    F = collections.defaultdict(list)
    for r in S:
        F[r["file_name"].split(".rf.")[0]].append(r)
    ex = []
    selected = []
    for stem, rows in sorted(F.items()):
        if len({r["split"] for r in rows}) > 1:
            ex.append({"family": stem, "reason": "source_family_cross_split", "images": len(rows)})
            continue
        if any(len(BOX[r["file_name"]]) > 100 for r in rows):
            ex.append(
                {"family": stem, "reason": "family_contains_more_than_100_targets", "images": len(rows)}
            )
            continue
        valid = []
        for r in rows:
            boxes = BOX[r["file_name"]]
            good = bool(boxes)
            for b in boxes:
                x, y, w, h = [
                    float(b[k]) for k in ("bbox_x_center", "bbox_y_center", "bbox_width", "bbox_height")
                ]
                if (
                    min(x - w / 2, y - h / 2) < -1e-12
                    or max(x + w / 2, y + h / 2) > 1 + 1e-12
                    or min(w, h) <= 0
                ):
                    good = False
            if good:
                valid.append(r)
        if not valid:
            ex.append({"family": stem, "reason": "no_valid_box_member", "images": len(rows)})
            continue
        r = min(
            valid,
            key=lambda r: hashlib.sha256(("dimer-ph-rice-pests-v1:" + r["file_name"]).encode()).hexdigest(),
        )
        r = dict(r, family=stem, rank=hashlib.sha256(("dimer-ph-rice-pests-v1:" + stem).encode()).hexdigest())
        selected.append(r)
    selected = sorted(selected, key=lambda r: r["rank"])
    candidates = []
    for role, n in [("train", 150), ("valid", 55), ("test", 55)]:
        candidates.extend([r for r in selected if r["split"] == role][:n])
    (P / "selection_candidates.json").write_text(
        json.dumps(candidates, indent=2), encoding="utf-8", newline="\n"
    )
    (P / "preselection_exclusions.json").write_text(json.dumps(ex, indent=2), encoding="utf-8", newline="\n")
    CACHE = P / "images"
    CACHE.mkdir(exist_ok=True)

    def fetch(r):
        e = INDEX["datasets/" + r["relative_path"]]
        path = CACHE / r["file_name"]
        if not path.exists():
            if args.offline:
                raise FileNotFoundError(f"Offline candidate missing: {path}")
            length = 30 + len(e["name"].encode()) + e["compressed"] + 1024
            for attempt in range(5):
                try:
                    req = urllib.request.Request(
                        URL, headers={"Range": f"bytes={e['offset']}-{e['offset'] + length - 1}"}
                    )
                    with urllib.request.urlopen(req, timeout=60) as response:
                        if response.status != 206:
                            raise ValueError("Range not honored")
                        data = response.read(length + 1)
                    h = struct.unpack_from("<4s5H3L2H", data, 0)
                    if h[0] != b"PK\x03\x04":
                        raise ValueError("Bad local header")
                    name = data[30 : 30 + h[-2]].decode()
                    if name != e["name"]:
                        raise ValueError("Member mismatch")
                    begin = 30 + h[-2] + h[-1]
                    payload = data[begin : begin + e["compressed"]]
                    if e["method"] == 8:
                        payload = zlib.decompress(payload, -15)
                    if len(payload) != e["size"] or f"{zlib.crc32(payload):08x}" != e["crc32"]:
                        raise ValueError("CRC mismatch")
                    path.write_bytes(payload)
                    break
                except Exception:
                    if attempt == 4:
                        raise
                    time.sleep(2 * (attempt + 1))
        payload = path.read_bytes()
        if len(payload) != e["size"] or f"{zlib.crc32(payload):08x}" != e["crc32"]:
            raise ValueError("Cache mismatch")
        im = ImageOps.exif_transpose(Image.open(io.BytesIO(payload))).convert("RGB")
        small = im.convert("L").resize((9, 8))
        pix = list(small.getdata())
        bits = 0
        for y in range(8):
            for x in range(8):
                bits = bits << 1 | int(pix[y * 9 + x] > pix[y * 9 + x + 1])
        return dict(
            r,
            entry=e,
            sha256=hashlib.sha256(payload).hexdigest(),
            decoded_sha256=hashlib.sha256(im.tobytes()).hexdigest(),
            dhash=f"{bits:016x}",
            width=im.width,
            height=im.height,
        )

    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
        downloaded = []
        for result in pool.map(fetch, candidates):
            downloaded.append(result)
            (P / "downloaded.json").write_text(
                json.dumps(downloaded, indent=2), encoding="utf-8", newline="\n"
            )
            if len(downloaded) % 25 == 0 or len(downloaded) == len(candidates):
                print("Verified candidate images", len(downloaded), flush=True)
    kept = []
    counts = collections.Counter()
    for r in downloaded:
        role = {"valid": "validation"}.get(r["split"], r["split"])
        limit = {"train": 120, "validation": 40, "test": 40}[role]
        if counts[role] >= limit:
            continue
        duplicate = next(
            (
                q
                for q in kept
                if r["sha256"] == q["sha256"]
                or r["decoded_sha256"] == q["decoded_sha256"]
                or (int(r["dhash"], 16) ^ int(q["dhash"], 16)).bit_count() <= 4
            ),
            None,
        )
        if duplicate:
            ex.append(
                {
                    "family": r["family"],
                    "reason": "conservative_dhash_or_exact_duplicate",
                    "against": duplicate["family"],
                }
            )
            continue
        r["role"] = role
        kept.append(r)
        counts[role] += 1
    if counts != {"train": 120, "validation": 40, "test": 40}:
        raise ValueError(dict(counts))
    records = []
    for r in kept:
        image_id = "rice_" + r["rank"][:16]
        objects = []
        for b in BOX[r["file_name"]]:
            x, y, w, h = [
                float(b[k]) for k in ("bbox_x_center", "bbox_y_center", "bbox_width", "bbox_height")
            ]
            xyxy = [
                (x - w / 2) * r["width"],
                (y - h / 2) * r["height"],
                (x + w / 2) * r["width"],
                (y + h / 2) * r["height"],
            ]
            if min(xyxy) < 0 or xyxy[2] > r["width"] or xyxy[3] > r["height"]:
                raise ValueError("Out of bounds")
            objects.append(
                {
                    "object_id": image_id + "_" + b["annotation_index"],
                    "bbox_xyxy": xyxy,
                    "species": "rice_black_bug" if b["class_label"].startswith("RBB") else "white_stemborer",
                    "original_label": b["class_label"],
                    "ignore": False,
                }
            )
        records.append(
            {
                "image_id": image_id,
                "relative_path": "images/" + r["file_name"],
                "sha256": r["sha256"],
                "decoded_sha256": r["decoded_sha256"],
                "dhash": r["dhash"],
                "bytes": r["entry"]["size"],
                "width": r["width"],
                "height": r["height"],
                "capture_group_id": r["family"],
                "grouping_basis": "source_filename_family_not_verified_capture",
                "split": r["role"],
                "source_record": "https://doi.org/10.5281/zenodo.20066074",
                "licence": "CC-BY-4.0",
                "attribution": "Zenodo 20066074 dataset creators; see embedded source metadata",
                "annotation_status": "published_unverified",
                "objects": objects,
                "zip_member": r["entry"],
            }
        )
    manifest = {
        "format_version": 1,
        "scope": "exploratory_published_annotations",
        "classes": ["rice_black_bug", "white_stemborer"],
        "archive": {
            "url": URL,
            "bytes": 1005857218,
            "md5": "24960c245a3a21b63be95314cc7467da",
            "sha256": None,
            "integrity": "Selected member SHA-256 and size are pinned; whole archive SHA-256 not measured.",
        },
        "records": records,
        "crops": [],
        "selection_salt": "dimer-ph-rice-pests-v1",
        "source_split_preserved": True,
        "exclusions": ex,
        "limitations": [
            "Source filename families are not verified independent capture events.",
            "Published boxes and species-sex labels are not verified human-complete references.",
            "Source images are processed Roboflow exports, not untouched camera originals.",
            "Capture location is absent; Philippine affiliation does not establish every capture origin.",
        ],
    }
    (OUT / "rice_data.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8", newline="\n")
    (P / "freeze_summary.json").write_text(
        json.dumps(
            {"images": dict(counts), "objects": sum(len(r["objects"]) for r in records), "exclusions": ex},
            indent=2,
        ),
        encoding="utf-8",
        newline="\n",
    )
    print("FROZEN", dict(counts), flush=True)
    path = OUT / "rice_data.json"
    m = json.loads(path.read_text())
    attribution = (
        "Deleña, Reymark; Balleras, Gina; Dequiña, Brent Jason; Jumawan, R"
        "exter; Flores, Christine. Annotated light-trap rice pest image da"
        "taset with detector-derived region-of-interest crops for species–"
        "sex recognition of rice black bug and white stemborer. Version 1."
        " Zenodo, DOI 10.5281/zenodo.20066074. CC BY 4.0."
    )
    for r in m["records"]:
        r["attribution"] = attribution
    m["attribution"] = attribution
    m["license_url"] = "https://creativecommons.org/licenses/by/4.0/"
    m["source_metadata"] = {
        "doi": "10.5281/zenodo.20066074",
        "version": "1",
        "licence": "cc-by-4.0",
        "capture_origin_verified": False,
        "human_annotation_completeness_verified": False,
    }
    path.write_text(json.dumps(m, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")
    instances = {
        s: collections.Counter(o["species"] for r in m["records"] if r["split"] == s for o in r["objects"])
        for s in ["train", "validation", "test"]
    }
    images = collections.Counter(r["split"] for r in m["records"])
    metadata = {}
    for p in sorted((P / "extracted/datasets/04_metadata").glob("*.csv")):
        metadata[p.name] = {"sha256": hashlib.sha256(p.read_bytes()).hexdigest(), "bytes": p.stat().st_size}
    audit = {
        "scope": m["scope"],
        "qualification": {
            "passed": True,
            "scope": m["scope"],
            "capture_independence_verified": False,
            "human_reference_completeness_verified": False,
            "field_surveillance_qualified": False,
            ("meaning"): (
                "Passed the user-approved exploratory published-annotation benchmark data checks only."
            ),
        },
        "manifest_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "gates": {
            "rights": {
                "status": "passed",
                "evidence": "Zenodo record API license cc-by-4.0; user screenshot confirms CC BY 4.0.",
            },
            "original_images": {
                "status": "passed",
                ("evidence"): (
                    "Selected whole detection images downloaded and decoded; these are"
                    " 640x640 processed Roboflow exports, not original camera files."
                ),
            },
            "crop_lineage": {
                "status": "passed",
                ("evidence"): (
                    "Canonical crops derive from manifest boxes with unique image/obje"
                    "ct IDs; all 91577 distributed ROI filenames map to source detecti"
                    "on image names, with zero split mismatches. Canonical run does no"
                    "t use distributed ROI pixels."
                ),
            },
            "capacity": {
                "status": "passed",
                "images": dict(images),
                "instances": instances,
                "groups": dict(images),
                "group_kind": "source-filename families only",
            },
            "philippine_capture_provenance": {
                "status": "unresolved",
                "evidence": "PhilRice Midsayap creator affiliations; all7098 location values absent.",
            },
            "capture_independence": {
                "status": "unresolved",
                ("evidence"): (
                    "All7098 capture_date,capture_time,light_trap_id,camera_device,ope"
                    "rator,notes fields absent; filename grouping cannot establish spe"
                    "cimen independence."
                ),
            },
            "reference_completeness": {
                "status": "unresolved",
                "evidence": "Published YOLO box table; no human verification/completeness documentation.",
            },
        },
        "source_inventory": {
            "archive_entries": 112940,
            "archive_files": 112898,
            "detection_images": 7098,
            "detection_label_files": 7097,
            "raw_folder_images": 7068,
            "raw_manifest_rows": 7098,
            "roi_images": 91577,
            "annotation_rows": 91577,
            "source_filename_families": 1685,
            "cross_role_source_families": 1,
            "images_above_100_targets": 4,
            "maximum_targets": 106,
        },
        "selection": {
            "salt": m["selection_salt"],
            "source_roles_preserved": True,
            ("variant_rule"): (
                "Within each source stem retain minimum SHA256(salt + colon + file"
                "name); rank families by SHA256(salt + colon + stem)."
            ),
            ("sample_rule"): (
                "Role quotas120/40/40; conservative exclusion of exact hashes, dec"
                "oded hashes or dHash Hamming distance <=4 to any previously selec"
                "ted image; refill using next deterministic candidate."
            ),
            "exclusions": m["exclusions"],
            "discarded_source_crops": True,
            ("missing_label_policy"): (
                "Canonical references use published bounding_box_annotations.csv; "
                "missing label file is never interpreted as zero targets."
            ),
            "target_free_images": sum(not r["objects"] for r in m["records"]),
        },
        "source_metadata_files": metadata,
        "integrity": {
            "central_directory_sha256": hashlib.sha256((P / "zip-directory.bin").read_bytes()).hexdigest(),
            "full_archive_downloaded": False,
            "archive_md5_publisher": "24960c245a3a21b63be95314cc7467da",
            "member_crc_and_size_verified": True,
            "selected_member_sha256_pinned": True,
        },
        "limitations": m["limitations"]
        + [
            (
                "Single-insect and multi-insect photographs are mixed. Dorsal/vent"
                "ral views may represent the same specimen across filename familie"
                "s."
            ),
            (
                "dHash is an approximate duplicate screen and cannot prove biologi"
                "cal or acquisition independence."
            ),
            "Source resize/contrast/augmentation cannot be undone; no claim to unprocessed camera imagery.",
            (
                "All performance is agreement with published annotations, not inde"
                "pendently verified pest count accuracy."
            ),
        ],
    }
    (OUT / "rice_dataset_audit.json").write_text(
        json.dumps(audit, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n"
    )
    (OUT / "return.json").write_text(
        json.dumps(
            {
                "status": "exploratory_frozen",
                "manifest": str(OUT / "rice_data.json"),
                "audit": str(OUT / "rice_dataset_audit.json"),
                "images": dict(images),
                "instances": instances,
                "manifest_sha256": audit["manifest_sha256"],
                "qualification": audit["qualification"],
                "limitations": audit["limitations"],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
        newline="\n",
    )
    (OUT / "return.md").write_text(
        (
            "# Rice pest source audit\n\nFrozen 200 whole-image Roboflow exports"
            ": 120 train / 40 validation / 40 test, one variant per filename f"
            "amily. Actual selected image bytes, CRCs, SHA-256 and decoded ima"
            "ge hashes verified. dHash distance <=4 candidates conservatively "
            "excluded. No model outputs used.\n\nRights: CC BY 4.0. Capture prov"
            "enance, biological independence and human-complete reference anno"
            "tations remain unresolved. This is an exploratory benchmark again"
            "st published labels, as explicitly approved by the user. No field"
            " surveillance qualification is claimed.\n\nSource inventory: 7098 d"
            "etection images, 7097 detection label files, 91577 annotation row"
            "s/ROI images, 7068 raw-folder copies. Metadata rows7098 have no d"
            "ates/times/locations/trap IDs. Four images exceed100 boxes; their"
            " entire source family is excluded. One train/test filename family"
            " is excluded. 133 families have no geometrically valid variant; n"
            "o boxes clipped.\n\nCanonical crops will be regenerated from pinned"
            " manifest boxes, not distributed detector-derived ROI pixels. Mem"
            "bers were fetched by HTTPS ranges and checked against ZIP CRC bef"
            "ore local SHA-256 pinning; whole archive MD5 is publisher-supplie"
            "d and not locally verified.\n"
        ),
        encoding="utf-8",
        newline="\n",
    )
    print(
        json.dumps(
            {"images": dict(images), "instances": instances, "manifest_sha256": audit["manifest_sha256"]},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
