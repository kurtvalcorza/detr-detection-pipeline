"""Pure CPU contracts for the rice-pest capstone; these do not certify source data."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path, PurePosixPath

import numpy as np
from scipy.optimize import linear_sum_assignment

SPECIES = ("rice_black_bug", "white_stemborer")
ROLES = ("train", "validation", "test")


def sha256(path: Path) -> str:
    """Hash a file without loading it into memory."""
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def digest_json(value: object) -> str:
    """Hash canonical JSON, refusing NaN and infinity."""
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()
    ).hexdigest()


def safe_path(root: Path, name: str) -> Path:
    """Refuse portable archive traversal and symlink escapes."""
    if not isinstance(name, str) or not name or "\\" in name or ":" in name or "\x00" in name:
        raise ValueError("Unsafe relative path")
    relative = PurePosixPath(name)
    if relative.is_absolute() or any(part in ("", ".", "..") for part in name.split("/")):
        raise ValueError("Unsafe relative path")
    resolved = Path(root).resolve() / relative
    resolved = resolved.resolve()
    if not resolved.is_relative_to(Path(root).resolve()):
        raise ValueError("Path escapes root")
    return resolved


def _boxes(value: object) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if array.size == 0:
        return np.empty((0, 4))
    if array.ndim != 2 or array.shape[1] != 4 or not np.isfinite(array).all():
        raise ValueError("Boxes must be finite N x 4 xyxy")
    if np.any(array[:, 2:] <= array[:, :2]):
        raise ValueError("Boxes require positive area")
    return array


def validate_manifest(manifest: dict, require_capacity: bool = False) -> dict:
    """Validate normalized references; provenance evidence still needs an independent audit.

    Complete annotation assertions are required even for zero-target images.
    Any declared specimen/session/duplicate lineage must have one split owner.
    """
    if manifest.get("classes") != list(SPECIES):
        raise ValueError("Class order mismatch")
    records = manifest.get("records", [])
    exploratory = manifest.get("scope") == "exploratory_published_annotations"
    if exploratory and require_capacity:
        raise ValueError("Exploratory source families cannot certify independent capture-group capacity")
    if not records:
        raise ValueError("Empty image manifest")
    ids, paths, object_ids, lineage = set(), set(), set(), {}
    parents = {}
    counts = {role: np.zeros(2, dtype=int) for role in ROLES}
    groups = {role: set() for role in ROLES}
    presence = {(role, species): set() for role in ROLES for species in SPECIES}
    for row in records:
        image_id, split, group = row["image_id"], row["split"], row["capture_group_id"]
        if not image_id or image_id in ids or split not in ROLES or not group:
            raise ValueError("Invalid image ID, role, or capture group")
        ids.add(image_id)
        safe_path(Path("."), row["relative_path"])
        if row["relative_path"] in paths:
            raise ValueError("Duplicate image path")
        paths.add(row["relative_path"])
        if not re.fullmatch(r"[0-9a-f]{64}", row["sha256"]):
            raise ValueError("Invalid image hash")
        if not 0 < row["bytes"] <= 25 * 1024**2:
            raise ValueError("Image byte limit")
        if any(type(row[key]) is not int or not 32 <= row[key] <= 4096 for key in ("width", "height")):
            raise ValueError("Image dimension limit")
        if not all(row.get(key) for key in ("source_record", "licence", "attribution")):
            raise ValueError("Missing source attribution")
        allowed_status = {"human_verified_complete", "human_created_complete"}
        if exploratory:
            allowed_status.add("published_unverified")
        if row.get("annotation_status") not in allowed_status:
            raise ValueError("Reference annotations must be human-verified and complete")
        for key in (
            "capture_group_id",
            "sha256",
            "decoded_sha256",
            "duplicate_cluster_id",
            "source_family_id",
            "capture_session_id",
            "specimen_id",
            "tray_id",
            "burst_id",
        ):
            value = row.get(key)
            if value:
                old = lineage.setdefault((key, value), split)
                if old != split:
                    raise ValueError(f"Cross-split lineage: {key}")
        groups[split].add(group)
        parents[image_id] = row
        for obj in row["objects"]:
            if not obj["object_id"] or obj["object_id"] in object_ids:
                raise ValueError("Duplicate object ID")
            object_ids.add(obj["object_id"])
            box = _boxes([obj["bbox_xyxy"]])[0]
            if np.any(box < 0) or box[2] > row["width"] or box[3] > row["height"]:
                raise ValueError("Out-of-bounds reference box")
            if obj["species"] not in SPECIES or not obj.get("original_label"):
                raise ValueError("Unknown or missing reference label")
            if type(obj.get("ignore")) is not bool or (obj["ignore"] and not obj.get("ignore_reason")):
                raise ValueError("Explicit ignore flag and reason required")
            if obj.get("sex") is not None and obj.get("sex_verified") is not True:
                raise ValueError("Unverified sex label")
            if not obj["ignore"]:
                counts[split][SPECIES.index(obj["species"])] += 1
                presence[(split, obj["species"])].add(group)
    crop_ids = set()
    for crop in manifest.get("crops", []):
        if not crop["crop_id"] or crop["crop_id"] in crop_ids:
            raise ValueError("Duplicate crop ID")
        crop_ids.add(crop["crop_id"])
        parent = parents.get(crop["parent_image_id"])
        if parent is None:
            raise ValueError("Unmapped crop")
        obj = next((o for o in parent["objects"] if o["object_id"] == crop["parent_object_id"]), None)
        if obj is None or obj["ignore"]:
            raise ValueError("Invalid crop reference object")
        if crop.get("split", parent["split"]) != parent["split"]:
            raise ValueError("Crop split leakage")
        box = _boxes([crop["bbox_xyxy"]])[0]
        if np.any(box < 0) or box[2] > parent["width"] or box[3] > parent["height"]:
            raise ValueError("Out-of-bounds crop")
        if not crop.get("preprocessing"):
            raise ValueError("Missing crop preprocessing")
    if require_capacity:
        for role, minimum_groups, minimum_instances in (
            ("train", 6, 60),
            ("validation", 3, 20),
            ("test", 3, 20),
        ):
            if len(groups[role]) < minimum_groups or np.any(counts[role] < minimum_instances):
                raise ValueError(f"Insufficient independent support: {role}")
            if role != "train" and any(len(presence[(role, s)]) < 2 for s in SPECIES):
                raise ValueError("Species requires two evaluation groups")
    return {
        "images": len(records),
        "crops": len(crop_ids),
        "objects": len(object_ids),
        "groups": {k: len(v) for k, v in groups.items()},
        "instances": {k: v.tolist() for k, v in counts.items()},
        "capacity_checked": require_capacity,
        "dataset_qualified": False,
        "scope": manifest.get("scope", "strict_capture_group_study"),
        "limitations": (
            [
                "Published annotations are unverified; completeness and label accuracy are unknown.",
                "Source-family splits do not establish independent capture sessions or specimens.",
            ]
            if exploratory
            else []
        ),
    }


def _counts(reference: object, predicted: object) -> tuple[np.ndarray, np.ndarray]:
    truth, prediction = np.asarray(reference, dtype=float), np.asarray(predicted, dtype=float)
    if truth.ndim != 2 or truth.shape[1] != 2 or not len(truth) or prediction.shape != truth.shape:
        raise ValueError("Counts must be aligned nonempty N x 2 arrays")
    if (
        not np.isfinite(truth).all()
        or not np.isfinite(prediction).all()
        or np.any(truth < 0)
        or np.any(prediction < 0)
    ):
        raise ValueError("Counts must be finite and nonnegative")
    if np.any(truth != np.floor(truth)):
        raise ValueError("Reference counts must be integers")
    return truth, prediction


def count_metrics(reference: object, predicted: object) -> dict:
    """Include every image/species pair, including zeros and fractional baselines."""
    truth, prediction = _counts(reference, predicted)
    error = prediction - truth
    return {
        "images": len(truth),
        "image_species_pairs": truth.size,
        "mae": float(np.abs(error).mean()),
        "bias": float(error.mean()),
        "exact_count_rate": float((error == 0).mean()),
        "all_species_exact_image_rate": float(np.all(error == 0, axis=1).mean()),
        "per_species_mae": np.abs(error).mean(axis=0).tolist(),
        "per_species_bias": error.mean(axis=0).tolist(),
        "per_species_exact_rate": (error == 0).mean(axis=0).tolist(),
        "total_target_mae": float(np.abs(error.sum(axis=1)).mean()),
    }


def pairwise_iou(reference: object, predicted: object) -> np.ndarray:
    """Compute continuous-coordinate IoU (no pixel +1 convention)."""
    a, b = _boxes(reference), _boxes(predicted)
    intersection = np.maximum(
        0, np.minimum(a[:, None, 2:], b[None, :, 2:]) - np.maximum(a[:, None, :2], b[None, :, :2])
    ).prod(axis=2)
    aa, bb = (a[:, 2:] - a[:, :2]).prod(axis=1), (b[:, 2:] - b[:, :2]).prod(axis=1)
    return intersection / (aa[:, None] + bb[None, :] - intersection)


def match_boxes(reference: object, predicted: object, iou_threshold: float = 0.5) -> dict:
    """Maximum cardinality, then IoU sum, with stable input-index tie resolution.

    Fixed SciPy assignment is deterministic for fixed ordered inputs. The cardinality
    bonus is greater than the maximum possible total IoU change. No perturbation
    changes the secondary scientific objective.
    """
    if not 0 < iou_threshold <= 1:
        raise ValueError("Invalid IoU threshold")
    iou = pairwise_iou(reference, predicted)
    n, m = iou.shape
    eligible = iou >= iou_threshold
    score = np.where(eligible, min(n, m) + 1 + iou, 0)
    row, col = linear_sum_assignment(score, maximize=True)
    pairs = [(int(r), int(c), float(iou[r, c])) for r, c in zip(row, col, strict=True) if eligible[r, c]]
    return {
        "matches": pairs,
        "misses": sorted(set(range(n)) - {p[0] for p in pairs}),
        "spurious": sorted(set(range(m)) - {p[1] for p in pairs}),
    }


def error_decomposition(reference: list[dict], predicted: list[dict]) -> dict:
    """Decompose only audited nonignored images; do not invent ignore semantics.

    Images with ignored objects need an explicit evaluator-specific handling policy;
    refusing them here prevents ignored insects being silently counted as spurious.
    """
    if any(obj.get("ignore", False) for obj in reference):
        raise ValueError("Ignore regions require an explicit evaluation policy")
    if any(obj.get("species") not in SPECIES for obj in reference + predicted):
        raise ValueError("Unknown species")
    result = match_boxes([o["bbox_xyxy"] for o in reference], [o["bbox_xyxy"] for o in predicted])
    confusion = np.zeros((2, 2), dtype=int)
    for r, p, _ in result["matches"]:
        confusion[SPECIES.index(reference[r]["species"]), SPECIES.index(predicted[p]["species"])] += 1
    result.update(
        missed=len(result["misses"]),
        spurious_count=len(result["spurious"]),
        wrong_species=int(confusion.sum() - np.trace(confusion)),
        matched_confusion=confusion.tolist(),
    )
    return result


def select_count_threshold(rows: list[dict], split: str = "validation") -> dict:
    """Choose predeclared validation grid minimum MAE; exact ties go higher."""
    if split != "validation" or len(rows) != 19:
        raise ValueError("Selection requires the complete validation threshold grid")
    thresholds = [float(row["threshold"]) for row in rows]
    if sorted(round(t, 12) for t in thresholds) != [i / 20 for i in range(1, 20)]:
        raise ValueError("Threshold grid mismatch")
    if any(not np.isfinite(row["mae"]) or row["mae"] < 0 for row in rows):
        raise ValueError("Invalid validation MAE")
    return dict(min(rows, key=lambda row: (row["mae"], -row["threshold"])))


def paired_group_bootstrap(
    reference: object,
    prediction_a: object,
    prediction_b: object,
    groups: list[str],
    n_boot: int = 2000,
    seed: int = 42,
) -> dict:
    """Resample whole groups, preserving image weighting; difference is A minus B."""
    truth, a = _counts(reference, prediction_a)
    _, b = _counts(reference, prediction_b)
    if len(groups) != len(truth) or any(not isinstance(g, str) or not g for g in groups) or n_boot < 1:
        raise ValueError("Invalid bootstrap groups or replicate count")
    unique = sorted(set(groups))
    indices = [np.flatnonzero(np.asarray(groups) == group) for group in unique]
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(n_boot):
        selected = np.concatenate([indices[i] for i in rng.integers(len(unique), size=len(unique))])
        ma, mb = np.abs(a[selected] - truth[selected]).mean(), np.abs(b[selected] - truth[selected]).mean()
        values.append((ma, mb, ma - mb))
    intervals = np.quantile(values, [0.025, 0.975], axis=0)
    return {
        "groups": len(unique),
        "images": len(truth),
        "replicates": n_boot,
        "seed": seed,
        "difference_direction": "a_minus_b",
        "unstable_few_groups": len(unique) < 10,
        "mae_a_ci": intervals[:, 0].tolist(),
        "mae_b_ci": intervals[:, 1].tolist(),
        "difference_ci": intervals[:, 2].tolist(),
        "difference": float(np.abs(a - truth).mean() - np.abs(b - truth).mean()),
    }
