"""Standalone, qualification-gated Philippine rice-pest capstone runtime.

Model stages require CUDA. Importing this module downloads nothing. Reference
labels are never used to select settings on the test split.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import gc
import importlib.metadata
import io
import json
import math
import os
import platform
import shutil
import time
import zipfile
from pathlib import Path

import numpy as np
import rice_core as core
from PIL import Image, ImageOps

SPECIES = ("rice_black_bug", "white_stemborer")
PINS = {
    "detr": ("facebook/detr-resnet-50", "1d5f47bd3bdd2c4bbfa585418ffe6da5028b4c0b"),
    "bioclip": ("imageomics/bioclip-2", "2957b322090f9cb17ae72c71981c7218a28d81e0"),
    "siglip": ("google/siglip2-base-patch16-224", "5ffaac51d5e2f3367f7dab0cad4be4cb07c0caa2"),
}
# Model stages run on the T4; CPU tests override this to exercise the same code paths.
DEVICE = "cuda"
STAGES = (
    "prepare",
    "features",
    "classifier",
    "detector",
    "policy",
    "evaluate",
    "activity",
    "reload",
    "report",
)


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def csv_write(path, rows, fields=None):
    rows = list(rows)
    fields = fields or (list(rows[0]) if rows else [])
    with Path(path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def context(root):
    manifest, audit = read(root / "data_manifest.json"), read(root / "dataset_audit.json")
    exploratory = manifest.get("scope") == "exploratory_published_annotations"
    required = (
        ("rights", "original_images", "crop_lineage", "capacity")
        if exploratory
        else (
            "rights",
            "philippine_provenance",
            "original_images",
            "reference_annotations",
            "completeness",
            "crop_lineage",
            "independence",
            "capacity",
        )
    )
    if not audit.get("qualification", {}).get("passed") or any(
        audit.get("gates", {}).get(key, {}).get("status") != "passed" for key in required
    ):
        raise RuntimeError(
            "Dataset qualification is incomplete. Resolve the recorded audit gates before model execution."
        )
    if audit.get("manifest_sha256") != core.sha256(root / "data_manifest.json"):
        raise ValueError("Dataset audit does not bind the exact data manifest")
    if exploratory and audit.get("qualification", {}).get("scope") != "exploratory_published_annotations":
        raise ValueError("Exploratory scope must be explicitly recorded in the qualification")
    core.validate_manifest(manifest, require_capacity=not exploratory)
    if any(obj.get("ignore", False) for record in manifest["records"] for obj in record["objects"]):
        raise ValueError("This bounded count benchmark requires records without ignored target regions")
    return manifest, manifest["records"]


def active_objects(record):
    return [obj for obj in record["objects"] if not obj.get("ignore", False)]


def image(root, record):
    path = core.safe_path(root / "data", record["relative_path"])
    if path.stat().st_size != record["bytes"] or core.sha256(path) != record["sha256"]:
        raise ValueError(f"Image digest mismatch: {record['image_id']}")
    with Image.open(path) as source:
        if source.getexif().get(274, 1) != 1:
            raise ValueError("Normalize EXIF orientation and box coordinates during dataset freeze")
        result = source.convert("RGB")
    if result.size != (record["width"], record["height"]):
        raise ValueError("Decoded image dimensions disagree with annotation coordinates")
    return result


def crops(root, records):
    result, lineage = [], []
    for record in records:
        original = image(root, record)
        for obj in active_objects(record):
            coords = obj["bbox_xyxy"]
            effective = [
                math.floor(coords[0]),
                math.floor(coords[1]),
                math.ceil(coords[2]),
                math.ceil(coords[3]),
            ]
            result.append(original.crop(tuple(effective)))
            lineage.append(
                {
                    "crop_id": f"{record['image_id']}:{obj['object_id']}",
                    "parent_image_id": record["image_id"],
                    "parent_object_id": obj["object_id"],
                    "split": record["split"],
                    "capture_group_id": record["capture_group_id"],
                    "species": obj["species"],
                    "bbox_xyxy": coords,
                    "effective_crop_xyxy": effective,
                }
            )
    return result, lineage


def softmax(x):
    x = np.asarray(x, dtype=float)
    exp = np.exp(x - x.max(axis=1, keepdims=True))
    return exp / exp.sum(axis=1, keepdims=True)


def classification(y, scores):
    pred = np.asarray(scores).argmax(axis=1)
    cm = np.zeros((2, 2), dtype=int)
    np.add.at(cm, (np.asarray(y, dtype=int), pred), 1)
    recall = np.divide(cm.diagonal(), cm.sum(axis=1), out=np.zeros(2), where=cm.sum(axis=1) > 0)
    precision = np.divide(cm.diagonal(), cm.sum(axis=0), out=np.zeros(2), where=cm.sum(axis=0) > 0)
    f1 = np.divide(2 * precision * recall, precision + recall, out=np.zeros(2), where=precision + recall > 0)
    return {
        "accuracy": float(np.mean(np.asarray(y) == pred)),
        "macro_f1": float(f1.mean()),
        "confusion": cm.tolist(),
        "per_class": [
            {
                "species": s,
                "precision": float(precision[k]),
                "recall": float(recall[k]),
                "f1": float(f1[k]),
                "support": int(cm[k].sum()),
            }
            for k, s in enumerate(SPECIES)
        ],
    }


def torch_runtime():
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("Select a fresh Colab T4 GPU runtime for model execution")
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    return torch


def stage_model(root, key):
    from huggingface_hub import hf_hub_download

    manifest = read(root / "model_manifest.json")[key]
    if (manifest["modelId"], manifest["revision"]) != PINS[key]:
        raise ValueError("Model identity mismatch")
    folder = root / "weights" / key
    folder.mkdir(parents=True, exist_ok=True)
    for item in manifest["files"]:
        path = core.safe_path(folder, item["path"])
        if not path.exists():
            hf_hub_download(
                manifest["modelId"], item["path"], revision=manifest["revision"], local_dir=folder
            )
        if path.stat().st_size != item["bytes"] or core.sha256(path) != item["sha256"]:
            raise ValueError(f"Snapshot digest mismatch: {key}/{item['path']}")
    return folder


def bioclip(root):
    torch_runtime()
    from open_clip import image_transform
    from open_clip.model import CLIP
    from open_clip.tokenizer import SimpleTokenizer
    from safetensors.torch import load_file

    folder = stage_model(root, "bioclip")
    cfg = read(folder / "open_clip_config.json")
    mc, pc = cfg["model_cfg"], cfg["preprocess_cfg"]
    model = CLIP(
        embed_dim=mc["embed_dim"],
        vision_cfg=mc["vision_cfg"],
        text_cfg=mc["text_cfg"],
        quick_gelu=bool(mc.get("quick_gelu", False)),
    )
    model.load_state_dict(load_file(str(folder / "open_clip_model.safetensors")), strict=True)
    preprocess = image_transform(
        mc["vision_cfg"]["image_size"],
        is_train=False,
        mean=tuple(pc["mean"]),
        std=tuple(pc["std"]),
        resize_mode=pc.get("resize_mode", "shortest"),
        interpolation=pc.get("interpolation", "bicubic"),
    )
    return model.cuda().eval(), preprocess, SimpleTokenizer(context_length=mc["text_cfg"]["context_length"])


def embed(model, preprocess, images):
    import torch

    vectors = []
    with torch.inference_mode():
        for start in range(0, len(images), 8):
            pixels = torch.stack([preprocess(im) for im in images[start : start + 8]]).cuda()
            vectors.append(model.encode_image(pixels, normalize=True).float().cpu().numpy())
    return (
        np.concatenate(vectors)
        if vectors
        else np.empty((0, model.text_projection.shape[1]), dtype=np.float32)
    )


def prepare(root):
    _, records = context(root)
    _, lineage = crops(root, records)
    csv_write(
        root / "outputs" / "crop_lineage.csv",
        [{**x, "bbox_xyxy": json.dumps(x["bbox_xyxy"])} for x in lineage],
    )
    csv_write(
        root / "outputs" / "split_manifest.csv",
        [{k: r[k] for k in ("image_id", "capture_group_id", "split", "sha256")} for r in records],
    )
    write(root / "outputs" / "lineage.json", lineage)
    csv_write(
        root / "outputs" / "sample_summary.csv",
        [
            {
                "split": split,
                "images": sum(r["split"] == split for r in records),
                "source_families": len({r["capture_group_id"] for r in records if r["split"] == split}),
                **{
                    species: sum(x["split"] == split and x["species"] == species for x in lineage)
                    for species in SPECIES
                },
                "max_objects_per_image": max(
                    (len(active_objects(r)) for r in records if r["split"] == split), default=0
                ),
            }
            for split in ("train", "validation", "test")
        ],
    )
    audit = read(root / "dataset_audit.json")
    csv_write(
        root / "outputs" / "audit_gates.csv",
        [
            {"gate": name, "status": gate.get("status"), "evidence": gate.get("evidence", "see audit JSON")}
            for name, gate in audit.get("gates", {}).items()
        ],
    )
    write(
        root / "outputs" / "environment.json",
        {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "pid": os.getpid(),
            "packages": {
                name: importlib.metadata.version(name)
                for name in (
                    "torch",
                    "torchvision",
                    "transformers",
                    "open_clip_torch",
                    "numpy",
                    "scipy",
                    "Pillow",
                    "pycocotools",
                    "safetensors",
                )
            },
            "status": "Candidate; hosted end-to-end verification required",
        },
    )


def features(root):
    torch = torch_runtime()
    _, records = context(root)
    pictures, lineage = crops(root, records)
    prompts = ["a photo of a rice black bug.", "a photo of a white stemborer."]
    model, transform, tokenizer = bioclip(root)
    vectors = embed(model, transform, pictures)
    with torch.inference_mode():
        text = model.encode_text(tokenizer(prompts).cuda(), normalize=True).cpu().numpy()
    scale = float(model.logit_scale.exp().item())
    np.savez(
        root / "outputs" / "features.npz",
        images=vectors,
        text=text,
        scale=scale,
        zero_shot=softmax(scale * vectors @ text.T),
    )
    del model
    gc.collect()
    torch.cuda.empty_cache()
    from transformers import AutoModel, AutoProcessor

    folder = stage_model(root, "siglip")
    model = (
        AutoModel.from_pretrained(
            folder, local_files_only=True, trust_remote_code=False, use_safetensors=True
        )
        .float()
        .cuda()
        .eval()
    )
    processor = AutoProcessor.from_pretrained(folder, local_files_only=True, trust_remote_code=False)
    scores = []
    with torch.inference_mode():
        for start in range(0, len(pictures), 8):
            inputs = processor(
                text=prompts, images=pictures[start : start + 8], padding="max_length", return_tensors="pt"
            ).to("cuda")
            scores.extend(torch.sigmoid(model(**inputs).logits_per_image).cpu().numpy())
    np.save(root / "outputs" / "siglip.npy", np.asarray(scores))
    train = np.array([x["split"] == "train" for x in lineage])
    y = np.array([SPECIES.index(x["species"]) for x in lineage])
    rgb = np.array(
        [
            np.concatenate(
                [
                    np.asarray(ImageOps.fit(im, (64, 64)), dtype=float).mean(axis=(0, 1)),
                    np.asarray(ImageOps.fit(im, (64, 64)), dtype=float).std(axis=(0, 1)),
                ]
            )
            for im in pictures
        ]
    )
    centroids = np.stack([rgb[train & (y == k)].mean(axis=0) for k in range(2)])
    np.save(root / "outputs" / "rgb.npy", -np.linalg.norm(rgb[:, None] - centroids[None], axis=2))
    val = np.array([x["split"] == "validation" for x in lineage])
    majority = int(np.bincount(y[train], minlength=2).argmax())
    majority_scores = np.tile(np.eye(2)[majority], (len(y), 1))
    np.save(root / "outputs" / "majority.npy", majority_scores)
    write(
        root / "outputs" / "validation_crops.json",
        {
            key: classification(y[val], value[val])
            for key, value in {
                "bioclip": softmax(scale * vectors @ text.T),
                "siglip": np.asarray(scores),
                "rgb": -np.linalg.norm(rgb[:, None] - centroids[None], axis=2),
                "majority": majority_scores,
            }.items()
        },
    )


def fit_head(features, labels_array, text, scale, train, val, *, device="cpu", epochs=20):
    """Pure bounded fitter; CPU fixtures verify selection without model downloads."""
    import torch

    torch.manual_seed(42)
    train, val = np.asarray(train), np.asarray(val)
    if len(train) > 1000:
        train = np.sort(np.random.default_rng(42).choice(train, 1000, replace=False))
    x = torch.tensor(features, device=device, dtype=torch.float32)
    labels = torch.tensor(labels_array, device=device, dtype=torch.long)
    head = torch.nn.Linear(x.shape[1], 2).to(device)
    with torch.no_grad():
        head.weight.copy_(torch.tensor(text * scale, device=device))
        head.bias.zero_()
    optimizer = torch.optim.AdamW(head.parameters(), lr=1e-4, weight_decay=0.01)
    best_loss, best, history = float("inf"), None, []
    rng = np.random.default_rng(42)
    for epoch in range(epochs + 1):
        if epoch:
            head.train()
            order = rng.permutation(train)
            for start in range(0, len(order), 32):
                ix = order[start : start + 32]
                optimizer.zero_grad(set_to_none=True)
                loss = torch.nn.functional.cross_entropy(head(x[ix]), labels[ix])
                loss.backward()
                optimizer.step()
        head.eval()
        with torch.no_grad():
            loss_value = float(torch.nn.functional.cross_entropy(head(x[val]), labels[val]).item())
            train_value = float(torch.nn.functional.cross_entropy(head(x[train]), labels[train]).item())
        history.append(
            {"epoch": epoch, "validation_cross_entropy": loss_value, "training_cross_entropy": train_value}
        )
        if loss_value < best_loss:
            best_loss, selected = loss_value, epoch
            best = {key: value.detach().cpu().clone() for key, value in head.state_dict().items()}
    return best, selected, history


def classifier(root):
    torch_runtime()
    from safetensors.torch import save_file

    lineage = read(root / "outputs" / "lineage.json")
    data = np.load(root / "outputs" / "features.npz", allow_pickle=False)
    y = np.array([SPECIES.index(x["species"]) for x in lineage])
    train = np.flatnonzero([x["split"] == "train" for x in lineage])
    val = np.flatnonzero([x["split"] == "validation" for x in lineage])
    best, selected, history = fit_head(
        data["images"], y, data["text"], data["scale"], train, val, device="cuda"
    )
    selected_train = (
        np.sort(np.random.default_rng(42).choice(train, 1000, replace=False)) if len(train) > 1000 else train
    )
    write(
        root / "outputs" / "classifier_training_subset.json",
        {"seed": 42, "maximum_crops": 1000, "crop_ids": [lineage[int(i)]["crop_id"] for i in selected_train]},
    )
    folder = root / "outputs" / "classifier_adapter"
    folder.mkdir(exist_ok=True)
    save_file(best, str(folder / "head.safetensors"))
    write(
        folder / "manifest.json",
        {
            "format": "dimer-rice-frozen-head-v1",
            "classes": list(SPECIES),
            "base": list(PINS["bioclip"]),
            "preprocessing": "pinned-bioclIP-config-normalized-features",
            "sha256": core.sha256(folder / "head.safetensors"),
            "epoch": selected,
            "trainable_parameters": sum(v.numel() for v in best.values()),
        },
    )
    np.save(
        root / "outputs" / "adapted.npy",
        softmax(data["images"] @ best["weight"].numpy().T + best["bias"].numpy()),
    )
    write(root / "outputs" / "classifier_history.json", history)


def load_head(root):
    from safetensors.numpy import load_file

    folder = root / "outputs" / "classifier_adapter"
    meta = read(folder / "manifest.json")
    if (
        meta["format"] != "dimer-rice-frozen-head-v1"
        or meta["classes"] != list(SPECIES)
        or meta["base"] != list(PINS["bioclip"])
        or meta["preprocessing"] != "pinned-bioclIP-config-normalized-features"
    ):
        raise ValueError("Classifier adapter identity mismatch")
    if core.sha256(folder / "head.safetensors") != meta["sha256"]:
        raise ValueError("Classifier tensor digest mismatch")
    tensors = load_file(str(folder / "head.safetensors"))
    if (
        set(tensors) != {"weight", "bias"}
        or tensors["weight"].shape != (2, 768)
        or tensors["bias"].shape != (2,)
        or not all(v.dtype == np.float32 and np.isfinite(v).all() for v in tensors.values())
    ):
        raise ValueError("Classifier tensor contract mismatch")
    return tensors


def detr_model(root, adapted=False):
    from transformers import DetrForObjectDetection, DetrImageProcessor

    torch_runtime()
    folder = stage_model(root, "detr")
    processor = DetrImageProcessor.from_pretrained(folder, local_files_only=True, trust_remote_code=False)
    if dict(processor.size) != {"shortest_edge": 800, "longest_edge": 1333}:
        raise ValueError("Pinned DETR resizing differs from the frozen training recipe")
    model = (
        DetrForObjectDetection.from_pretrained(
            folder,
            num_labels=1,
            id2label={0: "target_pest"},
            label2id={"target_pest": 0},
            ignore_mismatched_sizes=True,
            use_pretrained_backbone=False,
            local_files_only=True,
            trust_remote_code=False,
            use_safetensors=True,
        )
        .float()
        .cuda()
    )
    if adapted:
        from safetensors.torch import load_file

        adapter = root / "outputs" / "detector_adapter"
        meta = read(adapter / "manifest.json")
        if (
            meta.get("format") != "dimer-rice-detr-adapter-v1"
            or meta.get("preprocessing") != dict(processor.size)
            or meta.get("frozen_prefix") != "model.backbone."
            or meta["base"] != list(PINS["detr"])
            or meta["classes"] != ["target_pest"]
            or meta["sha256"] != core.sha256(adapter / "adapter.safetensors")
        ):
            raise ValueError("Detector adapter identity mismatch")
        tensors = load_file(str(adapter / "adapter.safetensors"))
        import torch

        if any(not torch.isfinite(tensor).all() for tensor in tensors.values()):
            raise ValueError("Nonfinite detector adapter tensor")
        expected = {key for key in model.state_dict() if not key.startswith("model.backbone.")}
        if set(tensors) != expected:
            raise ValueError("Detector adapter tensor inventory mismatch")
        result = model.load_state_dict(tensors, strict=False)
        if result.unexpected_keys or any(not k.startswith("model.backbone.") for k in result.missing_keys):
            raise ValueError("Detector adapter shape mismatch")
    return model.eval(), processor


def validate_raw_detections(logits, pred_boxes, *, queries, classes):
    """Refuse malformed or nonfinite DETR tensors before any score threshold is applied.

    The pinned postprocessor keeps entries whose score exceeds the threshold, so a NaN
    score is silently dropped and an all-NaN image would look like a valid empty result.
    Returns the fewest queries the threshold-0 postprocessor may keep.
    """
    logits = np.asarray(logits, dtype=np.float64)
    pred_boxes = np.asarray(pred_boxes, dtype=np.float64)
    if logits.shape != (1, queries, classes + 1) or pred_boxes.shape != (1, queries, 4):
        raise ValueError(
            f"Malformed DETR output: logits {logits.shape}, boxes {pred_boxes.shape}; "
            f"expected (1, {queries}, {classes + 1}) and (1, {queries}, 4)"
        )
    if not (np.isfinite(logits).all() and np.isfinite(pred_boxes).all()):
        raise ValueError("Nonfinite DETR logits or boxes; refusing to treat them as zero detections")
    if (pred_boxes < 0).any() or (pred_boxes > 1).any():
        raise ValueError("DETR normalised boxes fall outside [0, 1]")
    # Explicit filtering policy: threshold 0 may drop only queries whose target-class
    # probability underflows in float32; anything above 1e-30 must be retained.
    return int((softmax(logits[0])[:, :-1].max(axis=1) > 1e-30).sum())


def validate_postprocessed(scores, boxes, labels, minimum, queries):
    scores, boxes, labels = np.asarray(scores), np.asarray(boxes), np.asarray(labels)
    kept = len(scores)
    if (
        not minimum <= kept <= queries
        or scores.shape != (kept,)
        or boxes.shape != (kept, 4)
        or labels.shape != (kept,)
    ):
        raise ValueError(
            f"Postprocessed detections {scores.shape}/{boxes.shape} disagree with "
            f"{minimum}-{queries} valid queries"
        )
    if not (np.isfinite(scores).all() and np.isfinite(boxes).all()):
        raise ValueError("Nonfinite postprocessed detections")
    if ((scores < 0) | (scores > 1)).any() or (labels != 0).any():
        raise ValueError("Postprocessed detection scores or labels outside the declared domain")


def validate_species_scores(probabilities, rows):
    probabilities = np.asarray(probabilities, dtype=np.float64)
    if probabilities.shape != (rows, len(SPECIES)):
        raise ValueError(f"Species scores {probabilities.shape} disagree with {rows} detections")
    if not np.isfinite(probabilities).all():
        raise ValueError("Nonfinite species scores; refusing to count these detections")
    if ((probabilities < 0) | (probabilities > 1)).any() or not np.allclose(
        probabilities.sum(axis=1), 1, atol=1e-4
    ):
        raise ValueError("Species scores are not a probability distribution")


def detections(root, model, processor, records):
    import torch

    result = []
    with torch.inference_mode():
        for record in records:
            original = image(root, record)
            inputs = processor(images=original, return_tensors="pt").to(DEVICE)
            output = model(**inputs)
            minimum = validate_raw_detections(
                output.logits.float().cpu().numpy(),
                output.pred_boxes.float().cpu().numpy(),
                queries=model.config.num_queries,
                classes=model.config.num_labels,
            )
            post = processor.post_process_object_detection(
                output,
                threshold=0.0,
                target_sizes=torch.tensor([[original.height, original.width]], device=DEVICE),
            )[0]
            scores, boxes = post["scores"].cpu().numpy(), post["boxes"].cpu().numpy()
            validate_postprocessed(
                scores, boxes, post["labels"].cpu().numpy(), minimum, model.config.num_queries
            )
            # Preserve all ranked queries; thresholds applied later, no NMS.
            order = np.argsort(-scores, kind="stable")
            result.append(
                {
                    "image_id": record["image_id"],
                    "boxes": boxes[order].tolist(),
                    "scores": scores[order].tolist(),
                }
            )
    return result


def coco_metrics(records, predictions):
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    if not sum(len(active_objects(r)) for r in records):
        return {"ap50": None, "ap": None, "reason": "No reference targets"}
    annotations, predicted, images = [], [], []
    for index, (record, prediction) in enumerate(zip(records, predictions, strict=True)):
        images.append({"id": index, "width": record["width"], "height": record["height"]})
        for obj in record["objects"]:
            x1, y1, x2, y2 = obj["bbox_xyxy"]
            annotations.append(
                {
                    "id": len(annotations) + 1,
                    "image_id": index,
                    "category_id": 1,
                    "bbox": [x1, y1, x2 - x1, y2 - y1],
                    "area": (x2 - x1) * (y2 - y1),
                    "iscrowd": int(obj.get("ignore", False)),
                }
            )
        for box, score in zip(prediction["boxes"], prediction["scores"], strict=True):
            x1, y1, x2, y2 = box
            predicted.append(
                {"image_id": index, "category_id": 1, "bbox": [x1, y1, x2 - x1, y2 - y1], "score": score}
            )
    with contextlib.redirect_stdout(io.StringIO()):
        truth = COCO()
        truth.dataset = {
            "images": images,
            "annotations": annotations,
            "categories": [{"id": 1, "name": "target_pest"}],
            "info": {},
        }
        truth.createIndex()
        if predicted:
            detected = truth.loadRes(predicted)
        else:
            detected = COCO()
            detected.dataset = {**truth.dataset, "annotations": []}
            detected.createIndex()
        evaluator = COCOeval(truth, detected, "bbox")
        evaluator.params.maxDets = [1, 10, 100]
        evaluator.evaluate()
        evaluator.accumulate()
        evaluator.summarize()
    return {
        "ap": float(evaluator.stats[0]),
        "ap50": float(evaluator.stats[1]),
        "max_detections": 100,
        "ignore_semantics": "Explicit ignored regions represented as COCO crowd regions",
    }


def detector(root):
    torch = torch_runtime()
    from safetensors.torch import save_file

    _, records = context(root)
    train = [r for r in records if r["split"] == "train"]
    val = [r for r in records if r["split"] == "validation"]
    model, processor = detr_model(root)
    for name, parameter in model.named_parameters():
        parameter.requires_grad = not name.startswith("model.backbone.")
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=1e-4, weight_decay=1e-4)
    folder = root / "outputs" / "detector_adapter"
    folder.mkdir(exist_ok=True)
    best, history = -1.0, []
    rng = np.random.default_rng(42)
    for epoch in range(16):
        losses = []
        if epoch:
            model.train()
            # Frozen convolutional statistics must not update during adaptation.
            model.model.backbone.eval()
            order = rng.permutation(len(train))
            batches = [order[start : start + 2] for start in range(0, len(order), 2)]
            for window_start in range(0, len(batches), 4):
                window = batches[window_start : window_start + 4]
                optimizer.zero_grad(set_to_none=True)
                for batch_ix in window:
                    batch = [train[int(i)] for i in batch_ix]
                    annotations = []
                    for k, record in enumerate(batch):
                        objects = []
                        for obj in active_objects(record):
                            x1, y1, x2, y2 = obj["bbox_xyxy"]
                            objects.append(
                                {
                                    "image_id": k,
                                    "category_id": 0,
                                    "bbox": [x1, y1, x2 - x1, y2 - y1],
                                    "area": (x2 - x1) * (y2 - y1),
                                    "iscrowd": 0,
                                }
                            )
                        annotations.append({"image_id": k, "annotations": objects})
                    encoded = processor(
                        images=[image(root, r) for r in batch], annotations=annotations, return_tensors="pt"
                    )
                    labels = [{k: v.cuda() for k, v in target.items()} for target in encoded["labels"]]
                    loss = model(
                        pixel_values=encoded["pixel_values"].cuda(),
                        pixel_mask=encoded["pixel_mask"].cuda(),
                        labels=labels,
                    ).loss
                    (loss / len(window)).backward()
                    losses.append(float(loss.detach().cpu()))
                torch.nn.utils.clip_grad_norm_(params, 0.1)
                optimizer.step()
        model.eval()
        measured = coco_metrics(val, detections(root, model, processor, val))
        ap50 = measured["ap50"]
        if ap50 is None:
            raise ValueError("Validation detection AP undefined")
        history.append(
            {
                "epoch": epoch,
                "validation_ap50": ap50,
                "training_loss": float(np.mean(losses)) if losses else None,
            }
        )
        if ap50 > best:
            best, selected = ap50, epoch
            tensors = {
                key: value.detach().cpu().contiguous().clone()
                for key, value in model.state_dict().items()
                if not key.startswith("model.backbone.")
            }
            save_file(tensors, str(folder / "adapter.safetensors"))
    write(
        folder / "manifest.json",
        {
            "format": "dimer-rice-detr-adapter-v1",
            "base": list(PINS["detr"]),
            "classes": ["target_pest"],
            "sha256": core.sha256(folder / "adapter.safetensors"),
            "epoch": selected,
            "preprocessing": dict(processor.size),
            "frozen_prefix": "model.backbone.",
            "trainable_parameters": sum(p.numel() for p in params),
        },
    )
    write(root / "outputs" / "detector_history.json", history)
    del model
    gc.collect()
    torch.cuda.empty_cache()


def count_reference(records):
    return np.array(
        [[sum(o["species"] == s for o in active_objects(r)) for s in SPECIES] for r in records], dtype=int
    )


def compose(root, records):
    """Infer full images sequentially; reference annotations are not model inputs."""
    torch = torch_runtime()
    model, processor = detr_model(root, adapted=True)
    predicted = detections(root, model, processor, records)
    del model, processor
    gc.collect()
    torch.cuda.empty_cache()
    model, transform, tokenizer = bioclip(root)
    head = load_head(root)
    with torch.inference_mode():
        text = (
            model.encode_text(
                tokenizer(["a photo of a rice black bug.", "a photo of a white stemborer."]).to(DEVICE),
                normalize=True,
            )
            .cpu()
            .numpy()
        )
    scale = float(model.logit_scale.exp().item())
    if not (np.isfinite(text).all() and math.isfinite(scale)):
        raise ValueError("Nonfinite BioCLIP text embeddings or logit scale")
    for record, output in zip(records, predicted, strict=True):
        original = image(root, record)
        # Clip only predicted crop extraction, retaining original predicted boxes
        # for evaluation. Reference invalid boxes are refused at qualification.
        crop_images = []
        for box in output["boxes"]:
            x1, y1, x2, y2 = box
            bounded = (
                math.floor(max(0, min(original.width - 1, x1))),
                math.floor(max(0, min(original.height - 1, y1))),
                math.ceil(max(1, min(original.width, x2))),
                math.ceil(max(1, min(original.height, y2))),
            )
            if bounded[2] <= bounded[0] or bounded[3] <= bounded[1]:
                raise ValueError("Invalid predicted crop extent")
            crop_images.append(original.crop(bounded))
        vectors = embed(model, transform, crop_images)
        if vectors.shape != (len(crop_images), head["weight"].shape[1]) or not np.isfinite(vectors).all():
            raise ValueError(f"Invalid BioCLIP crop embeddings for {record['image_id']}")
        adapted = softmax(vectors @ head["weight"].T + head["bias"])
        zero_shot = softmax(scale * vectors @ text.T)
        validate_species_scores(adapted, len(crop_images))
        validate_species_scores(zero_shot, len(crop_images))
        output["adapted"], output["zero_shot"] = adapted.tolist(), zero_shot.tolist()
    return predicted


def count_predictions(predictions, threshold, field="adapted", majority=None):
    counts = np.zeros((len(predictions), 2), dtype=int)
    for index, row in enumerate(predictions):
        for score, species_scores in zip(row["scores"], row[field], strict=True):
            if score >= threshold:
                label = int(np.argmax(species_scores)) if majority is None else majority
                counts[index, label] += 1
    return counts


def review_policy(y, probabilities):
    probabilities = np.asarray(probabilities)
    margin = np.abs(probabilities[:, 0] - probabilities[:, 1])
    rows = []
    for threshold in sorted(set([0.0, *margin.tolist()])):
        accepted = margin >= threshold
        coverage = float(accepted.mean())
        accuracy = (
            float(np.mean(probabilities.argmax(axis=1)[accepted] == np.asarray(y)[accepted]))
            if accepted.any()
            else None
        )
        rows.append({"threshold": threshold, "coverage": coverage, "accuracy": accuracy})
    # Specification section 7, decision 5: target >=80% selective accuracy at >=50% coverage;
    # if unattainable, maximise accuracy, then coverage, then margin over every nonempty set.
    target = [r for r in rows if r["coverage"] >= 0.5 and r["accuracy"] >= 0.8]
    candidates = target or [r for r in rows if r["accuracy"] is not None]
    if not candidates:
        raise ValueError("Review policy needs at least one validation crop")
    selected = (
        max(target, key=lambda r: (r["coverage"], r["threshold"]))
        if target
        else max(candidates, key=lambda r: (r["accuracy"], r["coverage"], r["threshold"]))
    )
    rule = (
        "target met: highest coverage with >=80% selective accuracy and >=50% coverage"
        if target
        else "fallback: target unattainable; maximise accuracy, then coverage, then margin"
    )
    return {**selected, "target_met": bool(target), "selection_rule": rule, "curve": rows}


def policy(root):
    manifest, records = context(root)
    val = [r for r in records if r["split"] == "validation"]
    predictions = compose(root, val)
    write(root / "outputs" / "validation_detections.json", predictions)
    rows = [
        {
            "threshold": i / 20,
            **core.count_metrics(count_reference(val), count_predictions(predictions, i / 20)),
        }
        for i in range(1, 20)
    ]
    selected = core.select_count_threshold(rows)
    lineage = read(root / "outputs" / "lineage.json")
    ix = np.flatnonzero([r["split"] == "validation" for r in lineage])
    y = np.array([SPECIES.index(r["species"]) for r in lineage])[ix]
    review = review_policy(y, np.load(root / "outputs" / "adapted.npy", allow_pickle=False)[ix])
    result = {
        "scope": manifest.get("scope"),
        "detector_threshold": selected["threshold"],
        "review_margin": review["threshold"],
        "review_target_met": review["target_met"],
        "review_selection_rule": review["selection_rule"],
        "review_validation_coverage": review["coverage"],
        "review_validation_accuracy": review["accuracy"],
        "selection_split": "validation",
        "selected_at_unix": time.time(),
        "classifier_sha256": core.sha256(root / "outputs" / "classifier_adapter" / "head.safetensors"),
        "detector_sha256": core.sha256(root / "outputs" / "detector_adapter" / "adapter.safetensors"),
        "data_sha256": core.sha256(root / "data_manifest.json"),
        "validation_threshold_grid": rows,
        "review_validation": review,
    }
    result["digest"] = core.digest_json(result)
    write(root / "outputs" / "selected_policy.json", result)


def locked_policy(root):
    policy = read(root / "outputs" / "selected_policy.json")
    digest = policy.pop("digest")
    if core.digest_json(policy) != digest or policy["selection_split"] != "validation":
        raise ValueError("Policy lock changed")
    for key, path in (
        ("classifier_sha256", "outputs/classifier_adapter/head.safetensors"),
        ("detector_sha256", "outputs/detector_adapter/adapter.safetensors"),
        ("data_sha256", "data_manifest.json"),
    ):
        if policy[key] != core.sha256(root / path):
            raise ValueError("Selected checkpoint/data changed after policy lock")
    return {**policy, "digest": digest}


def oracle_counts(records, lineage, probabilities):
    by_id = {row["image_id"]: i for i, row in enumerate(records)}
    counts = np.zeros((len(records), 2), dtype=int)
    for row, scores in zip(lineage, probabilities, strict=True):
        if row["parent_image_id"] in by_id:
            counts[by_id[row["parent_image_id"]], int(np.argmax(scores))] += 1
    return counts


def chosen_objects(prediction, policy):
    result = []
    for k, (box, detection_score, scores) in enumerate(
        zip(prediction["boxes"], prediction["scores"], prediction["adapted"], strict=True)
    ):
        if detection_score < policy["detector_threshold"]:
            continue
        margin = abs(scores[0] - scores[1])
        result.append(
            {
                "query_rank": k,
                "bbox_xyxy": box,
                "detector_score": detection_score,
                "species": SPECIES[int(np.argmax(scores))],
                "species_scores": scores,
                "margin": margin,
                "review": margin < policy["review_margin"],
            }
        )
    return result


CHART_FIGURES = ("classifier.png", "detector.png", "evaluate.png", "activity.png")
SHORT = {"rice_black_bug": "RBB", "white_stemborer": "WSB"}
OUTCOMES = ("correct", "wrong_species", "missed", "spurious")


def object_outcomes(record, objects):
    """One row per reference or retained prediction: correct, wrong_species, missed or spurious."""
    references = active_objects(record)
    err = core.error_decomposition(references, objects)
    rows = []

    def row(outcome, reference, prediction, iou):
        scores = prediction["species_scores"] if prediction else None
        return {
            "image_id": record["image_id"],
            "outcome": outcome,
            "reference_species": reference["species"] if reference else "",
            "predicted_species": prediction["species"] if prediction else "",
            "detector_score": round(prediction["detector_score"], 4) if prediction else "",
            "species_score": round(max(scores), 4) if prediction else "",
            "species_margin": round(prediction["margin"], 4) if prediction else "",
            "review": ("REVIEW" if prediction["review"] else "accepted") if prediction else "",
            "iou": round(iou, 3) if iou is not None else "",
            "reference_box": reference["bbox_xyxy"] if reference else None,
            "predicted_box": prediction["bbox_xyxy"] if prediction else None,
        }

    for r, q, iou in err["matches"]:
        same = references[r]["species"] == objects[q]["species"]
        rows.append(row("correct" if same else "wrong_species", references[r], objects[q], iou))
    rows += [row("missed", references[r], None, None) for r in err["misses"]]
    rows += [row("spurious", None, objects[q], None) for q in err["spurious"]]
    return rows


def species_reconciliation(image_id, rows, objects):
    """Per-species identities: reference = correct + wrong_out + missed;
    raw = correct + wrong_in + spurious."""
    result = []
    for species in SPECIES:
        entry = {
            "image_id": image_id,
            "species": species,
            "reference": sum(r["reference_species"] == species for r in rows),
            "raw": sum(o["species"] == species for o in objects),
            "accepted": sum(o["species"] == species and not o["review"] for o in objects),
            "referred": sum(o["species"] == species and o["review"] for o in objects),
            "correct": sum(r["outcome"] == "correct" and r["reference_species"] == species for r in rows),
            "wrong_in": sum(
                r["outcome"] == "wrong_species" and r["predicted_species"] == species for r in rows
            ),
            "wrong_out": sum(
                r["outcome"] == "wrong_species" and r["reference_species"] == species for r in rows
            ),
            "missed": sum(r["outcome"] == "missed" and r["reference_species"] == species for r in rows),
            "spurious": sum(r["outcome"] == "spurious" and r["predicted_species"] == species for r in rows),
        }
        if (
            entry["reference"] != entry["correct"] + entry["wrong_out"] + entry["missed"]
            or entry["raw"] != entry["correct"] + entry["wrong_in"] + entry["spurious"]
        ):
            raise ValueError("Error decomposition does not reconcile with counts")
        result.append(entry)
    return result


def error_summaries(root, records, predictions, policy_value):
    """Test-set decomposition learners read beside the panels: per-species counts and confusion."""
    totals = {
        s: dict.fromkeys(
            (
                "reference",
                "raw",
                "accepted",
                "referred",
                "correct",
                "wrong_in",
                "wrong_out",
                "missed",
                "spurious",
            ),
            0,
        )
        for s in SPECIES
    }
    confusion = np.zeros((3, 3), dtype=int)  # rows: reference RBB/WSB/none; cols: predicted RBB/WSB/none
    for record, prediction in zip(records, predictions, strict=True):
        objects = chosen_objects(prediction, policy_value)
        rows = object_outcomes(record, objects)
        for entry in species_reconciliation(record["image_id"], rows, objects):
            for key in totals[entry["species"]]:
                totals[entry["species"]][key] += entry[key]
        for r in rows:
            ref = SPECIES.index(r["reference_species"]) if r["reference_species"] else 2
            pred = SPECIES.index(r["predicted_species"]) if r["predicted_species"] else 2
            confusion[ref, pred] += 1
    summary = [{"species": s, **totals[s]} for s in SPECIES]
    summary.append({"species": "all", **{k: sum(t[k] for t in totals.values()) for k in totals[SPECIES[0]]}})
    csv_write(root / "outputs" / "error_summary.csv", summary)
    labels = [*SPECIES, "no match"]
    csv_write(
        root / "outputs" / "matched_species_confusion.csv",
        [
            {
                "reference": ("reference " + labels[i]) if i < 2 else "no reference (spurious)",
                **{
                    f"predicted {labels[j]}" if j < 2 else "not detected (missed)": int(confusion[i, j])
                    for j in range(3)
                },
            }
            for i in range(3)
        ],
    )


def evaluate(root):
    manifest, records = context(root)
    policy_value = locked_policy(root)
    test = [r for r in records if r["split"] == "test"]
    predictions = compose(root, test)
    write(root / "outputs" / "test_detections.json", predictions)
    lineage = read(root / "outputs" / "lineage.json")
    features_data = np.load(root / "outputs" / "features.npz", allow_pickle=False)
    adapted = np.load(root / "outputs" / "adapted.npy", allow_pickle=False)
    train = [r for r in records if r["split"] == "train"]
    majority = int(count_reference(train).sum(axis=0).argmax())
    baseline = np.tile(count_reference(train).mean(axis=0), (len(test), 1))
    systems = {
        "training_mean": baseline,
        "zero_count": np.zeros_like(baseline),
        "oracle_bioclip": oracle_counts(test, lineage, features_data["zero_shot"]),
        "oracle_adapted": oracle_counts(test, lineage, adapted),
        "detector_majority": count_predictions(
            predictions, policy_value["detector_threshold"], majority=majority
        ),
        "detector_bioclip": count_predictions(
            predictions, policy_value["detector_threshold"], field="zero_shot"
        ),
        "detector_adapted": count_predictions(predictions, policy_value["detector_threshold"]),
    }
    truth = count_reference(test)
    metrics = {name: core.count_metrics(truth, counts) for name, counts in systems.items()}
    bootstrap = core.paired_group_bootstrap(
        truth, systems["detector_adapted"], baseline, [r["capture_group_id"] for r in test]
    )
    bootstrap["group_interpretation"] = (
        "source-image/duplicate groups; capture independence unverified"
        if manifest.get("scope") == "exploratory_published_annotations"
        else "verified capture groups"
    )
    write(root / "outputs" / "metrics.json", metrics)
    write(root / "outputs" / "bootstrap.json", bootstrap)
    csv_write(
        root / "outputs" / "metrics.csv",
        [
            {"system": name, **{k: v for k, v in value.items() if not isinstance(v, list)}}
            for name, value in metrics.items()
        ],
    )
    crop_y = np.array([SPECIES.index(r["species"]) for r in lineage])
    ix = np.flatnonzero([r["split"] == "test" for r in lineage])
    crop_methods = {
        "bioclip": features_data["zero_shot"],
        "adapted": adapted,
        "siglip": np.load(root / "outputs" / "siglip.npy", allow_pickle=False),
        "rgb": np.load(root / "outputs" / "rgb.npy", allow_pickle=False),
        "majority": np.load(root / "outputs" / "majority.npy", allow_pickle=False),
    }
    write(
        root / "outputs" / "test_crop_metrics.json",
        {key: classification(crop_y[ix], score[ix]) for key, score in crop_methods.items()},
    )
    csv_write(
        root / "outputs" / "crop_predictions.csv",
        [
            {
                "crop_id": lineage[i]["crop_id"],
                "system": key,
                "reference": SPECIES[crop_y[i]],
                "predicted": SPECIES[int(score[i].argmax())],
                "score_rbb": float(score[i, 0]),
                "score_wsb": float(score[i, 1]),
            }
            for key, score in crop_methods.items()
            for i in ix
        ],
    )
    decomposition, counts_rows, detection_rows = [], [], []
    for index, (record, prediction) in enumerate(zip(test, predictions, strict=True)):
        objects = chosen_objects(prediction, policy_value)
        errors = core.error_decomposition(record["objects"], objects)
        decomposition.append(
            {
                "image_id": record["image_id"],
                "missed": errors["missed"],
                "spurious": errors["spurious_count"],
                "wrong_species": errors["wrong_species"],
                "matches": len(errors["matches"]),
                "matched_confusion": json.dumps(errors["matched_confusion"]),
            }
        )
        for s, species in enumerate(SPECIES):
            counts_rows.append(
                {
                    "image_id": record["image_id"],
                    "capture_group_id": record["capture_group_id"],
                    "species": species,
                    "reference": int(truth[index, s]),
                    "raw": int(systems["detector_adapted"][index, s]),
                    "accepted": sum(o["species"] == species and not o["review"] for o in objects),
                    "unresolved_total": sum(o["review"] for o in objects),
                    "training_mean": float(baseline[index, s]),
                }
            )
        detection_rows.append({"image_id": record["image_id"], "detections": objects})
    csv_write(root / "outputs" / "counts.csv", counts_rows)
    csv_write(root / "outputs" / "error_decomposition.csv", decomposition)
    with (root / "outputs" / "detections.jsonl").open("w", encoding="utf-8") as handle:
        for row in detection_rows:
            handle.write(json.dumps(row) + "\n")
    detection_metrics = coco_metrics(test, predictions)
    matched = sum(x["matches"] for x in decomposition)
    misses = sum(x["missed"] for x in decomposition)
    spurious = sum(x["spurious"] for x in decomposition)
    detection_metrics.update(
        precision=matched / (matched + spurious) if matched + spurious else None,
        recall=matched / (matched + misses) if matched + misses else None,
    )
    write(root / "outputs" / "detection_metrics.json", detection_metrics)
    error_summaries(root, test, predictions, policy_value)
    write(
        root / "outputs" / "reload_expected.json",
        {"pid": os.getpid(), "policy_digest": policy_value["digest"], "predictions": predictions[:2]},
    )
    panels(root, test, predictions, policy_value)


PANEL_STYLE = {
    "correct": ("#1a9850", "-"),
    "wrong_species": ("#d01c8b", "-"),
    "spurious": ("#ff7f00", "-"),
    "missed": ("#00bfff", "--"),
}


def panels(root, records, predictions, policy_value):
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Rectangle

    folder = root / "outputs" / "local_figures"
    folder.mkdir(exist_ok=True)
    selected, found = [], {}
    for record, prediction in zip(records, predictions, strict=True):
        err = core.error_decomposition(record["objects"], chosen_objects(prediction, policy_value))
        categories = {
            "misses": err["missed"] > 0,
            "spurious": err["spurious_count"] > 0,
            "wrong_species": err["wrong_species"] > 0,
            "success": err["missed"] + err["spurious_count"] + err["wrong_species"] == 0,
        }
        for category, present in categories.items():
            if present and category not in found:
                found[category] = record["image_id"]
                if record["image_id"] not in [r[0]["image_id"] for r in selected]:
                    selected.append((record, prediction))
    absent = sorted(set(("misses", "spurious", "wrong_species", "success")) - set(found))
    write(
        root / "outputs" / "error_panel_inventory.json",
        {
            "selected": found,
            "absent_categories": absent,
            "note": "An absent category was not found in the held-out split; none is manufactured.",
        },
    )
    example_rows, example_counts = [], []
    for record, prediction in selected:
        objects = chosen_objects(prediction, policy_value)
        rows = object_outcomes(record, objects)
        reconciled = species_reconciliation(record["image_id"], rows, objects)
        example_counts += reconciled
        shown_for = [c for c, image_id in found.items() if image_id == record["image_id"]]
        fig, ax = plt.subplots(figsize=(10, 8.5))
        ax.imshow(image(root, record))
        tag = 0
        # Crowded trays get compact #tags; the table below carries the full labels and scores.
        compact = sum(r["outcome"] != "correct" for r in rows) > 12
        for row in sorted(rows, key=lambda r: OUTCOMES.index(r["outcome"])):
            colour, style = PANEL_STYLE[row["outcome"]]
            box = row["predicted_box"] or row["reference_box"]
            x1, y1, x2, y2 = box
            ax.add_patch(
                Rectangle(
                    (x1, y1),
                    x2 - x1,
                    y2 - y1,
                    fill=False,
                    edgecolor=colour,
                    linestyle=style,
                    linewidth=1.6 if row["outcome"] != "correct" else 0.9,
                )
            )
            if row["outcome"] == "correct":
                continue
            tag += 1
            row["tag"] = tag
            if row["outcome"] == "wrong_species":
                rx1, ry1, rx2, ry2 = row["reference_box"]
                ax.add_patch(
                    Rectangle(
                        (rx1, ry1),
                        rx2 - rx1,
                        ry2 - ry1,
                        fill=False,
                        edgecolor="white",
                        linestyle="--",
                        linewidth=0.8,
                    )
                )
                text = (
                    f"#{tag} ref {SHORT[row['reference_species']]} → pred {SHORT[row['predicted_species']]}"
                )
            elif row["outcome"] == "missed":
                text = f"#{tag} missed ref {SHORT[row['reference_species']]}"
            else:
                text = f"#{tag} extra pred {SHORT[row['predicted_species']]}"
            if row["review"] == "REVIEW":
                text += " REVIEW"
            if compact:
                ax.text(x1, y1, f"#{tag}", fontsize=5, color="black", backgroundcolor=colour, va="top")
            else:
                ax.text(x1, y1 - 2, text, fontsize=6, color="black", backgroundcolor=colour, va="bottom")
        for row in rows:
            if row["outcome"] != "correct":
                example_rows.append({"panel_tag": row["tag"], **{k: v for k, v in row.items() if k != "tag"}})
        totals = {k: sum(e[k] for e in reconciled) for k in ("reference", "raw", "accepted", "referred")}
        counted = {o: sum(r["outcome"] == o for r in rows) for o in OUTCOMES}
        ax.set_title(
            f"Test {record['image_id']} · example of: {', '.join(shown_for)}\n"
            f"reference {totals['reference']} · raw predicted {totals['raw']} · accepted {totals['accepted']}"
            f" · referred {totals['referred']} | correct {counted['correct']} · wrong species "
            f"{counted['wrong_species']} · missed {counted['missed']} · spurious {counted['spurious']}\n"
            + (
                "Crowded image: #tags only; see the error-example table for species and scores\n"
                if compact
                else ""
            )
            + record["attribution"],
            fontsize=8,
        )
        handles = [
            Line2D([], [], color=PANEL_STYLE["correct"][0], label="correct species (matched)"),
            Line2D(
                [],
                [],
                color=PANEL_STYLE["wrong_species"][0],
                label="wrong species (matched; white = reference)",
            ),
            Line2D([], [], color=PANEL_STYLE["missed"][0], linestyle="--", label="missed reference box"),
            Line2D([], [], color=PANEL_STYLE["spurious"][0], label="spurious prediction (no reference)"),
        ]
        ax.legend(
            handles=handles,
            loc="upper center",
            bbox_to_anchor=(0.5, -0.01),
            ncol=2,
            fontsize=7,
            title="RBB = rice black bug · WSB = white stemborer · scores per #tag in the table below",
            title_fontsize=7,
            frameon=False,
        )
        ax.axis("off")
        fig.savefig(folder / f"{record['image_id']}.png", dpi=120, bbox_inches="tight")
        plt.close(fig)
    fields = [
        "panel_tag",
        "image_id",
        "outcome",
        "reference_species",
        "predicted_species",
        "detector_score",
        "species_score",
        "species_margin",
        "review",
        "iou",
    ]
    csv_write(
        root / "outputs" / "error_examples.csv", [{k: r[k] for k in fields} for r in example_rows], fields
    )
    csv_write(
        root / "outputs" / "error_example_counts.csv",
        example_counts,
        list(species_reconciliation("", [], [])[0]),
    )


# The validation threshold grid is i/20 for i = 1..19; display thresholds stay inside it,
# because a threshold of 1.0 drops every box and only reproduces the zero-count baseline.
DISPLAY_THRESHOLD_BOUNDS = (0.05, 0.95)
REVIEW_ILLUSTRATION_COVERAGE = (0.9, 0.8)


def display_thresholds(locked):
    low, high = DISPLAY_THRESHOLD_BOUNDS
    return sorted({round(max(low, locked - 0.1), 6), locked, round(min(high, locked + 0.1), 6)})


def review_margin_rows(test, predictions, policy_value):
    """Display-only: what stricter review margins would refer, next to the locked margin.

    Illustrative margins come from the validation curve (the margin whose validation coverage is
    closest to, without exceeding, 90% and 80%). Test detections are counted at the locked
    detector threshold; the locked review margin and counts are not changed.
    """
    review = policy_value.get("review_validation") or {}
    curve = [r for r in review.get("curve", []) if r.get("accuracy") is not None]
    locked_margin = policy_value["review_margin"]
    margins = [(locked_margin, "canonical", review.get("coverage"), review.get("accuracy"))]
    for target in REVIEW_ILLUSTRATION_COVERAGE:
        eligible = [r for r in curve if r["coverage"] <= target]
        if not eligible:
            continue
        row = max(eligible, key=lambda r: (r["coverage"], -r["threshold"]))
        if all(abs(row["threshold"] - m[0]) > 1e-12 for m in margins):
            role = f"illustrative ~{target:.0%} coverage"
            margins.append((row["threshold"], role, row["coverage"], row["accuracy"]))
    rows = []
    for margin, role, coverage, accuracy in margins:
        referred = {"correct": 0, "wrong_species": 0, "spurious": 0}
        retained = 0
        for record, prediction in zip(test, predictions, strict=True):
            objects = chosen_objects(prediction, {**policy_value, "review_margin": margin})
            retained += len(objects)
            for outcome in object_outcomes(record, objects):
                if outcome["review"] == "REVIEW":
                    referred[outcome["outcome"]] += 1
        rows.append(
            {
                "review_margin": round(margin, 6),
                "role": role,
                "validation_coverage": "" if coverage is None else round(coverage, 4),
                "validation_selective_accuracy": "" if accuracy is None else round(accuracy, 4),
                "test_retained": retained,
                "test_referred": sum(referred.values()),
                "referred_correct": referred["correct"],
                "referred_wrong_species": referred["wrong_species"],
                "referred_spurious": referred["spurious"],
                "note": "Retrospective test illustration; locked review margin unchanged",
            }
        )
    return rows


def activity(root):
    _, records = context(root)
    test = [r for r in records if r["split"] == "test"]
    canonical = core.sha256(root / "outputs" / "test_detections.json")
    policy_value = locked_policy(root)
    predictions = read(root / "outputs" / "test_detections.json")
    locked = policy_value["detector_threshold"]
    thresholds = display_thresholds(locked)
    role = {t: "canonical" if t == locked else ("lower" if t < locked else "higher") for t in thresholds}
    rows, per_image = [], {}
    for threshold in thresholds:
        count_result = core.count_metrics(count_reference(test), count_predictions(predictions, threshold))
        errors = [
            core.error_decomposition(
                r["objects"], chosen_objects(p, {**policy_value, "detector_threshold": threshold})
            )
            for r, p in zip(test, predictions, strict=True)
        ]
        per_image[threshold] = count_predictions(predictions, threshold)
        rows.append(
            {
                "display_threshold": threshold,
                "role": role[threshold],
                "mae": count_result["mae"],
                "missed": sum(e["missed"] for e in errors),
                "spurious": sum(e["spurious_count"] for e in errors),
                "wrong_species": sum(e["wrong_species"] for e in errors),
                "note": "Retrospective test illustration; canonical policy unchanged",
            }
        )
    csv_write(root / "outputs" / "activity_thresholds.csv", rows)
    # Paired view: the held-out image whose raw count moves most across the three
    # settings (earliest on ties), so a learner can follow one photograph end to end.
    spread = [
        int(max(per_image[t][i].sum() for t in thresholds) - min(per_image[t][i].sum() for t in thresholds))
        for i in range(len(test))
    ]
    chosen = int(np.argmax(spread)) if test else None
    paired = []
    if chosen is not None:
        record, prediction = test[chosen], predictions[chosen]
        for threshold in thresholds:
            objects = chosen_objects(prediction, {**policy_value, "detector_threshold": threshold})
            for entry in species_reconciliation(
                record["image_id"], object_outcomes(record, objects), objects
            ):
                paired.append(
                    {
                        "image_id": record["image_id"],
                        "display_threshold": threshold,
                        "role": role[threshold],
                        "species": entry["species"],
                        "reference": entry["reference"],
                        "raw": entry["raw"],
                        "missed": entry["missed"],
                        "spurious": entry["spurious"],
                        "wrong_species": entry["wrong_out"],
                    }
                )
    csv_write(
        root / "outputs" / "activity_paired_counts.csv",
        paired,
        [
            "image_id",
            "display_threshold",
            "role",
            "species",
            "reference",
            "raw",
            "missed",
            "spurious",
            "wrong_species",
        ],
    )
    margins = review_margin_rows(test, predictions, policy_value)
    csv_write(root / "outputs" / "activity_review_margins.csv", margins)
    if (
        core.sha256(root / "outputs" / "test_detections.json") != canonical
        or locked_policy(root)["digest"] != policy_value["digest"]
    ):
        raise ValueError("Activity changed canonical predictions or policy")


def reload(root):
    _, records = context(root)
    policy_value = locked_policy(root)
    expected = read(root / "outputs" / "reload_expected.json")
    if expected["pid"] == os.getpid() or expected["policy_digest"] != policy_value["digest"]:
        raise ValueError("Reload requires a fresh process and unchanged policy")
    ids = [p["image_id"] for p in expected["predictions"]]
    by_id = {r["image_id"]: r for r in records}
    actual = compose(root, [by_id[key] for key in ids])
    for left, right in zip(expected["predictions"], actual, strict=True):
        for field, atol in (("boxes", 1e-3), ("scores", 1e-5), ("adapted", 1e-5), ("zero_shot", 1e-5)):
            if not np.allclose(left[field], right[field], atol=atol, rtol=1e-5):
                raise ValueError(f"Reload parity failed for {field}")
        lobjects, robjects = chosen_objects(left, policy_value), chosen_objects(right, policy_value)
        if [(o["query_rank"], o["species"], o["review"]) for o in lobjects] != [
            (o["query_rank"], o["species"], o["review"]) for o in robjects
        ]:
            raise ValueError("Reload threshold/label/referral decisions changed")
    write(
        root / "outputs" / "verification.json",
        {
            "fresh_process": True,
            "reference_pid": expected["pid"],
            "reload_pid": os.getpid(),
            "original_image_ids": ids,
            "box_atol": 1e-3,
            "score_atol": 1e-5,
            "rtol": 1e-5,
            "identical_labels_counts_referrals": True,
        },
    )


def report(root):
    manifest, records = context(root)
    policy_value = locked_policy(root)
    out = root / "outputs"
    verified = read(out / "verification.json")
    if not verified.get("fresh_process"):
        raise ValueError("Fresh-process verification missing")
    with (out / "counts.csv").open(encoding="utf-8", newline="") as handle:
        count_rows = list(csv.DictReader(handle))
    if not count_rows or len(count_rows) % 2:
        raise ValueError("Count CSV row inventory mismatch")
    csv_reference = np.array([int(row["reference"]) for row in count_rows]).reshape(-1, 2)
    csv_prediction = np.array([int(row["raw"]) for row in count_rows]).reshape(-1, 2)
    recomputed = core.count_metrics(csv_reference, csv_prediction)
    if recomputed != read(out / "metrics.json")["detector_adapted"]:
        raise ValueError("Count CSV-to-metric parity failed")
    test = [record for record in records if record["split"] == "test"]
    expected_keys = [(record["image_id"], species) for record in test for species in SPECIES]
    if [(row["image_id"], row["species"]) for row in count_rows] != expected_keys:
        raise ValueError("CSV identifiers/class order differ from manifest")
    if not np.array_equal(csv_reference, count_reference(test)):
        raise ValueError("CSV reference counts differ from annotations")
    predictions = read(out / "test_detections.json")
    if [p["image_id"] for p in predictions] != [r["image_id"] for r in test]:
        raise ValueError("Raw prediction identifiers differ from manifest")
    if not np.array_equal(csv_prediction, count_predictions(predictions, policy_value["detector_threshold"])):
        raise ValueError("Raw prediction-to-count CSV parity failed")
    for index, row in enumerate(predictions):
        objects = chosen_objects(row, policy_value)
        for species_index, species in enumerate(SPECIES):
            exported = count_rows[2 * index + species_index]
            if int(exported["accepted"]) != sum(
                o["species"] == species and not o["review"] for o in objects
            ) or int(exported["unresolved_total"]) != sum(o["review"] for o in objects):
                raise ValueError("Referral count parity failed")
    verified["csv_metric_parity"] = True
    # Chart PNGs carry no source photographs; previews and annotated panels stay excluded.
    charts = [out / "figures" / name for name in CHART_FIGURES if (out / "figures" / name).is_file()]
    write(
        out / "run_summary.json",
        {
            "status": "Executed; maintainer review required for release",
            "scope": manifest.get("scope"),
            "policy_digest": policy_value["digest"],
            "limitations": [
                "Published annotations are not independently verified",
                "Capture independence and completeness are unverified",
                "Source-image/duplicate groups do not establish capture independence",
                "Pretraining overlap is unresolved",
                "Counts are image contents, not field abundance or intervention advice",
            ]
            if manifest.get("scope") == "exploratory_published_annotations"
            else [
                "Pretraining overlap unresolved",
                "Small held-out sample does not establish field validity",
            ],
            "metrics": read(out / "metrics.json"),
            "verification": verified,
            "charts_included": [path.relative_to(out).as_posix() for path in charts],
            "stage_resources": {
                name: read(out / f"receipt_{name}.json")
                for name in STAGES[:-1]
                if (out / f"receipt_{name}.json").is_file()
            },
        },
    )
    for name in ("data_manifest.json", "dataset_audit.json", "model_manifest.json"):
        shutil.copyfile(root / name, out / name)
    (out / "DATA_LICENSE.txt").write_text(
        "Dataset: https://doi.org/10.5281/zenodo.20066074\n"
        "Creative Commons Attribution 4.0 International\n"
        "https://creativecommons.org/licenses/by/4.0/\n"
        "Attributions for each selected source image are recorded in data_manifest.json.\n"
        "Source images, annotated image panels and photo previews are excluded from this archive;\n"
        "only the metric charts listed in run_summary.json (no photographs) are included.\n",
        encoding="utf-8",
    )
    files = sorted(
        p
        for p in out.rglob("*")
        if p.is_file()
        and "local_figures" not in p.parts
        and p.name not in {"results.zip", "checksums.json"}
        and p.suffix in {".json", ".jsonl", ".csv", ".safetensors", ".txt"}
    )
    files = sorted(files + charts)
    if sum(p.stat().st_size for p in files) > 100 * 1024**2:
        raise ValueError("Derived artifact archive exceeds the 100 MiB bound")
    checksums = {p.relative_to(out).as_posix(): core.sha256(p) for p in files}
    write(out / "checksums.json", checksums)
    with zipfile.ZipFile(out / "results.zip", "w", zipfile.ZIP_DEFLATED) as archive:
        for path in files + [out / "checksums.json"]:
            archive.write(path, path.relative_to(out).as_posix())
    import hashlib

    with zipfile.ZipFile(out / "results.zip") as archive:
        if archive.testzip() is not None:
            raise ValueError("ZIP CRC verification failed")
        for name, digest in checksums.items():
            if hashlib.sha256(archive.read(name)).hexdigest() != digest:
                raise ValueError("ZIP member digest mismatch")


def execution_identity(root):
    paths = [
        root / "data_manifest.json",
        root / "dataset_audit.json",
        root / "model_manifest.json",
        Path(__file__),
        Path(core.__file__),
    ]
    paths.extend(
        p
        for p in (
            root / "requirements.txt",
            root / "source.json",
            root / "rice_assets.py",
            root / "rice_figures.py",
            Path(__file__).parent / "rice_assets.py",
            Path(__file__).parent / "rice_figures.py",
        )
        if p.is_file()
    )
    return core.digest_json({str(p.name): core.sha256(p) for p in paths})


def verify_receipts(root, before_stage):
    """Refuse stale source, artifacts or partially rerun dependency chains."""
    identity = execution_identity(root)
    previous = None
    for stage in STAGES[: STAGES.index(before_stage)]:
        receipt_path = root / "outputs" / f"receipt_{stage}.json"
        receipt = read(receipt_path)
        if (
            not receipt.get("completed")
            or receipt.get("identity") != identity
            or receipt.get("previous") != previous
        ):
            raise ValueError(f"Missing/stale {stage} receipt; rerun from the first changed stage")
        for relative, digest in receipt.get("outputs", {}).items():
            if core.sha256(core.safe_path(root / "outputs", relative)) != digest:
                raise ValueError(f"Changed output from {stage}: {relative}")
        previous = core.sha256(receipt_path)
    return previous


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--stage", choices=STAGES, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    (root / "outputs").mkdir(parents=True, exist_ok=True)
    context(root)
    previous = verify_receipts(root, args.stage)
    out = root / "outputs"
    before = {p.relative_to(out).as_posix(): core.sha256(p) for p in out.rglob("*") if p.is_file()}
    prior_receipt = out / f"receipt_{args.stage}.json"
    previously_owned = set(read(prior_receipt).get("outputs", {})) if prior_receipt.exists() else set()
    # Remove downstream receipts only, preserving artifacts for diagnosis. Their
    # absence prevents stale outputs from masquerading as a completed fresh run.
    for stage in STAGES[STAGES.index(args.stage) :]:
        (out / f"receipt_{stage}.json").unlink(missing_ok=True)
    started = time.monotonic()
    if args.stage in {"features", "classifier", "detector", "policy", "evaluate", "reload"}:
        torch_runtime().cuda.reset_peak_memory_stats()
    globals()[args.stage](root)
    produced = {
        p.relative_to(out).as_posix(): core.sha256(p)
        for p in out.rglob("*")
        if p.is_file() and not p.name.startswith("receipt_")
    }
    changed = {
        name: digest
        for name, digest in produced.items()
        if before.get(name) != digest or name in previously_owned
    }
    write(
        root / "outputs" / f"receipt_{args.stage}.json",
        {
            "stage": args.stage,
            "seconds": time.monotonic() - started,
            "pid": os.getpid(),
            "data_sha256": core.sha256(root / "data_manifest.json"),
            "runtime_sha256": core.sha256(Path(__file__)),
            "identity": execution_identity(root),
            "previous": previous,
            "outputs": changed,
            "completed": True,
            "peak_allocated_gpu_bytes": int(__import__("torch").cuda.max_memory_allocated())
            if args.stage in {"features", "classifier", "detector", "policy", "evaluate", "reload"}
            else None,
            "gpu_name": __import__("torch").cuda.get_device_name(0)
            if args.stage in {"features", "classifier", "detector", "policy", "evaluate", "reload"}
            else None,
        },
    )


if __name__ == "__main__":
    main()
