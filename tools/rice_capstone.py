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


def detections(root, model, processor, records):
    import torch

    result = []
    with torch.inference_mode():
        for record in records:
            original = image(root, record)
            inputs = processor(images=original, return_tensors="pt").to("cuda")
            output = model(**inputs)
            post = processor.post_process_object_detection(
                output,
                threshold=0.0,
                target_sizes=torch.tensor([[original.height, original.width]], device="cuda"),
            )[0]
            # Preserve all 100 ranked queries; thresholds applied later, no NMS.
            order = np.argsort(-post["scores"].cpu().numpy(), kind="stable")
            result.append(
                {
                    "image_id": record["image_id"],
                    "boxes": post["boxes"].cpu().numpy()[order].tolist(),
                    "scores": post["scores"].cpu().numpy()[order].tolist(),
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
                tokenizer(["a photo of a rice black bug.", "a photo of a white stemborer."]).cuda(),
                normalize=True,
            )
            .cpu()
            .numpy()
        )
    scale = float(model.logit_scale.exp().item())
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
        output["adapted"] = softmax(vectors @ head["weight"].T + head["bias"]).tolist()
        output["zero_shot"] = softmax(scale * vectors @ text.T).tolist()
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
    eligible = [r for r in rows if r["coverage"] >= 0.5]
    target = [r for r in eligible if r["accuracy"] >= 0.8]
    selected = (
        max(target, key=lambda r: (r["coverage"], r["threshold"]))
        if target
        else max(eligible, key=lambda r: (r["accuracy"], r["coverage"], r["threshold"]))
    )
    return {**selected, "target_met": bool(target), "curve": rows}


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
    write(
        root / "outputs" / "reload_expected.json",
        {"pid": os.getpid(), "policy_digest": policy_value["digest"], "predictions": predictions[:2]},
    )
    panels(root, test, predictions, policy_value)


def panels(root, records, predictions, policy_value):
    import matplotlib.pyplot as plt
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
    write(
        root / "outputs" / "error_panel_inventory.json",
        {
            "selected": found,
            "absent_categories": sorted(set(("misses", "spurious", "wrong_species", "success")) - set(found)),
        },
    )
    for record, prediction in selected:
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.imshow(image(root, record))
        for obj in active_objects(record):
            x1, y1, x2, y2 = obj["bbox_xyxy"]
            ax.add_patch(
                Rectangle(
                    (x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor="white", linestyle="--", linewidth=1
                )
            )
        objects = chosen_objects(prediction, policy_value)
        for obj in objects:
            x1, y1, x2, y2 = obj["bbox_xyxy"]
            ax.add_patch(Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor="orange", linewidth=1))
            ax.text(
                x1,
                y1,
                f"{obj['species']} {obj['detector_score']:.2f}" + (" REVIEW" if obj["review"] else ""),
                fontsize=5,
                color="black",
                backgroundcolor="white",
            )
        ax.set_title(
            f"Test {record['image_id']} | reference dashed / prediction solid\n"
            f"Reference {len(active_objects(record))}; predicted {len(objects)} | {record['attribution']}",
            fontsize=8,
        )
        ax.axis("off")
        fig.savefig(folder / f"{record['image_id']}.png", dpi=120, bbox_inches="tight")
        plt.close(fig)


def activity(root):
    _, records = context(root)
    test = [r for r in records if r["split"] == "test"]
    policy_value = locked_policy(root)
    predictions = read(root / "outputs" / "test_detections.json")
    rows = []
    for threshold in sorted(
        set(
            [
                max(0.0, policy_value["detector_threshold"] - 0.1),
                policy_value["detector_threshold"],
                min(1.0, policy_value["detector_threshold"] + 0.1),
            ]
        )
    ):
        count_result = core.count_metrics(count_reference(test), count_predictions(predictions, threshold))
        errors = [
            core.error_decomposition(
                r["objects"], chosen_objects(p, {**policy_value, "detector_threshold": threshold})
            )
            for r, p in zip(test, predictions, strict=True)
        ]
        rows.append(
            {
                "display_threshold": threshold,
                "mae": count_result["mae"],
                "missed": sum(e["missed"] for e in errors),
                "spurious": sum(e["spurious_count"] for e in errors),
                "note": "Retrospective test illustration; canonical policy unchanged",
            }
        )
    csv_write(root / "outputs" / "activity_thresholds.csv", rows)


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
        "Source images and annotated image panels are excluded from this archive.\n",
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
