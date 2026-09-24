"""COCO object detection and bounded detection fine-tuning with the pinned DETR ResNet-50 checkpoint.

The class loads the processor and model only from a digest-verified local snapshot (``weights/<key>/``),
always with ``trust_remote_code=False``: the DETR architecture comes from the pinned ``transformers``
release, the backbone is the ``timm`` ResNet-50 the checkpoint's config names (built with
``use_pretrained_backbone=False``, so its weights come from the snapshot and nothing else is fetched), the
weights are SafeTensors, and no model-repository code is executed.

Until ``tools/pin_snapshot.py`` has recorded an immutable revision and every file's SHA-256, the package
refuses to stage, verify or load weights: an unpinned snapshot is never trusted.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

MODEL_ID = "facebook/detr-resnet-50"
MODEL_REVISION = "unpinned"
MODEL_LICENSE = "apache-2.0"
MODEL_KEY = "detr-resnet-50"
DEFAULT_WEIGHTS_DIR = Path(__file__).resolve().parents[2] / "weights" / MODEL_KEY
MANIFEST_NAME = "dimer-base-manifest.json"
ARTIFACT_FORMAT = "detr-adapter-v1"
UNPINNED = "unpinned"
PIN_COMMAND = "python tools/pin_snapshot.py"

# The 91 category slots of the checkpoint's config.json id2label, in id order. The slots follow the
# original COCO category numbering: id 0 is "N/A" and ten further ids ("street sign", "hat", "shoe",
# "eye glasses", "plate", "mirror", "window", "desk", "door", "blender") are categories that COCO 2017
# never annotated, so the checkpoint was trained with boxes for the other 80 names only.
LABELS = (
    "N/A",
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "airplane",
    "bus",
    "train",
    "truck",
    "boat",
    "traffic light",
    "fire hydrant",
    "street sign",
    "stop sign",
    "parking meter",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "hat",
    "backpack",
    "umbrella",
    "shoe",
    "eye glasses",
    "handbag",
    "tie",
    "suitcase",
    "frisbee",
    "skis",
    "snowboard",
    "sports ball",
    "kite",
    "baseball bat",
    "baseball glove",
    "skateboard",
    "surfboard",
    "tennis racket",
    "bottle",
    "plate",
    "wine glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot dog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "couch",
    "potted plant",
    "bed",
    "mirror",
    "dining table",
    "window",
    "desk",
    "toilet",
    "door",
    "tv",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell phone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "blender",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush",
)
UNANNOTATED_LABEL_IDS: tuple[int, ...] = (0, 12, 26, 29, 30, 45, 66, 68, 69, 71, 83)

# Detection threshold: the value the pinned README's example passes to post_process_object_detection
# (threshold=0.9). DETR scores each query with a softmax over its classes plus a "no object" class and
# reports the largest non-"no object" probability; the value was not calibrated for any deployment and
# the deployment owns tuning it on labelled images.
DETECTION_THRESHOLD = 0.9
EVAL_DETECTION_THRESHOLD = 0.05
MAX_DETECTIONS = 100
MAX_EVAL_DETECTIONS = 100
MAX_IMAGE_SIDE = 4096
MIN_IMAGE_SIDE = 16
MAX_CLASSES = 1000
MAX_RECORDS = 5000

# Training defaults for the bounded tutorial adaptation.
DEFAULT_EPOCHS = 10
DEFAULT_BATCH_SIZE = 4
DEFAULT_LEARNING_RATE = 1e-4
DEFAULT_WEIGHT_DECAY = 1e-4
DEFAULT_GRAD_CLIP = 0.1
DEFAULT_SEED = 20260924
BACKBONE_PREFIX = "model.backbone."

COCO_IOU_THRESHOLDS: tuple[float, ...] = tuple(round(0.50 + 0.05 * i, 2) for i in range(10))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def is_pinned() -> bool:
    """True once MODEL_REVISION names an immutable 40-hex commit."""
    revision = MODEL_REVISION
    return len(revision) == 40 and all(c in "0123456789abcdef" for c in revision)


def _require_pinned(action: str) -> None:
    if not is_pinned():
        raise RuntimeError(
            f"refusing to {action}: {MODEL_ID} has no pinned revision yet (MODEL_REVISION = "
            f"{MODEL_REVISION!r}); run `{PIN_COMMAND}` to record the commit and every file's SHA-256"
        )


def _read_manifest(root: Path) -> dict[str, Any]:
    manifest_path = root / MANIFEST_NAME
    if not manifest_path.is_file():
        raise FileNotFoundError(f"manifest not found: {manifest_path}")
    with open(manifest_path, encoding="utf-8") as fh:
        return json.load(fh)


def verify_snapshot(path: str | Path | None = None) -> dict[str, Any]:
    """Check a local snapshot against its DIMER manifest; raise naming the first mismatch."""
    _require_pinned("verify the snapshot")
    root = Path(path) if path is not None else DEFAULT_WEIGHTS_DIR
    manifest = _read_manifest(root)
    if manifest.get("modelId") != MODEL_ID:
        raise ValueError(f"manifest modelId {manifest.get('modelId')!r} != {MODEL_ID!r}")
    if manifest.get("revision") != MODEL_REVISION:
        raise ValueError(f"manifest revision {manifest.get('revision')!r} != {MODEL_REVISION!r}")
    for entry in manifest["files"]:
        if not entry.get("sha256"):
            raise ValueError(f"{entry['path']}: manifest records no sha256; run `{PIN_COMMAND}`")
        file_path = root / entry["path"]
        if not file_path.is_file():
            raise FileNotFoundError(f"snapshot file missing: {file_path}")
        size = file_path.stat().st_size
        if size != entry["bytes"]:
            raise ValueError(f"{entry['path']}: size {size} != manifest {entry['bytes']}")
        digest = _sha256(file_path)
        if digest != entry["sha256"]:
            raise ValueError(f"{entry['path']}: sha256 {digest} != manifest {entry['sha256']}")
    return {
        "path": str(root),
        "model_id": manifest["modelId"],
        "revision": manifest["revision"],
        "files": len(manifest["files"]),
        "total_bytes": manifest.get("totalBytes"),
    }


def _hub_download(relative_path: str, root: Path) -> None:
    """Fetch one manifest-listed file at MODEL_REVISION straight into the snapshot directory."""
    from huggingface_hub import hf_hub_download

    hf_hub_download(MODEL_ID, relative_path, revision=MODEL_REVISION, local_dir=str(root))


def stage_missing_files(
    path: str | Path | None = None,
    *,
    allow_download: bool = False,
    downloader: Callable[[str, Path], None] | None = None,
) -> list[str]:
    """Fetch manifest-listed files that are absent locally (a fresh clone commits the manifest but
    git-ignores the weights). Returns the relative paths fetched; `verify_snapshot` still runs after."""
    _require_pinned("stage weights")
    root = Path(path) if path is not None else DEFAULT_WEIGHTS_DIR
    manifest = _read_manifest(root)
    if manifest.get("modelId") != MODEL_ID or manifest.get("revision") != MODEL_REVISION:
        raise ValueError(
            f"manifest names {manifest.get('modelId')}@{manifest.get('revision')}, "
            f"package pins {MODEL_ID}@{MODEL_REVISION}; refusing to stage"
        )
    missing = [entry["path"] for entry in manifest["files"] if not (root / entry["path"]).is_file()]
    if not missing:
        return []
    if not allow_download:
        raise FileNotFoundError(
            f"snapshot at {root} is missing {missing}; "
            f"pass allow_download=True to fetch them at {MODEL_REVISION}"
        )
    fetch = downloader or _hub_download
    for relative_path in missing:
        fetch(relative_path, root)
    return missing


def box_iou(a: Sequence[float], b: Sequence[float]) -> float:
    """Intersection-over-union of two xyxy pixel boxes; the building block for average precision."""
    if len(a) != 4 or len(b) != 4:
        raise ValueError("boxes must be [x0, y0, x1, y1]")
    if a[2] < a[0] or a[3] < a[1] or b[2] < b[0] or b[3] < b[1]:
        raise ValueError("boxes must satisfy x0 <= x1 and y0 <= y1")
    inter_w = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    inter_h = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = inter_w * inter_h
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return float(inter / union) if union > 0 else 0.0


def average_precision(
    predictions: Sequence[Sequence[Mapping[str, Any]]],
    references: Sequence[Mapping[str, Any]],
    class_names: Sequence[str],
    *,
    iou_thresholds: Sequence[float] = COCO_IOU_THRESHOLDS,
) -> dict[str, Any]:
    """Compact COCO-style average precision over a scored dataset.

    For each class and IoU threshold, detections are matched greedily to references by descending
    score; each detection matches at most one reference box, and precision is sampled at 101 recall
    points. Classes with no reference box are left out of the mean. There are no area ranges and no
    crowd handling, so the values are close to, but not identical with, pycocotools output.
    """
    if len(predictions) != len(references):
        raise ValueError(f"{len(predictions)} prediction lists but {len(references)} references")
    names = list(class_names)
    recall_points = np.linspace(0.0, 1.0, 101)
    per_threshold: dict[float, dict[str, float]] = {}

    for threshold in iou_thresholds:
        per_class: dict[str, float] = {}
        for name in names:
            scored: list[tuple[float, bool]] = []
            n_references = 0
            for dets, reference in zip(predictions, references, strict=True):
                ref_boxes = [
                    box
                    for box, label in zip(reference["boxes"], reference["labels"], strict=True)
                    if label == name
                ]
                n_references += len(ref_boxes)
                claimed = [False] * len(ref_boxes)
                candidates = sorted((d for d in dets if d["label"] == name), key=lambda d: -float(d["score"]))
                for det in candidates:
                    best, best_iou = -1, 0.0
                    for j, ref_box in enumerate(ref_boxes):
                        if claimed[j]:
                            continue
                        value = box_iou(det["box"], ref_box)
                        if value > best_iou:
                            best, best_iou = j, value
                    hit = best >= 0 and best_iou >= threshold
                    if hit:
                        claimed[best] = True
                    scored.append((float(det["score"]), hit))
            if n_references == 0:
                continue
            if not scored:
                per_class[name] = 0.0
                continue
            scored.sort(key=lambda pair: -pair[0])
            true_positives = np.cumsum([1 if hit else 0 for _score, hit in scored])
            false_positives = np.cumsum([0 if hit else 1 for _score, hit in scored])
            recall = true_positives / n_references
            precision = true_positives / np.maximum(true_positives + false_positives, 1)
            precision = np.maximum.accumulate(precision[::-1])[::-1]
            sampled = np.zeros_like(recall_points)
            indices = np.searchsorted(recall, recall_points, side="left")
            valid = indices < len(precision)
            sampled[valid] = precision[indices[valid]]
            per_class[name] = float(sampled.mean())
        per_threshold[threshold] = per_class

    scored_classes = sorted({name for values in per_threshold.values() for name in values})
    means = {
        threshold: (float(np.mean(list(values.values()))) if values else 0.0)
        for threshold, values in per_threshold.items()
    }
    return {
        "ap": float(np.mean(list(means.values()))) if means else 0.0,
        "ap50": means.get(0.5, 0.0),
        "ap75": means.get(0.75, 0.0),
        "per_class_ap50": per_threshold.get(0.5, {}),
        "iou_thresholds": [float(t) for t in iou_thresholds],
        "scored_classes": scored_classes,
        "n_images": len(references),
        "n_references": sum(len(r["boxes"]) for r in references),
    }


def validate_image(image: Any) -> Image.Image:
    if not isinstance(image, Image.Image):
        raise TypeError(f"image must be a PIL.Image.Image, got {type(image).__name__}")
    width, height = image.size
    if min(width, height) < MIN_IMAGE_SIDE:
        raise ValueError(f"image side {min(width, height)} px < MIN_IMAGE_SIDE {MIN_IMAGE_SIDE}")
    if max(width, height) > MAX_IMAGE_SIDE:
        raise ValueError(f"image side {max(width, height)} px > MAX_IMAGE_SIDE {MAX_IMAGE_SIDE}")
    return image.convert("RGB")


def _check_threshold(value: Any, name: str = "threshold") -> float:
    if isinstance(value, bool) or not isinstance(value, int | float) or not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be a number in [0, 1], got {value!r}")
    return float(value)


def _check_class_names(class_names: Sequence[str]) -> tuple[str, ...]:
    names = tuple(class_names)
    if not 1 <= len(names) <= MAX_CLASSES:
        raise ValueError(f"class_names must hold 1..{MAX_CLASSES} names, got {len(names)}")
    for name in names:
        if not isinstance(name, str) or not name.strip():
            raise ValueError(f"class names must be non-empty strings, got {name!r}")
    duplicates = sorted({name for name in names if names.count(name) > 1})
    if duplicates:
        raise ValueError(f"class_names contains duplicates: {duplicates}")
    return names


INPUT_SCHEMA: dict[str, Any] = {
    "input": "one image as PIL.Image.Image (any mode, converted to RGB): a photograph or a rendered scene",
    "image_side_px": [MIN_IMAGE_SIDE, MAX_IMAGE_SIDE],
    "threshold": [0.0, 1.0],
    "labels": list(LABELS),
    "unannotated_label_ids": list(UNANNOTATED_LABEL_IDS),
    "max_detections": MAX_DETECTIONS,
    "preprocessing": (
        "image converted to RGB; the processor resizes so the shorter side is 800 px and the longer side at "
        "most 1333 px (aspect ratio preserved) and normalises with the ImageNet mean and standard deviation; "
        "returned boxes are mapped back to input pixels"
    ),
}

DATASET_SCHEMA: dict[str, Any] = {
    "record": "{'image': PIL.Image.Image, 'boxes': [[x0, y0, x1, y1], ...], 'labels': [class_name, ...]}",
    "boxes": "xyxy pixel coordinates inside the image, x0 < x1 and y0 < y1",
    "labels": "one name per box, each one of class_names",
    "records": [1, MAX_RECORDS],
    "image_side_px": [MIN_IMAGE_SIDE, MAX_IMAGE_SIDE],
}


def _check_inputs(image: Any, threshold: Any) -> tuple[Image.Image, float]:
    """Raise TypeError/ValueError naming the first violated ceiling; return the checked request."""
    return validate_image(image), _check_threshold(threshold)


def validate_inputs(
    image: Image.Image,
    *,
    threshold: float = DETECTION_THRESHOLD,
    names: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Validation stage: return the input manifest (schema, observations, request, verdict)."""
    _rgb, checked = _check_inputs(image, threshold)
    if names is not None and len(names) != 1:
        raise ValueError("names must have exactly one entry (detect takes one image)")
    return {
        "schema": dict(INPUT_SCHEMA),
        "inputs": [{"id": names[0] if names else "image-0", "mode": image.mode, "size": list(image.size)}],
        "threshold": checked,
        "verdict": "accepted",
        "findings": [],
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
    }


def validate_dataset(
    records: Sequence[Mapping[str, Any]],
    class_names: Sequence[str],
    *,
    epochs: int = DEFAULT_EPOCHS,
) -> dict[str, Any]:
    """Validation stage for labelled detection records: raise on the first broken record, else return
    the dataset manifest. Classes that never occur are reported as findings, not silently accepted."""
    names = _check_class_names(class_names)
    if not records:
        raise ValueError("dataset must hold at least one record")
    if len(records) > MAX_RECORDS:
        raise ValueError(f"dataset holds {len(records)} records > MAX_RECORDS {MAX_RECORDS}")
    if isinstance(epochs, bool) or not isinstance(epochs, int) or not 1 <= epochs <= 100:
        raise ValueError(f"epochs must be an int in 1..100, got {epochs!r}")
    valid_classes = set(names)
    total_boxes = 0
    per_class = dict.fromkeys(names, 0)

    for idx, record in enumerate(records):
        if not isinstance(record, Mapping) or not {"image", "boxes", "labels"} <= set(record):
            raise ValueError(f"record {idx} must be a mapping with 'image', 'boxes' and 'labels'")
        try:
            img = validate_image(record["image"])
        except (TypeError, ValueError) as exc:
            raise type(exc)(f"record {idx}: {exc}") from exc
        boxes = record["boxes"]
        labels = record["labels"]
        if len(boxes) != len(labels):
            raise ValueError(f"record {idx}: {len(boxes)} boxes but {len(labels)} labels")
        width, height = img.size
        for b_idx, box in enumerate(boxes):
            if len(box) != 4:
                raise ValueError(f"record {idx} box {b_idx} must have 4 elements, got {len(box)}")
            x0, y0, x1, y1 = (float(v) for v in box)
            if not all(np.isfinite([x0, y0, x1, y1])):
                raise ValueError(f"record {idx} box {b_idx} has a non-finite coordinate")
            if not (0.0 <= x0 < x1 <= width and 0.0 <= y0 < y1 <= height):
                raise ValueError(
                    f"record {idx} box {b_idx} [{x0}, {y0}, {x1}, {y1}] is empty or "
                    f"outside image bounds {(width, height)}"
                )
        for label in labels:
            if label not in valid_classes:
                raise ValueError(f"record {idx} has unknown class {label!r}; expected one of {list(names)}")
            per_class[label] += 1
        total_boxes += len(boxes)

    absent = [name for name, count in per_class.items() if count == 0]
    findings = [f"class {name!r} has no box in this dataset" for name in absent]
    return {
        "schema": dict(DATASET_SCHEMA),
        "n_records": len(records),
        "n_boxes": total_boxes,
        "class_names": list(names),
        "boxes_per_class": per_class,
        "epochs": epochs,
        "findings": findings,
        "verdict": "accepted",
    }


def read_detection_records(directory: str | Path) -> list[dict[str, Any]]:
    """Read BYOD records from ``<directory>/annotations.json`` plus the image files it names.

    ``annotations.json`` is a list of ``{"file": "relative/name.png", "boxes": [[x0, y0, x1, y1], ...],
    "labels": [name, ...]}`` objects. File names must stay inside ``directory``; absolute paths and
    ``..`` segments are refused before any image is opened. The result still has to pass
    ``validate_dataset``.
    """
    root = Path(directory).resolve()
    index_path = root / "annotations.json"
    if not index_path.is_file():
        raise FileNotFoundError(f"{index_path} not found; expected annotations.json next to the images")
    with open(index_path, encoding="utf-8") as fh:
        entries = json.load(fh)
    if not isinstance(entries, list) or not entries:
        raise ValueError("annotations.json must hold a non-empty list of records")
    if len(entries) > MAX_RECORDS:
        raise ValueError(f"annotations.json lists {len(entries)} records > MAX_RECORDS {MAX_RECORDS}")
    records: list[dict[str, Any]] = []
    for idx, entry in enumerate(entries):
        if not isinstance(entry, dict) or not {"file", "boxes", "labels"} <= set(entry):
            raise ValueError(f"annotations.json entry {idx} must have 'file', 'boxes' and 'labels'")
        relative = Path(str(entry["file"]))
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(
                f"annotations.json entry {idx}: file {entry['file']!r} must be relative to {root}"
            )
        image_path = (root / relative).resolve()
        if root not in image_path.parents:
            raise ValueError(f"annotations.json entry {idx}: file {entry['file']!r} resolves outside {root}")
        with Image.open(image_path) as handle:
            image = handle.convert("RGB")
        records.append(
            {"image": image, "boxes": entry["boxes"], "labels": entry["labels"], "id": str(relative)}
        )
    return records


def evaluation_report(
    result: Mapping[str, Any],
    ground_truth_boxes: Mapping[str, Sequence[Sequence[float]]] | None = None,
    *,
    sample_kind: str = "synthetic",
) -> dict[str, Any]:
    """Single-image evaluation stage: machine-readable report with per-object box_iou."""
    detections = list(result["detections"])
    labels = tuple(result.get("class_names") or LABELS)
    base = {
        "task": f"object detection over {len(labels)} class slots on one image",
        "decision_rule": (
            "a query survives when its largest class probability, taken from a softmax over the classes and "
            "a 'no object' class, reaches the threshold; the value is not a calibrated probability for the "
            "deployment's images, and each query reports exactly one label"
        ),
        "threshold": result.get("threshold", DETECTION_THRESHOLD),
        "sample_kind": sample_kind,
        "n_detections": len(detections),
        "baselines": [],
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
    }
    if not ground_truth_boxes:
        return {
            **base,
            "metrics": [],
            "verdict": "not-measurable",
            "reason": "no ground-truth object boxes were supplied for the evaluated image",
            "needs": (
                "labelled boxes per class on your own images, scored per object with box_iou and aggregated "
                "into COCO-style average precision (AP@[.50:.95], AP50) at stated IoU thresholds"
            ),
        }
    metrics = []
    for label, boxes in ground_truth_boxes.items():
        if label not in labels:
            raise ValueError(f"unknown reference label {label!r}; expected one of {len(labels)} classes")
        same_label = [det for det in detections if det["label"] == label]
        for index, box in enumerate(boxes):
            ious = [box_iou(det["box"], box) for det in same_label]
            best = max(range(len(ious)), key=ious.__getitem__) if ious else None
            metrics.append(
                {
                    "id": "box_iou",
                    "reference": f"{label}-{index}",
                    "value": ious[best] if best is not None else 0.0,
                    "matched_score": same_label[best]["score"] if best is not None else None,
                    "n_detected_same_label": len(same_label),
                    "estimation": "one reference box per object on a single image, no dispersion estimate",
                }
            )
    return {
        **base,
        "metrics": metrics,
        "verdict": "sample-sanity",
        "reason": (
            f"{len(metrics)} reference box(es) on one tutorial image; geometry sanity evidence, "
            "not a detection benchmark"
        ),
        "needs": (
            "a labelled image set from the deployment domain (cameras, scenes, object classes) for any "
            "average-precision or precision/recall claim"
        ),
    }


def _coco_annotation(index: int, record: Mapping[str, Any], class_to_id: Mapping[str, int]) -> dict[str, Any]:
    """One record as the COCO-detection annotation the DETR processor converts into training targets."""
    annotations = []
    for box, label in zip(record["boxes"], record["labels"], strict=True):
        x0, y0, x1, y1 = (float(v) for v in box)
        annotations.append(
            {
                "bbox": [x0, y0, x1 - x0, y1 - y0],
                "category_id": class_to_id[label],
                "area": (x1 - x0) * (y1 - y0),
                "iscrowd": 0,
            }
        )
    return {"image_id": index, "annotations": annotations}


@dataclass
class DetrDetectionPipeline:
    """COCO-class object detection and transfer fine-tuning over DETR (ResNet-50)."""

    model: Any
    processor: Any
    device: str
    class_names: tuple[str, ...] = LABELS
    source: str = "snapshot"
    base_state_digest: str | None = None
    adapted: bool = False
    reinitialised: tuple[str, ...] = field(default_factory=tuple)
    frozen_prefixes: tuple[str, ...] = field(default_factory=tuple)

    @classmethod
    def from_pretrained(
        cls,
        device: str | None = None,
        weights_dir: str | Path | None = None,
        allow_download: bool = False,
        class_names: Sequence[str] | None = None,
        seed: int = DEFAULT_SEED,
    ) -> DetrDetectionPipeline:
        _require_pinned("load the model")
        root = Path(weights_dir) if weights_dir is not None else DEFAULT_WEIGHTS_DIR
        if not (root / MANIFEST_NAME).is_file():
            raise FileNotFoundError(
                f"no snapshot manifest at {root}; stage {MODEL_ID}@{MODEL_REVISION} "
                f"under weights/{MODEL_KEY} "
                "(allow_download=True fetches the manifest-listed files)"
            )
        stage_missing_files(root, allow_download=allow_download)
        verify_snapshot(root)
        names = _check_class_names(class_names) if class_names is not None else LABELS

        import torch
        from transformers import DetrForObjectDetection, DetrImageProcessor

        resolved_device = device or ("cuda:0" if torch.cuda.is_available() else "cpu")
        common = {"trust_remote_code": False, "local_files_only": True}
        processor = DetrImageProcessor.from_pretrained(str(root), **common)
        base_digest = _sha256(root / "model.safetensors") if (root / "model.safetensors").is_file() else None

        reinitialised: tuple[str, ...] = ()
        if class_names is not None:
            # Deterministic re-heading: a fresh class_labels_classifier (num_labels + 1 outputs, the last
            # being "no object") replaces the 92-way COCO head; every other tensor comes from the snapshot.
            torch.manual_seed(seed)
            model = DetrForObjectDetection.from_pretrained(
                str(root),
                num_labels=len(names),
                id2label=dict(enumerate(names)),
                label2id={name: i for i, name in enumerate(names)},
                ignore_mismatched_sizes=True,
                use_pretrained_backbone=False,
                **common,
            )
            reinitialised = ("class_labels_classifier.weight", "class_labels_classifier.bias")
        else:
            model = DetrForObjectDetection.from_pretrained(str(root), use_pretrained_backbone=False, **common)

        model = model.to(resolved_device).eval()
        return cls(
            model=model,
            processor=processor,
            device=resolved_device,
            class_names=names,
            source=str(root),
            base_state_digest=base_digest,
            adapted=False,
            reinitialised=reinitialised,
        )

    def _run(self, image: Image.Image, threshold: float) -> list[dict[str, Any]]:
        import torch

        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        was_training = self.model.training
        self.model.eval()
        with torch.inference_mode():
            outputs = self.model(**inputs)
        if was_training:
            self.model.train()
        result = self.processor.post_process_object_detection(
            outputs, threshold=threshold, target_sizes=[(image.height, image.width)]
        )[0]
        detections = []
        for box, label_idx, score in zip(result["boxes"], result["labels"], result["scores"], strict=True):
            idx = int(label_idx)
            label_name = self.class_names[idx] if idx < len(self.class_names) else f"class_{idx}"
            detections.append(
                {"box": [float(v) for v in box.tolist()], "label": label_name, "score": float(score)}
            )
        return detections

    def detect(self, image: Image.Image, *, threshold: float = DETECTION_THRESHOLD) -> dict[str, Any]:
        """Detect objects on one image; boxes are xyxy pixel coordinates in the input image."""
        rgb, checked = _check_inputs(image, threshold)
        detections = self._run(rgb, checked)
        if len(detections) > MAX_DETECTIONS:
            raise RuntimeError(
                f"backend returned {len(detections)} detections > num_queries {MAX_DETECTIONS}"
            )
        for det in detections:
            if (
                set(det) != {"box", "label", "score"}
                or len(det["box"]) != 4
                or det["label"] not in self.class_names
            ):
                raise RuntimeError(f"backend returned a malformed detection: {det!r}")
        return {
            "detections": sorted(detections, key=lambda d: -d["score"]),
            "threshold": checked,
            "width": rgb.width,
            "height": rgb.height,
            "class_names": list(self.class_names),
            "adapted": self.adapted,
            "model_id": MODEL_ID,
            "model_revision": MODEL_REVISION,
        }

    def detect_many(
        self,
        images: Sequence[Image.Image],
        *,
        threshold: float = EVAL_DETECTION_THRESHOLD,
    ) -> list[list[dict[str, Any]]]:
        """Detections for several images, defaulting to the evaluation threshold."""
        return [self.detect(image, threshold=threshold)["detections"] for image in images]

    def finetune(
        self,
        records: Sequence[Mapping[str, Any]],
        *,
        epochs: int = DEFAULT_EPOCHS,
        batch_size: int = DEFAULT_BATCH_SIZE,
        learning_rate: float = DEFAULT_LEARNING_RATE,
        seed: int = DEFAULT_SEED,
        freeze_backbone: bool = True,
        progress: Callable[[dict[str, Any]], None] | None = None,
    ) -> dict[str, Any]:
        """Bounded fine-tuning on ``records`` with DETR's own set-prediction loss.

        The loss is the upstream ``DetrForObjectDetection`` loss: Hungarian matching of queries to
        reference boxes, then cross-entropy over the classes plus "no object" (weighted by the config's
        ``eos_coefficient``), L1 and generalised-IoU box terms. Mutates this pipeline in place (``adapted``
        becomes True) and leaves the model in eval mode.
        """
        import torch

        validate_dataset(records, self.class_names, epochs=epochs)
        if not isinstance(batch_size, int) or isinstance(batch_size, bool) or batch_size < 1:
            raise ValueError(f"batch_size must be a positive int, got {batch_size!r}")
        if isinstance(learning_rate, bool) or not isinstance(learning_rate, int | float):
            raise ValueError(f"learning_rate must be a number, got {learning_rate!r}")
        if not 0.0 < float(learning_rate) <= 1.0:
            raise ValueError(f"learning_rate must be in (0, 1], got {learning_rate!r}")

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        rng = np.random.default_rng(seed)

        for name, parameter in self.model.named_parameters():
            parameter.requires_grad = not (freeze_backbone and name.startswith(BACKBONE_PREFIX))
        self.frozen_prefixes = (BACKBONE_PREFIX,) if freeze_backbone else ()
        trainable = [p for p in self.model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(trainable, lr=float(learning_rate), weight_decay=DEFAULT_WEIGHT_DECAY)
        class_to_id = {name: i for i, name in enumerate(self.class_names)}

        epoch_losses: list[float] = []
        for epoch in range(epochs):
            self.model.train()
            order = rng.permutation(len(records))
            running_loss, n_batches = 0.0, 0
            for start in range(0, len(records), batch_size):
                batch = [records[int(i)] for i in order[start : start + batch_size]]
                images = [validate_image(r["image"]) for r in batch]
                annotations = [_coco_annotation(start + k, r, class_to_id) for k, r in enumerate(batch)]
                encoded = self.processor(images=images, annotations=annotations, return_tensors="pt")
                labels = [
                    {key: value.to(self.device) for key, value in target.items()}
                    for target in encoded["labels"]
                ]
                optimizer.zero_grad(set_to_none=True)
                outputs = self.model(
                    pixel_values=encoded["pixel_values"].to(self.device),
                    pixel_mask=encoded["pixel_mask"].to(self.device),
                    labels=labels,
                )
                loss = outputs.loss
                loss.backward()
                torch.nn.utils.clip_grad_norm_(trainable, DEFAULT_GRAD_CLIP)
                optimizer.step()
                running_loss += float(loss.detach().cpu())
                n_batches += 1
            epoch_losses.append(running_loss / max(1, n_batches))
            if progress is not None:
                progress({"epoch": epoch + 1, "epochs": epochs, "loss": epoch_losses[-1]})

        self.model.eval()
        self.adapted = True
        return {
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": float(learning_rate),
            "weight_decay": DEFAULT_WEIGHT_DECAY,
            "grad_clip_norm": DEFAULT_GRAD_CLIP,
            "optimizer": "AdamW",
            "seed": seed,
            "precision": "float32",
            "freeze_backbone": freeze_backbone,
            "frozen_prefixes": list(self.frozen_prefixes),
            "trainable_parameters": sum(p.numel() for p in trainable),
            "total_parameters": sum(p.numel() for p in self.model.parameters()),
            "epoch_losses": epoch_losses,
            "final_loss": epoch_losses[-1] if epoch_losses else None,
            "loss": "upstream DetrForObjectDetection loss (Hungarian matching; cross-entropy + L1 + GIoU)",
            "device": self.device,
            "class_names": list(self.class_names),
            "reinitialised_tensors": list(self.reinitialised),
            "model_id": MODEL_ID,
            "model_revision": MODEL_REVISION,
        }

    def evaluate(
        self,
        records: Sequence[Mapping[str, Any]],
        *,
        threshold: float = EVAL_DETECTION_THRESHOLD,
        iou_thresholds: Sequence[float] = COCO_IOU_THRESHOLDS,
        max_detections: int = MAX_EVAL_DETECTIONS,
    ) -> dict[str, Any]:
        """Score a labelled dataset: average precision plus evaluation metadata."""
        checked = _check_threshold(threshold, "threshold")
        predictions = self.detect_many([r["image"] for r in records], threshold=checked)
        raw_counts = [len(dets) for dets in predictions]
        capped = [dets[:max_detections] for dets in predictions]
        metrics = average_precision(capped, records, self.class_names, iou_thresholds=iou_thresholds)
        return {
            **metrics,
            "threshold": checked,
            "max_detections": max_detections,
            "detections_before_cap": raw_counts,
            "adapted": self.adapted,
            "class_names": list(self.class_names),
            "estimation": (
                f"one pass over {len(records)} held-out images; no resampling, no dispersion estimate"
            ),
            "implementation": (
                "package-local average_precision: greedy score-ordered matching and 101-point interpolation, "
                "without pycocotools area ranges or crowd handling"
            ),
            "model_id": MODEL_ID,
            "model_revision": MODEL_REVISION,
        }

    def save_artifact(self, path: str | Path, *, notes: str | None = None) -> dict[str, Any]:
        """Write the adapted tensors as one SafeTensors file with the provenance in its metadata.

        Tensors under ``frozen_prefixes`` are left out: they equal the verified base snapshot, which
        ``load_artifact`` loads first. The artifact is therefore an adapter bound to the base revision.
        """
        from safetensors.torch import save_file

        if not self.adapted:
            raise RuntimeError("nothing to export: the pipeline has not been fine-tuned")
        artifact_path = Path(path)
        artifact_path.parent.mkdir(parents=True, exist_ok=True)
        tensors = {
            name: value.detach().cpu().contiguous().clone()
            for name, value in self.model.state_dict().items()
            if not any(name.startswith(prefix) for prefix in self.frozen_prefixes)
        }
        metadata = {
            "format": ARTIFACT_FORMAT,
            "model_id": MODEL_ID,
            "model_revision": MODEL_REVISION,
            "model_key": MODEL_KEY,
            "class_names": json.dumps(list(self.class_names)),
            "frozen_prefixes": json.dumps(list(self.frozen_prefixes)),
            "base_state_digest": self.base_state_digest or "",
            "notes": notes or "",
        }
        save_file(tensors, str(artifact_path), metadata=metadata)
        return {
            "path": str(artifact_path),
            "bytes": artifact_path.stat().st_size,
            "sha256": _sha256(artifact_path),
            "format": ARTIFACT_FORMAT,
            "class_names": list(self.class_names),
            "tensors": len(tensors),
            "frozen_prefixes": list(self.frozen_prefixes),
            "base_state_digest": self.base_state_digest,
            "model_id": MODEL_ID,
            "model_revision": MODEL_REVISION,
        }

    @staticmethod
    def read_artifact_metadata(path: str | Path) -> dict[str, Any]:
        """Read and check the artifact's provenance header without loading any tensor."""
        from safetensors import safe_open

        with safe_open(str(path), framework="pt") as handle:
            metadata = dict(handle.metadata() or {})
        if metadata.get("format") != ARTIFACT_FORMAT:
            raise ValueError(f"artifact format {metadata.get('format')!r} != {ARTIFACT_FORMAT!r}")
        if (metadata.get("model_id"), metadata.get("model_revision")) != (MODEL_ID, MODEL_REVISION):
            raise ValueError(
                f"artifact was built on {metadata.get('model_id')}@{metadata.get('model_revision')}, "
                f"package pins {MODEL_ID}@{MODEL_REVISION}"
            )
        if metadata.get("model_key") != MODEL_KEY:
            raise ValueError(
                f"artifact was built on {metadata.get('model_key')!r}, package pins {MODEL_KEY!r}"
            )
        return {
            **metadata,
            "class_names": json.loads(metadata["class_names"]),
            "frozen_prefixes": json.loads(metadata["frozen_prefixes"]),
        }

    def apply_artifact(self, path: str | Path) -> None:
        """Load adapter tensors onto this (base) pipeline; refuse any tensor set that does not fit."""
        from safetensors.torch import load_file

        metadata = self.read_artifact_metadata(path)
        if tuple(metadata["class_names"]) != tuple(self.class_names):
            raise ValueError("artifact class_names differ from this pipeline's class_names")
        expected_base = metadata.get("base_state_digest") or None
        if expected_base and self.base_state_digest and expected_base != self.base_state_digest:
            raise ValueError("artifact was exported against a different base model.safetensors digest")
        tensors = load_file(str(path), device="cpu")
        prefixes = tuple(metadata["frozen_prefixes"])
        result = self.model.load_state_dict(tensors, strict=False)
        if result.unexpected_keys:
            raise ValueError(
                f"artifact carries tensors the model does not have: {result.unexpected_keys[:5]}"
            )
        stray = [key for key in result.missing_keys if not any(key.startswith(p) for p in prefixes)]
        if stray:
            raise ValueError(f"artifact is missing trainable tensors: {stray[:5]}")
        self.model.eval()
        self.adapted = True
        self.frozen_prefixes = prefixes
        self.source = f"artifact:{Path(path).name}"

    @classmethod
    def load_artifact(
        cls,
        path: str | Path,
        *,
        weights_dir: str | Path | None = None,
        device: str | None = None,
    ) -> DetrDetectionPipeline:
        """Rebuild an adapted pipeline: verified base snapshot first, then the adapter tensors."""
        metadata = cls.read_artifact_metadata(path)
        pipe = cls.from_pretrained(
            device=device, weights_dir=weights_dir, class_names=metadata["class_names"]
        )
        pipe.apply_artifact(path)
        return pipe
