"""Deterministic in-code sample data: the COCO demonstration scene and the adaptation dataset.

Nothing here is downloaded and nothing needs torch — Pillow and numpy only — so the tutorial's
default path has no dataset dependency and the same generators are exercised by the repository's unit
tests.

Two separate label vocabularies live here and must not be conflated:

* ``tutorial_scene`` returns references drawn from the checkpoint's own **COCO** classes; it
  demonstrates the pretrained model and is not training data.
* ``sign_dataset`` returns records labelled with ``SIGN_CLASSES``, a three-class vocabulary that does
  **not** exist in COCO. It is the adaptation dataset, and a model fine-tuned on it answers in those
  three names only.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont

COCO_SCENE_SIZE = (640, 480)
ADAPT_SCENE_SIZE = (640, 640)

# The adaptation vocabulary. Deliberately not COCO names: "stop sign" exists in COCO, "yield-sign" and
# "speed-limit-sign" do not, and the hyphenated spellings keep the two vocabularies visually distinct
# in output. A model fine-tuned on this dataset predicts only these three.
SIGN_CLASSES: tuple[str, ...] = ("stop-sign", "yield-sign", "speed-limit-sign")


def _font(size: int) -> Any:
    return ImageFont.load_default(size=size)


def tutorial_scene(
    width: int = COCO_SCENE_SIZE[0], height: int = COCO_SCENE_SIZE[1]
) -> tuple[Image.Image, dict[str, list[list[float]]]]:
    """The COCO demonstration scene: drawn objects and their reference boxes, keyed by COCO label.

    These are drawn references on a rendered picture, not a labelled photographic dataset: they are
    enough for a per-object ``box_iou`` sanity check and nothing more.
    """
    img = Image.new("RGB", (width, height), (135, 190, 235))
    d = ImageDraw.Draw(img)
    d.rectangle([0, int(height * 0.6875), width, height], fill=(96, 128, 72))
    d.rectangle([0, int(height * 0.625), width, int(height * 0.6875)], fill=(110, 110, 110))
    refs: dict[str, list[list[float]]] = {}

    # Stop sign
    cx, cy, r = 110, 150, 62
    pts = [
        (cx + r * math.cos(math.pi / 8 + k * math.pi / 4), cy + r * math.sin(math.pi / 8 + k * math.pi / 4))
        for k in range(8)
    ]
    d.rectangle([cx - 5, cy, cx + 5, int(height * 0.6875)], fill=(90, 90, 90))
    d.polygon(pts, fill=(200, 20, 30), outline=(255, 255, 255))
    f = _font(30)
    d.text((cx - d.textlength("STOP", font=f) / 2, cy - 17), "STOP", fill="white", font=f)
    refs["stop sign"] = [[float(cx - r), float(cy - r), float(cx + r), float(cy + r)]]

    # Traffic light
    x0, y0 = 270, 60
    d.rectangle([x0 + 22, y0 + 150, x0 + 30, int(height * 0.6875)], fill=(70, 70, 70))
    d.rectangle([x0, y0, x0 + 52, y0 + 150], fill=(25, 25, 25), outline=(60, 60, 60))
    for k, col in enumerate([(230, 30, 30), (240, 200, 30), (40, 200, 60)]):
        d.ellipse([x0 + 8, y0 + 8 + k * 47, x0 + 44, y0 + 44 + k * 47], fill=col)
    refs["traffic light"] = [[float(x0), float(y0), float(x0 + 52), float(y0 + 150)]]

    # Clock
    cx, cy, r = 480, 140, 70
    d.rectangle([cx - 6, cy, cx + 6, int(height * 0.6875)], fill=(120, 80, 40))
    d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=(250, 250, 245), outline=(20, 20, 20), width=5)
    f2 = _font(16)
    for h in range(1, 13):
        a = math.radians(h * 30 - 90)
        d.text(
            (cx + (r - 18) * math.cos(a) - 5, cy + (r - 18) * math.sin(a) - 8),
            str(h),
            fill="black",
            font=f2,
        )
    d.line(
        [(cx, cy), (cx + 0.5 * r * math.cos(math.radians(-60)), cy + 0.5 * r * math.sin(math.radians(-60)))],
        fill="black",
        width=5,
    )
    d.line(
        [(cx, cy), (cx + 0.8 * r * math.cos(math.radians(30)), cy + 0.8 * r * math.sin(math.radians(30)))],
        fill="black",
        width=3,
    )
    refs["clock"] = [[float(cx - r), float(cy - r), float(cx + r), float(cy + r)]]

    # Sports ball
    cx, cy, r = 330, 400, 45
    d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=(235, 120, 30), outline=(40, 20, 10), width=3)
    d.line([(cx - r, cy), (cx + r, cy)], fill=(40, 20, 10), width=3)
    d.line([(cx, cy - r), (cx, cy + r)], fill=(40, 20, 10), width=3)
    d.arc([cx - r * 1.6, cy - r, cx - r * 0.2, cy + r], 300, 60, fill=(40, 20, 10), width=3)
    d.arc([cx + r * 0.2, cy - r, cx + r * 1.6, cy + r], 120, 240, fill=(40, 20, 10), width=3)
    refs["sports ball"] = [[float(cx - r), float(cy - r), float(cx + r), float(cy + r)]]

    return img, refs


def blank_scene(width: int = 640, height: int = 640) -> Image.Image:
    """A featureless white image: the degenerate input every detector should be asked about."""
    return Image.new("RGB", (width, height), (255, 255, 255))


def noise_scene(seed: int = 0, width: int = 640, height: int = 640) -> Image.Image:
    """Uniform RGB noise: structure-free input, for the same reason as ``blank_scene``."""
    rng = np.random.default_rng(seed)
    return Image.fromarray(rng.integers(0, 256, (height, width, 3), dtype=np.uint8))


def _octagon(draw: ImageDraw.ImageDraw, cx: float, cy: float, r: float, font: Any) -> list[float]:
    points = [
        (cx + r * math.cos(math.pi / 8 + i * math.pi / 4), cy + r * math.sin(math.pi / 8 + i * math.pi / 4))
        for i in range(8)
    ]
    draw.polygon(points, fill=(196, 30, 34), outline=(255, 255, 255))
    text = "STOP"
    draw.text(
        (cx - draw.textlength(text, font=font) / 2, cy - r * 0.27), text, fill=(255, 255, 255), font=font
    )
    half = r * math.cos(math.pi / 8)
    return [cx - half, cy - half, cx + half, cy + half]


def _yield_sign(draw: ImageDraw.ImageDraw, cx: float, cy: float, r: float) -> list[float]:
    points = [(cx, cy + r), (cx - r * 0.95, cy - r * 0.75), (cx + r * 0.95, cy - r * 0.75)]
    draw.polygon(points, fill=(255, 255, 255), outline=(198, 32, 36))
    inner = [(cx, cy + r * 0.62), (cx - r * 0.62, cy - r * 0.5), (cx + r * 0.62, cy - r * 0.5)]
    draw.line([*inner, inner[0]], fill=(198, 32, 36), width=int(max(4, r * 0.22)))
    return [cx - r * 0.95, cy - r * 0.75, cx + r * 0.95, cy + r]


def _speed_limit_sign(draw: ImageDraw.ImageDraw, cx: float, cy: float, r: float, limit: int) -> list[float]:
    draw.ellipse(
        [cx - r, cy - r, cx + r, cy + r],
        fill=(255, 255, 255),
        outline=(198, 32, 36),
        width=int(max(4, r * 0.2)),
    )
    font = _font(int(max(12, r * 0.9)))
    text = str(limit)
    draw.text((cx - draw.textlength(text, font=font) / 2, cy - r * 0.55), text, fill=(30, 30, 30), font=font)
    return [cx - r, cy - r, cx + r, cy + r]


def sign_dataset(
    n_images: int = 40,
    *,
    seed: int = 0,
    max_objects: int = 3,
    size: tuple[int, int] = ADAPT_SCENE_SIZE,
) -> list[dict[str, Any]]:
    """A deterministic labelled dataset over ``SIGN_CLASSES`` for the bounded fine-tune.

    Each record is ``{"id": str, "image": PIL.Image, "boxes": [[x0, y0, x1, y1], ...], "labels": [name,
    ...]}`` — the record shape ``validate_dataset`` and ``finetune`` accept, and the shape a BYOD caller
    must produce from their own labelled images. Signs are placed on a non-overlapping grid so the boxes
    are exact by construction rather than approximate.

    It is synthetic drawn data, so a model fine-tuned on it learns to find *these renderings*. That is
    the point of a bounded tutorial adaptation and the reason its metrics are not a claim about real
    traffic signs.
    """
    if not 1 <= n_images <= 500:
        raise ValueError(f"n_images must be in 1..500, got {n_images}")
    if not 1 <= max_objects <= 6:
        raise ValueError(f"max_objects must be in 1..6, got {max_objects}")
    rng = np.random.default_rng(seed)
    width, height = size
    slots = [(x, y) for y in (150, 400) for x in (140, 360, 560)]
    records: list[dict[str, Any]] = []
    for index in range(n_images):
        image = Image.new("RGB", size, (232, 236, 240))
        draw = ImageDraw.Draw(image)
        tint = rng.integers(200, 245, 3)
        draw.rectangle([0, 0, width, height], fill=tuple(int(v) for v in tint))
        draw.rectangle([0, int(height * 0.72), width, height], fill=(118, 122, 126))
        n_objects = int(rng.integers(1, max_objects + 1))
        chosen = rng.permutation(len(slots))[:n_objects]
        boxes: list[list[float]] = []
        labels: list[str] = []
        for slot_index in chosen:
            cx, cy = slots[int(slot_index)]
            cx += float(rng.integers(-28, 29))
            cy += float(rng.integers(-28, 29))
            radius = float(rng.integers(46, 71))
            cx = min(max(cx, radius + 6), width - radius - 6)
            cy = min(max(cy, radius + 6), height - radius - 100)
            kind = SIGN_CLASSES[int(rng.integers(0, len(SIGN_CLASSES)))]
            draw.rectangle(
                [cx - 5, cy + radius * 0.6, cx + 5, min(height - 1, cy + radius + 90)], fill=(112, 112, 116)
            )
            if kind == "stop-sign":
                box = _octagon(draw, cx, cy, radius, _font(int(max(11, radius * 0.34))))
            elif kind == "yield-sign":
                box = _yield_sign(draw, cx, cy, radius)
            else:
                box = _speed_limit_sign(draw, cx, cy, radius, int(rng.choice([30, 50, 60, 80])))
            boxes.append(
                [
                    float(max(0.0, box[0])),
                    float(max(0.0, box[1])),
                    float(min(width, box[2])),
                    float(min(height, box[3])),
                ]
            )
            labels.append(kind)
        records.append({"id": f"sign-{seed}-{index:03d}", "image": image, "boxes": boxes, "labels": labels})
    return records


def split_dataset(
    records: list[dict[str, Any]],
    *,
    train_fraction: float = 0.75,
    seed: int = 0,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Deterministically partition records into train and held-out splits (random, record-level).

    A random split assumes the records are independent. Records that share a source photograph, a
    scene or a camera session must be split by that group instead, or the held-out score leaks.
    """
    if not 0.0 < train_fraction < 1.0:
        raise ValueError(f"train_fraction must be between 0 and 1, got {train_fraction}")
    if len(records) < 2:
        raise ValueError(f"at least 2 records are required to split, got {len(records)}")
    rng = np.random.default_rng(seed)
    indices = rng.permutation(len(records))
    n_train = min(len(records) - 1, max(1, int(len(records) * train_fraction)))
    train_idx = {int(i) for i in indices[:n_train]}
    train = [r for i, r in enumerate(records) if i in train_idx]
    held_out = [r for i, r in enumerate(records) if i not in train_idx]
    return train, held_out
