"""Read-only learner figures; never changes canonical policies or model outputs."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402
from PIL import Image  # noqa: E402


def read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def render(root: Path, stage: str) -> None:
    out = root / "outputs"
    folder = out / "figures"
    folder.mkdir(exist_ok=True)
    if stage == "prepare":
        manifest = read(root / "data_manifest.json")
        train = [r for r in manifest["records"] if r["split"] == "train"]
        fig, axes = plt.subplots(3, 4, figsize=(15, 12))
        for ax, row in zip(axes.flat, train[:12], strict=True):
            with Image.open(root / "data" / row["relative_path"]) as source:
                ax.imshow(source.convert("RGB"))
            for obj in row["objects"]:
                x1, y1, x2, y2 = obj["bbox_xyxy"]
                ax.add_patch(Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor="red"))
            ax.set_title(f"{row['capture_group_id']}\n{len(row['objects'])} published boxes", fontsize=8)
            ax.axis("off")
        fig.suptitle("Training image exports and published reference boxes · not independently verified")
        fig.text(
            0.5,
            0.01,
            "Deleña et al. · Zenodo 10.5281/zenodo.20066074 · CC BY 4.0 · resized previews",
            ha="center",
        )
        fig.tight_layout(rect=(0, 0.025, 1, 0.96))
        fig.subplots_adjust(hspace=0.22)
    elif stage == "crops":
        # What the species classifier receives: crops cut from published reference boxes.
        manifest = read(root / "data_manifest.json")
        by_id = {r["image_id"]: r for r in manifest["records"]}
        lineage = [x for x in read(out / "lineage.json") if x["split"] == "train"]
        species = ("rice_black_bug", "white_stemborer")
        fig, axes = plt.subplots(2, 6, figsize=(15, 6))
        for row_axes, name in zip(axes, species, strict=True):
            chosen, parents = [], set()
            for crop in lineage:
                if crop["species"] == name and crop["parent_image_id"] not in parents:
                    chosen.append(crop)
                    parents.add(crop["parent_image_id"])
                if len(chosen) == len(row_axes):
                    break
            for ax in row_axes:
                ax.axis("off")
            for ax, crop in zip(row_axes, chosen, strict=False):
                record = by_id[crop["parent_image_id"]]
                with Image.open(root / "data" / record["relative_path"]) as source:
                    ax.imshow(source.convert("RGB").crop(tuple(crop["effective_crop_xyxy"])))
                x1, y1, x2, y2 = crop["effective_crop_xyxy"]
                ax.set_title(f"{name.replace('_', ' ')}\n{x2 - x1}×{y2 - y1} px", fontsize=8)
        fig.suptitle("Classifier input: training crops from published reference boxes · labels unverified")
        fig.text(
            0.5,
            0.01,
            "Deleña et al. · Zenodo 10.5281/zenodo.20066074 · CC BY 4.0 · crops of resized exports",
            ha="center",
        )
        fig.tight_layout(rect=(0, 0.03, 1, 0.95))
    elif stage in {"classifier", "detector"}:
        history = read(out / f"{stage}_history.json")
        keys = (
            ["training_cross_entropy", "validation_cross_entropy"]
            if stage == "classifier"
            else ["training_loss", "validation_ap50"]
        )
        fig, axes = plt.subplots(1, len(keys), figsize=(12, 4), squeeze=False)
        for ax, key in zip(axes.flat, keys, strict=True):
            rows = [r for r in history if r.get(key) is not None]
            if not rows:
                raise ValueError(f"Missing learning evidence: {key}")
            ax.plot([r["epoch"] for r in rows], [r[key] for r in rows], marker="o")
            ax.set(xlabel="Epoch", ylabel=key.replace("_", " "), title=key.replace("_", " "))
        fig.suptitle("Validation selects the checkpoint; test results do not select epochs")
        fig.tight_layout()
    elif stage == "evaluate":
        values = read(out / "metrics.json")
        names = list(values)
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        axes[0].barh(names, [values[n]["mae"] for n in names])
        axes[0].set(
            xlabel="MAE per image/species (lower is better)", title="Held-out source-family benchmark"
        )
        with (out / "counts.csv").open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        reference = np.array([float(r["reference"]) for r in rows])
        prediction = np.array([float(r["raw"]) for r in rows])
        axes[1].scatter(reference, prediction, alpha=0.7)
        ceiling = max(float(reference.max()), float(prediction.max()), 1)
        axes[1].plot([0, ceiling], [0, ceiling], linestyle="--", color="black")
        axes[1].set(
            xlabel="Published reference count",
            ylabel="Raw predicted count",
            title=f"{len(rows)} image/species pairs",
        )
        fig.tight_layout()
    elif stage == "activity":
        with (out / "activity_thresholds.csv").open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        fig, axes = plt.subplots(1, 4, figsize=(16, 4))
        for ax, key in zip(axes.flat, ("mae", "missed", "spurious", "wrong_species"), strict=True):
            ax.plot([float(r["display_threshold"]) for r in rows], [float(r[key]) for r in rows], marker="o")
            ax.set(xlabel="Display threshold", ylabel=key, title=key)
        fig.suptitle("Retrospective illustration · locked policy unchanged")
        fig.tight_layout()
    else:
        raise ValueError("Unknown figure stage")
    fig.savefig(folder / f"{stage}.png", dpi=120, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--stage", required=True)
    args = parser.parse_args()
    render(args.root, args.stage)
