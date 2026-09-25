# DETR ResNet-50 object detection pipeline

DIMER pipeline for **DETR with a ResNet-50 backbone** (`facebook/detr-resnet-50`), the reference end-to-end set-prediction detector trained on COCO 2017. The pipeline loads the checkpoint only from a digest-verified local snapshot, returns pixel-space boxes with the model's softmax score under a caller-owned threshold, and adds a bounded fine-tuning workflow that re-heads DETR onto a new class vocabulary and exports a SafeTensors adapter.

> **The upstream snapshot is pinned** to Hub commit `1d5f47bd3bdd2c4bbfa585418ffe6da5028b4c0b` (pinned 2026-09-25). The manifest records every file's byte size and SHA-256, and each LFS digest matched the Hub's record. No execution with the pinned weights is recorded yet (see [Release status](#release-status)).

## Upstream alignment

- Model: `facebook/detr-resnet-50`
- Revision: `1d5f47bd3bdd2c4bbfa585418ffe6da5028b4c0b`
- Upstream weight license: Apache-2.0
- Upstream task: object detection over the COCO 2017 categories (91 label slots, 80 trained with boxes)
- Repository adaptation: bounded gradient fine-tuning of the transformer and heads, with the ResNet-50 backbone frozen by default

## Quick start

```python
from PIL import Image
from detr_detection_pipeline import DetrDetectionPipeline, sign_dataset, split_dataset, SIGN_CLASSES

pipe = DetrDetectionPipeline.from_pretrained(allow_download=True)  # stages + verifies weights/detr-resnet-50
result = pipe.detect(Image.open("street.jpg"), threshold=0.9)      # threshold is caller-owned
for det in result["detections"]:                                   # sorted by score; boxes are xyxy pixels
    print(det["label"], det["box"], round(det["score"], 3))

train, held_out = split_dataset(sign_dataset(40), train_fraction=0.75)
adapter = DetrDetectionPipeline.from_pretrained(class_names=SIGN_CLASSES)
print(adapter.evaluate(held_out)["ap50"])                          # baseline
adapter.finetune(train)                                            # 10 epochs, backbone frozen
print(adapter.evaluate(held_out)["ap50"])                          # adapted
adapter.save_artifact("outputs/detr_adapter.safetensors")
```

Install into a Python 3.12 environment that already holds the pinned dependencies with `pip install -e . --no-deps`, and run `pytest` for the offline test suite (no weights needed; `tests/test_tiny_model.py` builds a tiny random-weight DETR to exercise fine-tuning, evaluation and adapter reload).

## Pinning the snapshot

The snapshot is pinned (see [Upstream alignment](#upstream-alignment)). To move to a newer upstream commit, from the repository root with network access to huggingface.co:

1. Run `python tools/pin_snapshot.py` (or `--revision <commit>`). It resolves `main` to a commit, downloads the four manifest files at that commit into `weights/detr-resnet-50/`, checks each LFS file against the Hub's SHA-256, and writes the commit and digests into the manifest and `MODEL_REVISION`.
2. Commit, then run `python tools/build_notebook.py` and commit the regenerated notebook.
3. Update the commit and digests cited in `README.md`, `MODEL_CARD.md`, `STATUS.md`, `docs/WEIGHTS.md`, `tutorials/README.md` and `docs/release-verification.md`.
4. Run `python tools/validate_release_assets.py` and `pytest`. A new pin invalidates any recorded execution, so the status returns to Candidate until the new commit is run.

## Weights layout

```
weights/detr-resnet-50/
  dimer-base-manifest.json   # modelId, revision, per-file bytes + SHA-256 (4 files)
  config.json                # DetrForObjectDetection: timm resnet50 backbone, 100 queries, 91 label slots
  preprocessor_config.json   # shorter side 800 px, longer side <= 1333 px, ImageNet normalisation
  model.safetensors          # git-ignored, 166,587,896 bytes
  README.md                  # staged with the weights
```

## Input ceilings and threshold

`MIN_IMAGE_SIDE = 16`, `MAX_IMAGE_SIDE = 4096`, `MAX_DETECTIONS = 100` (the checkpoint's `num_queries`), `LABELS` (91 slots in `id2label` order; `UNANNOTATED_LABEL_IDS` lists the 11 that COCO 2017 never annotated), `DETECTION_THRESHOLD = 0.9` (the pinned README example's value). Adaptation datasets hold 1–5,000 records. See `MODEL_CARD.md` for who owns the threshold and what the score means.

## Tutorials

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kurtvalcorza/detr-detection-pipeline/blob/main/tutorials/detr_detection_colab.ipynb)

`tutorials/detr_detection_colab.ipynb` is declared `E2E` / `GUIDED` under DIMER Notebook Specification 2.1 and is **standalone** (§4): `tools/build_notebook.py` generates it, and it carries the package modules, the model identity, the manifest and the runtime pins, so it runs without this repository. Its default `Run all` path detects on a drawn COCO scene, probes a blank and a noise image, validates a 40-image drawn sign dataset, measures a baseline, fine-tunes, evaluates the held-out split with COCO-style AP, detects on unseen images, and exports and reloads the adapter. BYOD image and dataset branches are off by default. See `tutorials/README.md` and `docs/release-verification.md`.

## Release status

**Candidate.** The snapshot is pinned (`1d5f47b`), but no execution with the pinned weights is recorded. Static checks, unit tests and the tiny-model test do not constitute notebook execution evidence; `docs/release-verification.md` defines the release gate.

## Documentation

- `MODEL_CARD.md`: MODEL_CARD_SPEC 1.2 card, provenance, input/output contract.
- `docs/WEIGHTS.md`: weight provenance, pinning and hosting notes.
- `STATUS.md`: release status.

## Licensing

This repository's code is Apache-2.0 (see `LICENSE`). The upstream weights are Apache-2.0; see `docs/WEIGHTS.md` and `MODEL_CARD.md`.

## AI Assistance Disclosure

This repository’s code and accompanying documentation were developed with generative AI assistance for code development and technical writing under maintainer direction. The maintainer remains responsible for reviewing the implementation, validating results, and making release decisions. AI assistance does not constitute independent verification, provider endorsement, or release approval.
