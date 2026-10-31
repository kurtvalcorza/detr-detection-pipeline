# DETR ResNet-50 object detection pipeline

DIMER pipeline for **DETR with a ResNet-50 backbone** (`facebook/detr-resnet-50`), the reference end-to-end set-prediction detector trained on COCO 2017. The pipeline loads the checkpoint only from a digest-verified local snapshot, returns pixel-space boxes with the model's softmax score under a caller-owned threshold, and adds a bounded fine-tuning workflow that re-heads DETR onto a new class vocabulary and exports a SafeTensors adapter.

> **The upstream snapshot is pinned** to Hub commit `1d5f47bd3bdd2c4bbfa585418ffe6da5028b4c0b` (pinned 2026-09-25). The manifest records every file's byte size and SHA-256, and each LFS digest matched the Hub's record. Default-path execution recorded on 2026-09-25 (Kaggle T4); REL12 BYOD exercise pending before promotion (see [Release status](#release-status)).

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

`tutorials/detr_detection_colab.ipynb` is declared `E2E` / `GUIDED` under DIMER Notebook Specification 2.5 and is **standalone** (§4); since 2026-10-04 the tutorial builds its own isolated `uv` Python 3.12.12 environment from the hash-locked `tutorials/requirements-colab.lock.txt` and runs every later cell there, so nothing is installed into the kernel and `Run all` needs no restart (**Linux x86_64 runtimes only**): `tools/build_notebook.py` generates it, and it carries the package modules, the model identity, the manifest and the runtime pins, so it runs without this repository. Its default `Run all` path detects on a drawn COCO scene, probes a blank and a noise image, validates a 40-image drawn sign dataset, measures a baseline, fine-tunes, evaluates the held-out split with COCO-style AP and, separately, what the adapted model returns at the operating threshold 0.9 (nothing, in the recorded runs: its scores stay below about 0.43) and at an adapted threshold chosen on the training split only (0.2905 on CPU: 1 of 5 unseen signs found), runs it on unseen images, and exports and reloads the adapter. BYOD image and dataset branches are off by default. See `tutorials/README.md` and `docs/release-verification.md`.

The separate [Philippine Rice Pest Surveillance capstone](tutorials/DIMER_Philippine_Rice_Pest_Surveillance_Capstone.ipynb)
composes adapted DETR with a BioCLIP species head and compares SigLIP and simple baselines.
Its 200-image sample contains 2,509 published boxes from a CC BY 4.0 PhilRice-affiliated dataset.
It is an **exploratory published-annotation benchmark**: source-family grouping does not establish
independent capture events or verified annotation completeness. The new notebook remains
**Candidate; passing fresh Colab T4 default Run all recorded on the current notebook; maintainer review required**. See [build and data-audit notes](docs/rice-capstone.md).

## Release status


**Candidate.** The snapshot is pinned (`1d5f47b`). Default-path execution recorded on 2026-09-25 (Kaggle T4): the exact notebook blob `a98705edf1cb` (commit `e39d680`) ran top-to-bottom with BYOD off; adapted held-out AP 0.2222 / AP50 0.2668 against 0.0063 / 0.0172 for the re-headed baseline on 10 drawn sign images (17 reference boxes), one seeded split, one runtime; the adapted model returned no detection on any of 3 new drawn images at the default threshold 0.9. That run needed a manual restart after the install cell, so it is not a one-pass `Run all` and not promotion evidence; the 2026-10-04 review-fix revision (uv isolated environment, commit `c3c81cb`), the tier C blob `da6b14bd4382` (commit `8fd8d5e`, 2026-10-25) and the previous blob `712d22d5ca82` (commit `c439bd4`, 2026-10-25, 171.5 s; adapted AP50 0.3617) each ran in one pass on a fresh Colab Tesla T4 (Colab CLI, 16/16 code cells, not a browser `Run all`). Blob `43868fa1a59e` (`c5f14ef`, final prose pass 2026-10-26: the Section 7 baseline and the Section 14 worked answer no longer state what a run is expected to print) is **Candidate** and passed its confirming hosted Colab T4 run on 2026-10-26 (PR #7 head `3b7e6d0`, Colab CLI, one pass, 0 errors; 16/16 code cells, 135.3 s wall; adapted `ap50` 0.4486, `ap` 0.3561; 0 boxes at 0.9; 3/5 unseen signs at the adapted threshold 0.1945; drift recorded in `docs/release-verification.md`); remaining gates REL12 and REL2. The previous blob `c19823537fa4` (`87eb45f`, historical wording) passed its confirming hosted Colab T4 run on 2026-10-25 (PR #7 head `dc65d30`, 16/16 code cells, one pass, 0 errors; `ap50` 0.3944, `ap` 0.346; drift recorded in `docs/release-verification.md`). REL12 BYOD exercise pending before promotion: release step 7 has not been run. Static checks, unit tests and the tiny-model test do not constitute notebook execution evidence; `docs/release-verification.md` defines the release gate.

The current blob `08904043` (code commit `2da6948`) shows a refused BYOD input as one `Refused: <file>: <reason>` line with no traceback, removes the earlier run's results and `outputs/` files and stops Run all at that cell, while genuine failures keep their traceback (NOTEBOOK_SPEC DAT19/DAT20); on 2026-10-29 its default path (16 code cells, exit 0, 0 errors) and its scripted DAT20 refusal (`USE_BYOD_IMAGE=True;BYOD_IMAGE_PATH=/etc/hostname`: exactly one `Refused:` line at cell `detr_detection-33`, no traceback, earlier cells clean, VM released) both passed on a hosted Colab Tesla T4 (Colab CLI `colab exec -f`, not a browser `Run all`). The remaining gates are REL12 (a hosted BYOD run with an accepted input) and REL2 (a browser run of the interactive journeys); earlier blobs' runs are history (REL14).

## Documentation

- `MODEL_CARD.md`: MODEL_CARD_SPEC 1.2 card, provenance, input/output contract.
- `docs/WEIGHTS.md`: weight provenance, pinning and hosting notes.
- `STATUS.md`: release status.

## Licensing

This repository's code is Apache-2.0 (see `LICENSE`). The upstream weights are Apache-2.0; see `docs/WEIGHTS.md` and `MODEL_CARD.md`.

## AI Assistance Disclosure

This repository’s code and accompanying documentation were developed with generative AI assistance for code development and technical writing under maintainer direction. The maintainer remains responsible for reviewing the implementation, validating results, and making release decisions. AI assistance does not constitute independent verification, provider endorsement, or release approval.
