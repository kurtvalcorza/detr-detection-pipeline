# Tutorials

[![GitHub](https://img.shields.io/badge/GitHub-181717?style=flat&logo=github&logoColor=white)](https://github.com/kurtvalcorza/detr-detection-pipeline)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kurtvalcorza/detr-detection-pipeline/blob/main/tutorials/detr_detection_colab.ipynb)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-facebook%2Fdetr--resnet--50-ffcc4d?style=flat)](https://huggingface.co/facebook/detr-resnet-50)
[![Upstream](https://img.shields.io/badge/Upstream-facebookresearch%2Fdetr-181717?style=flat&logo=github&logoColor=white)](https://github.com/facebookresearch/detr)
[![arXiv](https://img.shields.io/badge/arXiv-2005.12872-b31b1b.svg)](https://arxiv.org/abs/2005.12872)

Notebook specification: **DIMER Notebook Specification 2.1**. The notebook is **standalone** (§4) and declares its profile and pedagogical mode (§3.4). `tools/build_notebook.py` generates it from `tools/notebook_template.py`, and it carries the package's modules, the model identity, the snapshot manifest and the runtime pins, so the exported `.ipynb` works without this repository. Do not edit the notebook by hand: edit the package or the template and regenerate (`python tools/build_notebook.py`; CI and the validator enforce `--check`).

| Notebook | Profile | Mode | Carrier | Capability | Default runtime | Sample | BYOD | Run-all | Release status |
|---|---|---|---|---|---|---|---|---|---|
| `detr_detection_colab.ipynb` | `E2E` | `GUIDED` | standalone (generated) | COCO detection with `facebook/detr-resnet-50`, and a bounded fine-tune that re-heads DETR onto a three-class sign vocabulary, evaluates baseline and adapted AP@[.50:.95] and AP50 on a held-out split, infers on unseen images, and exports and reloads a SafeTensors adapter | CUDA GPU (T4 class) documented; CPU also runs | automatic (640×480 COCO scene and 40-image sign dataset, both drawn in code) | single image or labelled dataset directory, off by default; location fields `BYOD_IMAGE_PATH`, `BYOD_DATASET_DIR`; uploads stay in the runtime | not yet run on the pinned snapshot (`1d5f47b`); see `../docs/release-verification.md` | **Candidate** |

## Conformance notes

- **Standalone carrier (§4):** the default path performs no clone, repository install or repository import. Section 2 carries `samples.py` and `pipeline.py` verbatim in dependency order (tagged `metadata.dimer.embedded_module`; the only rewrite makes the default weights directory working-directory-relative). Section 3 carries the model identity and the manifest inline and asserts that they agree with the module before anything is fetched. Section 1 carries the exact `pyproject.toml` runtime pins.
- **Parity (PAR1–PAR3):** `tests/test_notebook_parity.py` and `tools/validate_release_assets.py` fail when a carried module, the inline manifest or the inline pins differ from the repository, or when the notebook differs from the generator's output.
- **Model acquisition (MOD1–MOD8):** the snapshot is pinned to an immutable revision (the three loaders would raise before any download if `MODEL_REVISION` were reset to `"unpinned"`). `stage_missing_files(WEIGHTS_DIR, allow_download=True)` fetches only the absent manifest entries at the immutable revision, `verify_snapshot` re-hashes every entry, and `from_pretrained(weights_dir=WEIGHTS_DIR)` loads with `local_files_only=True`, `trust_remote_code=False` and `use_pretrained_backbone=False`.
- **Stages:** `validate_inputs` writes the input manifest, including one rejection finding from the `threshold=1.5` probe; `evaluation_report` writes the single-image `sample-sanity` report; `validate_dataset` validates the adaptation records; `evaluate` computes AP before and after `finetune`; `save_artifact` writes `outputs/detr_adapter.safetensors`; `load_artifact` rebuilds from the verified base and the adapter, and the notebook compares detections within a stated tolerance.
- **Scores (UNC1–UNC4):** every `score` is a softmax probability over the classes plus "no object", not a calibration for the reader's images. `threshold` is a `# @param` field passed explicitly on every `detect` call.
- **Non-interactive execution (EXE1–EXE4):** seeds, schedule, threshold, BYOD switches and BYOD locations are form fields; outputs go to `outputs/` under the working directory.
- `tools/validate_release_assets.py` performs source validation only. It does not satisfy the clean-runtime execution requirement; a release review must confirm that a recorded clean run in `docs/release-verification.md` matches the notebook revision under review before the status is promoted to `Release-grade`.

## AI Assistance Disclosure

This repository’s code and accompanying documentation were developed with generative AI assistance for code development and technical writing under maintainer direction. The maintainer remains responsible for reviewing the implementation, validating results, and making release decisions. AI assistance does not constitute independent verification, provider endorsement, or release approval.
