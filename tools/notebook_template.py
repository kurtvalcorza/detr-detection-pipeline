"""Per-repository template for tools/build_notebook.py (NOTEBOOK_SPEC 2.2 §4 standalone carrier).

Only the task-specific prose and stage cells live here. Runtime install, the embedded package, and the
model pin/stage/verify cells are produced by the generator from repository sources so they cannot
drift from the package.

This is an `E2E` template, so it must state `run_all` itself, and its default path really adapts:
NOTEBOOK_SPEC 2.2 RUN7/FT2 make a bounded fine-tune mandatory rather than optional for this profile.
Every value a reader can change is a `# @param` form field, and each file-reading BYOD branch has a
location field that bypasses the upload dialog when set (EXE1, EXE2).

Review 2026-10-02 (DTR-M1..M4, DTR-m1..m3), ported from the sibling conditional-detr-detection-pipeline fix
(fd008c1, merged 879e0b1): the notebook runs in the fleet's uv isolated environment (generator /2.1), carries
the GUIDED layer (audience, task contract, how to use, roadmap, predictions, worked answers, a change-one-thing
activity, troubleshooting, glossary, conclusion template), says what the adapted model does at the operating
threshold instead of claiming it detects the new classes, chooses an adapted threshold on the training split
only (Kurt, 2026-10-04), rebuilds the re-headed model before every fine-tune, and writes BYOD results with a
reload check. All of it is tutorial-side: src/ is unchanged.
"""
# ruff: noqa: E501  -- markdown prose and code-cell text are kept on single lines for readable rendering

TEMPLATE = {
    "package": "detr_detection_pipeline",
    "repo_name": "detr-detection-pipeline",
    "stem": "detr_detection",
    "notebook_name": "detr_detection_colab.ipynb",
    "profile": "E2E",
    "mode": "GUIDED",
    # GDL11 (NOTEBOOK_SPEC 2.2 §3.5): Sections 1-3 labelled Infrastructure and the carried module source collapsed.
    "infrastructure_labels": True,
    "isolated_runtime": True,
    # The fleet's uv isolated-environment mechanism (bart-mnli-zero-shot-classification-pipeline ee128d2, generator /2.1):
    # managed CPython, a size- and SHA-256-verified uv wheel, and a lock compiled from the pyproject pins with
    # `uv pip compile pyproject.toml --python-version 3.12 --python-platform x86_64-manylinux_2_28 --generate-hashes
    # --only-binary :all: -o tutorials/requirements-colab.lock.txt`, transitive versions constrained to the
    # depth-anything-depth-estimation-pipeline lock already run on Colab T4 (timm and scipy are the only additions).
    "managed_python": "3.12.12",
    "uv": {
        "version": "0.12.15",
        "url": "https://files.pythonhosted.org/packages/1e/fd/432451d732917c49152a291de3ef171aa6b0f1a22d39780fb2c1f085ca4c/uv-0.12.15-py3-none-manylinux_2_17_x86_64.manylinux2014_x86_64.whl",
        "bytes": 20081404,
        "sha256": "aee9802f46bae436bd91751bb33ddeb379ef1596b5c19df193219d545d244b60",
    },
    "lock": "tutorials/requirements-colab.lock.txt",
    "pipeline_class": "DetrDetectionPipeline",
    "weights_key": "detr-resnet-50",
    "modules": [
        "samples.py",
        "pipeline.py",
    ],
    "entry_module": "pipeline.py",
    "runtime_imports": ["torch", "transformers", "timm", "scipy", "numpy", "PIL"],
    "title": "DETR ResNet-50 (COCO) — DIMER object detection and bounded detection fine-tuning (standalone)",
    "badges": [
        (
            "GitHub",
            "https://img.shields.io/badge/GitHub-181717?style=flat&logo=github&logoColor=white",
            "https://github.com/kurtvalcorza/detr-detection-pipeline",
        ),
        (
            "Open In Colab",
            "https://colab.research.google.com/assets/colab-badge.svg",
            "https://colab.research.google.com/github/kurtvalcorza/detr-detection-pipeline/blob/main/tutorials/detr_detection_colab.ipynb",
        ),
        (
            "Hugging Face",
            "https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-facebook%2Fdetr--resnet--50-ffcc4d?style=flat",
            "https://huggingface.co/facebook/detr-resnet-50",
        ),
        (
            "Upstream",
            "https://img.shields.io/badge/Upstream-facebookresearch%2Fdetr-181717?style=flat&logo=github&logoColor=white",
            "https://github.com/facebookresearch/detr",
        ),
        ("arXiv", "https://img.shields.io/badge/arXiv-2005.12872-b31b1b.svg", "https://arxiv.org/abs/2005.12872"),
        ("License", "https://img.shields.io/badge/License-Apache--2.0-green.svg", "https://github.com/kurtvalcorza/detr-detection-pipeline/blob/main/LICENSE"),
    ],
    "capability": "object detection over the COCO classes with the DETR ResNet-50 reference model, and a bounded detection fine-tune that re-heads DETR onto your own class vocabulary, evaluates it against a held-out split with COCO-style average precision, and exports a reloadable SafeTensors adapter",
    "intro": (
        "DETR (DEtection TRansformer) with a ResNet-50 backbone (`facebook/detr-resnet-50`) is the reference end-to-end set-prediction "
        "detector: a ResNet-50 feature map feeds a 6-layer transformer encoder and a 6-layer decoder with 100 learned object queries, and "
        "each query emits one box and one class distribution over the COCO category slots plus a **no object** class. Training matches "
        "queries to reference boxes one-to-one with the Hungarian algorithm, so the model needs no anchor boxes and no non-maximum "
        "suppression (NMS). At inference the processor resizes the image so its shorter side is 800 px, and every query whose best class "
        "probability reaches a caller-owned threshold is returned as an xyxy box in input pixels.\n\n"
        "**The default path really adapts the model:** it re-heads DETR onto a three-class traffic sign vocabulary that does not exist in "
        "COCO (`stop-sign`, `yield-sign`, `speed-limit-sign`), measures a pre-adaptation baseline, runs a bounded fine-tune with the ResNet "
        "backbone frozen, scores the result on a held-out split with COCO-style average precision (AP@[.50:.95] and AP50), runs the adapted "
        "model on unseen images, exports the changed tensors as a SafeTensors adapter, and reloads that adapter onto a fresh copy of the "
        "verified base model to check that it reproduces the same detections. Every number you see is measured in this notebook runtime. "
        "One result is worth knowing in advance: after this short fine-tune the adapted model ranks the right boxes first (high held-out AP) "
        "but scores them far below the default threshold of 0.9, so at that threshold it returns **no** boxes on the unseen images. "
        "Section 9 therefore also chooses a separate **adapted threshold** for the new classes on the training split only, and Sections "
        "9–12 show the adapted model at both thresholds side by side.\n\n"
        "**Who this is for.** A learner who knows basic Python and PIL, has met bounding boxes and intersection-over-union, and wants to see how "
        "a pretrained detector is re-headed and fine-tuned for a new class vocabulary, and how the result is measured without fooling "
        "themselves. No prior experience with DETR, transformers or fine-tuning is assumed; each term is explained where it is first used and "
        "again in the **Glossary** at the end.\n\n"
        "**Input → Model → Output.**\n\n"
        "| | Inference | Adaptation |\n"
        "|---|---|---|\n"
        "| Input | one RGB image, sides 16..4,096 px, and a score threshold you choose | 30 training images, each with xyxy pixel boxes and labels from a three-class sign vocabulary |\n"
        "| Model | DETR ResNet-50 with its 91-slot COCO class head | the same model with a fresh three-class head; transformer, class head and box head trained, ResNet backbone frozen |\n"
        "| Output | at most `MAX_DETECTIONS` (100, one per query) xyxy boxes, each with a label and a softmax score at or above the threshold | held-out AP before and after, detections on unseen images, and a SafeTensors adapter that reloads to the same detections |\n\n"
        "**How to use this notebook.** Choose a runtime (a GPU runtime is much faster; CPU works), then **Runtime → Run all**. Sections 1–3 are "
        "**infrastructure** — the isolated environment, the carried code (collapsed) and the model verification — and can be run without study. "
        "The learning path starts in Section 4. Form fields (`# @param`) are the only values meant to be edited; the defaults reproduce the "
        "recorded path. Before Sections 4, 5, 7, 8, 9 and 10 run you are asked to **predict** what they will print; the following section opens "
        "with **What to notice** and a collapsible **Check your reasoning** block with a worked answer from a recorded run. Section 8 always "
        "starts training from the freshly re-headed model, so re-running it after a change compares the new setting against the same starting "
        "point. Section 14 is a **change-one-thing activity**; it and each optional experiment name the cell to change and the cell to re-run "
        "from (**Runtime → Run after**). **Troubleshooting**, a **Glossary** and a **Conclusion** template are at the end. Your notes are "
        "optional and are not required submissions.\n\n"
        "**Roadmap:** 4 the pretrained detector on a drawn scene → 5 blank and noise probes → 6 build and validate the sign dataset → 7 split, "
        "re-head and measure the baseline → 8 bounded fine-tune → 9 held-out AP, and what it does not say → 10 unseen images at the operating "
        "threshold → 11 export and reload the adapter → 12 outputs → 13 optional BYOD → 14 **change one thing: unfreeze the backbone** → conclude."
    ),
    "learning_objectives": (
        "by the end you should be able to (1) describe the path *image → ResNet-50 features → transformer encoder and decoder with 100 queries "
        "→ a box and a softmax over the classes plus no-object per query* and say why a threshold decides what is returned (Section 4); (2) explain "
        "why any box on a blank or noise image is a false positive, and what the count at the evaluation threshold measures (Section 5); "
        "(3) explain why a new vocabulary needs a new class head and why its baseline is near zero (Sections 6–7); (4) read a training loss as "
        "optimisation evidence and held-out AP50 and AP@[.50:.95] as task evidence (Sections 8–9); (5) explain why a held-out AP50 near 0.3 "
        "can coexist with zero detections at threshold 0.9, by telling a ranking metric from a fixed-threshold decision, and why a threshold chosen on the training split changes the decision but not the ranking (Sections 9–10); "
        "(6) check that an exported adapter reproduces the evaluated model (Section 11); and (7) predict and then measure what unfreezing the "
        "backbone changes when both runs start from the same re-headed model (Section 14). Along the way the notebook builds a locked runtime, "
        "stages and digest-verifies the immutable upstream model revision, and writes machine-readable outputs with provenance."
    ),
    "exclusions": (
        "real-world traffic sign detection (the adaptation dataset is drawn in code, so the model learns these renderings and nothing about "
        "road photographs); COCO benchmark results (the average-precision helper here is a compact implementation without pycocotools area "
        "ranges or crowd handling, and it is run on synthetic data only); score calibration or a deployment threshold for the adapted model (the training-split threshold of Section 9 is a tutorial choice on 30 drawn images, not a deployment setting); full-schedule DETR "
        "training (the upstream schedule is 300 epochs on 16 GPUs; the tutorial runs a few epochs on 30 images); panoptic or instance "
        "segmentation; video tracking."
    ),
    "prerequisites": [
        "- **Learner:** basic Python and PIL, and Colab or Jupyter familiarity; no prior experience with object detection models or fine-tuning. The notebook explains queries, Hungarian matching, the softmax score with its no-object class, re-heading, AP50 and AP@[.50:.95] where they are first used; the Glossary repeats them.",
        "- **Runtime:** a fresh supported runtime (Google Colab, Kaggle or Linux Jupyter — **Linux x86_64 only**; the notebook builds its own isolated Python 3.12.12 environment, so a Windows or macOS kernel is not supported). A CUDA GPU such as a Colab or Kaggle T4 is recommended for the fine-tune and is used automatically when present; the notebook also runs on CPU, more slowly. Measured, each figure with its environment: a Kaggle Tesla T4 run of the earlier in-kernel-install revision (notebook blob `a98705ed`, 2026-09-25) took 357.5 s in two passes — 267.4 s until the install cell stopped for a restart, then 90.0 s for every cell after the restart; a local CPU run of the same blob (24-thread Windows workstation, 2026-10-03, install skipped, files pre-staged) took 212 s of cell time, 185 s of it the fine-tune; a local CPU run of this revision (same workstation, 2026-10-04, install skipped, files pre-staged, host shared with other jobs) took 863 s of cell time, 548 s of it the fine-tune. This revision's isolated-environment build has not yet been timed on a hosted runtime; expect it to add several minutes (an estimate). The locked install (PyTorch 2.14.0 with its CUDA libraries) and the ~167 MB checkpoint are the largest downloads.",
        "- **Knowledge:** basic Python and PIL; bounding boxes as xyxy pixel coordinates; intersection-over-union (IoU); and how to read average precision (AP50 and AP@[.50:.95]).",
        "- **Data:** the default path generates everything in code with `samples.py` and downloads no dataset: one 640×480 COCO demonstration scene and a 40-image labelled sign dataset. BYOD is optional and off by default. Expected BYOD input: one image, or `annotations.json` — a list of `{'file': 'name.png', 'boxes': [[x0, y0, x1, y1], ...], 'labels': [name, ...]}` objects — and the image files it names. Through the upload dialog, upload `annotations.json` and the images as **flat files (no folders)**, or one `.zip` that holds them (folders are allowed inside a zip; a zip may hold at most 10,001 files and 2 GiB). A location field (`BYOD_DATASET_DIR`) may name a directory or a `.zip`.",
        "- **Privacy:** Do not upload confidential or restricted data to a hosted notebook environment unless you are authorized to do so; uploaded inputs stay in this runtime and are not sent to any inference API. The default path uploads nothing.",
    ],
    "run_all": (
        "Selecting **Run all** in a fresh supported runtime builds an isolated environment from the hash-locked pins (the kernel's own "
        "packages are left alone, so no restart is needed), stages and digest-verifies the pinned checkpoint, runs COCO detection on a drawn "
        "scene, validates the 40-image sign dataset, splits it into training and held-out parts, measures the pre-adaptation baseline, **runs "
        "the bounded fine-tune**, re-evaluates on the held-out split, runs the adapted model on unseen images, exports the adapter, reloads it "
        "onto a fresh base model to verify the detections, and writes machine-readable outputs with provenance. Nothing is skipped behind a "
        "default-off flag, and no clone, DIMER worker, credential, upload dialog or configuration edit is required (NOTEBOOK_SPEC 2.2 §5, "
        "RUN7, FT2). CUDA is used when present; a GPU runtime is recommended. Measured times are listed under Prerequisites."
    ),
    "byod": (
        "Two optional BYOD branches are included, and both are off by default (`USE_BYOD_IMAGE = False`, `USE_BYOD_DATASET = False`). "
        "`USE_BYOD_IMAGE` runs your own image through the same validation, detection and evaluation-report stages as the sample scene. "
        "`USE_BYOD_DATASET` takes your own labelled detection records through the full adaptation workflow — validate, split, baseline, "
        "fine-tune, evaluate, export, reload and compare — under NOTEBOOK_SPEC 2.2 DAT14, and writes its own result JSON and detections CSV. "
        "Set `BYOD_IMAGE_PATH` or `BYOD_DATASET_DIR` (a directory or a `.zip`) to read from a location without an upload dialog (EXE2)."
    ),
    "cells": [
        # ---------------------------------------------------------------- 4. COCO scene
        {
            "md": (
                "## 4. What the pretrained detector does on a drawn scene\n\n"
                "Before adapting anything, inspect the model you start from. The carried `samples` module draws a deterministic street scene with "
                "four objects and their reference boxes: a **stop sign**, a **traffic light**, an analogue **clock**, and an orange **sports ball**. "
                "All four are COCO classes.\n\n"
                "The detection threshold is a **caller-owned request parameter**, not a pipeline constant. Each score is the query's largest class "
                "probability from a **softmax over the classes and a no-object class, not a calibrated** probability for your images. The default "
                "`0.9` is the value the pinned model README uses, and it is passed explicitly on every call.\n\n"
                "The scene is drawn, not photographed, so a miss here is a finding about renderings, not about photographs. The evaluation report "
                "records every miss with `box_iou = 0.0`. COCO mean average precision needs a labelled image set; on one scene the verdict is `sample-sanity`.\n\n"
                "**Predict before running:** how many of the four drawn objects will the COCO head find at threshold 0.9 with IoU ≥ 0.5? Which one "
                "do you expect a model trained on photographs to miss in a flat drawing?"
            ),
            "code": (
                "import hashlib\n"
                "import io\n"
                "import json\n"
                "import os\n"
                "from pathlib import Path\n\n"
                "os.makedirs('outputs', exist_ok=True)\n"
                "OUTPUTS = Path('outputs')\n\n"
                'threshold = 0.9  # @param {{type:"number"}}\n\n'
                "print({{'MIN_IMAGE_SIDE': MIN_IMAGE_SIDE, 'MAX_IMAGE_SIDE': MAX_IMAGE_SIDE, 'MAX_DETECTIONS': MAX_DETECTIONS,\n"
                "       'label_slots': len(LABELS), 'unannotated_slots': len(UNANNOTATED_LABEL_IDS), 'DETECTION_THRESHOLD': DETECTION_THRESHOLD}})\n\n"
                "scene, references = tutorial_scene()\n"
                "buffer = io.BytesIO()\n"
                "scene.save(buffer, format='PNG')\n"
                "print({{'sample_kind': 'synthetic', 'size': list(scene.size), 'sha256': hashlib.sha256(buffer.getvalue()).hexdigest()[:16],\n"
                "       'references': {{label: len(boxes) for label, boxes in references.items()}}}})\n\n"
                "input_manifest = validate_inputs(scene, threshold=threshold, names=['tutorial-scene'])\n"
                "try:\n"
                "    validate_inputs(scene, threshold=1.5)\n"
                "except ValueError as exc:\n"
                "    input_manifest['findings'].append({{'probe': 'threshold=1.5', 'rejected': str(exc)}})\n"
                "print({{'verdict': input_manifest['verdict'], 'findings': input_manifest['findings'], 'inputs': input_manifest['inputs']}})\n\n"
                "coco_result = pipe.detect(scene, threshold=threshold)\n"
                "for det in coco_result['detections']:\n"
                "    print(f\"{{det['label']:>14s}} {{det['score']:.3f}}  [{{', '.join(f'{{v:.0f}}' for v in det['box'])}}]\")\n\n"
                "coco_report = evaluation_report(coco_result, references, sample_kind='synthetic')\n"
                "print({{'verdict': coco_report['verdict'], 'n_detections': coco_report['n_detections']}})\n"
                "for metric in coco_report['metrics']:\n"
                "    print(f\"  {{metric['reference']:>18s}}  box_iou {{metric['value']:.3f}}  same-label detections {{metric['n_detected_same_label']}}\")\n"
                "hits = sum(1 for m in coco_report['metrics'] if m['value'] >= 0.5)\n"
                "print(f'{{hits}}/{{len(coco_report[\"metrics\"])}} drawn objects matched at IoU >= 0.5')\n"
                "scene"
            ),
        },
        # ---------------------------------------------------------------- 5. Degenerate inputs
        {
            "md": (
                "**What to notice (Section 4):** the matched count, which objects scored at or above 0.9, and the `box_iou` of each reference.\n\n"
                "<details><summary>Check your reasoning</summary>In the recorded runs (Kaggle T4, 2026-09-25, and a local CPU run, 2026-10-03) the "
                "head returned `stop sign` 0.999 (IoU 0.847), `traffic light` 0.997 (IoU 0.910) and `clock` 0.973 (IoU 0.898), so 3 of the 4 drawn "
                "objects matched; the sports ball was missed and is recorded with `box_iou` 0.0. A flat orange disc is weaker evidence for a "
                "photograph-trained detector than an octagon with a word on it, a box of coloured lamps or a clock face.</details>\n\n"
                "## 5. Degenerate input probes: blank canvas and noise\n\n"
                "Ask a detector about structure-free input before trusting it. A model that returns confident boxes on a blank canvas or on "
                "uniform noise will return them on empty real frames too. The cell counts detections on both images at the default threshold "
                "and at the evaluation threshold `EVAL_DETECTION_THRESHOLD` (0.05) that average precision is computed at.\n\n"
                "**The count is capped.** `detect` returns at most `MAX_DETECTIONS` (100) boxes — one per object query, each under its best "
                "class — so a count of 100 would mean \"every query scored above the threshold\", not \"the model saw 100 objects\".\n\n"
                "**What to look for:** any detection above the default threshold on these two images is a false positive by construction.\n\n"
                "**Predict before running:** will either image return a box at 0.9? Will noise return any at 0.05?"
            ),
            "code": (
                "degenerate = {{}}\n"
                "for name, image in (('blank', blank_scene()), ('noise', noise_scene(0))):\n"
                "    standard = pipe.detect(image, threshold=threshold)['detections']\n"
                "    lenient = pipe.detect(image, threshold=EVAL_DETECTION_THRESHOLD)['detections']\n"
                "    degenerate[name] = {{\n"
                "        'at_default_threshold': len(standard),\n"
                "        'at_evaluation_threshold': len(lenient),\n"
                "        'cap': MAX_DETECTIONS,\n"
                "        'top': [(d['label'], round(d['score'], 3)) for d in lenient[:3]],\n"
                "    }}\n"
                "print(json.dumps(degenerate, indent=2))"
            ),
        },
        # ---------------------------------------------------------------- 6. Dataset & Validation
        {
            "md": (
                "**What to notice (Section 5):** the counts at 0.9 against the counts at 0.05.\n\n"
                "<details><summary>Check your reasoning</summary>Both probes returned 0 boxes at 0.9 and 0 at 0.05 as well (Kaggle T4, 2026-09-25, "
                "and local CPU runs). DETR's score is a softmax that includes a **no object** class, and on structure-free input every query puts "
                "almost all of its probability there, so not even the lenient 0.05 evaluation threshold lets a box through. That is good behaviour "
                "on these two images; it is not evidence about empty real frames, which still contain texture and edges.</details>\n\n"
                "## 6. Labelled adaptation dataset and validation\n\n"
                "Suppose your task needs sign classes that COCO does not have. `sign_dataset` draws a deterministic 40-image dataset over "
                "`SIGN_CLASSES`: `stop-sign`, `yield-sign` and `speed-limit-sign`.\n\n"
                "**Keep the two vocabularies apart.** COCO has `stop sign` (with a space). The adaptation vocabulary uses `stop-sign` (hyphenated), "
                "`yield-sign` and `speed-limit-sign`. They are different class identities, and the adapted model answers in the new names only.\n\n"
                "`validate_dataset` checks every record before any model runs: the record keys, the image size ceilings, that every box is finite, "
                "non-empty and inside its image, and that every label is in the vocabulary. It returns a dataset manifest with the box count per "
                "class and a finding for any class that has no box. `EPOCHS` lives here because the manifest records it."
            ),
            "code": (
                'N_IMAGES = 40  # @param {{type:"integer"}}\n'
                'DATASET_SEED = 0  # @param {{type:"integer"}}\n'
                'EPOCHS = 10  # @param {{type:"integer"}}\n\n'
                "records = sign_dataset(N_IMAGES, seed=DATASET_SEED)\n"
                "dataset_manifest = validate_dataset(records, SIGN_CLASSES, epochs=EPOCHS)\n"
                "print(json.dumps({{k: v for k, v in dataset_manifest.items() if k != 'schema'}}, indent=2))\n\n"
                "preview = Image.new('RGB', (480, 320))\n"
                "for index, record in enumerate(records[:6]):\n"
                "    preview.paste(record['image'].resize((160, 160)), (160 * (index % 3), 160 * (index // 3)))\n"
                "preview"
            ),
        },
        # ---------------------------------------------------------------- 7. Split & Baseline
        {
            "md": (
                "## 7. Split, re-head, and measure the pre-adaptation baseline\n\n"
                "The dataset is split at random into a training part (75%, 30 images) and a held-out part (25%, 10 images). A random split is "
                "valid here because every image is drawn independently; records that share a photograph or a camera session must be split by that "
                "group instead. The held-out images are never shown to the optimizer, and no hyperparameter is selected on them.\n\n"
                "`from_pretrained(class_names=SIGN_CLASSES)` loads the verified checkpoint and replaces only the class head "
                "(`class_labels_classifier`, now 3 classes plus no-object) with a freshly initialised layer, seeded by `SEED`. This is **re-heading**. "
                "The backbone, the transformer and the box head keep their COCO weights.\n\n"
                "**The baseline is expected to be near zero.** A freshly initialised class head has no information about the new classes, so "
                "this number is the floor the fine-tune has to beat, not a property of DETR.\n\n"
                "**Predict before running:** what baseline AP50 do you expect on the 10 held-out images?"
            ),
            "code": (
                'HOLDOUT = 0.25  # @param {{type:"number"}}\n'
                'SEED = 0  # @param {{type:"integer"}}\n\n'
                "train_records, held_out = split_dataset(records, train_fraction=1.0 - HOLDOUT, seed=SEED)\n"
                "overlap = {{r['id'] for r in train_records}} & {{r['id'] for r in held_out}}\n"
                "assert not overlap, f'split leaked records: {{sorted(overlap)}}'\n"
                "print({{'train': len(train_records), 'held_out': len(held_out),\n"
                "       'train_boxes': sum(len(r['boxes']) for r in train_records),\n"
                "       'held_out_boxes': sum(len(r['boxes']) for r in held_out)}})\n\n"
                "adapter = DetrDetectionPipeline.from_pretrained(weights_dir=WEIGHTS_DIR, class_names=SIGN_CLASSES, seed=SEED)\n"
                "print({{'class_names': list(adapter.class_names), 'device': adapter.device, 'reinitialised': list(adapter.reinitialised)}})\n\n"
                "baseline = adapter.evaluate(held_out)\n"
                "print(json.dumps({{'ap': round(baseline['ap'], 4), 'ap50': round(baseline['ap50'], 4),\n"
                "                  'per_class_ap50': {{k: round(v, 4) for k, v in baseline['per_class_ap50'].items()}},\n"
                "                  'n_references': baseline['n_references']}}, indent=2))"
            ),
        },
        # ---------------------------------------------------------------- 8. Bounded Fine-Tuning
        {
            "md": (
                "**What to notice (Section 7):** the split sizes, the two re-initialised class-head tensors, and the baseline AP.\n\n"
                "<details><summary>Check your reasoning</summary>The recorded runs split 30 / 10 images (65 / 17 boxes) and measured a baseline "
                "`ap` of 0.0063 and `ap50` of 0.0172 (Kaggle T4 and local CPU alike): the fresh head spreads each query's probability over the "
                "three new classes and no-object without any knowledge of them, so the few boxes that pass the 0.05 evaluation threshold are "
                "ranked by chance. Near zero is the expected floor.</details>\n\n"
                "## 8. Bounded detection fine-tuning\n\n"
                "This cell runs the real adaptation step in this runtime. It is gradient fine-tuning with DETR's own loss: each image's 100 queries "
                "are matched one-to-one to its reference boxes by the **Hungarian algorithm**; the matched queries are trained with cross-entropy "
                "toward their reference class and the unmatched ones toward **no object** (that term is down-weighted by the config's "
                "`eos_coefficient` 0.1), plus an L1 box term and a generalised-IoU box term on the matched queries.\n\n"
                "- **The backbone is frozen.** Parameters under `model.backbone.` keep their COCO values; the transformer, the class head and the "
                "box head are trained. The cell prints the trainable and total parameter counts. (`FREEZE_BACKBONE` is the Section 14 activity.)\n"
                "- **Schedule:** `EPOCHS` epochs of AdamW at learning rate `1e-4` (weight decay `1e-4`), batch size 4, gradient-norm clipping at 0.1, "
                "float32, seed `SEED`.\n"
                "- **Adaptation always starts from the re-headed model.** `finetune` changes the model in place. If you re-run this cell after the "
                "model has been fine-tuned, it first rebuilds `adapter` from the verified snapshot with the same `SEED` — the same fresh class head "
                "that Section 7's baseline measured — so a changed setting is compared from the same starting point instead of training further.\n\n"
                "**Read the loss as optimisation evidence only.** A falling loss says the optimizer is fitting the training images; the held-out "
                "average precision in the next section is the task evidence.\n\n"
                "**Predict before running:** will the first epoch's loss be above or below 1.0, and will the tenth be about half of it, a quarter of "
                "it, or near zero?"
            ),
            "code": (
                "import time\n\n"
                'LEARNING_RATE = 1e-4  # @param {{type:"number"}}\n'
                'BATCH_SIZE = 4  # @param {{type:"integer"}}\n'
                'FREEZE_BACKBONE = True  # @param {{type:"boolean"}}\n\n'
                "if adapter.adapted:\n"
                "    # A re-run: start again from the re-headed base (same SEED, same class head as the Section 7 baseline).\n"
                "    adapter = DetrDetectionPipeline.from_pretrained(weights_dir=WEIGHTS_DIR, class_names=SIGN_CLASSES, seed=SEED)\n"
                "    print({{'rebuilt_from_verified_snapshot': True, 'adapted': adapter.adapted}})\n"
                "train_started = time.perf_counter()\n"
                "run = adapter.finetune(\n"
                "    train_records,\n"
                "    epochs=EPOCHS,\n"
                "    batch_size=BATCH_SIZE,\n"
                "    learning_rate=LEARNING_RATE,\n"
                "    seed=SEED,\n"
                "    freeze_backbone=FREEZE_BACKBONE,\n"
                "    progress=lambda row: print(f\"epoch {{row['epoch']}}/{{row['epochs']}}  loss {{row['loss']:.4f}}\"),\n"
                ")\n"
                "train_seconds = round(time.perf_counter() - train_started, 1)\n"
                "print(json.dumps({{**{{key: run[key] for key in ('freeze_backbone', 'trainable_parameters', 'total_parameters', 'epochs',\n"
                "                                             'batch_size', 'learning_rate', 'optimizer', 'precision', 'device')}},\n"
                "                  'first_epoch_loss': round(run['epoch_losses'][0], 4), 'final_loss': round(run['final_loss'], 4),\n"
                "                  'train_seconds': train_seconds}}, indent=2))"
            ),
        },
        # ---------------------------------------------------------------- 9. Evaluate Held-Out
        {
            "md": (
                "**What to notice (Section 8):** the epoch-1 loss, how fast it falls, and the trainable share of the parameters.\n\n"
                "<details><summary>Check your reasoning</summary>The recorded runs started at 1.6564 (Kaggle T4) and 2.1077 (local CPU) and "
                "ended at 0.5602 and 0.4860 after 10 epochs — down to roughly a quarter to a third, not near zero: 30 images and 10 epochs are a "
                "short schedule. With the backbone frozen, 18,047,240 of 41,502,152 parameters were trainable. The T4 and CPU trajectories differ "
                "more than floating-point noise because the fresh class head and the training kernels are not identical across devices; the "
                "conclusions below hold for both.</details>\n\n"
                "## 9. Evaluate on the held-out split, and what AP does not say\n\n"
                "`evaluate` re-runs on the same held-out images with the same thresholds as the baseline, so the two rows are comparable. `ap50` "
                "is average precision at IoU 0.50; `ap` averages AP over the ten IoU thresholds 0.50–0.95, so it also rewards tight boxes. Reading "
                "only `ap50` hides loose boxes; reading only `ap` hides whether objects were found at all. These are tutorial metrics from one pass "
                "over 10 synthetic images, with no dispersion estimate.\n\n"
                "**AP is computed at the evaluation threshold.** `evaluate` scores every detection at or above `EVAL_DETECTION_THRESHOLD` (0.05), "
                "at most 100 per image, ranked by score. AP is a **ranking** metric: it rewards putting correct boxes above wrong ones, whatever "
                "their absolute scores. It therefore says nothing about how many boxes the model returns at a deployment threshold such as 0.9. "
                "So the cell also asks the **decision** question on the same held-out images: at `threshold`, how many boxes come back, how many "
                "match a same-label reference at IoU ≥ 0.5 (precision and recall), and what is the highest adapted score? A run-history table "
                "keeps one row per evaluation, so a Section 14 re-run prints side by side with the default run.\n\n"
                "**An adapted threshold, chosen on the training split only.** The COCO threshold 0.9 belongs to the pretrained head. For the "
                "re-headed model the cell picks `ADAPTED_THRESHOLD` on the **30 training images**: every score above 0.05 is a candidate, and the "
                "one with the highest F1 (same label, IoU ≥ 0.5) on the training split is kept. The held-out and unseen images are never used to "
                "choose it — choosing on them would make their numbers optimistic. The held-out check is then printed at both thresholds; "
                "`threshold` (0.9) is unchanged.\n\n"
                "**Predict before running:** will the adapted AP50 be above 0.5? Will any held-out box survive `threshold = 0.9`? Will the "
                "adapted threshold be nearer 0.9 or nearer 0.05?"
            ),
            "code": (
                "adapted = adapter.evaluate(held_out)\n"
                "print(f\"{{'metric':<8s}} {{'baseline':>10s}} {{'adapted':>10s}} {{'change':>10s}}\")\n"
                "for key in ('ap', 'ap50', 'ap75'):\n"
                "    print(f\"{{key:<8s}} {{baseline[key]:>10.4f}} {{adapted[key]:>10.4f}} {{adapted[key] - baseline[key]:>+10.4f}}\")\n"
                "print('per-class AP50:', {{k: round(v, 4) for k, v in adapted['per_class_ap50'].items()}})\n"
                "print({{'evaluation_threshold': adapted['threshold'], 'max_detections_scored_per_image': adapted['max_detections']}})\n\n"
                "import math\n\n"
                "def _count_matches(detections, record, at):\n"
                "    \"\"\"(returned, correct, references) for one image: detections scoring >= `at`, matched once each to a same-label reference at IoU >= 0.5.\"\"\"\n"
                "    kept = [d for d in detections if d['score'] >= at]\n"
                "    unused = list(zip(record['boxes'], record['labels'], strict=True))\n"
                "    references, correct = len(unused), 0\n"
                "    for det in kept:\n"
                "        hit = next((ref for ref in unused if ref[1] == det['label'] and box_iou(det['box'], ref[0]) >= 0.5), None)\n"
                "        if hit is not None:\n"
                "            unused.remove(hit)\n"
                "            correct += 1\n"
                "    return len(kept), correct, references\n\n"
                "def threshold_check(model, records, at):\n"
                "    \"\"\"The decision view: boxes returned at one fixed threshold, how many match a same-label reference (IoU >= 0.5), top score.\"\"\"\n"
                "    returned = correct = n_references = 0\n"
                "    top_score = 0.0\n"
                "    for record in records:\n"
                "        lenient = model.detect(record['image'], threshold=min(at, EVAL_DETECTION_THRESHOLD))['detections']\n"
                "        top_score = max([top_score] + [d['score'] for d in lenient])\n"
                "        r, c, n = _count_matches(lenient, record, at)\n"
                "        returned, correct, n_references = returned + r, correct + c, n_references + n\n"
                "    return {{'threshold': at, 'returned': returned, 'correct': correct, 'references': n_references,\n"
                "            'precision': round(correct / returned, 4) if returned else None,\n"
                "            'recall': round(correct / n_references, 4) if n_references else None,\n"
                "            'highest_score': round(top_score, 4)}}\n\n"
                "def select_adapted_threshold(model, train):\n"
                "    \"\"\"Max-F1 score threshold over the TRAINING split's detections (never the held-out or unseen images); ties keep the higher threshold.\"\"\"\n"
                "    cached = [(model.detect(record['image'], threshold=EVAL_DETECTION_THRESHOLD)['detections'], record) for record in train]\n"
                "    best = {{'threshold': None, 'f1': -1.0}}\n"
                "    for candidate in sorted({{d['score'] for detections, _ in cached for d in detections}}):\n"
                "        returned = correct = references = 0\n"
                "        for detections, record in cached:\n"
                "            r, c, n = _count_matches(detections, record, candidate)\n"
                "            returned, correct, references = returned + r, correct + c, references + n\n"
                "        f1 = 2 * correct / (returned + references) if returned + references else 0.0\n"
                "        if f1 >= best['f1']:\n"
                "            best = {{'threshold': candidate, 'f1': f1, 'precision': correct / returned if returned else None, 'recall': correct / references if references else None}}\n"
                "    if best['threshold'] is None or best['f1'] <= 0.0:\n"
                "        return {{'threshold': threshold, 'selected_on': 'training split', 'rule': 'no training detection matched; fell back to threshold', 'train_f1': 0.0}}\n"
                "    return {{'threshold': math.floor(best['threshold'] * 1e4) / 1e4, 'selected_on': 'training split', 'rule': 'max F1 at IoU >= 0.5, same label',\n"
                "            'train_f1': round(best['f1'], 4), 'train_precision': round(best['precision'], 4), 'train_recall': round(best['recall'], 4),\n"
                "            'train_images': len(train)}}\n\n"
                "adapted_threshold = select_adapted_threshold(adapter, train_records)\n"
                "ADAPTED_THRESHOLD = adapted_threshold['threshold']\n"
                "print('adapted operating threshold, chosen on the training split only:', adapted_threshold)\n\n"
                "operating_check = threshold_check(adapter, held_out, threshold)\n"
                "adapted_check = threshold_check(adapter, held_out, ADAPTED_THRESHOLD)\n"
                "print('held-out at the COCO threshold:     ', operating_check)\n"
                "print('held-out at the adapted threshold:  ', adapted_check)\n\n"
                "RUN_HISTORY = globals().get('RUN_HISTORY') or []\n"
                "RUN_HISTORY.append({{'freeze_backbone': run['freeze_backbone'], 'epochs': run['epochs'], 'learning_rate': run['learning_rate'],\n"
                "                    'first_epoch_loss': round(run['epoch_losses'][0], 4), 'final_loss': round(run['final_loss'], 4),\n"
                "                    'ap': round(adapted['ap'], 4), 'ap50': round(adapted['ap50'], 4),\n"
                "                    'returned_at_threshold': operating_check['returned'], 'adapted_threshold': ADAPTED_THRESHOLD,\n"
                "                    'correct_at_adapted_threshold': adapted_check['correct'], 'train_seconds': train_seconds}})\n"
                "print('run history (one row per evaluation; row 0 is the first run of this session):')\n"
                "for index, row in enumerate(RUN_HISTORY):\n"
                "    print(index, row)"
            ),
        },
        # ---------------------------------------------------------------- 10. New-data inference
        {
            "md": (
                "**What to notice (Section 9):** the AP change, and then the operating-threshold line under it.\n\n"
                "<details><summary>Check your reasoning</summary>The recorded runs raised `ap50` from 0.0172 to 0.2668 (Kaggle T4) and "
                "0.3205 (local CPU), and `ap` to 0.2222 and 0.2884: the fine-tune taught the head to rank some of the new signs' boxes "
                "above the wrong ones, but far from all of them. The operating-threshold line read `returned: 0` in a local CPU run of this revision "
                "(2026-10-04): none of the 17 held-out reference boxes came back at `threshold = 0.9`, and the highest adapted score on the 10 "
                "held-out images was 0.421. Ten epochs on 30 images move the softmax only part of the way from no-object toward the new "
                "classes, so the right boxes score well above the wrong ones (hence a non-zero AP) and far below 0.9. A non-zero AP and an empty "
                "output are both true; they answer different questions. The training split chose `ADAPTED_THRESHOLD = 0.2905` (training F1 "
                "0.57, precision 0.63, recall 0.52), and at that threshold 14 held-out boxes came back, "
                "4 of them correct (recall 0.24 of 17 references, precision 0.29).</details>\n\n"
                "## 10. Inference on unseen images, at the operating threshold\n\n"
                "Three new images come from a seed the dataset never used (`NEW_DATA_SEED = 99`). The adapted model is run at the same `threshold` "
                "as the COCO head (0.9), and each reference box is compared with the best same-label detection by IoU. Next to that, the cell "
                "reports the best-localised same-label detection at the evaluation threshold 0.05, with its score, so you can see the difference "
                "between *ranking* the right box first and *returning* it.\n\n"
                "**What the recorded runs show.** At 0.9 the adapted model returned **no** boxes on any of the three images, on the Kaggle T4 and "
                "on CPU (highest score 0.390). At 0.05 the best same-label box overlapped its reference with IoU 0.87–0.91 for three of the five signs, at scores 0.287–0.291, and two signs had no same-label box over them at all (local CPU run of this revision, 2026-10-04): where the model finds a sign the box is right, but every score is low. At the adapted threshold the same run found **1 of the 5** reference signs.\n\n"
                "The cell prints each image twice over — at `threshold` (0.9) and at `ADAPTED_THRESHOLD` (from Section 9) — and the found count "
                "at each.\n\n"
                "**Predict before running:** how many reference boxes will be found at 0.9, and how many at the adapted threshold? Which sign is "
                "most likely to be missed at both?"
            ),
            "code": (
                'NEW_DATA_SEED = 99  # @param {{type:"integer"}}\n\n'
                "new_records = sign_dataset(3, seed=NEW_DATA_SEED)\n"
                "new_data_rows = []\n"
                "for record in new_records:\n"
                "    out = adapter.detect(record['image'], threshold=threshold)\n"
                "    lenient = adapter.detect(record['image'], threshold=EVAL_DETECTION_THRESHOLD)['detections']\n"
                "    at_adapted = [d for d in lenient if d['score'] >= ADAPTED_THRESHOLD]\n"
                "    ious, ious_adapted, best_at_evaluation_threshold = [], [], []\n"
                "    for box, label in zip(record['boxes'], record['labels'], strict=True):\n"
                "        same_label = [d for d in out['detections'] if d['label'] == label]\n"
                "        ious.append(round(max((box_iou(d['box'], box) for d in same_label), default=0.0), 3))\n"
                "        ious_adapted.append(round(max((box_iou(d['box'], box) for d in at_adapted if d['label'] == label), default=0.0), 3))\n"
                "        iou, score = max(((box_iou(d['box'], box), d['score']) for d in lenient if d['label'] == label), default=(0.0, 0.0))\n"
                "        best_at_evaluation_threshold.append({{'iou': round(iou, 3), 'score': round(score, 3)}})\n"
                "    row = {{'id': record['id'], 'truth': record['labels'], 'threshold': threshold,\n"
                "           'detections': [(d['label'], round(d['score'], 3)) for d in out['detections']], 'same_label_iou': ious,\n"
                "           'adapted_threshold': ADAPTED_THRESHOLD,\n"
                "           'detections_at_adapted_threshold': [(d['label'], round(d['score'], 3)) for d in at_adapted],\n"
                "           'same_label_iou_at_adapted_threshold': ious_adapted,\n"
                "           'highest_score': round(max((d['score'] for d in lenient), default=0.0), 3),\n"
                "           'best_same_label_at_evaluation_threshold': best_at_evaluation_threshold}}\n"
                "    new_data_rows.append(row)\n"
                "    print(json.dumps(row))\n"
                "n_new = sum(len(row['truth']) for row in new_data_rows)\n"
                "found = sum(1 for row in new_data_rows for value in row['same_label_iou'] if value >= 0.5)\n"
                "found_adapted = sum(1 for row in new_data_rows for value in row['same_label_iou_at_adapted_threshold'] if value >= 0.5)\n"
                "print(f'{{found}}/{{n_new}} reference boxes found at the COCO threshold {{threshold}} (same label, IoU >= 0.5)')\n"
                "print(f'{{found_adapted}}/{{n_new}} reference boxes found at the adapted threshold {{ADAPTED_THRESHOLD}} (chosen on the training split)')"
            ),
        },
        # ---------------------------------------------------------------- 11. Export, reload & verify
        {
            "md": (
                "**What to notice (Section 10):** the found count at 0.9, each image's `highest_score`, and the IoU and score of the best "
                "same-label detection at 0.05.\n\n"
                "<details><summary>Check your reasoning</summary>0 of 5 reference boxes were found at 0.9 and every `detections` list was empty; the highest scores per image were 0.390, 0.291 and 0.289 (local CPU run of this revision, 2026-10-04; the Kaggle T4 run of the earlier revision also returned nothing at 0.9). At 0.05 the best same-label detections had IoU 0.912 (the second yield sign on image 000), 0.871 and 0.913 (image 001), at scores 0.287–0.291; the first yield sign on image 000 and the stop sign on image 002 had no same-label box over them. At the training-split threshold 0.2905 the same run found 1 of 5, the yield sign on image 001 (IoU 0.913, score 0.291); image 000's yield sign at 0.287 falls just below it, and image 000's highest box (0.390) carries the label `speed-limit-sign`, which matches no reference there, so the threshold lets one wrong box through. With every score packed between about 0.25 and 0.42, a small change in the threshold or in the run (a GPU run differs) moves the found count. The threshold changes what is returned, not how well the "
                "boxes are ranked or placed.</details>\n\n"
                "## 11. Adapter export, fresh reload, and equivalence check\n\n"
                "`save_artifact` writes `outputs/detr_adapter.safetensors`: every tensor the fine-tune could change, plus a metadata header naming "
                "the base model, its pinned revision, the base `model.safetensors` SHA-256, the class names and the frozen prefixes. The frozen "
                "backbone is left out because it equals the verified base snapshot.\n\n"
                "`load_artifact` then builds a **fresh** pipeline from the verified base snapshot, loads the adapter tensors onto it, and refuses "
                "an adapter whose format, base identity, base digest or tensor set does not fit. The cell compares the reloaded detections with the "
                "in-memory model's on an unseen image, with a stated tolerance: loading succeeding is not the check, reproducing the detections is. "
                "The comparison runs at the evaluation threshold 0.05, because at 0.9 there would be nothing to compare (Section 10); if nothing passes 0.05 either, it compares the 10 top-scoring pairs, so the check is never empty. The same "
                "`reload_equivalence` helper checks the BYOD adapter in Section 13."
            ),
            "code": (
                "artifact_path = OUTPUTS / 'detr_adapter.safetensors'\n"
                "descriptor = adapter.save_artifact(artifact_path, notes='DETR ResNet-50 sign adaptation tutorial adapter')\n"
                "print(json.dumps(descriptor, indent=2))\n\n"
                "reloaded = DetrDetectionPipeline.load_artifact(artifact_path, weights_dir=WEIGHTS_DIR)\n"
                "print({{'reloaded_source': reloaded.source, 'adapted': reloaded.adapted, 'class_names': list(reloaded.class_names)}})\n\n"
                "TOLERANCE = 1e-3\n\n"
                "def reload_equivalence(first, second, image, tolerance=TOLERANCE):\n"
                "    \"\"\"Both pipelines must return the same detections on `image` at the evaluation threshold (or, if none passes it, the same top 10 pairs), within `tolerance`.\"\"\"\n"
                "    at, limit = EVAL_DETECTION_THRESHOLD, None\n"
                "    if not first.detect(image, threshold=at)['detections']:  # nothing passes 0.05 (e.g. a short BYOD run)\n"
                "        at, limit = 0.0, 10  # compare the 10 top-scoring pairs instead, so the check is never empty\n"
                "    det_orig = first.detect(image, threshold=at)['detections'][:limit]\n"
                "    det_reloaded = second.detect(image, threshold=at)['detections'][:limit]\n"
                "    if len(det_orig) != len(det_reloaded):\n"
                "        raise RuntimeError(f'the reloaded adapter returned {{len(det_reloaded)}} detections, the in-memory model {{len(det_orig)}}: the export is not faithful; restart the session and choose Run all')\n"
                "    for d1, d2 in zip(det_orig, det_reloaded, strict=True):\n"
                "        if d1['label'] != d2['label'] or not np.allclose(d1['box'], d2['box'], atol=tolerance) or abs(d1['score'] - d2['score']) > tolerance:\n"
                "            raise RuntimeError(f'reloaded detection {{d2}} differs from {{d1}} beyond tolerance {{tolerance}}: the export is not faithful')\n"
                "    return {{'detections_compared': len(det_orig), 'threshold': at, 'tolerance': tolerance, 'equivalent': True}}\n\n"
                "reload_check = reload_equivalence(adapter, reloaded, new_records[0]['image'])\n"
                "print(reload_check)"
            ),
        },
        # ---------------------------------------------------------------- 12. Outputs & provenance
        {
            "md": (
                "## 12. Write machine-readable outputs and provenance\n\n"
                "The cell writes:\n"
                "- `outputs/detr_detection_input_manifest.json`\n"
                "- `outputs/detr_detection_evaluation_report.json`\n"
                "- `outputs/detr_detection_result.json` (identity, runtime versions, device, dataset manifest, split, baseline and adapted metrics, "
                "the evaluation threshold, the held-out check at the operating threshold, the run history, fine-tuning configuration, new-data rows, "
                "adapter descriptor and reload check)\n"
                "- `outputs/detr_detection_detections.csv` (the COCO scene's detections and the adapted model's detections on the unseen "
                "images at `threshold` and at `ADAPTED_THRESHOLD`; a `threshold` column says which. At 0.9 the adapted model returned no box in the "
                "recorded runs, so its rows come from the adapted threshold)\n"
                "- `outputs/detr_detection_annotated.png`\n"
                "- `outputs/detr_adapter.safetensors` (written in Section 11)"
            ),
            "code": (
                "import csv\n"
                "from PIL import ImageDraw\n\n"
                "with open(OUTPUTS / '{stem}_input_manifest.json', 'w', encoding='utf-8') as f:\n"
                "    json.dump(input_manifest, f, indent=2)\n\n"
                "with open(OUTPUTS / '{stem}_evaluation_report.json', 'w', encoding='utf-8') as f:\n"
                "    json.dump(coco_report, f, indent=2)\n\n"
                "annotated = scene.copy()\n"
                "draw = ImageDraw.Draw(annotated)\n"
                "for det in coco_result['detections']:\n"
                "    x0, y0, x1, y1 = det['box']\n"
                "    draw.rectangle([x0, y0, x1, y1], outline='red', width=3)\n"
                "    draw.text((x0 + 4, y0 + 4), f\"{{det['label']}} {{det['score']:.2f}}\", fill='red')\n"
                "annotated.save(OUTPUTS / '{stem}_annotated.png')\n\n"
                "with open(OUTPUTS / '{stem}_detections.csv', 'w', newline='', encoding='utf-8') as f:\n"
                "    writer = csv.writer(f)\n"
                "    writer.writerow(['image', 'rank', 'label', 'score', 'x0', 'y0', 'x1', 'y1', 'threshold'])\n"
                "    for rank, d in enumerate(coco_result['detections']):\n"
                "        writer.writerow(['tutorial-scene', rank, d['label'], f\"{{d['score']:.4f}}\", *(f\"{{v:.1f}}\" for v in d['box']), threshold])\n"
                "    for at in (threshold, ADAPTED_THRESHOLD):\n"
                "        for row, record in zip(new_data_rows, new_records, strict=True):\n"
                "            lenient = adapter.detect(record['image'], threshold=EVAL_DETECTION_THRESHOLD)['detections']\n"
                "            for rank, d in enumerate(d for d in lenient if d['score'] >= at):\n"
                "                writer.writerow([row['id'], rank, d['label'], f\"{{d['score']:.4f}}\", *(f\"{{v:.1f}}\" for v in d['box']), at])\n\n"
                "result_export = {{\n"
                "    'notebook_source': NOTEBOOK_SOURCE,\n"
                "    'repository_revision': NOTEBOOK_SOURCE['repository_revision'],\n"
                "    'model_id': MODEL_ID,\n"
                "    'model_revision': MODEL_REVISION,\n"
                "    'model_license': MODEL_LICENSE,\n"
                "    'runtime': {{'python': platform.python_version(), 'torch': torch.__version__, 'transformers': transformers.__version__,\n"
                "                'timm': timm.__version__, 'cuda': torch.cuda.is_available()}},\n"
                "    'device': pipe.device,\n"
                "    'threshold': threshold,\n"
                "    'evaluation_threshold': EVAL_DETECTION_THRESHOLD,\n"
                "    'coco_detections': coco_result['detections'],\n"
                "    'degenerate_probes': degenerate,\n"
                "    'adaptation': {{\n"
                "        'dataset': {{k: v for k, v in dataset_manifest.items() if k != 'schema'}},\n"
                "        'dataset_seed': DATASET_SEED,\n"
                "        'split': {{'train': len(train_records), 'held_out': len(held_out), 'seed': SEED, 'holdout': HOLDOUT}},\n"
                "        'finetune': {{**run, 'train_seconds': train_seconds}},\n"
                "        'baseline': baseline,\n"
                "        'adapted': adapted,\n"
                "        'held_out_at_operating_threshold': operating_check,\n"
                "        'adapted_threshold': adapted_threshold,\n"
                "        'held_out_at_adapted_threshold': adapted_check,\n"
                "        'run_history': RUN_HISTORY,\n"
                "        'new_data': new_data_rows,\n"
                "        'artifact': descriptor,\n"
                "        'reload_check': reload_check,\n"
                "    }},\n"
                "}}\n"
                "with open(OUTPUTS / '{stem}_result.json', 'w', encoding='utf-8') as f:\n"
                "    json.dump(result_export, f, indent=2, default=str)\n\n"
                "for p in sorted(OUTPUTS.iterdir()):\n"
                "    if p.is_file():\n"
                "        print(f'  {{p.name:<44s}} {{p.stat().st_size:>12,d}} bytes')"
            ),
        },
        # ---------------------------------------------------------------- 13. BYOD
        {
            "md": (
                "## 13. Optional: Bring Your Own Data (BYOD)\n\n"
                "Both branches are off by default, so `Run all` never stops here. Before you turn one on, read the contract:\n\n"
                "- **Image branch** (`USE_BYOD_IMAGE`): one image file PIL can open, each side between `MIN_IMAGE_SIDE` (16) and `MAX_IMAGE_SIDE` "
                "(4096) px. It runs through `validate_inputs`, `detect` and `evaluation_report` — the report is `not-measurable`, because no "
                "reference boxes come with it. Upload exactly one image; every upload replaces the previous one, so a second upload is the image "
                "that runs.\n"
                "- **Dataset branch** (`USE_BYOD_DATASET`): `annotations.json` (a list of `{{'file', 'boxes', 'labels'}}` objects, xyxy pixel boxes) "
                "and the images it names, at least 2 records and at most 5,000. File names must stay inside the dataset folder. **Upload the files "
                "flat — `annotations.json` and the images, no folders — or upload one `.zip`** that holds them (folders inside the zip are fine; at most "
                "10,001 files and 2 GiB, and member paths may not be absolute or contain `..`). `BYOD_CLASS_NAMES` is a comma-separated vocabulary; "
                "leave it empty to use the sorted set of labels in the annotations. The branch runs the same validate → split → baseline → "
                "fine-tune → evaluate → export → reload stages as the sample, compares the reloaded adapter's detections with the in-memory "
                "model's, and writes `outputs/byod_detr_result.json` (annotation digest, split, fine-tune configuration, baseline and "
                "adapted metrics at the evaluation threshold, its own adapted threshold chosen on the BYOD training split, the held-out check at "
                "`threshold` and at that adapted threshold, adapter descriptor and reload check) and "
                "`outputs/byod_detr_detections.csv` (held-out detections at the evaluation threshold).\n\n"
                "Set `BYOD_IMAGE_PATH` (a file) or `BYOD_DATASET_DIR` (a directory or a `.zip`) to read from a mounted or local location; leave them "
                "empty on Colab to get an upload dialog instead. Uploaded files are written under `outputs/byod/` in this runtime and are not sent "
                "anywhere else. The first lines of the cell show the validator refusing two malformed inputs with messages that name the failed rule. "
                "After changing a field here, re-run this cell only."
            ),
            "code": (
                'USE_BYOD_IMAGE = False  # @param {{type:"boolean"}}\n'
                'BYOD_IMAGE_PATH = ""  # @param {{type:"string"}}\n'
                'USE_BYOD_DATASET = False  # @param {{type:"boolean"}}\n'
                'BYOD_DATASET_DIR = ""  # @param {{type:"string"}}\n'
                'BYOD_CLASS_NAMES = ""  # @param {{type:"string"}}\n\n'
                "import shutil\n"
                "import zipfile\n\n"
                "for desc, probe in (\n"
                "    ('non-image object', lambda: validate_inputs('/not/an/image.png')),\n"
                "    ('box outside the image', lambda: validate_dataset([{{'image': blank_scene(), 'boxes': [[0, 0, 9999, 10]], 'labels': [SIGN_CLASSES[0]]}}], SIGN_CLASSES)),\n"
                "):\n"
                "    try:\n"
                "        probe()\n"
                "    except (TypeError, ValueError) as exc:\n"
                "        print(f'refused as expected: {{desc}} -> {{type(exc).__name__}}: {{exc}}')\n\n"
                "BYOD_DIR = OUTPUTS / 'byod'\n"
                "BYOD_MAX_ARCHIVE_FILES = 2 * MAX_RECORDS + 1\n"
                "BYOD_MAX_EXPANDED_BYTES = 2 * 1024**3\n\n"
                "def _upload_into(target):\n"
                "    \"\"\"Colab upload dialog into a fresh `target`: an earlier upload never mixes with this one.\"\"\"\n"
                "    try:\n"
                "        from google.colab import files  # type: ignore[import-not-found]\n"
                "    except ImportError as exc:\n"
                "        raise RuntimeError('There is no upload dialog outside Google Colab: set BYOD_IMAGE_PATH or BYOD_DATASET_DIR to a local path.') from exc\n"
                "    shutil.rmtree(target, ignore_errors=True)\n"
                "    target.mkdir(parents=True)\n"
                "    uploaded = files.upload()\n"
                "    if not uploaded:\n"
                "        raise ValueError('Nothing was uploaded: run the cell again and choose the files.')\n"
                "    for name, data in uploaded.items():\n"
                "        (target / Path(name).name).write_bytes(data)\n"
                "    return target\n\n"
                "def _safe_unzip(archive, target):\n"
                "    \"\"\"Extract a BYOD .zip member by member into a fresh `target`, refusing unsafe paths and oversized archives.\"\"\"\n"
                "    shutil.rmtree(target, ignore_errors=True)\n"
                "    target.mkdir(parents=True)\n"
                "    with zipfile.ZipFile(archive) as bundle:\n"
                "        members = [m for m in bundle.infolist() if not m.is_dir()]\n"
                "        if len(members) > BYOD_MAX_ARCHIVE_FILES:\n"
                "            raise ValueError(f'{{Path(archive).name}} holds {{len(members)}} files, more than {{BYOD_MAX_ARCHIVE_FILES}}')\n"
                "        if sum(m.file_size for m in members) > BYOD_MAX_EXPANDED_BYTES:\n"
                "            raise ValueError(f'{{Path(archive).name}} expands to more than {{BYOD_MAX_EXPANDED_BYTES:,d}} bytes')\n"
                "        for member in members:\n"
                "            name = member.filename.replace('\\\\', '/')\n"
                "            parts = [p for p in name.split('/') if p not in ('', '.')]\n"
                "            if name.startswith('/') or ':' in name or '..' in parts:\n"
                "                raise ValueError(f'{{Path(archive).name}}: unsafe member path {{member.filename!r}}')\n"
                "            destination = target.joinpath(*parts)\n"
                "            destination.parent.mkdir(parents=True, exist_ok=True)\n"
                "            destination.write_bytes(bundle.read(member))\n"
                "    return target\n\n"
                "def _dataset_root(source):\n"
                "    \"\"\"The folder holding annotations.json: `source` itself, the inside of one .zip, or its single sub-folder that has one.\"\"\"\n"
                "    source = Path(source)\n"
                "    if source.is_file() and source.suffix.lower() == '.zip':\n"
                "        source = _safe_unzip(source, BYOD_DIR / 'unzipped')\n"
                "    elif source.is_dir() and not (source / 'annotations.json').is_file():\n"
                "        archives = sorted(source.glob('*.zip'))\n"
                "        if len(archives) == 1:\n"
                "            source = _safe_unzip(archives[0], BYOD_DIR / 'unzipped')\n"
                "    if (source / 'annotations.json').is_file():\n"
                "        return source\n"
                "    found = sorted(source.rglob('annotations.json'))\n"
                "    if len(found) != 1:\n"
                "        raise ValueError(f'expected one annotations.json under {{source}}, found {{len(found)}}: upload annotations.json and the images as flat files, or one .zip')\n"
                "    return found[0].parent\n\n"
                "if USE_BYOD_IMAGE:\n"
                "    if BYOD_IMAGE_PATH:\n"
                "        image_path = Path(BYOD_IMAGE_PATH)\n"
                "    else:\n"
                "        uploaded_images = sorted(p for p in _upload_into(BYOD_DIR / 'image').iterdir() if p.is_file())\n"
                "        if len(uploaded_images) != 1:\n"
                "            raise ValueError(f'upload exactly one image for the image branch, got {{len(uploaded_images)}}')\n"
                "        image_path = uploaded_images[0]\n"
                "    with Image.open(image_path) as handle:\n"
                "        byod_image = handle.convert('RGB')\n"
                "    print(validate_inputs(byod_image, threshold=threshold, names=[image_path.name])['verdict'])\n"
                "    byod_result = pipe.detect(byod_image, threshold=threshold)\n"
                "    for det in byod_result['detections'][:20]:\n"
                "        print(f\"{{det['label']:>16s}} {{det['score']:.3f}}  [{{', '.join(f'{{v:.0f}}' for v in det['box'])}}]\")\n"
                "    print(evaluation_report(byod_result, None, sample_kind='byod')['verdict'])\n"
                "else:\n"
                "    print('BYOD image branch is off; set USE_BYOD_IMAGE = True to run detection on your own image.')\n\n"
                "if USE_BYOD_DATASET:\n"
                "    dataset_dir = _dataset_root(BYOD_DATASET_DIR if BYOD_DATASET_DIR else _upload_into(BYOD_DIR / 'dataset'))\n"
                "    try:\n"
                "        byod_records = read_detection_records(dataset_dir)\n"
                "    except FileNotFoundError as exc:\n"
                "        raise FileNotFoundError(f'{{exc}} -- every file named in annotations.json must be present; through the upload dialog, upload the files flat (no folders) or as one .zip') from exc\n"
                "    byod_names = [n.strip() for n in BYOD_CLASS_NAMES.split(',') if n.strip()] or sorted({{l for r in byod_records for l in r['labels']}})\n"
                "    byod_manifest = validate_dataset(byod_records, byod_names, epochs=EPOCHS)\n"
                "    print(json.dumps({{k: v for k, v in byod_manifest.items() if k != 'schema'}}, indent=2))\n"
                "    byod_train, byod_held = split_dataset(byod_records, train_fraction=1.0 - HOLDOUT, seed=SEED)\n"
                "    byod_pipe = DetrDetectionPipeline.from_pretrained(weights_dir=WEIGHTS_DIR, class_names=byod_names, seed=SEED)\n"
                "    byod_baseline = byod_pipe.evaluate(byod_held)\n"
                "    byod_run = byod_pipe.finetune(byod_train, epochs=EPOCHS, batch_size=BATCH_SIZE, learning_rate=LEARNING_RATE, seed=SEED, freeze_backbone=FREEZE_BACKBONE)\n"
                "    byod_adapted = byod_pipe.evaluate(byod_held)\n"
                "    byod_operating = threshold_check(byod_pipe, byod_held, threshold)\n"
                "    byod_threshold = select_adapted_threshold(byod_pipe, byod_train)\n"
                "    byod_adapted_check = threshold_check(byod_pipe, byod_held, byod_threshold['threshold'])\n"
                "    print({{'baseline_ap50': round(byod_baseline['ap50'], 4), 'adapted_ap50': round(byod_adapted['ap50'], 4),\n"
                "           'evaluation_threshold': EVAL_DETECTION_THRESHOLD, 'at_operating_threshold': byod_operating,\n"
                "           'adapted_threshold': byod_threshold, 'at_adapted_threshold': byod_adapted_check}})\n"
                "    byod_artifact = OUTPUTS / 'byod_detr_adapter.safetensors'\n"
                "    byod_descriptor = byod_pipe.save_artifact(byod_artifact, notes='BYOD adaptation adapter')\n"
                "    byod_reloaded = DetrDetectionPipeline.load_artifact(byod_artifact, weights_dir=WEIGHTS_DIR)\n"
                "    byod_reload_check = reload_equivalence(byod_pipe, byod_reloaded, byod_held[0]['image'])\n"
                "    print('BYOD adapter exported, reloaded and compared:', byod_descriptor['sha256'][:16], byod_reload_check)\n"
                "    with open(OUTPUTS / 'byod_detr_detections.csv', 'w', newline='', encoding='utf-8') as f:\n"
                "        writer = csv.writer(f)\n"
                "        writer.writerow(['image', 'rank', 'label', 'score', 'x0', 'y0', 'x1', 'y1'])\n"
                "        for record in byod_held:\n"
                "            for rank, d in enumerate(byod_pipe.detect(record['image'], threshold=EVAL_DETECTION_THRESHOLD)['detections']):\n"
                "                writer.writerow([record['id'], rank, d['label'], f\"{{d['score']:.4f}}\", *(f\"{{v:.1f}}\" for v in d['box'])])\n"
                "    byod_export = {{\n"
                "        'notebook_source': NOTEBOOK_SOURCE, 'model_id': MODEL_ID, 'model_revision': MODEL_REVISION,\n"
                "        'annotations_sha256': hashlib.sha256((dataset_dir / 'annotations.json').read_bytes()).hexdigest(),\n"
                "        'dataset': {{k: v for k, v in byod_manifest.items() if k != 'schema'}},\n"
                "        'split': {{'train': [r['id'] for r in byod_train], 'held_out': [r['id'] for r in byod_held], 'seed': SEED, 'holdout': HOLDOUT}},\n"
                "        'finetune': byod_run, 'baseline': byod_baseline, 'adapted': byod_adapted,\n"
                "        'evaluation_threshold': EVAL_DETECTION_THRESHOLD, 'held_out_at_operating_threshold': byod_operating,\n"
                "        'adapted_threshold': byod_threshold, 'held_out_at_adapted_threshold': byod_adapted_check,\n"
                "        'artifact': byod_descriptor, 'reload_check': byod_reload_check,\n"
                "    }}\n"
                "    with open(OUTPUTS / 'byod_detr_result.json', 'w', encoding='utf-8') as f:\n"
                "        json.dump(byod_export, f, indent=2, default=str)\n"
                "    print('wrote', [p.name for p in sorted(OUTPUTS.glob('byod_detr_*'))])\n"
                "else:\n"
                "    print('BYOD dataset branch is off; set USE_BYOD_DATASET = True to adapt DETR on your own labelled images.')"
            ),
        },
    ],
    "closing": (
        "## 14. Your turn — change one thing: unfreeze the backbone\n\n"
        "Optional; **Predict → Change → Run → Observe → Explain**. It re-runs the fine-tune with the ResNet backbone trainable, which takes "
        "longer than Section 8 did (more parameters get gradients).\n\n"
        "1. **Predict:** with `FREEZE_BACKBONE = False`, will held-out AP50 go up or down against the default run? Will any held-out box now "
        "pass `threshold = 0.9`? Write your guess next to row 0 of the Section 9 run history.\n"
        "2. **Change:** in Section 8 set `FREEZE_BACKBONE = False`. Change nothing else.\n"
        "3. **Run:** select the Section 8 cell and choose **Runtime → Run after**. Section 8 rebuilds the re-headed model from the verified "
        "snapshot before training (it prints `rebuilt_from_verified_snapshot`), Section 9 adds a row to the run history, and Sections 10–12 "
        "re-run on the new adapter.\n"
        "4. **Observe:** in Section 8, the cell prints `rebuilt_from_verified_snapshot` and an epoch-1 loss well above 1 (2.66 in a local CPU check with "
        "the backbone unfrozen) — not the roughly 0.72 that continuing from the adapted model gave in the 2026-10-02 review; that is the sign "
        "the run started from the re-headed model. The trainable parameter count rises to every parameter. In Section 9, compare "
        "rows 0 and 1 of the run history: `ap`, `ap50`, `returned_at_threshold`, `adapted_threshold`, `correct_at_adapted_threshold` and "
        "`train_seconds`.\n"
        "5. **Explain:** in one sentence, why can unfreezing more parameters change AP without changing whether anything passes 0.9?\n\n"
        "<details><summary>Check your reasoning</summary>In a local CPU check of this revision (2026-10-04, real weights, `EPOCHS` lowered to 1 for time) the re-run after a default run printed `rebuilt_from_verified_snapshot`, trained all 41,502,152 parameters, and its first epoch had a loss of 2.6567 — a fresh start (the default frozen run's first epoch: 2.1077), not the 0.7202 measured in the 2026-10-02 review when the cell continued training the adapted model. The full 10-epoch unfrozen comparison is not recorded for this revision, so your run history is the measurement: compare row 1's `ap50`, `returned_at_threshold`, `adapted_threshold` and `train_seconds` with row 0. Expect `returned_at_threshold` to stay 0 unless a held-out score passes 0.9. AP rewards ranking the right boxes first, and a threshold "
        "decides what is returned; a short fine-tune can improve the first without lifting any score near 0.9. One run on 10 held-out "
        "images carries no dispersion estimate, so read differences of a few hundredths as unresolved, not as a ranking of the two "
        "settings.</details>\n\n"
        "## Interpretation and limits\n\n"
        "Start from your own run: the Section 9 AP change, the operating-threshold line under it, and the Section 10 found count. Then read the "
        "reference answer.\n\n"
        "<details><summary>Reference answer from the recorded runs</summary>The pretrained COCO head found 3 of the 4 drawn COCO objects at "
        "0.9 and returned nothing on blank or noise images, even at 0.05. Re-heading onto three new sign classes started from a baseline AP50 "
        "of 0.02; a 10-epoch fine-tune with the backbone frozen raised held-out AP50 to about 0.27–0.32 and AP@[.50:.95] to about "
        "0.22–0.29 (Kaggle T4 and local CPU). The adapted scores, however, stayed below about 0.42, so at the operating "
        "threshold 0.9 the adapted model returned no box on the held-out images or on the three unseen images. The fine-tune taught the model "
        "to *rank* some of the new signs above the wrong boxes, not to score them confidently. A threshold chosen on the training split "
        "(0.2905 on CPU) recovered 1 of the 5 unseen signs and 4 of 17 held-out boxes; the adapter "
        "reloaded to the same detections within 1e-3.</details>\n\n"
        "**What this notebook established, in this runtime.** The pinned `facebook/detr-resnet-50` snapshot was verified against a committed "
        "SHA-256 manifest before loading. The pretrained detector was run on a drawn scene and on two structure-free probes, and its boxes were "
        "compared with the drawn references by IoU. A three-class sign vocabulary that COCO does not contain was then adapted by replacing the "
        "class head, training the transformer and heads with the backbone frozen, and scoring the held-out split with COCO-style average "
        "precision before and after, and at the operating threshold. The adapted model was run on unseen images, and the adapter was exported, "
        "reloaded onto a fresh base model and checked against the in-memory detections.\n\n"
        "**Scores and thresholds.** AP is computed over every detection at or above 0.05 and measures ranking; a fixed threshold such as 0.9 is "
        "a decision, and a model can rank the right boxes above the wrong ones while scoring all of them below the decision threshold — as "
        "the adapted model does here. Each score is the query's best class probability from a softmax that includes no-object, not a "
        "calibrated probability; the COCO head's 0.9 is the pinned model README's value for a model trained for 300 epochs on COCO, and a short fine-tune of a fresh "
        "head moves the softmax only part of the way from no-object toward the new classes. A threshold for an adapted detector has to be chosen on labelled data other than the held-out split — here "
        "the training split, which is small and was also fitted, so the chosen value is optimistic for the training images and is not a "
        "deployment setting; the held-out and unseen numbers at it are the honest check.\n\n"
        "**What a green run proves.** Successful execution proves that the recorded repository revision, the pinned dependency set and the "
        "pinned checkpoint together reproduce these stages in a fresh runtime, without the repository being cloned or installed and without "
        "any DIMER worker or service. It does **not** establish benchmark superiority, fitness for any deployment, or that the adapted model "
        "generalises beyond the synthetic images it was fitted to. The held-out AP is measured on 10 drawn images and carries no dispersion "
        "estimate. The scores are not calibrated probabilities.\n\n"
        "**Reproducibility.** Seeds are form fields (`DATASET_SEED`, `SEED`, `NEW_DATA_SEED`), the run is float32 with no data augmentation, "
        "and the class head is initialised under `SEED`. GPU kernels are not forced to be deterministic, so repeated GPU runs can differ in "
        "the last digits of the loss and the scores.\n\n"
        "**More experiments (optional).** Each one names the cell to change and where to re-run from; a re-run of Section 8 always starts "
        "from the re-headed model, and Section 9's run history keeps the earlier rows.\n\n"
        "- **Longer training:** set `EPOCHS = 20` in Section 6, select Section 6 and choose **Runtime → Run after**. Watch where held-out AP stops "
        "improving and whether the highest held-out score approaches 0.9.\n"
        "- **A different learning rate:** set `LEARNING_RATE = 3e-4` in Section 8 and **Run after** from Section 8.\n"
        "- **Your own labelled images:** turn on `USE_BYOD_DATASET` in Section 13 and re-run that cell only.\n\n"
        "## Troubleshooting\n\n"
        "- **Section 1 stops with \"needs a Linux x86_64 runtime\".** The locked environment is built from manylinux wheels; use Google Colab, "
        "Kaggle or a Linux Jupyter server.\n"
        "- **Section 1 fails while downloading.** The `uv` wheel, the managed Python and the locked packages come from PyPI and "
        "python-build-standalone; run the cell again. A size or SHA-256 mismatch is refused on purpose — if it repeats, the download is being altered.\n"
        "- **A restart prompt.** This notebook never needs one: nothing is installed into the kernel. If the isolated process exits (usually out "
        "of memory), restart the session and choose **Run all**.\n"
        "- **Section 3 stops on a download or a digest mismatch.** The checkpoint is fetched from the Hugging Face Hub at a pinned revision and "
        "re-hashed; a mismatch is never loaded. Run the cell again; if it repeats, delete `weights/` and run from Section 3.\n"
        "- **Out of memory in Section 8 or 14.** Lower `BATCH_SIZE` to 2 in Section 8, or keep `FREEZE_BACKBONE = True`, and **Run after** from "
        "Section 8. On a CPU runtime the fine-tune is slow, not broken.\n"
        "- **Section 8's first loss is far below 1.0 on a re-run.** The model was not rebuilt; this should not happen with the current cell. "
        "Restart the session and choose **Run all**.\n"
        "- **Section 10 prints empty detection lists at 0.9.** That is the recorded result at `threshold = 0.9` (Sections 9 and 10 explain it), "
        "not an error; look at the adapted-threshold columns next to it.\n"
        "- **BYOD is refused.** Each refusal names the rule: a missing `annotations.json`, a record without `labels`, a label outside "
        "`BYOD_CLASS_NAMES`, a box outside its image, a path with `..`, fewer than 2 records, or a file named in `annotations.json` that is "
        "missing — usually because the files were uploaded in folders; upload them flat or as one `.zip`, then re-run Section 13.\n\n"
        "## Glossary\n\n"
        "- **DETR (DEtection TRansformer):** a detector that predicts a fixed set of boxes with a transformer instead of proposing anchors "
        "and filtering them with non-maximum suppression.\n"
        "- **Object query:** one of the 100 learned decoder slots; each produces one box and one class distribution.\n"
        "- **Hungarian matching:** the one-to-one assignment of queries to reference boxes that minimises the matching cost; unmatched "
        "queries are trained toward \"no object\".\n"
        "- **Softmax score and no-object class:** each query's class distribution is a softmax over the classes plus a no-object class; "
        "the score of a box is its best class probability. Scores are not calibrated probabilities.\n"
        "- **Threshold:** the score at or above which a detection is returned; a caller's decision, not a model property.\n"
        "- **Re-heading:** replacing the class head with a freshly initialised one for a new vocabulary (here 3 classes plus no-object).\n"
        "- **AP50 / AP@[.50:.95]:** average precision with a match at IoU ≥ 0.50, and the mean over IoU thresholds 0.50–0.95; both rank "
        "detections by score, so they measure ranking, not a decision at one threshold.\n"
        "- **Held-out split:** images never shown to the optimizer and never used to choose a setting; the task evidence.\n"
        "- **Adapted threshold:** the score threshold for the re-headed model, chosen as the max-F1 value on the training split only; "
        "the held-out and unseen images are scored at it but never used to pick it.\n"
        "- **Adapter / reload equivalence:** the SafeTensors file of the tensors the fine-tune changed, and the check that a fresh base "
        "model with the adapter loaded returns the same detections.\n\n"
        "## Conclusion (your notes)\n\n"
        "Optional. Fill in from your own run, one sentence each:\n\n"
        "1. The pretrained COCO head matched ___ of 4 drawn objects at 0.9; the noise probe returned ___ boxes at 0.05.\n"
        "2. The fine-tune moved held-out AP50 from ___ to ___; at `threshold = 0.9` it returned ___ held-out boxes, and the highest adapted score was ___.\n"
        "3. The training split chose an adapted threshold of ___; at it, ___ of ___ held-out boxes were correct, and ___ of 5 unseen signs were found (___ at 0.9).\n"
        "4. In Section 14, unfreezing the backbone changed AP50 by ___ and the training time by ___.\n"
        "5. What I would need before using this adapter on real images: ___.\n\n"
        "## References\n\n"
        "- Carion, N., Massa, F., Synnaeve, G., Usunier, N., Kirillov, A. and Zagoruyko, S. (2020). *End-to-End Object Detection with Transformers.* [arXiv:2005.12872](https://arxiv.org/abs/2005.12872).\n"
        "- Upstream repository: [facebookresearch/detr](https://github.com/facebookresearch/detr) — Apache-2.0.\n"
        "- Hugging Face checkpoint: [facebook/detr-resnet-50](https://huggingface.co/facebook/detr-resnet-50) — Apache-2.0.\n"
        "- Lin, T.-Y. et al. (2014). *Microsoft COCO: Common Objects in Context.* [arXiv:1405.0312](https://arxiv.org/abs/1405.0312).\n"
        "- Repository model card: https://github.com/kurtvalcorza/detr-detection-pipeline/blob/main/MODEL_CARD.md\n"
        "- [`kurtvalcorza/detr-detection-pipeline`](https://github.com/kurtvalcorza/detr-detection-pipeline) — source repository for this pipeline."
    ),
}
