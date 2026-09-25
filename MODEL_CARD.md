---
license: apache-2.0
model_card_spec: "1.2"
pipeline_tag: object-detection
base_model: facebook/detr-resnet-50
date_published: "2020-05"
date_published_source: "month of the DETR paper and first code release (arXiv:2005.12872, submitted 2020-05-26, with the R50 checkpoint in facebookresearch/detr); the date the Hugging Face conversion was first published is not established by this repository"
---

# DETR ResNet-50, COCO — Object Detection with Bounded Fine-Tuning

[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-facebook%2Fdetr--resnet--50-ffcc4d?style=flat)](https://huggingface.co/facebook/detr-resnet-50)
[![Upstream GitHub](https://img.shields.io/badge/Upstream%20GitHub-facebookresearch%2Fdetr-181717?style=flat&logo=github&logoColor=white)](https://github.com/facebookresearch/detr)
[![arXiv Paper](https://img.shields.io/badge/arXiv-2005.12872-b31b1b.svg)](https://arxiv.org/abs/2005.12872)
[![License: Apache-2.0](https://img.shields.io/badge/License-Apache--2.0-blue.svg)](https://www.apache.org/licenses/LICENSE-2.0)

> [!WARNING]
> ⚠️ **Provided for research, training, and evaluation purposes only.** Model weights are redistributed unmodified under their upstream license, which controls your use, including any commercial use or redistribution; the accompanying code and notebooks are released under this repository's license. All of it is supplied **"as is"**, without warranty of any kind, and has not been validated for production, clinical, or safety-critical use. Running the notebooks downloads third-party weights and datasets governed by their own licenses and consumes compute on your own Colab/Kaggle account. To the maximum extent permitted by law, the maintainers of this repository and the DIMER platform accept no liability for any damages arising from their use. Hosting implies no affiliation with or endorsement by the original authors.

> [!IMPORTANT]
> The upstream snapshot is pinned to Hub commit `1d5f47bd3bdd2c4bbfa585418ffe6da5028b4c0b`, and the manifest records every file's SHA-256. No execution with the pinned weights has been recorded yet, so this card claims no measured value for this repository.

---

## Interactive Colab Tutorials

- **End-to-end detection and adaptation tutorial**:
  [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kurtvalcorza/detr-detection-pipeline/blob/main/tutorials/detr_detection_colab.ipynb) [`detr_detection_colab.ipynb`](https://github.com/kurtvalcorza/detr-detection-pipeline/blob/main/tutorials/detr_detection_colab.ipynb)
  *COCO detection on a drawn scene, degenerate-input probes, a bounded fine-tune that re-heads DETR onto a three-class sign vocabulary, held-out COCO-style average precision before and after, new-data inference, and SafeTensors adapter export and reload.*

---

#### Description

`facebook/detr-resnet-50` is the Hugging Face Transformers release of DETR (DEtection TRansformer) with a ResNet-50 backbone, from "End-to-End Object Detection with Transformers" (Carion et al., arXiv:2005.12872). The snapshot's `config.json` declares `DetrForObjectDetection` with a `timm` `resnet50` backbone, a 6-layer encoder and a 6-layer decoder at hidden size 256, and 100 learned object queries (`num_queries`). The Hub reports 41.6M parameters; this repository has not counted them.

DETR treats detection as set prediction. The backbone's last feature map, flattened and given sine position encodings, passes through the encoder. Each decoder query then attends to that memory and emits one box and one class distribution. The distribution covers 91 COCO category slots plus a "no object" class. At inference, `post_process_object_detection` takes each query's most probable class that is not "no object" and keeps the query when that probability reaches the threshold. There is no anchor box and no non-maximum suppression.

The pretrained weights come from supervised training on COCO 2017. This repository adds gradient fine-tuning on a caller's labelled boxes. `from_pretrained(class_names=...)` replaces the class head with a new layer for the caller's vocabulary. `finetune` then trains with the upstream set-prediction loss: Hungarian matching of queries to reference boxes, cross-entropy with a down-weighted "no object" term, and L1 plus generalised-IoU box terms. The ResNet backbone is frozen by default.

What this repository adds to the upstream weights:

- `verify_snapshot` and `stage_missing_files`: manifest checks and staging of the pinned files, both refusing to run if `MODEL_REVISION` is ever reset to the `"unpinned"` sentinel;
- `DetrDetectionPipeline.from_pretrained`: loading from the verified directory only, with `trust_remote_code=False`, `local_files_only=True` and `use_pretrained_backbone=False`;
- `detect` and `detect_many`: input checks, threshold checks, and score-sorted pixel-space output;
- `validate_inputs`, `validate_dataset`, `read_detection_records` and `evaluation_report`: the validation and single-image evaluation stages;
- `finetune`, `evaluate` (package-local `average_precision`), `save_artifact`, `apply_artifact` and `load_artifact`: the bounded adaptation workflow and its SafeTensors adapter;
- `tools/pin_snapshot.py`, `tools/build_notebook.py` and `tools/validate_release_assets.py`: pinning, notebook generation and static release checks.

#### Intended Use and Limitations

The uses below are the ones the package was built to support. Everything else is out of scope (Out-of-scope use cases) or prohibited (Use cases).

###### Primary Intended Uses

The task is closed-vocabulary object detection. `detect` takes one `PIL.Image.Image` and a threshold. It returns at most 100 detections, each an xyxy box in input pixels, a `label` from the model's class slots, and a `score`, sorted by score.

The pretrained vocabulary fits everyday scenes: people and vehicles in street imagery, animals, furniture and household objects indoors, sports and food. The adaptation path fits a small labelled set in a new vocabulary, for example signage, parts or equipment that COCO does not name.

The intended role is a reference detector and a teaching baseline. DETR is the model that later set-prediction detectors, such as Conditional DETR, Deformable DETR and RT-DETR, are compared against. A reader's own application can embed `DetrDetectionPipeline` as the first stage of counting, cropping or downstream classification.

###### Primary Intended Users

Intended users are machine-learning engineers, computer-vision researchers, students and instructors. Settings envisioned are research prototypes, teaching, and self-hosted applications that run the code in this repository.

A user is expected to know the following before relying on the output:

- the pretrained vocabulary is the 91 COCO category slots, of which 80 were trained with boxes; anything else is mislabelled or missed;
- a `score` is a softmax probability under the model's own training distribution, not a calibrated probability on the user's images;
- the threshold trades recall against false boxes and must be set per deployment;
- drawn graphics, documents, aerial, medical and thermal imagery are distribution shifts from COCO photographs;
- average precision, precision and recall can only be measured on a labelled set the user supplies;
- a fine-tune on a few dozen images demonstrates the workflow and does not produce a deployable detector.

###### Out-of-scope use cases

1. **Capability boundary:** no classes outside the loaded vocabulary and no text prompts (the public `owlv2-detection-pipeline` and `grounding-dino-detection-pipeline` repositories cover open-vocabulary detection). No instance or panoptic masks, although DETR has segmentation variants. No tracking, keypoints or video. No calibrated confidence.
2. **Input boundary:** `validate_image` rejects anything that is not a `PIL.Image.Image` (`TypeError`), and sides below `MIN_IMAGE_SIDE = 16` px or above `MAX_IMAGE_SIDE = 4096` px (`ValueError`). Thresholds outside `[0, 1]` are rejected. One image per `detect` call; at most `MAX_DETECTIONS = 100` detections, one per query.
3. **Input boundary:** the processor resizes every image so its shorter side is 800 px and its longer side at most 1333 px. The DETR paper reports weaker results on small objects than on large ones. Objects only a few pixels tall after that resize are not a use this repository supports.
4. **Data boundary for adaptation:** `validate_dataset` accepts 1–5,000 records (`MAX_RECORDS`), and `split_dataset` needs at least 2. The bounded tutorial fine-tune (10 epochs by default) is far shorter than the upstream 300-epoch schedule. It is not a way to train a production detector.
5. **Decision boundary:** not for decisions that act on detections without a person reviewing them. This covers vehicle control, security alerts, safety interlocks, and medical or industrial inspection. It also requires precision and recall measured locally on the deployment's own labelled images at the chosen threshold.

#### Factors

###### Groups

The pipeline is human-centric in one respect: `person` is a COCO class, so the pretrained model localises people in any image. Neither the upstream authors nor this repository evaluated person detection per group. Recall may differ by skin tone, age, body size, clothing, mobility aids and lighting; that difference is unknown, not known to be absent.

COCO's collection skews toward consumer photographs from a limited set of regions. Vehicles, foods, furniture and signage from under-represented regions may have lower recall; that is also unmeasured.

An adapted model inherits whatever group structure the caller's labelled data has. The operator who detects people, with the pretrained or an adapted model, owns a per-group audit on their own images before relying on the output.

###### Instrumentation

The upstream training data comes from consumer cameras: colour photographs at web resolution, with boxes drawn by crowd workers under COCO's annotation rules. Inference images arrive from whatever produced them: phones, CCTV or dashboard cameras, drones, or rendering engines.

Resolution, motion blur, compression, exposure, lens distortion and viewpoint all change the visual evidence. The resize to a shorter side of 800 px changes the pixel scale of every image. The pipeline checks only type and size. It cannot detect a night frame, a fisheye lens, a rendered scene or an empty frame.

The tutorial's sample data is itself an instrument: Pillow drawings with flat colours and a bundled font, far from any camera. Adaptation boxes are exact by construction; boxes a caller supplies carry whatever error their annotation process has, and a systematic offset in them is learned by the fine-tune as if it were correct.

###### Environment

**Operating environment.** Python 3.12 with the pins in `pyproject.toml`: `torch==2.14.0`, `torchvision==0.29.0`, `torchaudio==2.11.0`, `transformers==4.57.6`, `timm==1.0.29`, `scipy==1.18.1`, `safetensors==0.8.0`, `numpy==2.5.3`, `pillow==11.3.0`, `huggingface-hub==0.36.2`. Computation is float32. The code runs on CPU and uses CUDA automatically when available. `timm` builds the ResNet-50 backbone, and `scipy` supplies the Hungarian matcher the fine-tuning loss needs. No run with the pinned weights has been recorded yet, so no runtime, memory or throughput figure is given.

**Data environment.** The pretrained model assumes a photograph of an everyday scene containing COCO objects. An adapted model assumes inference images that resemble its training images in camera, scene and object appearance. The tutorial's adaptation data is synthetic, so a model adapted on it transfers to drawn signs of the same style and to nothing else. When these assumptions fail, the model still returns boxes. The pipeline reports no signal that the distribution has shifted.

#### Metrics

###### Performance Measures

`evaluate(records)` reports average precision computed by the package-local `average_precision`:

- `ap`: mean AP over the ten IoU thresholds 0.50, 0.55, …, 0.95 (COCO's AP@[.50:.95]);
- `ap50` and `ap75`: AP at IoU 0.50 and 0.75;
- `per_class_ap50`: AP at IoU 0.50 for each class that has at least one reference box.

AP summarises the precision–recall trade-off over all score levels, so it does not depend on one threshold. `ap50` measures whether objects are found at all; `ap` also rewards tight boxes. Reading `ap50` alone hides loose localisation, and reading `ap` alone hides whether a low score comes from misses or from loose boxes. The implementation uses greedy score-ordered matching and 101-point interpolation. It has no pycocotools area ranges or crowd handling, so its values are close to, but not identical with, the COCO evaluator.

`evaluation_report(result, ground_truth_boxes)` covers one image. It reports one `box_iou` per supplied reference box, against the best-overlapping detection **of the same label**, with the verdict `sample-sanity`. Without references it returns `not-measurable` and names the labelled data that would be needed.

The upstream README reports AP 42.0 on COCO 2017 validation. That value is upstream-reported, and this repository does not reproduce it. No value from this repository has been recorded yet.

###### Decision thresholds

`detect` applies one threshold: a query is kept when its largest class probability, excluding "no object", is at least `threshold`. The default, `DETECTION_THRESHOLD = 0.9`, is the value in the pinned README's example. It was not tuned or calibrated by this repository. Each query reports exactly one label, which is an implicit argmax over its class probabilities.

`evaluate` uses `EVAL_DETECTION_THRESHOLD = 0.05` and keeps at most `MAX_EVAL_DETECTIONS = 100` detections per image. A low threshold is needed so that AP can see the full precision–recall curve. It is not a deployment setting.

No acceptance threshold on AP is set anywhere in the repository. The deployment owns choosing `threshold` on its own labelled images. Lower it when a missed object costs more than a false box. Raise it when a false box triggers downstream action. Re-tune it after any change of camera, scene, class mix or adapter.

###### Approaches to uncertainty and variability

Every AP value is one pass over one held-out split: no repeated runs, no cross-validation, no bootstrap, and no confidence interval. The tutorial's held-out split has 10 synthetic images. One image more or less found moves its AP visibly, so the value is tutorial evidence only.

Sources of run-to-run variability:

- the dataset draw, the split and the new-data draw, controlled by `DATASET_SEED`, `SEED` and `NEW_DATA_SEED`;
- the class-head initialisation and the batch order, controlled by the `seed` argument;
- GPU kernel selection, which is not forced to be deterministic, so repeated GPU runs can differ slightly;
- dropout (0.1 in the pinned config), active during fine-tuning only.

A `score` is a softmax output, not a calibrated probability. A caller who needs calibrated confidence must fit a calibration map on labelled images from the deployment. A caller who needs an uncertainty estimate for AP must evaluate over many images, with repeated runs or bootstrap resampling.

#### Ethical considerations and biases

No external ethics board, red team, or population-specific review has examined this repository or, to our knowledge, the upstream checkpoint. Nothing below implies that one did.

###### Data

The upstream README states that the model was trained on COCO 2017 object detection: 118k training and 5k validation images. The COCO paper describes images collected from Flickr and annotated by crowd workers. COCO contains identifiable people, licence plates, house numbers and private settings. Personal data is therefore present in the training corpus by construction; it was not audited here.

This repository distributes code, tests, documentation, and two small configuration files copied from the upstream snapshot (`config.json`, `preprocessor_config.json`). It does not distribute `model.safetensors` (166,587,896 bytes), which is staged locally and git-ignored. It ships no photographs: the tutorial scene and the adaptation dataset are drawn in code.

An exported adapter contains weights fitted to the caller's training images. It does not contain the images, but it can reflect them. The operator must audit the images they detect on, or fine-tune on, for personal, proprietary or restricted content; the pipeline performs no such check.

###### Human Life

The pipeline is not intended for decisions in health, safety, criminal justice, employment, credit or housing. Neither this repository, the upstream authors, nor any regulator has validated or certified it for any of them.

Some sensitive uses are foreseeable although not intended: pedestrian detection for vehicle control, person detection for surveillance, crowd counting for policing, and PPE or hazard detection for safety interlocks. Any of them would be admissible only with human review of every acted-on detection. They would also need locally measured precision and recall stratified by the groups named above, a documented threshold and re-validation policy, and any regulatory clearance the domain requires.

###### Mitigations

- **Supply-chain integrity:** while `MODEL_REVISION` is `"unpinned"`, `verify_snapshot`, `stage_missing_files` and `from_pretrained` raise before any download or model import. Once pinned, `stage_missing_files` refuses a manifest whose `modelId` or `revision` differs from the package constants. It fetches only manifest-listed files, and only with `allow_download=True`. `verify_snapshot` checks every file's byte size and SHA-256 and refuses an entry with no recorded digest. `from_pretrained` loads only the verified directory, with `local_files_only=True`, `trust_remote_code=False` and `use_pretrained_backbone=False`. The upstream `pytorch_model.bin` pickle is not in the manifest and is never staged.
- **Tests of those refusals:** tests assert that an unpinned package, a missing snapshot and a tampered digest are all refused before `torch`, `transformers`, `timm` or `scipy` is imported. Another asserts that `LABELS` equals the snapshot's `id2label` order.
- **Input integrity:** `validate_inputs` and `detect` share one checker for type, image size and threshold. `validate_dataset` rejects a record with missing keys, an out-of-range image, a box count different from the label count, a non-finite, empty or out-of-bounds box, or an unknown label. It reports classes that have no box as findings. `read_detection_records` refuses absolute file names and `..` segments. `detect` raises on a malformed backend detection, an unknown label or more than 100 detections.
- **Adapter integrity:** `save_artifact` writes SafeTensors, not pickle, with the base identity, base-weights digest, class names and frozen prefixes in its header. `apply_artifact` refuses a different format, base identity or model key. It also refuses a different vocabulary, a different base digest, tensors the model does not have, and any missing trainable tensor.
- **Reproducibility:** exact `==` pins in `pyproject.toml`, carried into the notebook and checked by the parity tests. Seeds for data, split, head initialisation and batch order. Every result and adapter records `model_id` and `model_revision`.
- **Refusals:** no download without the explicit flag, no pickle deserialisation, no remote model code, and no export of an unadapted pipeline.
- **Statistical mitigations:** none is implemented. There is no class balancing, re-sampling or augmentation; `validate_dataset` reports class coverage and does not change it.

###### Risks and harms

- **Boxes on empty or unfamiliar input.** The model can return boxes for images that contain no object of any trained class. The operator and any downstream consumer bear the harm of a fabricated count or alert. Likelihood on real empty frames is unmeasured; the tutorial probes a blank and a noise image and records what it finds.
- **Missed objects.** An object that is small, occluded or rendered unusually is simply absent from the output, and no field flags the miss. The harm falls on whoever relies on the detection being complete.
- **Mislabelling within a closed vocabulary.** An object outside the vocabulary that resembles a class is labelled as that class. Systems that act on labels inherit the error.
- **Overfitting in adaptation.** A fine-tune on a few dozen images can score well on a held-out split drawn from the same source and fail on anything else. The operator who deploys it bears the harm, which is realised whenever training and deployment images differ.
- **Person detection and surveillance.** `person` is a first-class output with unaudited per-group recall. The people in the processed images bear the harm of misuse or unequal error.
- **Automation bias.** High softmax scores invite trust that an uncalibrated score has not earned. Operators who skip review turn a model error into a decision error.
- **Leakage through adaptation data.** A random split of records that share a source photograph or session puts near-duplicates on both sides. The resulting held-out AP overstates quality without any signal; `split_dataset` documents that grouped data must be split by group.
- **Resource use.** Inference at an 800 px shorter side on CPU can be slow on large images. A video stream can saturate a shared host.

###### Use cases

The following uses are prohibited even where the model would work:

- detecting people in order to surveil, track, profile or score them, or to support biometric or demographic profiling;
- unlawful discrimination in employment, housing, credit, insurance, education, healthcare access or law enforcement;
- processing images the operator has no right to process, or in breach of consent, privacy or data-protection obligations;
- deceptive uses that present detections as verified facts or as evidence;
- autonomous physical control or safety interlocks driven by unreviewed detections;
- any use that violates the upstream Apache-2.0 licence or the terms of the deployment running the pipeline.

## Immutable provenance

- Model: `facebook/detr-resnet-50`
- Revision: `1d5f47bd3bdd2c4bbfa585418ffe6da5028b4c0b` (pinned 2026-09-25 by `python tools/pin_snapshot.py`, which resolved the Hub's `main` to this commit, downloaded every manifest file at it, and recorded each file's SHA-256).
- Snapshot manifest: `weights/detr-resnet-50/dimer-base-manifest.json`, 4 files, `totalBytes` 166598669. The byte sizes and digests describe the files at the pinned commit.
- `model.safetensors`: 166,587,896 bytes; SHA-256 `830f5e2eeaada8c8c8281779dcc8ab12833972eb8514ed0a35be6c1d4420ad81` (matches the Hub's LFS record).
- `config.json`: 4,592 bytes; `DetrForObjectDetection`, `timm` `resnet50` backbone, 100 queries, 91 label slots.
- `preprocessor_config.json`: 290 bytes; `DetrImageProcessor`, shorter side 800 px, longer side at most 1333 px, ImageNet normalisation.
- `README.md`: 5,891 bytes; the upstream model card.
- Loader: `DetrForObjectDetection.from_pretrained(<verified dir>, local_files_only=True, trust_remote_code=False, use_pretrained_backbone=False)`, with `DetrImageProcessor` from the same directory.

## Input/output contract

- `DetrDetectionPipeline.from_pretrained(device=None, weights_dir=None, allow_download=False, class_names=None, seed=20260924)`: stage (only with `allow_download=True`), verify, load; with `class_names`, replace the class head deterministically under `seed`.
- `detect(image, *, threshold=0.9) -> dict`: keys `detections` (list of `{"box": [x0, y0, x1, y1], "label": str, "score": float}`, input pixels, descending score, at most 100), `threshold`, `width`, `height`, `class_names`, `adapted`, `model_id`, `model_revision`.
- `detect_many(images, *, threshold=0.05) -> list[list[dict]]`.
- `finetune(records, *, epochs=10, batch_size=4, learning_rate=1e-4, seed=20260924, freeze_backbone=True, progress=None) -> dict`: AdamW (weight decay `1e-4`), gradient-norm clipping at 0.1, float32; returns the configuration, parameter counts and per-epoch losses.
- `evaluate(records, *, threshold=0.05, iou_thresholds=(0.50, …, 0.95), max_detections=100) -> dict`: `ap`, `ap50`, `ap75`, `per_class_ap50`, `n_images`, `n_references`, `estimation`.
- `save_artifact(path, *, notes=None) -> dict`; `read_artifact_metadata(path) -> dict`; `apply_artifact(path)`; `load_artifact(path, *, weights_dir=None, device=None)`. The adapter format is `detr-adapter-v1`.
- Records: `{"image": PIL.Image.Image, "boxes": [[x0, y0, x1, y1], ...], "labels": [name, ...]}`; `read_detection_records(directory)` reads them from `annotations.json` plus image files.
- Constants: `MIN_IMAGE_SIDE = 16`, `MAX_IMAGE_SIDE = 4096`, `MAX_DETECTIONS = 100`, `MAX_RECORDS = 5000`, `MAX_CLASSES = 1000`, `LABELS` (91 slots), `UNANNOTATED_LABEL_IDS` (11 slots), `DETECTION_THRESHOLD = 0.9`, `EVAL_DETECTION_THRESHOLD = 0.05`.

## Verification records

No execution with the pinned weights has been recorded. The offline test suite runs a tiny random-weight DETR through fine-tuning, evaluation and adapter reload; that exercises the code path and is not a result about this model. `docs/release-verification.md` holds the release gate and the record table.

## References

- Carion et al. End-to-End Object Detection with Transformers. ECCV 2020. https://arxiv.org/abs/2005.12872
- Lin et al. Microsoft COCO: Common Objects in Context. ECCV 2014. https://arxiv.org/abs/1405.0312
- Upstream code: https://github.com/facebookresearch/detr
- Upstream card: https://huggingface.co/facebook/detr-resnet-50
- Transformers DETR documentation: https://huggingface.co/docs/transformers/model_doc/detr
