# Release verification

`tutorials/detr_detection_colab.ipynb` (`E2E`, **standalone** carrier) is a **release candidate** until the exact notebook revision has executed top-to-bottom in a clean supported runtime. Unit tests, the tiny-model test, JSON validation, code-cell compilation, the generator parity checks and `tools/validate_release_assets.py` are necessary checks but are **not** runtime evidence under DIMER Notebook Specification 2.2 (REL8). This file is the durable release-gate record for the notebook.

The upstream snapshot is pinned to `1d5f47bd3bdd2c4bbfa585418ffe6da5028b4c0b` and the notebook is regenerated with that revision and manifest. A default-path execution was recorded on 2026-09-25 (Kaggle T4; record below), but it completed only after a manual restart, so it is **not** a one-pass `Run all` and not promotion evidence. Since 2026-10-04 the notebook runs in an isolated `uv` environment (no in-kernel install, no restart; Linux x86_64 runtimes only) and carries the 2026-10-02 review's fixes; that revision had a local CPU pre-flight (below) and, on 2026-10-04, one sequential pass with no restart and 0 errors on a fresh Colab Tesla T4 through the Colab CLI (record below; a CLI execution, not a browser `Run all`). REL12 (step 7, the BYOD dataset exercise) and a browser one-pass `Run all` are still outstanding, so the status stays `Candidate`.

## Automatic coverage (static and unit, every pull request)

CI installs the pinned CPU-only torch wheel and the other runtime pins, then runs:

- `ruff check src tests tools`;
- `pytest`: snapshot verification and staging against synthetic manifests, the unpinned refusals, the import boundary, input and dataset validation, BYOD directory reading, average precision, the evaluation report, the notebook parity checks, the weight-facts check, and `tests/test_tiny_model.py` — a tiny random-weight DETR taken through `finetune`, `evaluate`, `save_artifact` and `apply_artifact`;
- `tools/validate_release_assets.py`: model card 1.2 structure and front matter, the pin state across the package, the manifest and the documents, identity consistency, the weight facts, the release-status tokens, and the notebook's structure, carried modules, parity, markers, BYOD gates and location fields;
- `tools/build_notebook.py --check`.

These are source, provenance and unit checks. None of them loads the pinned checkpoint, so none of them is execution evidence.

## Supported release verification procedure

Before changing the registry status from `Candidate` to `Release-grade`:

1. confirm the snapshot is pinned (`MODEL_REVISION` is a 40-hex commit and every manifest entry has a SHA-256) and that static CI is green on the exact commit under review;
2. open that exact notebook revision in a new GPU runtime (Colab or Kaggle, Tesla T4 or similar) with **no repository checkout** and a clean model cache;
3. run the notebook top-to-bottom **in one pass with Run all** — no restart, no re-run of any cell — without editing implementation cells, with every form field at its default (`USE_BYOD_IMAGE = False`, `USE_BYOD_DATASET = False`, `threshold = 0.9`, `EPOCHS = 10`);
4. verify that the isolated environment was built (Section 1 prints the isolated and kernel Python versions), that the runtime record reports `NOTEBOOK_SOURCE.repository_revision` equal to `metadata.dimer.generated_from.revision`, and that the imported versions equal the inline `PINS`; record `restarted: false` — a run that needed a restart is not a one-pass `Run all` and does not qualify;
5. verify that every default-path stage completes: the pinned install; the carried modules; staging of all four manifest entries and `verify_snapshot`; `validate_inputs` with the `threshold=1.5` rejection finding; COCO detection and the `sample-sanity` report; the blank and noise probes; `validate_dataset`; the split with its leakage assertion; the baseline; `finetune`; the adapted evaluation with the held-out check at the operating threshold and at the training-split adapted threshold; new-data inference; adapter export, reload and the tolerance check; and the six outputs in `outputs/`;
6. record the observed baseline and adapted `ap`, `ap50` and `ap75`, the per-epoch losses, the trainable and total parameter counts, the degenerate-probe counts and the wall time in the table below. No metric value is asserted in advance: a low adapted AP is a finding to record, not a failure by itself;
7. exercise the BYOD dataset branch once with a small labelled directory (`BYOD_DATASET_DIR`) and once with an incompatible `annotations.json`, and record both outcomes (REL12);
8. record the notebook Git blob id, commit, runtime (platform, GPU, Python, PyTorch, Transformers, timm), model identifier and revision, whether the model cache was clean, and any warning judged harmless with the reason;
9. record no access tokens or other secrets.

A known-failing default path in the supported runtime blocks release.

## Recorded executions

Notebook identity is the Git blob id of `tutorials/detr_detection_colab.ipynb` (verify with `git rev-parse <commit>:tutorials/detr_detection_colab.ipynb`). Each record uses the fields Date, Subject, Runtime, Procedure, Observed result and Caveats.

| Date (UTC) | Subject (commit / notebook blob) | Runtime | Procedure | Observed result | Caveats |
|---|---|---|---|---|---|
| 2026-09-25 (23:11:59–23:18:01) | commit `e39d680` / blob `a98705edf1cb` (`NOTEBOOK_SOURCE.repository_revision` `7ee32a4`, the source revision the notebook was generated from; `7ee32a4..e39d680` changes only the notebook and `tests/test_pin_snapshot.py`) | Kaggle Tesla T4 (`kurtvalcorza/dimer-nb2-detr-detection` v1), batch run; image `gcr.io/kaggle-gpu-images/python@sha256:37c64f7dd9c54116ecd1bcc88817c5469b88387388fade02bfa8bf3fc647d461` (image torch 2.10.0+cu128, transformers 5.0.0), Python 3.12.13, Tesla T4 15360 MiB, driver 580.159.04; after the inline pins torch 2.14.0+cu130 (CUDA 13.0), transformers 4.57.6, timm 1.0.29, scipy 1.18.1, numpy 2.5.3, Pillow 11.3.0; device `cuda:0`, float32 | serial suite: blob fetched at the 40-char SHA and Git-blob verified, HF cache clean at start, no repository checkout; top-to-bottom with every form field at its default (`USE_BYOD_IMAGE = False`, `USE_BYOD_DATASET = False`, `threshold = 0.9`, `EPOCHS = 10`); `facebook/detr-resnet-50` at `1d5f47bd3bdd2c4bbfa585418ffe6da5028b4c0b`, 4 manifest files fetched and `verify_snapshot` passed (`verified_files` 4) | **Passed only after a manual restart** (not a one-pass `Run all`, not promotion evidence; corrected 2026-10-04 from "PASSED (default path)") — 357.5 s wall (pass 1 267.4 s stopped after the install cell with `RuntimeError: Core dependencies changed while older modules were loaded` (cuda-bindings 12.9.4 → 13.4.3, numpy 2.0.2 → 2.5.3), kernel restarted after the install cell as the notebook instructs; pass 2 90.0 s), 14/14 post-restart code cells ok; `validate_inputs` recorded the `threshold=1.5` rejection; COCO scene (threshold 0.9): stop sign 0.9991, traffic light 0.9975, clock 0.9733, same-label `box_iou` 0.8465 / 0.9095 / 0.8983, sports ball not detected (0.0), 3/4 matched, `sample-sanity`; blank and noise probes 0 detections at 0.9 and at 0.05; sign dataset 40 records / 82 boxes (stop-sign 21, yield-sign 34, speed-limit-sign 27), `verdict` accepted, split seed 0 → 30 train / 10 held out (17 references); class head re-initialised to 4 slots; `finetune` backbone frozen, 18,047,240 of 41,502,152 parameters trainable, AdamW lr 1e-4, batch 4, 10 epochs, losses 1.6564, 1.0888, 0.9144, 0.8003, 0.7812, 0.7256, 0.6889, 0.6303, 0.5725, 0.5602; held-out evaluation (threshold 0.05, max 100 detections, package-local AP): baseline ap 0.0063 / ap50 0.0172 / ap75 0.0063 (per-class AP50 stop-sign 0.0, yield-sign 0.0218, speed-limit-sign 0.0297; 100 detections per image before the cap), adapted ap 0.2222 / ap50 0.2668 / ap75 0.2479 (per-class AP50 stop-sign 0.2079, yield-sign 0.3140, speed-limit-sign 0.2783; 1–7 detections per image); new-data inference on 3 drawn images (seed 99, 5 truth boxes) at threshold 0.9: **0 detections on all 3**, same-label IoU 0.0 for every truth box; adapter `detr_adapter.safetensors` 72,219,224 B sha256 `6a455a2193082a4d05f1d20edcf53cdf8e7c62c6199ae11a6526eb0796864878` (265 tensors, format `detr-adapter-v1`, base state digest `830f5e2eeaada8c8c8281779dcc8ab12833972eb8514ed0a35be6c1d4420ad81`), reload onto a fresh base: 7 detections compared, tolerance 0.001, equivalent; BYOD refusal probes: non-image → `TypeError`, out-of-bounds box → `ValueError`; six outputs written, sha256 `detr_detection_result.json` `4a18a1b29d82…`, `detr_detection_evaluation_report.json` `b1fde2f1f5a9…`, `detr_detection_input_manifest.json` `e2fcc09d2690…`, `detr_detection_detections.csv` `31a3158b52e0…`, `detr_detection_annotated.png` `b1724b3d2e44…` | One seeded split of 10 drawn images, one runtime, no dispersion estimate; the COCO part is one drawn scene, not a COCO evaluation. The adapted AP is low and the adapted model found nothing on the 3 new images at the default threshold — recorded as findings (step 6), not hidden. Harmless warnings: the checkpoint's unused `num_batches_tracked` backbone buffers, and the expected class-head shape mismatch (92 → 4) when re-heading. **REL12 not exercised** (both BYOD branches off), so the status stays `Candidate`. No tokens or secrets recorded. |

## 2026-10-04 review fixes: uv isolated environment, guided layer, adapted threshold (local pre-flight only)

The 2026-10-02 Notebook Review Framework v1 review (`docs/reviews/`, findings DTR-M1..M4, DTR-m1..m3) is addressed on the review branch by porting the sibling `conditional-detr-detection-pipeline` fix (`fd008c1`, merged as `879e0b1`, Colab T4 one-pass run recorded there). `tools/build_notebook.py` is generator /2.1 (NOTEBOOK_SPEC 2.2) and the template sets `isolated_runtime`: the first code cell downloads the pinned `uv` 0.12.15 wheel (size and SHA-256 checked), builds a managed CPython 3.12.12 environment and installs `tutorials/requirements-colab.lock.txt` (the same 49-package lock as `conditional-detr-detection-pipeline`, whose `pyproject.toml` dependencies are identical to this repository's) with `--require-hashes --only-binary :all:`; the second routes every later cell to one worker there. Nothing is installed into the kernel and Run all needs no restart; **Linux x86_64 runtimes only**. Section 9 chooses `ADAPTED_THRESHOLD` on the **training split only** (max F1, same label, IoU >= 0.5; Kurt's decision for the sibling's CDT-M2, option B, 2026-10-04) and Sections 9–12 show it next to the unchanged 0.9 view; Section 8 rebuilds the re-headed model before every re-run; the BYOD dataset branch writes its own result JSON and CSV and compares the reloaded adapter. `src/` is unchanged (the rice capstone stays byte-identical). The isolated worker's `google.colab` stubs carry a `ModuleSpec`.

| Date (UTC) | Subject | Runtime | Procedure | Observed result | Caveats |
|---|---|---|---|---|---|
| 2026-10-04 | review branch, regenerated notebook (learner code as committed) | local Windows 11 workstation CPU, 24 threads, heavily shared with other jobs; Python 3.12.12, torch 2.14.0+cpu, Transformers 4.57.6, timm 1.0.29 — the notebook's pins; real `1d5f47b` snapshot files pre-staged and re-hashed by Section 3 | the notebook's learner cells executed verbatim from the JSON with `DIMER_NOTEBOOK_CI_PREINSTALLED=1` (the two kernel cells — uv install and router — not executed: Linux-only), every field at its default; then the BYOD dataset branch from a `.zip` with a sub-folder (10 records, defaults) | 14/14 ok, 863 s of cell time (fine-tune 548 s). COCO scene 3/4 at 0.9; blank and noise 0 at 0.9 and 0.05; baseline `ap` 0.0063 / `ap50` 0.0172; losses 2.1077 → 0.4860; adapted `ap` 0.2884 / `ap50` 0.3205 (identical to the review's CPU run). `ADAPTED_THRESHOLD` 0.2905 (training F1 0.5714, precision 0.6296, recall 0.5231, 30 images). Held-out: 0.9 → 0 returned of 17 references (highest score 0.421); 0.2905 → 14 returned, 4 correct (precision 0.2857, recall 0.2353). Unseen: 0/5 at 0.9, **1/5 at 0.2905** (yield sign IoU 0.913, score 0.291; one wrong-label box at 0.390). Reload 2 compared, equivalent. BYOD: AP50 0.1958 → 0.4680, own threshold 0.1746, result JSON + CSV + adapter written, reload 100 compared, equivalent | pre-flight only, **not clean-runtime evidence**; adapted scores sit between about 0.25 and 0.42, so the hosted (GPU) threshold and the unseen count may differ, including 0/5 |
| 2026-10-04 | same | same | Section 14 check: after the default run (adapter reloaded from that run's export, so `adapter.adapted` is True), Section 8 re-run with `FREEZE_BACKBONE = False` and `EPOCHS = 1` (lowered for time, labelled) | printed `rebuilt_from_verified_snapshot`; 41,502,152 of 41,502,152 parameters trainable; epoch-1 loss 2.6567 (a fresh start; the review measured 0.7202 when the old cell continued training the adapted model) | the full 10-epoch unfrozen comparison was not run |

## 2026-10-04 Colab T4 run of the review-fix head (Colab CLI, one pass)

| Date (UTC) | Subject (commit / notebook blob) | Runtime | Procedure | Observed result | Caveats |
|---|---|---|---|---|---|
| 2026-10-04 (07:16:30–07:19:15) | commit `c3c81cbcbfab27a5c092d9ef3629f06cafa88730` / blob `6b2e5fc3d373f0638309863a6373b5193186490f` (`NOTEBOOK_SOURCE.repository_revision` `808b10c97552` = `metadata.dimer.generated_from.revision`) | Colab CLI 0.7.4 sequential execution, fresh Colab Tesla T4 VM (session `suite-detr-c3c81cb-dd82`, released after the run); kernel Python 3.13.15; isolated `uv` environment Python 3.12.12, 49 locked packages, setup 60 s; torch 2.14.0+cu130, Transformers 4.57.6, timm 1.0.29, scipy 1.18.1, numpy 2.5.3, Pillow 11.3.0; device `cuda:0`, float32 | blob fetched from `raw.githubusercontent.com` at the 40-char SHA and Git-blob verified before the VM was allocated; no repository checkout, clean model cache; `colab exec -f` ran the 16 code cells in order in one kernel (order from `exec.log` "Executing cell k/16"; the CLI sets no execution counts) with every field at its default (`USE_BYOD_IMAGE = False`, `USE_BYOD_DATASET = False`, `threshold = 0.9`, `EPOCHS = 10`); executed code-cell sources equal the blob's | **One pass, no restart, 0 errors** — 16/16 code cells, 165.1 s wall (cells 4–5 are the carried module definitions and print nothing). Snapshot: 4 manifest files fetched, `verified_files` 4. `validate_inputs` recorded the `threshold=1.5` rejection. COCO scene at 0.9: stop sign 0.999, traffic light 0.997, clock 0.973, `box_iou` 0.847 / 0.910 / 0.898, sports ball not detected, 3/4 matched, `sample-sanity`; blank and noise 0 at 0.9 and at 0.05. Sign dataset 40 records / 82 boxes, accepted; split 30 / 10 (65 / 17 boxes). Baseline `ap` 0.0063 / `ap50` 0.0172. Fine-tune (backbone frozen, 18,047,240 of 41,502,152 trainable, 33.7 s): losses 1.6564, 1.0888, 0.9157, 0.8218, 0.8045, 0.7688, 0.7231, 0.6761, 0.6328, 0.6103. Adapted `ap` 0.2983 / `ap50` 0.3851 / `ap75` 0.3527 (per-class AP50 stop-sign 0.4059, yield-sign 0.4638, speed-limit-sign 0.2857). `ADAPTED_THRESHOLD` **0.2231** (training split, F1 / precision / recall 0.5538, 30 images). Held-out: 0.9 → 0 returned of 17 (highest score 0.3078); 0.2231 → 15 returned, 5 correct (precision 0.3333, recall 0.2941). Unseen (seed 99, 5 references): **0/5 at 0.9, 1/5 at 0.2231** — the stop sign on image 002 (IoU 0.874, score 0.225); wrong-label boxes passed the adapted threshold on images 000 (`speed-limit-sign` 0.279) and 001 (two `stop-sign`, 0.273 and 0.253); highest scores per image 0.279, 0.273, 0.225. Adapter 72,219,224 B, sha256 `f433b1063499b1970294159e83c2b93fa3357c50c30a6b44545d35c77c8efa48`, 265 tensors; reload: 18 detections compared at 0.05, tolerance 0.001, equivalent. BYOD refusal probes: `TypeError`, `ValueError`; six outputs written | Evidence files in `execution-evidence/2026-10-04/`: `detr_detection_colab_c3c81cb_colab-cli-t4.ipynb` sha256 `2b8d501f7a42b3b276d592f214c82d7772993d173fe0262c96634080e0255408`, `detr_detection_colab_c3c81cb_run_summary.json` `a269d167a13b5e3cd67f49cda0bbffcf5591fadf2368dcce6fe5de7052ae4bcf`, `detr_detection_colab_c3c81cb_exec.log` `3e6c07d4e77bb12163d405c02eaf37c8c93d00dd82e8c87ffb6900a89db555da`. A Colab CLI execution, **not a browser `Run all`**: forms not rendered; optional journeys (BYOD branches, Section 14 activity) not exercised. The T4 numbers differ from the CPU pre-flight (threshold 0.2231 vs 0.2905; the recovered unseen sign is the stop sign on image 002, not the yield sign on image 001); the guided text quotes the CPU figures labelled as CPU, and the lesson (ranking improves, nothing passes 0.9, a training-split threshold recovers about 1 of 5) holds on T4. One seeded split, no dispersion estimate. Harmless warnings: unused `num_batches_tracked` buffers and the expected 92 → 4 class-head mismatch. No tokens or secrets recorded |

Outstanding before promotion: a browser one-pass **Run all** of the current blob on a fresh Colab or Kaggle GPU runtime (`restarted: false`), the REL12 BYOD exercise, and a human review of the guided layer. Status stays **Candidate**.

## Rice-pest capstone recorded executions

The rice-pest capstone (`tutorials/DIMER_Philippine_Rice_Pest_Surveillance_Capstone.ipynb`) keeps its full records in [`rice-capstone.md`](rice-capstone.md#recorded-executions); this table indexes them.

| Date (UTC) | Subject (commit / notebook blob) | Runtime | Procedure | Observed result | Caveats |
|---|---|---|---|---|---|
| 2026-10-12 (19:58:36 start, 1,402.2 s wall) | commit `49a7ff8c7db042d1be7e6f0df874068c5f834842` (PR #6 head) / blob `5ca0cc91f85c7f3e5afa2c2f767e7e7d68d35b76` | Colab CLI 0.7.4, fresh Colab Tesla T4 (session `suite-detr-49a7ff8-7c5e`, released) | blob fetched at the 40-char SHA and Git-blob verified; `colab exec -f` ran the 10 code cells in order with defaults | **Passed**: 10/10 code cells, 0 errors; every held-out metric, count (445 retained, 126 unmatched), threshold (0.9), review margin (0.00523372) and the existing bootstrap interval equal the `9f85155` run; new: two bootstrap intervals, referral report (0/337 validation, 0/445 test), stricter/looser margin labels, `images_checked` 2. Evidence: `execution-evidence/2026-10-12/DIMER_Philippine_Rice_Pest_Surveillance_Capstone_49a7ff8_colab-cli-t4.ipynb` sha256 `1c7f5672589a91b34fea519d5d8f72f6831aa768cc6c6371cb20fbb3706bad24` | CLI sequential execution, **not a browser Run all**; forms and uploads not exercised; BYOD not assessed. Covers the **previous design**: the later owner decisions (RPS-m5 = C activity controls, RPS-CX4 = A training-median baseline) change the blob and need a new T4 run. Status stays **Candidate** |
| 2026-10-14 (14:01:36 start, 1,160.6 s wall) | commit `0ee0de310d4cc7b7d6d0d0eb786ce289a6ff6d02` (PR #6 head) / blob `991ab9aa9deab64128196dd8d3e15f6ab590490b` | Colab CLI 0.7.4, fresh Colab Tesla T4 (session `suite-detr-0ee0de3-7c5e`, released) | blob fetched at the 40-char SHA and Git-blob verified; `colab exec -f` ran the 10 code cells in order with defaults (Section 8 controls off) | **Passed**: 10/10 code cells, 0 errors; every detector/classifier metric, count (445 retained, 126 unmatched), threshold (0.9), review margin (0.00523372), referral report and earlier bootstrap interval equal the `49a7ff8` run; new: `training_median` baseline MAE 5.4125 (offline 5.41) with paired bootstrap difference -3.3625, CI [-5.525, -1.375]; controls printout `{'display_threshold': None, 'review_margin': None}`; no `CompletedProcess` echo. Evidence: `execution-evidence/2026-10-14/DIMER_Philippine_Rice_Pest_Surveillance_Capstone_0ee0de3_colab-cli-t4.ipynb` sha256 `56cf506c90f45cf3e43d708e2e4ad2b53d44549ea04a6bf555e9c74c4fb014f3` | CLI sequential execution, **not a browser Run all**; forms and uploads not exercised; controls switched on and BYOD not assessed. Status stays **Candidate** |

## Final prose pass (2026-10-26)

**Code commit `ff68418`, notebook commit `c5f14ef`.** The current blob of `tutorials/detr_detection_colab.ipynb` is `43868fa1a59e7992f7ce7f442a71d5e264c8cb03`, **Candidate**; that exact blob (PR #7 head `3b7e6d0`) passed its confirming hosted run on 2026-10-26: Colab CLI sequential execution on a fresh Colab Tesla T4, one pass, no restart, 0 errors (16/16 code cells, 135.3 s wall; adapted `ap50` 0.4486, `ap` 0.3561; 0 boxes at 0.9; 3/5 unseen signs at the adapted threshold 0.1945). The paths this blob's runs cover: the default path only. Remaining gates: REL12 (a hosted BYOD run, one input accepted and one refused) and REL2 (a browser `Run all`). Runs of earlier blobs are history (REL14). See "2026-10-26 Colab T4 run of the final-prose-pass head" below. Kurt's 2026-10-26 ruling: fix only the prose rule breaks the final evidence flagged; recorded drift stays recorded, not fixed. Markdown only: the Section 7 baseline and the Section 14 worked answer no longer state what a run is expected to print. The runs of blob `c1982353` and earlier blobs are history (REL14); the records below are kept.

## Prose kept true across runs (2026-10-25)

**Historical wording (2026-10-26): code commit `ccd85d5`, notebook commit `87eb45f`.** The previous blob of `tutorials/detr_detection_colab.ipynb` was `c19823537fa46d8da85749587756f39e8d366aa5`, **Candidate**, regenerated under Kurt's 2026-10-26 ruling: every run-derived number in the prose names the hosted runs that printed it (Kaggle T4 of `a98705ed` 2026-09-25; Colab T4 of `6b2e5fc3` 2026-10-04, `da6b14bd` and `712d22d5` 2026-10-25), with no expectation or run-status sentence; it resolves the two prose mismatches recorded for the `712d22d5` run (final loss 0.6244; the stale "hosted runs pending" sentence). Markdown only. This exact blob (PR #7 head `dc65d30`) **passed** its confirming hosted run: Colab CLI sequential execution on a fresh Colab Tesla T4, 2026-10-25, 16/16 code cells in one pass, no restart, 0 errors, 272.9 s wall; adapted `ap50` 0.3944, `ap` 0.346, 0 boxes at 0.9, 2/5 unseen signs at the adapted threshold 0.2304; see "2026-10-25 Colab T4 run of the historical-wording head" below; evidence in `docs/execution-evidence/2026-10-25/`. This PR's runs cover the default path only. Remaining gates: REL12 (a hosted BYOD run with one accepted and one refused input) and REL2 (a browser Run all). Runs of earlier blobs are history (REL14). The records below stay as history.

**Code commit `06235e4`, notebook commit `af6856b`.** The previous blob of `tutorials/detr_detection_colab.ipynb` was `712d22d5ca82c31597707d81fe40b110ae5bf530`, **Candidate**; it ran in one pass on a fresh Colab Tesla T4 on 2026-10-25 (commit `c439bd4`, Colab CLI, 16/16 code cells, 0 errors, 171.5 s; see "2026-10-25 Colab T4 run of the prose-kept-true head" below). Markdown only: the worked answers name the recorded hosted runs (Kaggle T4 of `a98705ed`, 2026-09-25; Colab T4 of `6b2e5fc3`, 2026-10-04, and of `da6b14bd`, 2026-10-25) and give per run the values that moved: AP50 0.3851 / 0.3594 (Kaggle 0.2668), AP 0.2983 / 0.2993 (Kaggle 0.2222), final loss 0.6103 / 0.6078 (Kaggle 0.5602), adapted threshold 0.2231 / 0.2232 with its training metrics, held-out 15 / 5 and 14 / 4 at that threshold, unseen-image scores and IoUs, and 1 / 2 of 5 unseen signs found. Values equal in every run (first-epoch loss 1.6564, baseline AP50 0.0172, the split, 0 boxes at 0.9) are labelled with the runs. The records below are history.

## Tier C wording revision (2026-10-25)

**Tier C wording revision (2026-10-25, branch `fix/detr-detection-tier-c-2026-10-25`, code commit `b5cc7e1`).** Markdown, header and metadata only for `tutorials/detr_detection_colab.ipynb`: worked answers re-derived from the 2026-10-04 hosted run of `c3c81cb` (REL13), pretraining-overlap statement (DAT9), use notice, `notebook_kind`, IEEE references and the DIMER Notebook Specification 2.5 declaration. The previous notebook blob `c19823537fa4` (commit `87eb45f`, historical wording) has its hosted Colab T4 run pending; the previous notebook blob `712d22d5ca82` (prose values given per recorded hosted run, PR #7 head `c439bd4`) is **Candidate**; it passed a one-pass Colab T4 run of the default path on 2026-10-25 (Colab CLI, 16/16 code cells, 0 errors, 171.5 s; adapted `ap50` 0.3617, 0 held-out boxes at 0.9). The previous notebook blob `da6b14bd4382ccbbd483a8c0f91aca33505a78a3` (`8fd8d5e`, history under REL14) passed a one-pass Colab T4 run of the default path on 2026-10-25 (Colab CLI, 16/16 code cells, 0 errors, 136.6 s; see "2026-10-25 Colab T4 run of the tier C head" below), whose numbers differ from the `c3c81cb` run the worked answers quote. Remaining gates: REL12 (a BYOD run, one input accepted and one refused) and REL2 (a browser run of the interactive journeys). Under REL14 the 2026-10-04 run is history for blob `6b2e5fc3`.

The current blob `08904043` (code commit `2da6948`) shows a refused BYOD input as one `Refused: <file>: <reason>` line with no traceback, removes the earlier run's results and `outputs/` files and stops Run all at that cell, while genuine failures keep their traceback (NOTEBOOK_SPEC DAT19/DAT20); on 2026-10-29 its default path (16 code cells, exit 0, 0 errors) and its scripted DAT20 refusal (`USE_BYOD_IMAGE=True;BYOD_IMAGE_PATH=/etc/hostname`: exactly one `Refused:` line at cell `detr_detection-33`, no traceback, earlier cells clean, VM released) both passed on a hosted Colab Tesla T4 (Colab CLI `colab exec -f`, not a browser `Run all`). The remaining gates are REL12 (a hosted BYOD run with an accepted input) and REL2 (a browser run of the interactive journeys); earlier blobs' runs are history (REL14).

## 2026-10-25 Colab T4 run of the tier C head (Colab CLI, one pass)

Colab CLI sequential execution (`colab exec -f`, not a browser Run all), fresh Colab Tesla T4 (session `suite-detr-8fd8d5e-dd82`), PR #7 head
`8fd8d5e095c5c21334556792332b9fbba4077563`, notebook blob `da6b14bd4382ccbbd483a8c0f91aca33505a78a3` fetched at that commit and Git-blob verified;
started 2026-10-25T08:09:43Z, 136.6 s wall, exit code 0, input notebook unchanged, 16/16 code cells in order in `exec.log`, one pass, no restart,
0 error outputs, verdict PASS. Default path only (every form field at its default, both BYOD branches off).

Printed: input manifest `accepted` with the `threshold=1.5` probe rejected; COCO checkpoint on the drawn scene 3 detections, 3/4 drawn objects matched at IoU ≥ 0.5
(stop sign 0.847, traffic light 0.910, clock 0.898, sports ball 0.000); re-headed baseline `ap50` 0.0172 / `ap` 0.0063; 10 epochs with the backbone frozen
(18,047,240 trainable parameters), loss 1.6564 → 0.6078, `train_seconds` 31.3; adapted `ap50` 0.3594 (+0.3422; per class stop 0.4059, yield 0.3801, speed limit 0.2921), `ap` 0.2993;
adapted threshold chosen on the training split 0.2232 (train F1 0.629, precision 0.661, recall 0.6); held-out at the COCO threshold 0.9: 0 returned of 17 references
(highest score 0.2985); at 0.2232: 14 returned, 4 correct (precision 0.2857, recall 0.2353); unseen images: highest scores 0.212 / 0.266 / 0.228, 0/5 reference boxes found
at 0.9 and 2/5 at 0.2232 (the yield sign on image 001, IoU 0.933, score 0.253; the stop sign on image 002, IoU 0.928, score 0.228); reload of `detr_adapter.safetensors`
equivalent (16 detections compared, tolerance 0.001).

The notebook's worked answers (Sections 8–10 and the Interpretation) quote the 2026-10-04 run of `c3c81cb` (blob `6b2e5fc3`): final loss 0.6103, `ap50` 0.3851, `ap` 0.2983, highest held-out adapted score 0.3078, `ADAPTED_THRESHOLD` 0.2231 (training F1 / precision / recall 0.5538), 15 held-out boxes returned at that threshold with 5 correct (recall 0.29, precision 0.33), unseen highest scores 0.279 / 0.273 / 0.225, IoU 0.874 at the evaluation threshold, and 1 of 5 unseen signs found at the adapted threshold. This run printed different values for each of these (above). The notebook is not edited by this record.

Evidence in `docs/execution-evidence/2026-10-25/`. The local run directory was redacted to `<local-run-dir>` before the first commit; the executed notebook was checked against
`run_summary.json`'s `executed_sha256` before any edit. SHA-256 captured → committed:

| File | Captured | Committed |
|---|---|---|
| `detr_detection_colab_8fd8d5e_colab-cli-t4_default.ipynb` | `34654821322450f6fdf41803ce8dd787bd75aa3736982b2e613d70a5c8f9e9f8` | unchanged (no local path) |
| `detr_detection_colab_8fd8d5e_colab-cli-t4_default_exec.log` | `ad886dc0e70bcce6f0b9b87ce652315eef4d23ec09729f766889851d3b3ae1b4` | `832c2b6ee2878f3476b898626e9d1bda9e8697daeaf90a0b6fab80c2c2e7e490` (2 replacements) |
| `detr_detection_colab_8fd8d5e_colab-cli-t4_default_run_summary.json` | `582e8312bc1a55deda827057060e4ba0c5d7cd802607fc31445e8888d1deb261` | unchanged (no local path) |

## 2026-10-25 Colab T4 run of the prose-kept-true head (Colab CLI, one pass)

Colab CLI sequential execution (`colab exec -f`, not a browser Run all), fresh Colab Tesla T4 (session `suite-detr-c439bd4-dd82`), PR #7 head
`c439bd4490f2a4c237dd11f035a061ea15a65fcc`, notebook blob `712d22d5ca82c31597707d81fe40b110ae5bf530` fetched at that commit and Git-blob verified;
started 2026-10-25T14:14:37Z, 171.5 s wall (setup 58 s), exit code 0, input notebook unchanged, 16/16 code cells in order in `exec.log`, one pass, no restart,
0 error outputs, verdict PASS. Default path only (every form field at its default, both BYOD branches off).

Printed: input manifest `accepted` with the `threshold=1.5` probe rejected; COCO checkpoint on the drawn scene 3 detections, 3/4 drawn objects matched at IoU ≥ 0.5
(stop sign 0.847, traffic light 0.910, clock 0.898, sports ball 0.000); blank and noise 0 boxes at 0.9 and at 0.05; split 30 / 10 images (65 / 17 boxes); re-headed baseline `ap50` 0.0172 / `ap` 0.0063; 10 epochs with the backbone frozen
(18,047,240 of 41,502,152 parameters trainable), loss 1.6564 → 0.6244, `train_seconds` 31.6; adapted `ap50` 0.3617 (+0.3445; per class stop 0.0, yield 0.6346, speed limit 0.4505), `ap` 0.3093;
adapted threshold chosen on the training split 0.2225 (train F1 0.5399, precision 0.449, recall 0.6769); held-out at the COCO threshold 0.9: 0 returned of 17 references
(highest score 0.3089); at 0.2225: 27 returned, 8 correct (precision 0.2963, recall 0.4706); unseen images: highest scores 0.226 / 0.27 / 0.212, 0/5 reference boxes found
at 0.9 and 2/5 at 0.2225 (image 001: yield sign IoU 0.906 at 0.27 and speed-limit sign IoU 0.882 at 0.258); adapter `detr_adapter.safetensors` 72,219,224 bytes, 265 tensors, saved; reload equivalent (30 detections compared, tolerance 0.001).

This is the confirming run for the "Prose kept true across runs" revision. Blob `712d22d5` is **Candidate**; the paths this blob's runs cover: the default path only. Remaining gates: REL12 (a BYOD run, one input accepted and one refused) and REL2 (a browser run of the interactive journeys). The runs of blobs `da6b14bd` (2026-10-25), `6b2e5fc3` (2026-10-04) and `a98705ed` (2026-09-25) are history (REL14).

Evidence files for the 2026-10-25 default run of blob `712d22d5`, copied from the run directory and redacted byte-wise (local run paths replaced by `<local-run-dir>`, the Colab CLI install path by `<colab-cli-tool-dir>`; no JSON re-serialised; the executed notebook's captured hash equals `run_summary.json` `executed_sha256`):

| File (under `docs/execution-evidence/2026-10-25/`) | Captured SHA-256 | Committed SHA-256 |
|---|---|---|
| `detr_detection_colab_c439bd4_colab-cli-t4_default.ipynb` | `1f769a35e8699a67359c8ba468910f59e0d477a93c233fe6435ba77ace43ef35` | `1f769a35e8699a67359c8ba468910f59e0d477a93c233fe6435ba77ace43ef35` (unchanged, no local path) |
| `detr_detection_colab_c439bd4_colab-cli-t4_default_exec.log` | `69aeb7f5c9100ce7446805d3f5f45ea5a4d5466b024ae6ecbe32ae7c25c6f2ba` | `29bde0c6d4affca8dba532ef1cea46ccf474da225854a71f571db96b4479d12e` (2 local-path replacements) |
| `detr_detection_colab_c439bd4_colab-cli-t4_default_run_summary.json` | `ed4e4650f0c1e9a624cd120dce5e3abb38ee3481ae94fab74fd1125c198ace48` | `ed4e4650f0c1e9a624cd120dce5e3abb38ee3481ae94fab74fd1125c198ace48` (unchanged, no local path) |

**Prose confirmation (2026-10-26).** This run is the confirming run for the "prose kept true across runs" revision. Every run-derived number in the notebook prose was checked against what this run printed: each number given as agreed across runs equals this run's printed value, and each stated range contains it. These still fail (the notebook is unchanged in this evidence commit):

- Section 8 worked answer (md24): "the loss started at 1.6564 and ended between 0.5602 and 0.6103 after 10 epochs" in every recorded hosted run. This run ended at 0.6244, outside the stated range (first epoch 1.6564 agrees; "down to roughly a quarter to two fifths" holds at 0.377).
- Prerequisites (md1): "Hosted runs of this revision are pending." This run is a hosted run of the current blob `712d22d5`, so the sentence is now false; the timing sentence before it quotes only the 2026-10-04 run (165.1 s, setup 60 s, fine-tune 33.7 s; this run 171.5 s, 58 s, 31.6 s).

- Values the prose gives per named run and that moved again in this run (all consistent with the notebook's ranges where it states one): `ap50` 0.3617 (Interpretation range 0.36–0.39 holds), `ap` 0.3093 ("about 0.30"), highest held-out adapted score 0.3089 ("below about 0.31" holds), adapted threshold 0.2225 (train F1 0.5399, precision 0.449, recall 0.6769), 27 held-out boxes returned at it with 8 correct, unseen highest scores 0.226 / 0.27 / 0.212, 2 of 5 unseen signs found at 0.2225, 0/5 at 0.9 with every `detections` list empty (holds).
- Agreed values all equal this run: scene detections `stop sign` 0.999 (IoU 0.847), `traffic light` 0.997 (0.910), `clock` 0.973 (0.898), sports ball 0.0; 0 boxes on blank and noise at 0.9 and 0.05; split 30 / 10 (65 / 17 boxes); baseline `ap` 0.0063, `ap50` 0.0172; first-epoch loss 1.6564; 18,047,240 of 41,502,152 trainable; 0 held-out boxes returned at 0.9.

## 2026-10-25 Colab T4 run of the historical-wording head (Colab CLI, one pass)

Colab CLI sequential execution (`colab exec -f`, not a browser Run all), fresh Colab Tesla T4, PR #7 head
`dc65d302e485fbf7b173b86d0592801cdc5300f6`, notebook blob `c19823537fa46d8da85749587756f39e8d366aa5` fetched at that commit and Git-blob verified;
started 2026-10-25T20:07:11Z, 272.9 s wall (setup 178 s), exit code 0, input notebook unchanged, 16/16 code cells in order in `exec.log`, one pass, no restart,
0 error outputs, verdict PASS. Default path only (every form field at its default, both BYOD branches off).

Printed: input manifest `accepted` with the `threshold=1.5` probe rejected; COCO checkpoint on the drawn scene 3/4 drawn objects matched at IoU ≥ 0.5
(stop sign 0.847, traffic light 0.910, clock 0.898, sports ball 0.000); blank and noise 0 boxes at 0.9 and at 0.05; re-headed baseline `ap50` 0.0172 / `ap` 0.0063; 10 epochs with the backbone frozen,
loss 1.6564 → 0.6345, `train_seconds` 31.7; adapted `ap50` 0.3944, `ap75` 0.3944, `ap` 0.346; adapted threshold chosen on the training split 0.2304 (train F1 0.7338);
held-out at the COCO threshold 0.9: 0 returned of 17 references (highest score 0.3028); at 0.2304: 18 returned, 10 correct (precision 0.5556, recall 0.5882); unseen images: 0/5 reference boxes found
at 0.9 and 2/5 at 0.2304; adapter `detr_adapter.safetensors` 72,219,224 bytes, 265 tensors, saved; reload equivalent.

This is the confirming run for the historical-wording revision. Blob `c1982353` is **Candidate**; this PR's runs cover the default path only. Remaining gates: REL12 (a BYOD run, one input accepted and one refused) and REL2 (a browser run of the interactive journeys). The runs of earlier blobs are history (REL14).

Evidence files for the 2026-10-25 default run of blob `c1982353`, copied from the run directory and redacted byte-wise (local run paths replaced by `<local-run-dir>`, the Colab CLI install path by `<colab-cli-tool-dir>`; no JSON re-serialised; the executed notebook's captured hash equals `run_summary.json` `executed_sha256`):

| File (under `docs/execution-evidence/2026-10-25/`) | Captured SHA-256 | Committed SHA-256 |
|---|---|---|
| `detr_detection_colab_dc65d30_colab-cli-t4_default.ipynb` | `c3079736a4a99797fe9572d40f937640d7684c0e8fcf44ae9ead92c3b323bdc2` | `c3079736a4a99797fe9572d40f937640d7684c0e8fcf44ae9ead92c3b323bdc2` (unchanged, no local path) |
| `detr_detection_colab_dc65d30_colab-cli-t4_default_exec.log` | `135d48d9124a6c309045123ba1bd16ddb41e5496b9ed96ea15022539029674ac` | `7a48b5941a48544444c528ff78360fe694850beed906ccb1f063d59fc0fe30e1` (2 local-path replacements) |
| `detr_detection_colab_dc65d30_colab-cli-t4_default_run_summary.json` | `c3cc08016bbe3df6993988f60f1924e532122aab574b0c08ba7d566c27534e35` | `c3cc08016bbe3df6993988f60f1924e532122aab574b0c08ba7d566c27534e35` (unchanged, no local path) |

**Drift recorded, not fixed (2026-10-26, Kurt's 2026-10-26 "historical wording, then stop" ruling).** The notebook prose of blob `c1982353` labels every run-derived number with the named recorded runs that printed it, so these differences are not prose errors. This confirming run printed these values outside the recorded values and spans the prose names. The notebook is not edited:

- Prerequisites (md1): this run took 272.9 s of wall time, setup 178 s (fine-tune 31.7 s); the prose records 136.6 s to 171.5 s of wall time (setup 60 s and 58 s) for the three named Colab T4 runs.
- Section 8 worked answer: the loss ended at 0.6345; the prose records 0.5602 to 0.6244 after 10 epochs in every named hosted run (first epoch 1.6564 agrees; the fall to about two fifths holds at 0.383).
- Section 9 worked answer and Interpretation: adapted `ap50` 0.3944 and `ap` 0.346, above the recorded 0.3594–0.3851 and 0.2983–0.3093 (Interpretation: AP50 0.36–0.39, AP@[.50:.95] 0.30–0.31).
- Section 9 worked answer: adapted threshold 0.2304 (training F1 0.7338), 18 held-out boxes returned at it, 10 correct; the prose records thresholds 0.2225–0.2232 and 15, 14 and 27 boxes returned for the named runs.

## 2026-10-26 Colab T4 run of the final-prose-pass head (Colab CLI, one pass)

Colab CLI sequential execution (`colab exec -f`, not a browser Run all), fresh Colab Tesla T4, PR #7 head `3b7e6d06cd4c02159eb5d71d84d883d3341a671c`, notebook blob `43868fa1a59e7992f7ce7f442a71d5e264c8cb03` fetched at that commit and Git-blob verified; started 2026-10-26T00:02:30Z, 135.3 s wall (setup 51 s), exit code 0, input notebook unchanged, 16/16 code cells in order in `exec.log`, one pass, no restart, 0 error outputs, verdict PASS. Default path only (every form field at its default, both BYOD branches off).

Printed: input manifest `accepted` with the `threshold=1.5` probe rejected; COCO checkpoint on the drawn scene 3/4 drawn objects matched at IoU ≥ 0.5 (stop sign 0.847, traffic light 0.910, clock 0.898, sports ball 0.000); blank and noise 0 boxes at 0.9 and at 0.05; re-headed baseline `ap50` 0.0172 / `ap` 0.0063; 10 epochs with the backbone frozen, loss 1.6564 → 0.6743, `train_seconds` 31.5; adapted `ap50` 0.4486, `ap75` 0.4391, `ap` 0.3561; adapted threshold chosen on the training split 0.1945 (train F1 0.7097); held-out at the COCO threshold 0.9: 0 returned of 17 references (highest score 0.2769); at 0.1945: 11 returned, 6 correct (precision 0.5455, recall 0.3529); unseen images: 0/5 reference boxes found at 0.9 and 3/5 at 0.1945; adapter `detr_adapter.safetensors` 72,219,224 bytes, 265 tensors, saved; reload equivalent (52 detections, tolerance 0.001).

This is the confirming run of the final prose pass. Blob `43868fa1` is **Candidate**; this PR's runs cover the default path only. Remaining gates: REL12 (a hosted BYOD run, one input accepted and one refused) and REL2 (a browser `Run all`). Runs of earlier blobs are history (REL14).

Evidence files for the 2026-10-26 default run of blob `43868fa1` (confirming run of the final prose pass), copied from the run directory and redacted byte-wise (local run paths replaced by `<local-run-dir>`, the Colab CLI install path by `<colab-cli-tool-dir>`; no JSON re-serialised; the executed notebook's captured hash equals `run_summary.json` `executed_sha256`):

| File (under `docs/execution-evidence/2026-10-26/`) | Captured SHA-256 | Committed SHA-256 |
|---|---|---|
| `detr_detection_colab_3b7e6d0_colab-cli-t4_default.ipynb` | `de8bebd4472fac05771001baeee4abac50e5b41b85d3a02751a06e1a9dbe7b45` | `de8bebd4472fac05771001baeee4abac50e5b41b85d3a02751a06e1a9dbe7b45` (unchanged, no local path) |
| `detr_detection_colab_3b7e6d0_colab-cli-t4_default_exec.log` | `e356b43d564a684034a2a786e118855f0d5a9288d704648950214ea4e30c46f4` | `6499d11cb619535b9f0c6eb264c3aad09fe50a7cd04a1cb8d40ae5ef4a112bf8` (2 local-path replacements) |
| `detr_detection_colab_3b7e6d0_colab-cli-t4_default_run_summary.json` | `a7b3813d4be1acee156eff63171d91e59e28d73a509ed131f2cd14cd39402991` | `a7b3813d4be1acee156eff63171d91e59e28d73a509ed131f2cd14cd39402991` (unchanged, no local path) |

**Final prose pass: drift record (2026-10-26).** Under Kurt's 2026-10-26 ruling the notebook prose of blob `43868fa1` is not changed again: a value this run printed that differs from the values the prose attributes to the named earlier runs is recorded here as drift, not fixed. Checked against this run's output and equal to the prose's named-run values: the input manifest (`accepted`, `threshold=1.5` rejected), the drawn scene 3/4 matched at IoU ≥ 0.5 (0.847, 0.910, 0.898, sports ball 0.000), blank and noise 0 boxes at 0.9 and at 0.05, the 40 images and 82 boxes, the 30 / 10 split with 65 / 17 boxes, the re-headed baseline `ap50` 0.0172 / `ap` 0.0063, the first-epoch loss 1.6564, 0 held-out boxes at 0.9, the adapter (72,219,224 bytes, 265 tensors) and the reload check (52 detections compared, equivalent at tolerance 0.001).

Drift (this run versus the prose's named-run values):

- Prerequisites (md1): this run took 135.3 s of wall time, setup 51 s, fine-tune 31.5 s; the prose records 136.6 s to 171.5 s of wall time (setup 60 s and 58 s, fine-tune 33.7 s and 31.6 s) for the three named Colab T4 runs.
- Section 8 worked answer (md24): the loss ended at 0.6743; the prose records 0.5602 to 0.6244 after 10 epochs for the named runs (0.6103, 0.6078 and 0.6244 in the three Colab runs). The first epoch, 1.6564, agrees; the fall to 0.41 of the first-epoch loss is just above the "roughly a quarter to two fifths" the prose gives for those runs.
- Section 9 worked answer and Interpretation (md26, md34): adapted `ap50` 0.4486 and `ap` 0.3561 (`ap75` 0.4391), above the recorded 0.3594–0.3851 and 0.2983–0.3093 (Interpretation: AP50 0.36–0.39, AP@[.50:.95] 0.30–0.31). The highest adapted held-out score was 0.2769; the prose records 0.3078, 0.2985 and 0.3089.
- Section 9 worked answer (md26): the training split chose threshold 0.1945 (training F1 0.7097, precision 0.7458, recall 0.6769); at it 11 held-out boxes came back, 6 correct (precision 0.5455, recall 0.3529). The prose records thresholds 0.2231, 0.2232 and 0.2225, with 15, 14 and 27 boxes returned and 5, 4 and 8 correct.
- Section 10 (md26, md28) and Interpretation (md34): on the three unseen images the highest scores were 0.197, 0.228 and 0.241 (the prose gives each named run's highest as 0.279, 0.266 and 0.270), and at the adapted threshold this run found 3 of the 5 reference signs; the prose records 1, 2 and 2 of 5.

Prose statements that still break the historical rules (reported, not fixed):

- md34 (Activity step 4, borderline): "in Section 8, the cell prints `rebuilt_from_verified_snapshot` and an epoch-1 loss well above 1" states what an activity run will print; its only number is labelled (2.66, a local CPU check), but no hosted run of the activity is recorded for this revision (rule 1).

### Hosted Colab T4 runs of blob `08904043` (2026-10-29)

These runs executed the PR head `e43607cb1df6fa764ba84f5367dfd0ad17f3f3b3`, notebook `tutorials/detr_detection_colab.ipynb` blob `0890404337387045d88cf7baecac2fffcad31797`, fetched at the 40-character SHA and Git-blob verified, with the Colab CLI (`colab exec -f`) on a fresh Colab Tesla T4 session (not a browser `Run all`). Each `run_summary.json` records `verdict` `PASS`, `exec_exit_code` 0, `input_unchanged` true and `stop_verified` true (the VM was released), and the SHA-256 of each executed notebook equals the summary's `executed_sha256`.

**Default path (2026-10-29 03:15:41 UTC start).** Default settings, no overrides; session `suite-detr-e43607c-dd82`; wall 266.6 s (`wall_s`); `code_cells` 16, `executed_cells` 14, 0 errors. Retained evidence: [`docs/execution-evidence/2026-10-29/`](execution-evidence/2026-10-29/).

Printed by the default run (verbatim lines from the executed outputs):

```text
{'verified_files': 4, 'revision': '1d5f47bd3bdd2c4bbfa585418ffe6da5028b4c0b'}
{'verdict': 'sample-sanity', 'n_detections': 3}
metric     baseline    adapted     change
ap           0.0063     0.1678    +0.1615
ap50         0.0172     0.2046    +0.1874
ap75         0.0063     0.1943    +0.1880
per-class AP50: {'stop-sign': 0.0, 'yield-sign': 0.3281, 'speed-limit-sign': 0.2857}
held-out at the COCO threshold:      {'threshold': 0.9, 'returned': 0, 'correct': 0, 'references': 17, 'precision': None, 'recall': 0.0, 'highest_score': 0.3329}
held-out at the adapted threshold:   {'threshold': 0.238, 'returned': 16, 'correct': 2, 'references': 17, 'precision': 0.125, 'recall': 0.1176, 'highest_score': 0.3329}
{'detections_compared': 19, 'threshold': 0.05, 'tolerance': 0.001, 'equivalent': True}
```

**DAT20 refusal record (2026-10-29, blob `08904043`).** Overrides, set by `--set` on a copy of the blob: `USE_BYOD_IMAGE` `False` → `True` and `BYOD_IMAGE_PATH` `""` → `'/etc/hostname'` (the run copy differs from the blob in exactly those 2 source lines). The suite cut the run copy right after the refusing cell `detr_detection-33`, dropping 1 cells (0 of them code cells), which reproduces `Run all` stopping at the first error; the cut copy is `run_copy.ipynb`, SHA-256 `7641614dd14e4fa997d7faea6c6ff067695fad570a707708262d05cc9455938b` (Git blob `d859b95b`), equal to the summary's `run_copy_sha256`. The cut copy's earlier code cells completed without an error output; the refusing cell's only error output has `ename` `IsolatedCellRefusal` and no traceback, and its output holds exactly one `Refused:` line:

```text
Refused: hostname: PIL cannot open this file as an image (UnidentifiedImageError: cannot identify image file '/etc/hostname'); use one PNG or JPEG image; fix the input and run this cell again (the earlier results of this session and their outputs/ files were removed).
```

This run says nothing about the 1 cut cells, which did not run in it. Session `suite-detr-e43607c-dd82`, wall 154.4 s. Retained evidence: [`docs/execution-evidence/2026-10-29/`](execution-evidence/2026-10-29/).

**Local-path redaction.** Local paths were replaced byte-wise by `<local-run-dir>` (and the Colab CLI install path by `<colab-cli-tool-dir>/`) before the first commit; each executed notebook and run copy was checked against its `run_summary.json` before any edit.

| File (under `docs/execution-evidence/2026-10-29/`) | Captured SHA-256 | Committed SHA-256 |
|---|---|---|
| `detr_detection_colab_e43607c_colab-cli-t4_default.ipynb` | `96c4492c987cce7471b18e1e3c72a17f8587b9969ffff6bcc60c6183b4340e25` | `96c4492c987cce7471b18e1e3c72a17f8587b9969ffff6bcc60c6183b4340e25` (unchanged, no local path) |
| `detr_detection_colab_e43607c_colab-cli-t4_default_exec.log` | `c6a525323dfdf336ef7f3c8346551d988798a95d002c0cebf9151b949b1db162` | `ef128e500ec0367ae12858c9ceb44c96811d25112a2b6b9d4437f6313545fa15` (2 local-path replacements) |
| `detr_detection_colab_e43607c_colab-cli-t4_default_run_summary.json` | `03590315904ab006cfa5696fa08159a1a2f7ce434537cfd41b1e3a96c62aee37` | `03590315904ab006cfa5696fa08159a1a2f7ce434537cfd41b1e3a96c62aee37` (unchanged, no local path) |

| File (under `docs/execution-evidence/2026-10-29/`) | Captured SHA-256 | Committed SHA-256 |
|---|---|---|
| `detr_detection_colab_e43607c_colab-cli-t4_refusal.ipynb` | `fc047b4ed3035e21f2e063c9a8689bab4b5756c1efca1a273eb179c49e871f9a` | `fc047b4ed3035e21f2e063c9a8689bab4b5756c1efca1a273eb179c49e871f9a` (unchanged, no local path) |
| `detr_detection_colab_e43607c_colab-cli-t4_refusal_exec.log` | `eeeb221bdbd25dc6c27072b4d0f95dce981a14156843dbd378ddd033c978720d` | `ba3bb293c9b9ad9bb4ef819112b3070659e0f290b55bcdfabd7daea3f4858268` (2 local-path replacements) |
| `detr_detection_colab_e43607c_colab-cli-t4_refusal_run_summary.json` | `732a660ba5f609f1c60ecdedfd2cd2d15a7df3214db82b713c87ce37e0bc320a` | `732a660ba5f609f1c60ecdedfd2cd2d15a7df3214db82b713c87ce37e0bc320a` (unchanged, no local path) |
| `detr_detection_colab_e43607c_colab-cli-t4_refusal_run_copy.ipynb` | `7641614dd14e4fa997d7faea6c6ff067695fad570a707708262d05cc9455938b` | `7641614dd14e4fa997d7faea6c6ff067695fad570a707708262d05cc9455938b` (unchanged, no local path) |

Not exercised by these runs: an accepted BYOD input (REL12) and a browser `Run all` of the interactive journeys (REL2).
