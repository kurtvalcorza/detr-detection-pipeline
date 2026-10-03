# DETR Detection E2E Notebook — Review

**Verdict: Needs revision**  
**Review date:** 3 October 2026 (relay batch of 2 October 2026)  
**Repository:** `kurtvalcorza/detr-detection-pipeline`  
**Notebook:** `tutorials/detr_detection_colab.ipynb`  
**Reviewed commit:** `f76a35975b1493518df0124b939c48ed34967ed6` (`main`, confirmed with `gh api repos/kurtvalcorza/detr-detection-pipeline/commits/main`)  
**Notebook Git blob:** `a98705edf1cb7c8b537db384e781a4fd920452be`. This is the blob executed in the recorded Kaggle Tesla T4 run of 2026-09-25 (commit `e39d680`, `fetched_blob_verified: true`); the notebook's last change is `7e86e1f`. At the reviewed commit `tools/build_notebook.py --check` and `tools/validate_release_assets.py` both exit 0.  
**Finding prefix:** `DTR`  
**Framework:** Notebook Review Framework v1. **Requirements baseline:** NOTEBOOK_SPEC 2.2 (2026-09-26), `ml-worker` `origin/main`.

## Executive assessment

The default path is well engineered and it reproduces. The notebook carries the package's two modules (973 + 241 lines, one documented rewrite), asserts the inline manifest against the module identity, stages and re-hashes the pinned `facebook/detr-resnet-50` snapshot, runs the COCO head on a drawn scene with an input manifest and a threshold rejection probe, probes blank and noise images, validates a 40-image drawn sign dataset, splits it with a leakage assertion, re-heads the class layer, measures the baseline, fine-tunes with DETR's set-prediction loss and the backbone frozen, scores held-out AP, exports a SafeTensors adapter bound to the base digest, and reloads it onto a fresh base with a stated tolerance. Score semantics (softmax with a no-object class, not calibrated) and the synthetic-data limits are stated correctly.

A direct CPU run of every code cell at the documented defaults (install skipped through the notebook's own hook) agreed with the Kaggle record on everything except the fine-tune numbers, which moved within the same regime:

| Measure | This review (CPU, repo `.venv` at the pins, defaults, `EPOCHS = 10`) | Kaggle T4 record (blob `a98705ed`) |
|---|---|---|
| Code cells completed | 14/14 (212 s of cell time; fine-tune 185 s) | 14/14 on pass 2 (pass 1 stopped at the install guard) |
| COCO scene at 0.9 | stop sign 0.999 (IoU 0.847), traffic light 0.997 (0.910), clock 0.973 (0.898); sports ball missed; 3/4 | identical |
| Blank / noise at 0.9 and 0.05 | 0 / 0 / 0 / 0 | identical |
| Split | 30 / 10 images, 65 / 17 boxes | identical |
| Baseline `ap` / `ap50` | 0.0063 / 0.0172 | identical |
| Epoch losses (1 → 10) | 2.1077 → 0.4860 | 1.6564 → 0.5602 |
| Adapted held-out `ap` / `ap50` (at score threshold 0.05) | 0.2884 / 0.3205 | 0.2222 / 0.2668 |
| New-data detections at `threshold = 0.9` | **0 on all 3 images** | **0 on all 3 images** |
| Reload check | equivalent, 2 detections compared | equivalent, 7 compared |

Four problems stand in the way of `Ready for intended use`:

1. **No one-pass `Run all` (DTR-M1).** The recorded run stopped at the install cell's stale-module guard (`cuda-bindings` 12.9.4 → 13.4.3, `numpy` 2.0.2 → 2.5.3) and passed only after a restart; the notebook and the release procedure present this as designed.
2. **The adapted model finds nothing at the notebook's own threshold, and the notebook says it does (DTR-M2).** Every adapted score is ≤ 0.43 (held-out per-image maximum 0.30–0.42; new-data maximum 0.39). At `threshold = 0.9` held-out AP50 is 0.0 and the three unseen images return no boxes, yet Section 10 says "The adapted pipeline detects the new sign classes". `MODEL_CARD.md` and `STATUS.md` already record the zero; the notebook does not tell the learner.
3. **The suggested experiment silently continues training (DTR-M3).** "Change `FREEZE_BACKBONE` to `False` and compare held-out AP": re-running the fine-tune cell continues from the adapted weights (first-epoch loss 0.720 instead of 2.108) and the comparison is printed against the untrained baseline.
4. **Guided layer largely absent (DTR-M4).** Declared `GUIDED`, but there is no audience statement, how-to-use, roadmap, task contract, glossary, prediction prompt, checkpoint, troubleshooting or conclusion template, and the 1,214 carried lines are not labelled as infrastructure.

The REL12 BYOD release gate has not been run on a hosted runtime; this review exercised the BYOD branches locally (one positive and eight negative datasets, four images), see §4.

## 1. Review contract and evidence

| Item | Value |
|---|---|
| Declared profile / mode | `E2E` / `GUIDED` (metadata `dimer.notebook_profile` / `notebook_mode`, opening cell) |
| Declared spec | DIMER Notebook Specification **2.1** (metadata, opening cell, `NOTEBOOK_SOURCE`) |
| Spec baseline applied | NOTEBOOK_SPEC **2.2** |
| Intended audience | Not stated. Prerequisites: "basic Python and PIL; bounding boxes as xyxy pixel coordinates; intersection-over-union (IoU); and how to read average precision" |
| Supported runtime | "Google Colab or Jupyter, Python 3.12"; CUDA T4 documented, CPU "also runs"; float32 |
| Promised outcomes | Pinned install; carried modules; digest-verified snapshot; COCO detection on a drawn scene with `box_iou`; blank/noise probes; validated 40-image sign dataset; split and baseline; bounded fine-tune; held-out COCO-style AP; "the adapted pipeline detects the new sign classes" on unseen images; adapter export and fresh reload with equivalence; machine-readable outputs; BYOD image and dataset branches through "the same validate → split → baseline → fine-tune → evaluate → export → reload stages"; "Try next" experiments |
| Generator | `tools/build_notebook.py` (`build_notebook.py/2`) + `tools/notebook_template.py`; recorded generating revision `7ee32a4` |
| Release status | `Candidate` (`tutorials/README.md`, `STATUS.md`, `docs/release-verification.md`): default path recorded, REL12 pending |

### Evidence actually obtained

- **Source inspection.** All 31 cells (14 code; cells 5 and 7 are the carried `pipeline.py` and `samples.py`, one differing line — the documented `DEFAULT_WEIGHTS_DIR` rewrite). Also read: the generator and template, `pipeline.py` (`_run`, `detect`, `finetune`, `evaluate`, `read_detection_records`, `load_artifact`), `README.md`, `STATUS.md`, `MODEL_CARD.md`, `tutorials/README.md`, `docs/release-verification.md`. The repository has no `AGENTS.md`; `docs/execution-evidence/` holds only the rice capstone's runs.
- **Documented execution evidence.** `docs/release-verification.md` plus the archived executor output for Kaggle kernel `kurtvalcorza/dimer-nb2-detr-detection` v1 (`run_summary.json`, `executed-pass1.ipynb`, `executed.ipynb`, `outputs/`). Kaggle Tesla T4, 2026-09-25, **the reviewed blob**, clean HF cache, image torch 2.10.0+cu128 / numpy 2.0.2 / transformers 5.0.0. Pass 1 failed in cell 3 with the restart `RuntimeError` (267.4 s); pass 2 ran 14/14 (90.0 s); 357.5 s total. No Colab run, no BYOD run and no optional-experiment run is recorded.
- **Direct execution (this review).**
  - **Environment:** `run_probes.py`, Windows 11, CPU only (`CUDA_VISIBLE_DEVICES=-1`, `HF_HUB_OFFLINE=1`), 24 threads, the repository's own `.venv` (Python 3.12.12, torch 2.14.0+cpu, torchvision 0.29.0+cpu, transformers 4.57.6, timm 1.0.29, scipy 1.18.1, numpy 2.5.3, pillow 11.3.0, safetensors 0.8.0, huggingface-hub 0.36.2 — the notebook's pins, CPU wheel). Nothing was installed.
  - **Install skipped:** cell 3 ran with `DIMER_NOTEBOOK_CI_PREINSTALLED=1`, the notebook's executor hook.
  - **Not a clean runtime:** the four snapshot files were hard-linked into a scratch working directory; cell 9 wrote its manifest fresh and `verify_snapshot` re-hashed every file (nothing fetched).
  - **Executed:** every code cell at defaults (P1); adapted score distribution at several thresholds (P2); the "Try next" rerun with `FREEZE_BACKBONE = False` and `EPOCHS = 2` (lowered for time, labelled) (P3); BYOD dataset branch on an 11-record directory including one no-box image, `EPOCHS = 2` (P4); eight incompatible BYOD datasets (P5); BYOD image branch with one valid and three invalid inputs (P6). Field values were set in an executed copy of cell 29's source.
  - **CPU vs T4:** the class-head initialisation and fine-tune trajectory differ by device (epoch-1 loss 2.108 vs 1.656), so the adapted AP differs (0.32 vs 0.27 AP50). Everything before the fine-tune is identical. The drawn scene's PNG digest differs from Kaggle's (Windows font rasterisation); the detections and IoUs are identical.
- **Learner observation:** none. No claim here is about measured learning effectiveness.

## 2. Separate judgments

- **Technical correctness:** good on the default path (P1 14/14; reload equivalence holds; traversal and malformed-annotation inputs are refused with named rules; `MAX_DETECTIONS = 100` agrees with DETR's 100 queries). Two defects: the install pattern forces a restart (DTR-M1), and in-place `finetune` makes the documented experiment stale (DTR-M3).
- **Promise fulfilment:** the promised new-data inference is not delivered at the notebook's operating threshold and the prose says it is (DTR-M2); the "compare held-out AP" experiment yields a contaminated comparison (DTR-M3); the BYOD dataset branch reaches export but writes no results and does not verify its reload (DTR-m2).
- **Learner experience:** accurate, careful stage prose (vocabulary separation, softmax/no-object score semantics, loss as optimisation evidence, AP semantics), but the guided layer is mostly missing (DTR-M4) and nothing helps the learner read a held-out AP50 of 0.27–0.32 next to zero detections.
- **Spec conformance:** unresolved applicable MUSTs — RUN1, RUN10, ENV6, REL2 (DTR-M1); SRC3 knowingly stale instruction and INF2 (DTR-M2; the model card records the opposite); UX12 (DTR-m1); REL12 BYOD evidence absent from the release record. SHOULD deviations: GDL1–GDL4, GDL6, GDL7, GDL9–GDL14, UX8 (DTR-M4).

## 3. Promise and objective tracing

| Claim / objective | Implementation | Observable result | Learner interpretation | Status |
|---|---|---|---|---|
| One-pass `Run all` | cell 3 in-kernel `pip install` + stale-module guard | Kaggle pass 1 `RuntimeError`, restart, pass 2 14/14 | Section 1 prose says the cell "stops with a restart instruction" | **Not met** (DTR-M1) |
| Digest-verified pinned snapshot | cell 9 | 4/4 files verified, revision `1d5f47b…` | clear | Met |
| COCO detection on a drawn scene, per-object `box_iou` | cell 11 | 3/4 matched at IoU ≥ 0.5; sports ball recorded as 0.0 | prose prepares for misses on renderings | Met |
| Blank and noise probes | cell 13 | 0 detections at 0.9 and 0.05 on both | "What to look for" note | Met |
| Validated sign dataset; split without leakage | cells 15, 17 | 40 records / 82 boxes accepted; 30/10, overlap assertion | clear vocabulary separation | Met |
| Baseline near zero | cell 17 | ap 0.0063, ap50 0.0172 | explained as the floor | Met |
| Bounded fine-tune with stated config | cell 19 | 10 epochs, 18,047,240 of 41,502,152 trainable, AdamW 1e-4, batch 4 | loss labelled as optimisation evidence | Met |
| Held-out COCO-style AP | cell 21 | ap50 0.32 (CPU) / 0.27 (T4) at score threshold 0.05 | metric meaning explained; the 0.05 evaluation threshold is not mentioned in Section 9 | Met, but see DTR-M2 |
| "The adapted pipeline detects the new sign classes" on unseen images | cell 23 at `threshold = 0.9` | **0 detections on 3/3 images** (both runs); max score 0.39 | none; Interpretation is silent | **Not met** (DTR-M2) |
| Export + fresh reload equivalence | cell 25 | equivalent within 1e-3 (compared at 0.05) | "loading is not the check" stated | Met |
| Machine-readable outputs with provenance | cell 27 | 6 files; CSV holds only the 3 COCO-scene rows | listed | Met |
| BYOD image branch | cell 29 | valid image → `accepted`, 3 detections, `not-measurable`; 8 px, text-as-PNG, missing path refused | contract stated first | Met (local) |
| BYOD dataset branch, same stages | cell 29 | validate → split → baseline → fine-tune → evaluate → export → load ran; nothing written but the adapter; reload not compared | | Partly met (DTR-m2) |
| Try next: unfreeze the backbone and compare held-out AP | cell 19 field | rerun continues from adapted weights; AP50 0.476 vs baseline 0.017 | none | **Not met** (DTR-M3) |

| Learning objective (opening cell) | Learner activity | Evidence exercised |
|---|---|---|
| Install the pinned runtime; read what the carried package guarantees; stage and verify the revision | run cells | printed versions and verified-file count; no activity |
| Run COCO detection and score `box_iou`; probe blank and noise | run cells, read tables | outputs readable; no prediction asked |
| Build and validate a dataset; split; baseline; fine-tune with the set-prediction loss; score AP | run cells | outputs readable; no interpretation prompt |
| Run inference on unseen images | run cell | output is empty at 0.9; the objective is not exercised in a meaningful way (DTR-M2) |
| Export, reload and verify the adapter | run cell | equivalence printed |

All objectives are phrased as actions the code performs, not as observable learner outcomes (GDL5), and none is followed by a check of the learner's understanding.

## 4. Journeys

| Journey | Basis | Result |
|---|---|---|
| **First-time learner** | Source inspection, all 31 cells | Stage prose is accurate (vocabulary separation, softmax + no-object scores, Hungarian matching and loss terms, AP semantics). Missing: audience, how-to-use, roadmap, glossary, predictions, checkpoints, troubleshooting, conclusion template; carried cells unlabeled (DTR-M4). At Section 10 the learner meets three empty detection lists after a held-out AP50 of ~0.3 with no explanation (DTR-M2). "Runtimes are not measured in this revision" is stale (DTR-m1). |
| **Clean default** | Documented (Kaggle T4, reviewed blob) + direct (CPU, install skipped) | Kaggle: pass 1 failed at the install guard, pass 2 14/14 after a restart (DTR-M1). Direct: 14/14 at defaults, 212 s of cell time, numbers in the table above; six outputs written. No Colab run. |
| **Active learning** | Direct (P3) | Followed "Try next" by setting `FREEZE_BACKBONE = False` in cell 19 and re-running it (`EPOCHS = 2` for time): epoch losses 0.7202, 0.5645 (fresh run starts at 2.1077), `run['freeze_backbone']` = False although the first 10 epochs were frozen; cell 21 then printed ap50 0.4758 against baseline 0.0172 (DTR-M3). Changing `EPOCHS` and re-running from cell 15 skips cell 17 the same way (inferred from source; same code path). |
| **Reuse and recovery** | Direct (P4–P6); Colab upload dialog not verified | Dataset branch with 10 drawn records plus 1 no-box image reached validate → split → baseline → fine-tune → evaluate → export → load (AP50 0.1895 → 0.3333 at `EPOCHS = 2`, labelled; not a quality claim); only `byod_detr_adapter.safetensors` was written and `result.json` has no BYOD block (DTR-m2). Eight incompatible datasets refused before any model ran, each naming the rule: no `annotations.json`, entry without `labels`, label outside `BYOD_CLASS_NAMES`, missing image file (raw `FileNotFoundError` with the path), xywh boxes out of bounds, `../` traversal, one record ("at least 2 records are required to split"), COCO-format dict. Image branch: valid image accepted; 8 px (`MIN_IMAGE_SIDE`), text file named `.png` (`UnidentifiedImageError`), missing path refused. |

## 5. Findings

### Major

#### DTR-M1 — `Run all` needs a manual restart after the install cell, and the release record counts the restarted run

- **Cell/section:** cell 3, Section 1 (generator `tools/build_notebook.py`, install block lines ~50–70; Section 1 prose around line 406); `docs/release-verification.md` record and procedure.
- **Observed issue:** the cell `pip install`s ten pins into the running kernel, then raises `RuntimeError: Core dependencies changed while older modules were loaded … Restart the runtime, then rerun from the top.` when a loaded distribution changed. Section 1 prose presents this as the expected behaviour.
- **Consequence:** a learner selecting **Run all** on a stock Kaggle/Colab image hits an error in the first code cell and must restart and run again. RUN1, RUN10 and ENV6 forbid this.
- **Evidence:** documented — Kaggle T4 run of blob `a98705ed`, pass 1 `ok: false` with `cuda-bindings: loaded=12.9.4, installed=13.4.3; numpy: loaded=2.0.2, installed=2.5.3`, `restarted_after_install_cell: true`, pass 2 14/14; the record reports "PASSED (default path)". Source — `pip_install_in_kernel: true`, `uses_uv: false` (probe static block).
- **Recommended correction:** adopt the fleet's **uv isolated-environment pattern**, which is how the capstone and newer workshop notebooks already run in one pass: the setup cell bootstraps uv, creates an isolated managed interpreter (`uv venv --managed-python --python 3.12.12 <ROOT>/env`), installs a hash-locked `requirements.txt` (`uv pip install --require-hashes --only-binary :all:`), and runs the pinned stages in that environment, so the kernel's preloaded NumPy/torch are never replaced and no restart can be required. Reference implementation on `main`: `bioclip2-biodiversity-pipeline/tutorials/DIMER_Philippine_Biodiversity_Field_Survey_Capstone.ipynb` (also `ast-audio-classification-pipeline/tutorials/DIMER_Sound_Event_Classification_Workshop.ipynb`); this repository's own rice capstone already ships a hash lock (`tools/rice-requirements.lock`). Do not add another in-kernel install guard or loosen pins to dodge the restart. Implement it in `tools/build_notebook.py`, regenerate, re-qualify with a one-pass hosted Run all, and correct the release record so a restart-dependent run is not reported as a `Run all` PASS.
- **Acceptance check:** a fresh Kaggle or Colab GPU runtime completes every code cell in a single **Run all** with no restart and no error, recorded in `docs/release-verification.md` with the notebook blob id and `restarted: false`; `grep -n "Restart the runtime" tutorials/detr_detection_colab.ipynb` returns nothing.
- **Spec:** RUN1, RUN10, ENV6, REL2.

#### DTR-M2 — The adapted model returns no detections at the notebook's operating threshold, and the notebook says it detects the new classes

- **Cell/section:** Section 10 (cell 22 prose, cell 23), Section 9 (cells 20–21), Section 11 reload check (cell 25), Interpretation (cell 30). Generator: `tools/notebook_template.py` line ~279 (Section 10 prose), line ~261 (Section 9) and the interpretation block from line ~463.
- **Observed issue:** cell 23 reuses the COCO head's `threshold = 0.9` for the re-headed model. After the bounded 10-epoch fine-tune every adapted score stays far below it. Held-out AP is computed at `EVAL_DETECTION_THRESHOLD = 0.05` (not mentioned in Section 9), and the reload check also compares at 0.05, so every printed number looks healthy while the operating-threshold inference is empty. Section 10 states "The adapted pipeline detects the new sign classes"; the interpretation section is silent about the empty result. `MODEL_CARD.md` (lines 73 and 128) and `STATUS.md` already record "found nothing on 3 new drawn images at the default threshold", so the notebook prose is a knowingly stale instruction.
- **Consequence:** the central "use the adapted model on new data" demonstration produces three empty lists, the exported CSV contains no adapted-model rows, and the learner is told it worked. A learner is likely to conclude that the AP is wrong or the adapter broken, and gets no guidance on the difference between a ranking metric (AP over all score levels ≥ 0.05) and a decision at a fixed threshold.
- **Evidence:** documented — Kaggle record: 0 detections on all 3 new images at 0.9, same-label IoU 0.0 for every truth box. Direct (CPU, P2) — max adapted score per new image 0.3897 / 0.2913 / 0.2887; per held-out image 0.3009–0.4210; detections at ≥ 0.5: 0 on every image; held-out `ap50` at threshold 0.9 = 0.0 and at 0.3 = 0.2633, versus 0.3205 at 0.05; `detections.csv` holds the header and 3 COCO-scene rows only.
- **Recommended correction:** in the template, (a) state in Section 9 that AP is computed over all detections scoring ≥ 0.05 and therefore does not imply detections at a deployment threshold; (b) give the adapted model its own operating threshold chosen on the **training** split (or a validation slice of it, never the held-out split), record it as a separate field (e.g. `ADAPTED_THRESHOLD`) and in `result.json`, and use it in Sections 10–12; (c) print, next to AP, per-class precision/recall or detection counts at that operating threshold on the held-out split; (d) replace the Section 10 sentence with prose describing the observed outcome and how to read it; (e) add the score-scale point to the interpretation section. Longer training is optional and does not replace (a)–(d).
- **Acceptance check:** on a default hosted run, Section 10 reports at least one same-label detection with IoU ≥ 0.5 on the unseen images at the threshold the notebook uses there, that threshold and its selection split are printed and exported in `result.json`, and the notebook text contains no claim about new-data detections that the recorded output contradicts.
- **Spec:** SRC3, INF2, EVAL15, RUN8, UNC3, UX4.

#### DTR-M3 — The suggested "unfreeze the backbone and compare" experiment silently continues training the adapted model

- **Cell/section:** cell 19 (`FREEZE_BACKBONE`, `LEARNING_RATE`, `BATCH_SIZE` fields), cell 17 (adapter creation), Interpretation "Try next". Generator: `tools/notebook_template.py` around line 244 (cell 19 fields) and line ~477 ("Try next").
- **Observed issue:** the adapter is created in cell 17; `finetune` mutates it in place (`pipeline.py` docstring: "Mutates this pipeline in place"). The natural rerun after changing a field in cell 19 is cell 19 alone, which continues from the already adapted weights. No rerun instruction is given. Cell 21 then compares the result with the untouched baseline, and `run` reports `freeze_backbone: False` although 10 epochs ran frozen; an adapter exported afterwards would carry that inaccurate configuration.
- **Consequence:** the notebook's only learner experiment produces an invalid comparison that looks like a clear improvement (AP50 0.32 → 0.48), teaching the wrong conclusion about freezing.
- **Evidence:** direct (P3) — rerun of cell 19 with `FREEZE_BACKBONE = False` (`EPOCHS = 2`): epoch losses 0.7202, 0.5645 versus 2.1077 for a fresh first epoch; cell 21 printed `ap50 0.0172 → 0.4758`.
- **Recommended correction:** create the adapter inside the fine-tune cell (or reset it there from the verified snapshot with `from_pretrained(..., class_names=SIGN_CLASSES, seed=SEED)` and re-measure the baseline), so every run of that cell starts from the same re-headed base; keep the default-run metrics in a named variable so the experiment can print default vs changed side by side; add an explicit "change one thing → re-run cells N–M" instruction to the Try-next text.
- **Acceptance check:** after a default Run all, setting `FREEZE_BACKBONE = False` and re-running only the documented cells yields a first-epoch loss within 10 % of a fresh run's first-epoch loss, and the printed comparison shows default and changed results side by side.
- **Spec:** GDL10, UX5, UX7, SRC2, FT6.

#### DTR-M4 — Declared `GUIDED`, but the guided layer is largely absent

- **Cell/section:** opening cells 0–1, every section boundary, cells 5 and 7, end of notebook. Generator: `tools/notebook_template.py`.
- **Observed issue:** no intended-learner statement, no **How to use this notebook**, no roadmap, no Input → Model → Output contract, no glossary (DETR, object query, Hungarian matching, no-object class, generalised IoU, AP@[.50:.95], re-heading), no prediction before the baseline/fine-tune/new-data results, no interpretation checkpoint with sample answer, no troubleshooting section (GPU absent, download failure, out-of-memory, digest mismatch, BYOD errors), no evidence-based conclusion template; only two "look for" notes; the 1,214 carried lines in cells 5 and 7 are not labelled **Infrastructure** or collapsed (`cellView` absent on every code cell).
- **Consequence:** a self-paced learner new to detection fine-tuning gets an accurate script but little help deciding what matters, what to expect, or how to interpret the AP-versus-threshold result that DTR-M2 exposes.
- **Evidence:** source inspection; probe `guided_markers` (How to use / Roadmap / Glossary / Check your reasoning / Troubleshooting / conclusion / audience / Infrastructure all absent), `cellView_form_cells: []`.
- **Recommended correction:** add the GDL layer in the template following NOTEBOOK_SPEC's guided-mode requirements: audience and how-to-use, roadmap, task contract, glossary, a prediction before Sections 7, 9 and 10, "What to notice" after each principal stage, collapsible checkpoint answers, a Predict → Change one thing → Run → Observe → Explain activity built on DTR-M3's fix, troubleshooting, and a conclusion scaffold; title the carried and install cells `# @title Infrastructure: …` with `cellView: form`.
- **Acceptance check:** each of GDL1–GDL4, GDL6, GDL7, GDL9–GDL14 maps to a named cell in a checklist added to `tutorials/README.md`, and cells 3, 5, 7 and 9 carry `cellView: form` with an Infrastructure title.
- **Spec:** GDL1–GDL4, GDL6, GDL7, GDL9–GDL14, UX8.

### Minor

#### DTR-m1 — "Runtimes are not measured in this revision" is stale

- **Cell/section:** cell 1 Prerequisites (`tools/notebook_template.py` line 83).
- **Observed issue:** the release record measured this blob: 357.5 s total on Kaggle T4 (pass 1 267.4 s including the install, pass 2 90.0 s). This review measured 212 s of cell time on a 24-thread CPU with the install skipped (fine-tune 185 s).
- **Consequence:** the learner cannot plan for the download, install and fine-tune time, and the CPU fallback's cost is not stated.
- **Evidence:** documented record; direct P1 timings.
- **Recommended correction:** state measured times with their environment (T4 and CPU), labelled as measurements of a named run.
- **Acceptance check:** the Prerequisites cell quotes a wall time with runtime, date and revision that matches a row in `docs/release-verification.md`.
- **Spec:** UX12, SRC3.

#### DTR-m2 — The BYOD dataset branch writes no results, runs no new-data inference and only loads (does not verify) its adapter

- **Cell/section:** cell 29 dataset branch (`tools/notebook_template.py`, BYOD block from line ~425).
- **Observed issue:** after evaluate, the branch writes `outputs/byod_detr_adapter.safetensors` and calls `load_artifact` without comparing detections; baseline/adapted metrics, the dataset manifest, the split and the fine-tune configuration are printed but not exported, and no inference on held-back or new user images is shown. The sample path does all of these.
- **Consequence:** a user who adapts on their own data leaves with an adapter but no machine-readable metrics or provenance, and "exported and reloaded" overstates what was checked.
- **Evidence:** direct (P4): new outputs `['byod_detr_adapter.safetensors']` only; `result.json` has no BYOD block; stdout "BYOD adapter exported and reloaded: 93e330be9d64ced2".
- **Recommended correction:** reuse the sample path's reload comparison and output writer for BYOD (`byod_result.json` with dataset digest, split, config, baseline/adapted metrics at the evaluation and operating thresholds, and a held-out detections CSV).
- **Acceptance check:** a BYOD dataset run writes a BYOD result JSON and CSV under `outputs/`, and prints a reload equivalence check with a tolerance.
- **Spec:** DAT13, DAT14, OUT1, OUT8, OUT9, VER4, VER5.

#### DTR-m3 — BYOD upload path friction and one unguided error

- **Cell/section:** cell 29 `_upload_into`, image branch; `read_detection_records`.
- **Observed issue:** uploads are flattened to basenames, so an `annotations.json` that names `images/0001.png` cannot be satisfied through the dialog and the user is not told to keep files flat; zips are not accepted; the image branch takes the alphabetically first file in `outputs/byod/image`, so a second upload can silently re-run an earlier image. An annotation naming a missing image surfaces as a raw `FileNotFoundError` with no corrective hint.
- **Consequence:** first-time BYOD users on Colab hit a `FileNotFoundError` or get results for the wrong image.
- **Evidence:** source inspection; direct P5 `missing_image_file` → `FileNotFoundError: [Errno 2] No such file or directory: …`; the Colab upload dialog was not executed (not verified).
- **Recommended correction:** state "upload `annotations.json` and the images as flat files (no folders)" in Section 13, accept a zip, clear the image upload directory before each upload (or use the uploaded name), and wrap the missing-file case in a message naming the annotation entry and the expected location.
- **Acceptance check:** uploading two different images in succession with `USE_BYOD_IMAGE = True` processes the second one; Section 13 states the flat-file rule; a missing image file yields a message naming the `annotations.json` entry.
- **Spec:** DAT12, DAT19, UX10.

### Suggestions

- **DTR-S1** — Declare `notebook_spec` 2.2 instead of 2.1 once the guided layer lands.
- **DTR-S2** — Add a small score-threshold sweep table (held-out precision/recall at 0.05 / 0.1 / 0.3 / 0.5 / 0.9) so the learner sees why AP and fixed-threshold detection can disagree.
- **DTR-S3** — Draw the adapted detections on the three unseen images (and the held-out images) next to their references, as cell 27 does for the COCO scene.
- **DTR-S4** — Record per-stage wall times and the evaluation threshold in `result.json`.

## 6. Readiness

**Needs revision.** Open Majors DTR-M1 to DTR-M4. Remaining gates after the fixes: a one-pass hosted Run all of the regenerated blob (RUN1/RUN10/REL2), the REL12 BYOD exercise recorded in `docs/release-verification.md` (one compatible directory, one incompatible `annotations.json`), and a release record whose Section 10 output matches its prose.

## 7. Verified versus inferred

- **Verified by direct execution (CPU, install skipped, labelled above):** default path 14/14; adapted scores ≤ 0.43 and zero detections at 0.9 (and 0.5) on held-out and new images; the in-place rerun contamination; the BYOD positive run, all eight negative refusals and the image-branch refusals.
- **Verified from documented evidence:** the restart on Kaggle pass 1, the pass-2 numbers, and the zero new-data detections on T4.
- **Inferred from source:** the `EPOCHS`-from-cell-15 rerun contamination (same code path as P3); the Colab upload behaviour (DTR-m3); that T4 adapted scores sit in the same sub-0.5 range as CPU (the T4 record shows 0 detections at 0.9 but did not record score maxima).
- **Only Kurt can confirm:** whether the intended product behaviour is a separate adapted operating threshold or longer training (DTR-M2's correction offers the former as the minimum).
- **Most likely to be wrong:** DTR-M3's severity — a maintainer could argue the notebook never told learners to re-run only cell 19, making it a Minor missing rerun instruction; it is rated Major because the field lives in that cell and the resulting comparison is wrong without any warning.
