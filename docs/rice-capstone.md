# Rice-pest exploratory capstone

**Candidate: fresh Colab T4 default Run all remains pending.** The implementation is separate
from the original DETR tutorial. Original package code and original notebook are unchanged.

## Agreed scope and source audit

On 2026-09-27, the maintainer explicitly chose an exploratory benchmark using source-image
and duplicate grouping after the archive failed the specification's independent-capture gate.
The user supplied a CC BY 4.0 screenshot; the Zenodo API independently confirmed `cc-by-4.0`.

Source: Deleña, Reymark; Balleras, Gina; Dequiña, Brent Jason; Jumawan, Rexter; Flores, Christine.
*Annotated light-trap rice pest image dataset with detector-derived region-of-interest crops
for species–sex recognition of rice black bug and white stemborer*, version 1.
[Zenodo DOI 10.5281/zenodo.20066074](https://doi.org/10.5281/zenodo.20066074).

Measured archive inventory: 112,940 entries, 112,898 files, 7,098 detection images, 7,097
label files and 91,577 distributed ROI images. The 91,577-row annotation table supplies
reference boxes; canonical crops are reconstructed from these boxes, not taken from the
distributed crop pixels. The “raw” folder contains 7,068 copied image exports although its
manifest lists 7,098. Source preprocessing stretches images to 640×640; some training
variants have augmentation. This cannot be reversed.

All 7,098 acquisition-metadata rows have blank capture date, time, location, trap ID,
camera, operator and notes. Four images contain more than 100 boxes (maximum 106).
One filename family crosses the published train/test split. Human-complete annotation
verification is undocumented. Creator affiliations establish Philippine institutional
provenance, not the origin of every photograph. These findings remain visible in the notebook.

## Frozen sample

The frozen sample contains 200 images: 120 train, 40 validation and 40 test, preserving
published roles while selecting one variant per source filename family. Invalid-box families,
the overlapping family, the crowded family and six perceptually duplicate candidates were
excluded before model output. Exclusion details and source metadata hashes are in
`tools/rice_dataset_audit.json`.

There are 2,509 reference boxes: training 713 rice black bug / 991 white stemborer;
validation 186 / 164; test 271 / 184. Selected image bytes total 8,463,531. All selected
SHA-256, decoded image hashes, dimensions and 19,900 dHash pairs were audited. Approximate
duplicate screening cannot establish independent specimens; dorsal/ventral photos may
show the same insect under different filenames. The sample mixes single-insect photographs
and multi-insect trays.

The notebook fetches exact ZIP byte ranges and checks member names, compression, byte
lengths, CRCs, SHA-256 and image dimensions. It does not download arbitrary executable source
from the dataset. The published archive is 1,005,857,218 bytes with MD5
`24960c245a3a21b63be95314cc7467da`; its whole-file SHA-256 was not measured. Per-member SHA-256
pins define the actual experiment. Data/audit JSON use LF to preserve their digest relationship
across Windows and Linux checkouts.

## Implementation

`tools/build_rice_capstone.py` embeds the runtime, CPU contracts, ZIP fetcher, figure renderer,
data/audit/model manifests and a fully hashed Python 3.12 lock. Model files are staged at
immutable revisions with exact byte/hash verification. No repository clone, remote repository
imports, DIMER service, credential or private upload is required for the default path.

The default uses a target-pest DETR head with frozen ResNet backbone, 15 bounded epochs,
gradient accumulation and validation AP50 selection. BioCLIP features remain frozen;
at most 1,000 training crops fit a text-initialised head for 20 epochs, including epoch zero
in validation-loss selection. SigLIP, RGB, majority and training-mean count baselines remain
visible. Capture-group bootstrap becomes source-family bootstrap under the approved scope.

Validation alone selects checkpoints and thresholds. The held-out comparison separates
reference-box classification from full-image detection/classification/counting. Referred
detections remain in raw counts; accepted and unresolved counts are separate. Threshold
activities do not overwrite the canonical policy. Both adapters reload in a fresh process
and reprocess original held-out image exports. The capstone uses explicit versioned adapter
formats and does not claim drop-in compatibility with the original DETR worker adapter.

Exports include predictions, counts, error decomposition, metrics, attribution, manifests,
adapters, receipts and checksums. The default ZIP excludes source photos and annotated panels.
Results measure agreement with published annotations, not independently verified pest counts,
field abundance, crop damage or pesticide-intervention needs. Completion records are optional.

## Verification and next qualification step

The starting repository suite passed 56 tests using CPU PyTorch 2.11.0. New offline tests
exercise metrics, matching, lineage, ZIP/cache refusals, synthetic head fitting, COCO evaluation,
adapter tampering and prediction/CSV/report parity. Synthetic predictions are software fixtures,
never model-performance evidence. No model weights or local GPU execution are used in this build.

Final local verification on 2026-09-27: **102 tests passed** (56 baseline plus 46 new tests),
Ruff passed, both notebook generators passed parity checks, and release-asset validation
passed. The generated notebook passed nbformat validation and all code cells parsed;
its 21 cells contain no stored execution outputs. Materialising its exact embedded files
and running preparation against all 200 real images passed, and the annotation preview
was rendered and inspected. Independently recomputed image/decoded hashes and all 19,900
dHash pairs matched the frozen audit (minimum Hamming distance 5). One uncached source
image also passed the actual HTTP range download and integrity checks.

Run the saved notebook in a fresh Colab T4 runtime with defaults and Run all. Preserve the
executed notebook, selected policy, receipts, `run_summary.json`, `verification.json`, adapters
and `results.zip`. Actual runtime, peak memory, performance and full-model reload parity remain
unverified until that run. Initial 45-minute/20-GiB budgets are targets, not measured guarantees.

Regeneration/check commands:

```text
python tools/build_rice_capstone.py
python tools/build_rice_capstone.py --check
python tools/build_notebook.py --check
python tools/validate_release_assets.py
pytest -q -o addopts= tests
ruff check src tests tools
```

Maintainer refreezing uses `tools/freeze_rice_data.py --audit-dir outputs/rice-data-audit
--output-dir outputs/rice-refreeze --offline`. It requires the measured metadata/index and
candidate image cache; source hashes are checked. Omit `--offline` to retrieve missing
candidate members. Refreezing changes the scientific sample and is not part of learner Run all.
