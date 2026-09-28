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

## Specification amendment A1 (exploratory scope)

This amendment records how the implemented capstone departs from the supplied
`PHILIPPINE_RICE_PEST_SURVEILLANCE_CAPSTONE_SPEC.md`, version 1.0. It does not mark
any original gate as passed.

| Specification 1.0 requirement | Status under amendment A1 |
|---|---|
| Independently justified capture groups (§2, §5) | **Not met.** Replaced by source-filename families plus duplicate screening. Groups are not verified capture sessions or specimens. |
| Documented Philippine capture provenance (§2) | **Not met.** Creator affiliation is documented; capture location, date and trap fields are blank. |
| Human-created or human-verified complete annotations (§5, §11) | **Not met.** Published boxes are used as unverified references. |
| Qualification of the full surveillance study (§11) | **Not claimed.** Results measure agreement with published annotations only. |
| Review policy, §7 decision 5 | Implemented as written: ≥80% selective accuracy at ≥50% coverage, otherwise accuracy, then coverage, then margin. |

Scope decision: the maintainer chose the exploratory benchmark on 2026-09-27 after the
archive failed the independent-capture gate (see "Agreed scope and source audit"). The
runtime refuses strict capacity certification for this scope, and the dataset audit
records the three unresolved gates. A future full-scope study needs a new dataset that
meets specification 1.0 unchanged.

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

The notebook first downloads a hash-pinned bundle of the 200 selected images (see "Sample
mirror" below). If that fails, it fetches exact ZIP byte ranges from Zenodo and checks member
names, compression, byte lengths, CRCs, SHA-256 and image dimensions. Either way, every image is
re-verified against the manifest. It does not download arbitrary executable source from the
dataset. The published archive is 1,005,857,218 bytes with MD5
`24960c245a3a21b63be95314cc7467da` and SHA-256
`f4a0409977e0615e34f3cf2b936a5ca9dd3a91b153d239e314334ea40c2537f7` (measured 2026-09-28 on a
local copy whose size and MD5 matched Zenodo's). The manifest's `archive.sha256` stays `null` so
that `rice_data.json` keeps the digest bound by the audit. Per-member SHA-256 pins define the
actual experiment. Data/audit JSON use LF to preserve their digest relationship
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

## Review fixes (PR #2 review, 2026-09-28)

The capstone review of PR head `2b8544a` (notebook blob `939beef`) asked for two major and
four minor changes. All were made in the generator and runtime, never by hand in the notebook.

- **M1, detector output validation.** `detections()` checks raw DETR logits and boxes (batch,
  query count, class count, box shape, finite values, normalised range) before the pinned
  postprocessor can drop NaN scores. It then checks the postprocessed rows. `compose()` checks
  BioCLIP embeddings and both species-score distributions. A NaN, Inf or malformed output now
  stops the stage before AP, threshold selection, counts or a receipt. A finite result with
  every score below the operating threshold remains a valid zero count.
- **M2, error diagnosis.** Section 7 shows an error summary per species (reference = correct +
  wrong_out + missed; raw = correct + wrong_in + spurious), a matched-species confusion table,
  the panel inventory including absent categories, and a table of every tagged error in the
  example panels with reference species, predicted species, detector score, species score,
  margin and review state. Panels are colour-coded by outcome, titled with their category,
  and use compact tags in crowded images. Section 3 shows labelled training crops first.
- **m1, review fallback.** The fallback now follows specification §7 decision 5 over every
  nonempty margin, not only those with ≥50% coverage. The selected rule, coverage and accuracy
  are printed in Section 6. This can change the referral workload, not raw counts.
- **m2, learner guidance.** The 1.3-million-character carrier cell is collapsed with a title
  and links to readable source. A glossary defines the metrics. Section 2 shows sample support
  per split and the audit-gate statuses.
- **m3, paired count view.** The threshold activity adds wrong-species totals and follows the
  held-out image whose raw count moves most across lower/canonical/higher thresholds. It checks
  that canonical predictions and the policy lock are unchanged.
- **m4, charts in the bundle.** `results.zip` now includes the four metric charts. Photo
  previews, reference-crop grids and annotated panels stay excluded.

User-visible changes: new output files (`sample_summary.csv`, `audit_gates.csv`,
`error_summary.csv`, `matched_species_confusion.csv`, `error_examples.csv`,
`error_example_counts.csv`, `activity_paired_counts.csv`, `figures/crops.png`), new
`selected_policy.json` fields, a `wrong_species` column in `activity_thresholds.csv`, and
charts in `results.zip`. Invalid model outputs that were previously filtered now stop the run.

Verification (CPU only; not clean-runtime evidence): 119 tests passed with the CI-pinned
CPU stack (102 existing plus 17 new). The M1 tests run the real pinned Transformers 4.57.6
postprocessor on a tiny randomly initialised DETR; they are not pretrained-inference
evidence. Panels were rendered from synthetic geometric placeholders and inspected for a
wrong-species case, a zero-detection image and a crowded image. No model weights, source
photographs or GPU were used. A fresh Colab T4 Run all on the revised head remains pending.

## Sample mirror and error surfacing (2026-09-28)

The Colab run of `a4a0917` failed at data preparation (see "Recorded executions"). Three changes
follow from it, all in the generator and runtime:

- **Hash-pinned mirror.** `tools/rice_mirror.json`, carried into the notebook as `mirror.json`,
  pins a release asset of this repository:
  `https://github.com/kurtvalcorza/detr-detection-pipeline/releases/download/rice-capstone-sample-v1/rice_capstone_sample_v1.zip`,
  8,502,987 bytes, SHA-256 `48b44850c62ff07cce97154180965b9c57855cd5bb58f95b41da3bbb4e513e5d`.
  It holds the 200 selected images at their manifest `relative_path`, `ZIP_STORED`, with sorted
  names, fixed 1980-01-01 timestamps and no directory entries or extra fields. It was built from
  the local Zenodo archive after checking each member's offset, sizes, CRC, SHA-256 and decoded
  dimensions against `rice_data.json`; two builds were byte-identical. Every member is therefore
  byte-identical to its Zenodo ZIP member. `prepare_images` downloads it once (host allowlist,
  64 MiB ceiling, exact size and SHA-256), refuses unsafe or unexpected member names, re-checks
  each member's bytes and SHA-256 before writing, then runs every image through the same cache
  checks as before (bytes, SHA-256, format, dimensions). The bundle is CC BY 4.0 data with the
  attribution recorded in `rice_mirror.json` and the release notes. `rice_data.json` is
  unchanged, so the audit's `manifest_sha256` still binds it.
- **Zenodo fallback.** Any mirror failure prints one line and falls back to Zenodo, which
  remains the source of record. The fallback now reads each member in one bounded range request
  instead of two, uses two threads instead of four, honours `Retry-After` on HTTP 429/503 (capped
  at 90 s, up to six attempts) and does not retry other 4xx errors. Every length, `Content-Range`
  and CRC check is unchanged.
- **Visible errors.** A `checked()` helper runs the bootstrap installs, the Section 2 image
  helper and the figure renderer with captured output. It always prints stdout; on failure it
  prints stderr and raises `RuntimeError` with its tail, so the real exception appears in the
  notebook. `run()` already streamed combined output and is unchanged.

- **Compressed carrier.** Opening the notebook at `c113a9c` in Colab made the browser report
  "page unresponsive": the collapsed carrier cell held its embedded files as one
  1,348,754-character line. The generator now embeds them as zlib-compressed base64 in
  100-character lines, with the SHA-256 of the uncompressed JSON checked before any file is
  written. The notebook shrinks from 1,484,378 to about 297,000 bytes; the carrier cell has about
  2,500 lines, none longer than 192 characters. The decoded files are identical to the generator's
  carried files. `build_rice_capstone.py --check` passes with Windows zlib 1.3.1 and Linux
  zlib 1.3, so parity does not depend on the platform used for regeneration.

User-visible changes: the Section 2 summary gains a `source` field (`mirror` or `zenodo`), and a
failed helper now raises `RuntimeError` instead of `CalledProcessError`. The carrier cell's source
is no longer readable in place; the readable copies linked above it are unchanged.

Verification (CPU only, Windows; not clean-runtime evidence): 130 tests passed with the
CI-pinned CPU stack (119 before plus 11 new in `tests/test_rice_mirror.py`, all offline):
mirror success, bundle-hash mismatch fallback, a tampered member in a valid-hash bundle,
path traversal, host separation, 429 then 206 with `Retry-After`, persistent 429, no retry on
404, and a static check that Section 2 surfaces stderr. Real inputs: with the real manifest,
`rice_mirror.json` and the real bundle bytes served through a stubbed download, all 200 images
verified from the mirror without contacting Zenodo. Forcing the fallback against live Zenodo
fetched and verified all 200 images in 191 s with one range request per member; no 429 was
received, so `Retry-After` handling is covered only by the offline tests. After the release was
published, an anonymous download of the pinned URL (302 to `release-assets.githubusercontent.com`,
8,502,987 bytes) passed the size and SHA-256 checks and verified all 200 images from the mirror.

## Recorded executions

### Maintainer-supplied Colab execution of revision `a4a0917` — 2026-09-28 (FAILED at data preparation)

| Field | Value |
|---|---|
| File | `docs/execution-evidence/2026-09-28/DIMER_Philippine_Rice_Pest_Surveillance_Capstone_a4a0917_failed-prepare.ipynb`, SHA-256 `a62a769256b56ad97e48fe0ecb098891f6fc80d4c10bbb46a2967060ab04228a` (byte copy of the upload) |
| Source match | All 22 cells, ids, order and `dimer` metadata identical to the notebook at `a4a0917` (blob `83c20ae`); no `# @param` or other diffs |
| Runtime | Colab, `Tesla T4`; kernel Python 3.13; isolated environment built from the hashed lock |
| Executed cells | Preflight (1), carrier/bootstrap (2), Section 2 data preparation (3, error); cells 4–10 not executed |
| Result | `CalledProcessError`: the `prepare_images` helper exited with status 1. Its own traceback went to the kernel's raw stderr and is not in the notebook, so the exact exception is unknown. |
| Probable cause (not proven) | Zenodo sends `x-ratelimit-limit: 133` per 60 s window for this archive. Preparing 200 images issues 400 range requests over 4 threads; `read_range` retries twice after 1 s and 2 s, far shorter than `retry-after: 60`. The same helper and manifest passed outside Colab on 2026-09-28 (200 images, all digests verified, 102 s through a throttling proxy). |
| Evidence boundary | Saved outputs were inspected; execution was not independently repeated. No model stage ran, so this run is not evidence for M1–m4 behaviour. |

| Journey | Verdict |
|---|---|
| Preflight and isolated environment | Passed |
| Data preparation | **Failed** |
| Features, training, policy, evaluation, activity, reload, report | Not assessed in this run |

Fixed in: the sample-mirror change below (the helper's stderr is now shown in the cell, images come from a hash-pinned mirror, and the Zenodo fallback honours `Retry-After`). A new Colab T4 Run all on that head is pending.

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
