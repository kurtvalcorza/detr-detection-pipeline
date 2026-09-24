# Release verification

`tutorials/detr_detection_colab.ipynb` (`E2E`, **standalone** carrier) is a **release candidate** until the exact notebook revision has executed top-to-bottom in a clean supported runtime. Unit tests, the tiny-model test, JSON validation, code-cell compilation, the generator parity checks and `tools/validate_release_assets.py` are necessary checks but are **not** runtime evidence under DIMER Notebook Specification 2.1 (REL8). This file is the durable release-gate record for the notebook.

The upstream snapshot is not yet pinned, so the notebook cannot run yet: its model cell raises before any download. Pinning (`python tools/pin_snapshot.py`) and regenerating the notebook come before any execution recorded here.

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
3. run the notebook top-to-bottom without editing implementation cells, with every form field at its default (`USE_BYOD_IMAGE = False`, `USE_BYOD_DATASET = False`, `threshold = 0.9`, `EPOCHS = 10`);
4. verify that Section 1 reports `NOTEBOOK_SOURCE.repository_revision` equal to `metadata.dimer.generated_from.revision`, and that the installed versions equal the inline `PINS`;
5. verify that every default-path stage completes: the pinned install; the carried modules; staging of all four manifest entries and `verify_snapshot`; `validate_inputs` with the `threshold=1.5` rejection finding; COCO detection and the `sample-sanity` report; the blank and noise probes; `validate_dataset`; the split with its leakage assertion; the baseline; `finetune`; the adapted evaluation; new-data inference; adapter export, reload and the tolerance check; and the six outputs in `outputs/`;
6. record the observed baseline and adapted `ap`, `ap50` and `ap75`, the per-epoch losses, the trainable and total parameter counts, the degenerate-probe counts and the wall time in the table below. No metric value is asserted in advance: a low adapted AP is a finding to record, not a failure by itself;
7. exercise the BYOD dataset branch once with a small labelled directory (`BYOD_DATASET_DIR`) and once with an incompatible `annotations.json`, and record both outcomes (REL12);
8. record the notebook Git blob id, commit, runtime (platform, GPU, Python, PyTorch, Transformers, timm), model identifier and revision, whether the model cache was clean, and any warning judged harmless with the reason;
9. record no access tokens or other secrets.

A known-failing default path in the supported runtime blocks release.

## Recorded executions

Notebook identity is the Git blob id of `tutorials/detr_detection_colab.ipynb` (verify with `git rev-parse <commit>:tutorials/detr_detection_colab.ipynb`). Each record uses the fields Date, Subject, Runtime, Procedure, Observed result and Caveats.

| Date (UTC) | Subject (commit / notebook blob) | Runtime | Procedure | Observed result | Caveats |
|---|---|---|---|---|---|
| — | — | — | — | No execution recorded. The snapshot is not yet pinned. | — |
