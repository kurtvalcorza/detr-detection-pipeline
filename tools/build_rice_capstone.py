"""Generate the standalone, explicitly exploratory rice-pest capstone."""

# ruff: noqa: E501
from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NAME = "DIMER_Philippine_Rice_Pest_Surveillance_Capstone.ipynb"


def carried_files() -> dict[str, str]:
    names = {
        "capstone.py": "rice_capstone.py",
        "rice_core.py": "rice_core.py",
        "rice_assets.py": "rice_assets.py",
        "rice_figures.py": "rice_figures.py",
        "data_manifest.json": "rice_data.json",
        "dataset_audit.json": "rice_dataset_audit.json",
        "mirror.json": "rice_mirror.json",
        "model_manifest.json": "rice_models.json",
        "requirements.txt": "rice-requirements.lock",
    }
    files = {
        target: (ROOT / "tools" / source).read_bytes().decode("utf-8") for target, source in names.items()
    }
    files["LICENSE.txt"] = (ROOT / "LICENSE").read_text(encoding="utf-8")
    return files


PREFLIGHT = r"""
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import time
import urllib.request
import uuid
import zipfile
from IPython.display import Image, Markdown, FileLink, display

if platform.system() != 'Linux' or platform.machine() != 'x86_64':
    raise RuntimeError('Use a fresh Colab Linux x86-64 T4 runtime.')
try:
    GPU = subprocess.check_output(['nvidia-smi', '--query-gpu=name', '--format=csv,noheader'], text=True).strip()
except (FileNotFoundError, subprocess.CalledProcessError) as exc:
    raise RuntimeError('Select Runtime > Change runtime type > T4 GPU, then Run all.') from exc
if 'T4' not in GPU:
    raise RuntimeError('The canonical qualification target is a fresh T4 runtime.')
ROOT = Path.cwd() / 'outputs' / 'rice_pest' / uuid.uuid4().hex[:12]
ROOT.mkdir(parents=True)
if shutil.disk_usage(ROOT).free < 20 * 1024**3:
    raise RuntimeError('Allow at least 20 GiB free disk for the isolated environment, weights and outputs.')
print('GPU:', GPU, '\nRun directory:', ROOT)
"""

BOOTSTRAP = r"""
started = time.perf_counter()
UV_URL = 'https://files.pythonhosted.org/packages/1e/fd/432451d732917c49152a291de3ef171aa6b0f1a22d39780fb2c1f085ca4c/uv-0.12.15-py3-none-manylinux_2_17_x86_64.manylinux2014_x86_64.whl'
with urllib.request.urlopen(UV_URL, timeout=90) as response:
    wheel = response.read(20081405)
if len(wheel) != 20081404 or hashlib.sha256(wheel).hexdigest() != 'aee9802f46bae436bd91751bb33ddeb379ef1596b5c19df193219d545d244b60':
    raise RuntimeError('Environment bootstrap size/hash mismatch')
with zipfile.ZipFile(io.BytesIO(wheel)) as archive:
    member = next(name for name in archive.namelist() if name.endswith('.data/scripts/uv'))
    UV = ROOT / 'uv'
    UV.write_bytes(archive.read(member))
UV.chmod(0o700)
ENV = dict(os.environ, HF_HUB_DISABLE_IMPLICIT_TOKEN='1', HF_HUB_DISABLE_TELEMETRY='1', DO_NOT_TRACK='1', MPLBACKEND='Agg')
ENV.pop('HF_TOKEN', None)
ENV.pop('HUGGING_FACE_HUB_TOKEN', None)

def checked(args, **kwargs):
    # Capture the child's stderr so a failure shows its real exception in the cell, not only an exit status.
    result = subprocess.run([str(a) for a in args], env=ENV, capture_output=True, text=True, **kwargs)
    if result.stdout.strip():
        print(result.stdout.rstrip(), flush=True)
    if result.returncode:
        print(result.stderr.rstrip(), flush=True)
        raise RuntimeError(f'{Path(str(args[0])).name} failed ({result.returncode}): ' + result.stderr.strip()[-2000:])
    return result

checked([UV, 'venv', '--managed-python', '--python', '3.12.12', ROOT / 'env'])
PYTHON = ROOT / 'env/bin/python'
checked([UV, 'pip', 'install', '--python', PYTHON, '--require-hashes', '--only-binary', ':all:', '-r', ROOT / 'requirements.txt'])
checked([PYTHON, '-c', 'import torch; assert torch.cuda.is_available(); print(torch.__version__)'])
(ROOT / 'bootstrap.json').write_text(json.dumps({'seconds': time.perf_counter()-started, 'gpu': GPU}), encoding='utf-8')

def run(stage):
    log = ROOT / (stage + '.log')
    print('Running', stage, '— retain this log if anything fails:', log, flush=True)
    with log.open('w', encoding='utf-8') as handle:
        process = subprocess.Popen([str(PYTHON), '-u', str(ROOT / 'capstone.py'), '--root', str(ROOT), '--stage', stage], env=ENV, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        for line in process.stdout:
            handle.write(line)
            handle.flush()
            if len(line) < 2000:
                print(line.rstrip(), flush=True)
        process.wait()
    if process.returncode:
        raise RuntimeError(f'{stage} failed ({process.returncode}); inspect {log}. Do not skip the failed stage.')

def record(name):
    value = json.loads((ROOT / 'outputs' / name).read_text(encoding='utf-8'))
    print(json.dumps(value, indent=2)[:18000])
    display(FileLink(str(ROOT / 'outputs' / name)))

def table(name, limit=16):
    path = ROOT / 'outputs' / name
    with path.open(encoding='utf-8', newline='') as handle:
        reader = csv.DictReader(handle)
        columns, rows = reader.fieldnames, list(reader)
    if not columns:
        raise RuntimeError('Missing table columns: ' + name)
    visible = [c for c in columns if c not in ('scores', 'bbox_xyxy')][:12]
    clean = lambda value: str(value).replace('|', '/').replace('\n', ' ')
    lines = ['| ' + ' | '.join(visible) + ' |', '| ' + ' | '.join('---' for _ in visible) + ' |']
    lines += ['| ' + ' | '.join(clean(row[c]) for c in visible) + ' |' for row in rows[:limit]]
    display(Markdown('\n'.join(lines)))
    print('Rows:', len(rows), '(preview limited to', limit, ')')
    display(FileLink(str(path)))

def figures(stage):
    checked([PYTHON, ROOT / 'rice_figures.py', '--root', ROOT, '--stage', stage])
    for path in sorted((ROOT / 'outputs' / 'figures').glob(stage + '*.png')):
        display(Image(filename=str(path)))
    if stage == 'evaluate':
        for path in sorted((ROOT / 'outputs' / 'local_figures').glob('*.png')):
            display(Image(filename=str(path)))
"""


def build() -> dict:
    files = carried_files()
    dataset = json.loads(files["data_manifest.json"])
    rows = dataset["records"]
    counts = {split: sum(r["split"] == split for r in rows) for split in ("train", "validation", "test")}
    provenance = {
        "repository": "kurtvalcorza/detr-detection-pipeline",
        "generator": "build_rice_capstone.py/1",
        "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "files": {k: hashlib.sha256(v.encode()).hexdigest() for k, v in files.items()},
    }
    files["source.json"] = json.dumps(provenance, indent=2)
    cells = []

    def md(source):
        cells.append(
            {
                "cell_type": "markdown",
                "metadata": {},
                "id": f"md-{len(cells):02}",
                "source": source.strip().splitlines(True),
            }
        )

    def code(source, metadata=None):
        ast.parse(source)
        cells.append(
            {
                "cell_type": "code",
                "metadata": metadata or {},
                "id": f"code-{len(cells):02}",
                "source": source.strip().splitlines(True),
                "outputs": [],
                "execution_count": None,
            }
        )

    md(f"""# Philippine Rice Pest Surveillance
## From published insect annotations to an exploratory counting benchmark

**Scientific applications / composed-system capstone · E2E / WORKSHOP · Candidate**

Can a detector and crop classifier reproduce published rice-pest counts, and where do errors enter the system?

**Input:** full detection-image exports. **System:** DETR → insect crops → BioCLIP species head. **Output:** boxes, species predictions, counts and review flags. SigLIP and simple baselines help interpret the result.

This notebook uses **{len(rows)} images**, split into **{counts["train"]} train / {counts["validation"]} validation / {counts["test"]} test**, grouped by source-image family and duplicate checks. These groups are **not verified independent capture sessions or specimens**. The published annotations are unverified references, not independently checked ground truth. This is an exploratory benchmark, not a qualified field-surveillance system.

The source is a CC BY 4.0 [PhilRice-affiliated Zenodo dataset](https://doi.org/10.5281/zenodo.20066074). Its “raw” images are resized Roboflow exports; capture dates, trap IDs and locations are blank. Philippine institutional provenance is documented; exact capture location is not. Four source images exceed DETR's 100-query capacity and are excluded from this bounded study before model evaluation.

**Prerequisites:** basic Python/Colab familiarity. Choose a **fresh T4 GPU** and **Run all** with defaults. No token, repository clone, DIMER worker, private upload or manual restart is required. Initial budget: up to 45 minutes compute and 20 GiB free disk; these are targets pending hosted measurement.

**Predict before running:** will missed boxes or species mistakes contribute more to count error? Write one reason. Completion notes are optional, not required submissions.

**How to read this notebook:** each section says what goes in, what comes out and what to notice. Terms are defined in the glossary below; come back to it when a metric first appears.

**AI Assistance Disclosure:** Code and instructional text were developed with generative AI assistance under maintainer direction. The maintainer is responsible for review, validation and release decisions. AI assistance is not independent verification or provider endorsement.
""")
    md("""### Glossary
| Term | Meaning in this notebook |
|---|---|
| **Reference box** | A box and species label published with the dataset. Treated as the answer key, but not independently verified. |
| **IoU** (intersection over union) | Overlap between two boxes, from 0 (none) to 1 (identical). A prediction matches a reference box when IoU ≥ 0.5. |
| **AP50** | Detection average precision at IoU 0.5: how well ranked boxes find reference insects, from 0 to 1. It ignores species. |
| **Detector score** | DETR's confidence that a box contains *a target pest*. It says nothing about which species. |
| **Species score / margin** | The classifier's probability for its chosen species, and the gap between the two species' probabilities. A small margin sends the detection to **review**. |
| **Raw / accepted / referred count** | Raw counts every detection above the detector threshold. Accepted excludes detections sent to review; referred counts those review items. Review never deletes insects. |
| **Count MAE** | Mean absolute error of per-image, per-species counts. 0.5 means counts are off by half an insect per image and species on average. |
| **Missed / spurious / wrong species** | A reference with no matching prediction; a prediction with no matching reference; a matched pair whose species differ. |
| **Macro-F1** | The average of per-species F1 scores, so the rarer species counts as much as the common one. |
| **Cross-entropy** | The training loss for the species head; lower on validation means better-calibrated species scores. |
| **RGB centroid** | A deliberately simple baseline that classifies crops by average colour. Beating it shows the model uses more than colour. |
| **Bootstrap interval** | A range from resampling source families with replacement. It shows sampling uncertainty over these families, not over independent field captures. |""")
    code(PREFLIGHT)
    md("""## 1. Reconstruct the portable experiment
The notebook carries its own source, immutable data/model manifests and a hashed dependency lock. The separate Python environment avoids changing libraries already loaded by Colab. Hash failures stop execution rather than substituting a different experiment.

The next cell is **collapsed on purpose**: it holds about 1.3 million characters of embedded source and manifests, and you do not need to read it to follow the experiment. Readable copies are in the repository: the [runtime](https://github.com/kurtvalcorza/detr-detection-pipeline/blob/main/tools/rice_capstone.py), [metrics and matching](https://github.com/kurtvalcorza/detr-detection-pipeline/blob/main/tools/rice_core.py) and [figures](https://github.com/kurtvalcorza/detr-detection-pipeline/blob/main/tools/rice_figures.py). The same files are written into the run directory.""")
    code(
        "# @title Carrier cell: write embedded source, manifests and lock, then build the isolated environment\n"
        "FILES = "
        + repr(files)
        + "\nfor name, source in FILES.items():\n    target = ROOT / name\n    target.parent.mkdir(parents=True, exist_ok=True)\n    target.write_bytes(source.encode('utf-8'))\nprint('Carried files:', len(FILES))\n"
        + BOOTSTRAP,
        {"cellView": "form", "collapsed": True, "jupyter": {"source_hidden": True}},
    )
    md("""## 2. Audit the sample before using models
Only frozen ZIP members are downloaded. Their SHA-256, ZIP CRC, byte length and image dimensions are verified.

The 200 images come from a hash-pinned sample bundle published as a release asset of this repository (`mirror.json` records its URL, size and SHA-256). Every member is byte-identical to the Zenodo ZIP member named in the manifest and is re-checked against it. If the bundle cannot be fetched or fails a check, the cell prints one line and falls back to reading the frozen members directly from Zenodo, which remains the source of record. Source-family splits, annotations and limitations are checked before training.

The archive has 112,940 entries, mostly images and crops. We fetch a bounded sample rather than extracting it all. Per-member hashes establish the assets used; this run does not claim to hash the entire approximately 1 GB archive. Original creator attribution and licence references travel with the manifest.

**Notice:** image dimensions, objects per image, species support and source-family counts. The first table gives support per split: the species columns count reference boxes (insects), not images. The second lists the audit gates: *unresolved* gates are the reasons this is an exploratory benchmark. A missing annotation is not evidence that no pests are present.""")
    code(
        "checked([PYTHON, '-c', \"import json; from pathlib import Path; from rice_assets import prepare_images; root=Path('.'); load=lambda name: json.loads((root/name).read_text(encoding='utf-8')); print(prepare_images(root, load('data_manifest.json'), load('mirror.json')))\"], cwd=ROOT)\nrun('prepare')\ntable('sample_summary.csv')\ntable('audit_gates.csv')\ntable('split_manifest.csv')\nfigures('prepare')"
    )
    md("""## 3. Reference crops and representation baselines
**Input:** crops reconstructed from published reference boxes. **Models:** majority class, RGB colour centroid, BioCLIP 2 and SigLIP 2. **Output:** validation classification metrics.

Crop evaluation assumes an object has already been located. It cannot measure missed insects. Common-name prompts and crop preprocessing are fixed before test inspection. RGB success can indicate background shortcuts; large confidence does not establish biological correctness.

The first figure shows what the classifier receives: training crops cut from reference boxes, labelled with their published species. Small crops carry little detail.

**Predict:** will the biodiversity representation outperform the general representation? Compare macro-F1 and class support rather than confidence alone.

**How to read the result:** for each method, `accuracy` and `macro_f1` range from 0 to 1; `confusion` rows are reference species and columns predicted species (rice black bug first). The majority baseline's macro-F1 shows what "no information" looks like.""")
    code("figures('crops')\nrun('features')\nrecord('validation_crops.json')")
    md("""## 4. Adapt only the species head
The BioCLIP image tower stays frozen. A small linear head starts from text embeddings and trains for at most 20 epochs. Epoch zero is eligible; minimum validation cross-entropy selects the checkpoint, with earlier ties.

**Notice:** whether validation loss improves with training loss. More epochs are not automatically better. The test split is not used to choose this head.""")
    code("run('classifier')\nrecord('classifier_history.json')\nfigures('classifier')")
    md("""## 5. Adapt the detector
DETR starts from COCO weights, but its new target-pest head is untrained. Epoch zero is an untrained-head baseline, not zero-shot pest detection. The ResNet backbone stays frozen while transformer and detection heads train. The bounded recipe uses gradient accumulation and validation AP50 checkpoint selection.

**Input:** complete image exports and their boxes. **Output:** ranked target-pest boxes. The detector cannot represent more than 100 objects per image. We do not truncate reference boxes to make a metric look better.

**Notice:** validation localisation quality and the chosen epoch. Poor adaptation remains a reportable result.""")
    code("run('detector')\nrecord('detector_history.json')\nfigures('detector')")
    md("""## 6. Lock the composed policy
Checkpoint selection is finished. Choose the detection threshold on validation data to minimise per-image/per-species count MAE, with higher-threshold ties. Separately choose a species-score-margin review policy.

Every retained detection contributes to the raw count. Review flags create unresolved work; they do not make insects disappear. A missed detection cannot be referred by the classifier. Scores are not calibrated probabilities or operational pest-alert thresholds.

The review policy first looks for the margin with the highest coverage that keeps at least 80% selective accuracy while accepting at least 50% of validation crops. If no margin reaches that target, it falls back to the most accurate margin, then the widest coverage. The output states which rule applied; a fallback can send many detections to review.

**Predict:** will the full pipeline match the reference-crop classifier? Name a failure the crop classifier cannot expose.""")
    code(
        "run('policy')\nrecord('selected_policy.json')\n"
        "locked = json.loads((ROOT / 'outputs' / 'selected_policy.json').read_text(encoding='utf-8'))\n"
        "print('Detector threshold:', locked['detector_threshold'])\n"
        "print('Review margin:', locked['review_margin'], '|', locked['review_selection_rule'])\n"
        "print('Validation coverage / selective accuracy:', locked['review_validation_coverage'], '/', locked['review_validation_accuracy'])"
    )
    md("""## 7. Reveal the held-out benchmark
The primary metric averages absolute count error over images and both species, including zero counts. Compare it with the training-mean and zero-count baselines. Oracle localisation uses published boxes; it is not a deployable system or a guaranteed mathematical upper bound because counting errors can cancel.

Inspect crop macro-F1, detection AP, misses, spurious/duplicate boxes and wrong-species matches. Bootstrap intervals resample source families; they do not establish independent capture events, complete annotations, pretraining independence or wider field performance. Two-class top-2 accuracy would be uninformative and is not a headline metric.

**Where does count error enter?** After the metrics, three tables trace it:
1. **Error summary**, per species: `reference = correct + wrong_out + missed` and `raw = correct + wrong_in + spurious`. `wrong_out` is a reference insect of this species labelled as the other; `wrong_in` is the reverse. `accepted` and `referred` split the raw count by the review policy.
2. **Matched-species confusion**: rows are reference species, columns predicted species; the last column counts missed references, and the last row counts spurious predictions.
3. **Example panels**: one held-out photograph per error category (misses, spurious, wrong species, success). Each title says which category it illustrates and gives its counts. Green boxes are correct matches, magenta boxes are wrong-species matches (the white dashed box is the reference), dashed blue boxes are missed references, and orange boxes are spurious predictions. Each error box carries a `#tag`; the table after the panels lists each tag's reference species, predicted species, **detector score**, **species score**, margin and review state. The last table reconciles each example's per-species counts.

If a category is absent from the held-out split, the inventory says so; no example is manufactured.""")
    code(
        "run('evaluate')\ntable('metrics.csv')\nrecord('test_crop_metrics.json')\nrecord('detection_metrics.json')\nrecord('bootstrap.json')\n"
        "table('error_summary.csv')\ntable('matched_species_confusion.csv')\ntable('counts.csv')\nfigures('evaluate')\n"
        "record('error_panel_inventory.json')\ntable('error_examples.csv', limit=40)\ntable('error_example_counts.csv')"
    )
    md("""## 8. Change one thing: the display threshold
Predict what a lower detector threshold will do to misses, spurious boxes and count MAE. The activity compares lower/canonical/higher settings with the same models. It is retrospective test analysis, not permission to replace the locked result with the best-looking test setting.

Canonical predictions and policy remain unchanged. Explain any error cancellation: a correct total can still contain missed and extra insects.

The second table follows one held-out photograph: the image whose raw count changes most across the three settings. For each setting and species it shows the reference count, the raw count and the stage errors, so you can see which errors a threshold change adds or removes.""")
    code(
        "run('activity')\ntable('activity_thresholds.csv')\ntable('activity_paired_counts.csv')\nfigures('activity')"
    )
    md("""## 9. Export, reload and verify
Both adapters reload in a fresh process with pinned bases. Original held-out images are reprocessed; boxes, species, counts and review decisions must agree within declared tolerances. Hashes and class order must match before tensors are loaded.

The results archive includes derived evidence, attribution and the metric charts (learning curves, count comparison and threshold charts). Source photographs, photo previews and annotated panels are excluded by default. A passing archive check proves internal consistency, not independent biological validation.""")
    code(
        "run('reload')\nrecord('verification.json')\nrun('report')\nrecord('run_summary.json')\ndisplay(FileLink(str(ROOT / 'outputs' / 'results.zip')))"
    )
    md("""## 10. Conclude with evidence
Complete this optional statement using the exported results:

> On [N] held-out image exports grouped into [G] source families, the composed pipeline achieved count MAE [value and interval], compared with [baseline]. Misses, extra detections and species errors contributed [evidence]. Changing only the display threshold showed [trade-off]. These results describe agreement with this dataset's published annotations; they do not establish independent-capture generalisation, field abundance, crop damage or intervention thresholds.

**Limitations to retain:** source-image grouping cannot resolve repeated specimens in different originals; exported images were stretched to 640×640; annotations and completeness were not independently verified; source capture locations are unavailable; the sample excludes over-capacity images; pretrained-model overlap is unresolved. Report absent error categories and zero detections honestly.

**Continue exploring:** compare the full [DETR](https://github.com/kurtvalcorza/detr-detection-pipeline), [BioCLIP 2](https://github.com/kurtvalcorza/bioclip2-biodiversity-pipeline) and [SigLIP 2](https://github.com/kurtvalcorza/siglip2-vision-language-pipeline) pipelines with the models available through [DIMER](https://training.dimer1.asti.dost.gov.ph/signin). A stronger follow-up would collect verified trap/session/specimen metadata and independently checked exhaustive annotations.

**Optional extension:** four-way species–sex classification requires separately verified sex labels and adequate independent support. It is not enabled by default. Completion records are not required submissions.

### Troubleshooting
If a source hash, model hash or receipt fails, preserve the log and stop; do not bypass the check. If T4 memory or time limits are exceeded, record actual use and revise the recipe before a new clean run. Do not silently lower sample size, epochs or image resolution. A model that fails to beat the count baseline is an informative result, not an execution error.

Sources: [dataset and citation](https://doi.org/10.5281/zenodo.20066074), [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/), [DIMER notebook standard](https://github.com/kurtvalcorza/ml-worker/blob/main/integrations/dimer/fleet-specs/NOTEBOOK_SPEC.md). Exact source/model identities and creator attribution are carried in the manifests.
""")
    return {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.12"},
            "accelerator": "GPU",
            "colab": {"name": NAME, "gpuType": "T4", "provenance": []},
            "dimer": {
                "notebook_spec": "2.2",
                "notebook_profile": "E2E",
                "notebook_mode": "WORKSHOP",
                "standalone": True,
                "release_status": "Candidate",
                "study_scope": "exploratory_published_annotations",
                "generated_from": provenance,
            },
        },
        "cells": cells,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    target = ROOT / "tutorials" / NAME
    rendered = json.dumps(build(), indent=1, ensure_ascii=False) + "\n"
    if args.check:
        if not target.exists() or target.read_text(encoding="utf-8") != rendered:
            raise SystemExit("Rice capstone generation parity failed")
        print("Rice capstone generation parity: PASS")
    else:
        target.write_text(rendered, encoding="utf-8", newline="\n")
        print(target)


if __name__ == "__main__":
    main()
