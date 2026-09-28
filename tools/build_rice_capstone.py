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
subprocess.run([str(UV), 'venv', '--managed-python', '--python', '3.12.12', str(ROOT / 'env')], env=ENV, check=True)
PYTHON = ROOT / 'env/bin/python'
subprocess.run([str(UV), 'pip', 'install', '--python', str(PYTHON), '--require-hashes', '--only-binary', ':all:', '-r', str(ROOT / 'requirements.txt')], env=ENV, check=True)
subprocess.run([str(PYTHON), '-c', 'import torch; assert torch.cuda.is_available(); print(torch.__version__)'], env=ENV, check=True)
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
    visible = [c for c in columns if c not in ('scores', 'bbox_xyxy')][:9]
    clean = lambda value: str(value).replace('|', '/').replace('\n', ' ')
    lines = ['| ' + ' | '.join(visible) + ' |', '| ' + ' | '.join('---' for _ in visible) + ' |']
    lines += ['| ' + ' | '.join(clean(row[c]) for c in visible) + ' |' for row in rows[:limit]]
    display(Markdown('\n'.join(lines)))
    print('Rows:', len(rows), '(preview limited to', limit, ')')
    display(FileLink(str(path)))

def figures(stage):
    subprocess.run([str(PYTHON), str(ROOT / 'rice_figures.py'), '--root', str(ROOT), '--stage', stage], env=ENV, check=True)
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

    def code(source):
        ast.parse(source)
        cells.append(
            {
                "cell_type": "code",
                "metadata": {},
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

**AI Assistance Disclosure:** Code and instructional text were developed with generative AI assistance under maintainer direction. The maintainer is responsible for review, validation and release decisions. AI assistance is not independent verification or provider endorsement.
""")
    code(PREFLIGHT)
    md("""## 1. Reconstruct the portable experiment
The notebook carries its own source, immutable data/model manifests and a hashed dependency lock. The separate Python environment avoids changing libraries already loaded by Colab. Hash failures stop execution rather than substituting a different experiment.""")
    code(
        "FILES = "
        + repr(files)
        + "\nfor name, source in FILES.items():\n    target = ROOT / name\n    target.parent.mkdir(parents=True, exist_ok=True)\n    target.write_bytes(source.encode('utf-8'))\nprint('Carried files:', len(FILES))\n"
        + BOOTSTRAP
    )
    md("""## 2. Audit the sample before using models
Only frozen ZIP members are downloaded. Their SHA-256, ZIP CRC, byte length and image dimensions are verified. Source-family splits, annotations and limitations are checked before training.

The archive has 112,940 entries, mostly images and crops. We fetch a bounded sample rather than extracting it all. Per-member hashes establish the assets used; this run does not claim to hash the entire approximately 1 GB archive. Original creator attribution and licence references travel with the manifest.

**Notice:** image dimensions, objects per image, species support and source-family counts. A missing annotation is not evidence that no pests are present.""")
    code(
        "subprocess.run([str(PYTHON), '-c', \"import json; from pathlib import Path; from rice_assets import prepare_images; root=Path('.'); print(prepare_images(root,json.loads((root/'data_manifest.json').read_text())))\"], cwd=ROOT, env=ENV, check=True)\nrun('prepare')\ntable('split_manifest.csv')\nfigures('prepare')"
    )
    md("""## 3. Reference crops and representation baselines
**Input:** crops reconstructed from published reference boxes. **Models:** majority class, RGB colour centroid, BioCLIP 2 and SigLIP 2. **Output:** validation classification metrics.

Crop evaluation assumes an object has already been located. It cannot measure missed insects. Common-name prompts and crop preprocessing are fixed before test inspection. RGB success can indicate background shortcuts; large confidence does not establish biological correctness.

**Predict:** will the biodiversity representation outperform the general representation? Compare macro-F1 and class support rather than confidence alone.""")
    code("run('features')\nrecord('validation_crops.json')")
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

**Predict:** will the full pipeline match the reference-crop classifier? Name a failure the crop classifier cannot expose.""")
    code("run('policy')\nrecord('selected_policy.json')")
    md("""## 7. Reveal the held-out benchmark
The primary metric averages absolute count error over images and both species, including zero counts. Compare it with the training-mean and zero-count baselines. Oracle localisation uses published boxes; it is not a deployable system or a guaranteed mathematical upper bound because counting errors can cancel.

Inspect crop macro-F1, detection AP, misses, spurious/duplicate boxes and wrong-species matches. Bootstrap intervals resample source families; they do not establish independent capture events, complete annotations, pretraining independence or wider field performance. Two-class top-2 accuracy would be uninformative and is not a headline metric.""")
    code(
        "run('evaluate')\ntable('metrics.csv')\nrecord('test_crop_metrics.json')\nrecord('detection_metrics.json')\nrecord('bootstrap.json')\ntable('counts.csv')\nfigures('evaluate')"
    )
    md("""## 8. Change one thing: the display threshold
Predict what a lower detector threshold will do to misses, spurious boxes and count MAE. The activity compares lower/canonical/higher settings with the same models. It is retrospective test analysis, not permission to replace the locked result with the best-looking test setting.

Canonical predictions and policy remain unchanged. Explain any error cancellation: a correct total can still contain missed and extra insects.""")
    code("run('activity')\ntable('activity_thresholds.csv')\nfigures('activity')")
    md("""## 9. Export, reload and verify
Both adapters reload in a fresh process with pinned bases. Original held-out images are reprocessed; boxes, species, counts and review decisions must agree within declared tolerances. Hashes and class order must match before tensors are loaded.

The results archive includes derived evidence and attribution, excluding source photographs by default. A passing archive check proves internal consistency, not independent biological validation.""")
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
