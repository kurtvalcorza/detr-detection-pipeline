"""Owner decisions on PR #6: RPS-m5 = C (activity controls) and RPS-CX4 = A (training-median baseline).

CPU only, synthetic records and predictions: this exercises the runtime's bookkeeping, not models.
"""

import csv
import importlib.util
import io
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import build_rice_capstone as build  # noqa: E402
import rice_capstone as run  # noqa: E402

NOTEBOOK = ROOT / "tutorials" / build.NAME
# T4-recorded runtime before the decisions (Colab CLI run recorded in 3eefee7).
PREVIOUS_RUNTIME = "49a7ff8c7db042d1be7e6f0df874068c5f834842"
ACTIVITY_FILES = ("activity_thresholds.csv", "activity_paired_counts.csv", "activity_review_margins.csv")
CANONICAL_FILES = ("test_detections.json", "metrics.json", "bootstrap.json")


def _sources():
    return ["".join(c["source"]) for c in json.loads(NOTEBOOK.read_text(encoding="utf-8"))["cells"]]


def _section(number):
    cells = _sources()
    start = next(i for i, s in enumerate(cells) if s.startswith(f"## {number}."))
    out = [cells[start]]
    for text in cells[start + 1 :]:
        if text.startswith("## "):
            break
        out.append(text)
    return "\n".join(out)


def _rows(data):
    return list(csv.DictReader(io.StringIO(data.decode("utf-8"))))


def _fixture(root, monkeypatch, mod):
    """Synthetic records and predictions with misses, unmatched boxes and varied margins."""
    out = root / "outputs"
    out.mkdir(parents=True, exist_ok=True)
    rbb, wsb = mod.SPECIES
    boxes = [[2, 2, 12, 12], [20, 20, 30, 30], [40, 40, 50, 50], [60, 2, 70, 12]]
    records = []
    for i in range(6):
        objects = [
            {
                "object_id": f"{i}_{k}",
                "bbox_xyxy": boxes[k],
                "species": (rbb, wsb)[(i + k) % 2],
                "ignore": False,
            }
            for k in range(1 + i % 3)
        ]
        records.append(
            {
                "image_id": str(i),
                "width": 80,
                "height": 80,
                "split": "train" if i < 2 else "test",
                "capture_group_id": f"g{i}",
                "objects": objects,
            }
        )
    lineage = [
        {"crop_id": str(i), "parent_image_id": str(i), "species": (wsb, rbb)[i % 2], "split": r["split"]}
        for i, r in enumerate(records)
    ]
    manifest = {"scope": "exploratory_published_annotations", "records": records}
    for name in ("data_manifest.json", "dataset_audit.json", "model_manifest.json"):
        mod.write(root / name, manifest if name == "data_manifest.json" else {"test_fixture": True})
    curve = [
        {"threshold": 0.0, "coverage": 1.0, "accuracy": 0.8},
        {"threshold": 0.1, "coverage": 0.9, "accuracy": 0.85},
        {"threshold": 0.4, "coverage": 0.8, "accuracy": 0.9},
        {"threshold": 0.7, "coverage": 0.5, "accuracy": 1.0},
    ]
    policy = {
        "detector_threshold": 0.5,
        "review_margin": 0.1,
        "digest": "test_fixture",
        "review_validation": {"coverage": 0.9, "accuracy": 0.85, "curve": curve},
    }
    adapted = [[0.9, 0.1], [0.3, 0.7], [0.55, 0.45], [0.2, 0.8]]
    predictions = [
        {
            "image_id": r["image_id"],
            "boxes": boxes,
            "scores": [0.95, 0.75, 0.55, 0.35],
            "adapted": adapted,
            "zero_shot": adapted[::-1],
        }
        for r in records
        if r["split"] == "test"
    ]
    monkeypatch.setattr(mod, "context", lambda root: (manifest, records))
    monkeypatch.setattr(mod, "locked_policy", lambda root: policy)
    monkeypatch.setattr(mod, "compose", lambda root, rows: predictions)
    monkeypatch.setattr(mod, "panels", lambda *args: None)
    mod.write(out / "lineage.json", lineage)
    crop_scores = np.array([[0.9, 0.1], [0.1, 0.9]] * 3)
    np.savez(out / "features.npz", zero_shot=crop_scores)
    for name in ("adapted", "siglip", "rgb", "majority"):
        np.save(out / f"{name}.npy", crop_scores)
    return out


def _read_all(out, names):
    return {name: (out / name).read_bytes() for name in names}


def test_m5_controls_off_reproduce_defaults_and_on_only_add_learner_rows(tmp_path, monkeypatch):
    out = _fixture(tmp_path, monkeypatch, run)
    run.evaluate(tmp_path)
    canonical = _read_all(out, CANONICAL_FILES)
    run.activity(tmp_path)
    default = _read_all(out, ACTIVITY_FILES)
    run.write(tmp_path / run.ACTIVITY_CONTROLS, {"display_threshold": None, "review_margin": None})
    run.activity(tmp_path)
    assert _read_all(out, ACTIVITY_FILES) == default, "controls off must reproduce the default outputs"
    run.write(tmp_path / run.ACTIVITY_CONTROLS, {"display_threshold": 0.35, "review_margin": 0.5})
    run.activity(tmp_path)
    now = _read_all(out, ACTIVITY_FILES)
    assert now["activity_paired_counts.csv"] == default["activity_paired_counts.csv"]
    thresholds, margins = _rows(now["activity_thresholds.csv"]), _rows(now["activity_review_margins.csv"])
    assert [r["role"] for r in thresholds] == ["lower", "canonical", "higher", "learner"]
    assert thresholds[:3] == _rows(default["activity_thresholds.csv"])
    assert thresholds[3]["display_threshold"] == "0.35"
    assert margins[:-1] == _rows(default["activity_review_margins.csv"])
    assert margins[-1]["role"] == "learner (stricter locked)" and margins[-1]["review_margin"] == "0.5"
    # 0.5 lies between curve margins 0.4 and 0.7, so it accepts exactly the validation crops of 0.7.
    assert margins[-1]["validation_coverage"] == "0.5"
    assert _read_all(out, CANONICAL_FILES) == canonical


@pytest.mark.parametrize(
    "bad",
    [
        {"display_threshold": 0.33},
        {"display_threshold": 1.0},
        {"display_threshold": True},
        {"review_margin": 1.5},
        {"other": 1},
    ],
)
def test_m5_invalid_controls_are_refused(tmp_path, bad):
    run.write(tmp_path / run.ACTIVITY_CONTROLS, bad)
    with pytest.raises(ValueError):
        run.activity_controls(tmp_path)


def test_m5_notebook_explains_controls_before_the_cell():
    sources = _sources()
    index = next(i for i, s in enumerate(sources) if "run('activity')" in s)
    activity = sources[index]
    assert "USE_DISPLAY_THRESHOLD = False  # @param" in activity
    assert "USE_REVIEW_MARGIN = False  # @param" in activity
    assert activity.index("activity_controls.json") < activity.index("run('activity')")
    explanation = sources[index - 1]
    assert "off by default" in explanation and "**What to notice:**" in explanation


def test_referral_report_cells_do_not_leak_completed_process():
    lines = [line for s in _sources() for line in s.splitlines() if "referral_report" in line]
    assert len(lines) == 2 and all(line.startswith("_ = checked(") for line in lines)


def test_cx4_training_median_baseline_is_fitted_on_training_and_bootstrapped(tmp_path, monkeypatch):
    out = _fixture(tmp_path, monkeypatch, run)
    run.evaluate(tmp_path)
    _, records = run.context(tmp_path)
    train = [r for r in records if r["split"] == "train"]
    test = [r for r in records if r["split"] == "test"]
    median = np.median(run.count_reference(train), axis=0)
    expected = run.core.count_metrics(run.count_reference(test), np.tile(median, (len(test), 1)))
    metrics = run.read(out / "metrics.json")
    assert metrics["training_median"] == expected
    boot = run.read(out / "bootstrap.json")["comparisons"]["detector_adapted_minus_training_median"]
    assert boot["difference"] == pytest.approx(metrics["detector_adapted"]["mae"] - expected["mae"])
    assert "training_median" in _rows((out / "counts.csv").read_bytes())[0]
    assert "training-median" in _section(7) and "training-median" in _section(10)
    assert "stronger among those evaluated" not in "\n".join(_sources())


def test_previous_runtime_outputs_are_reproduced(tmp_path, monkeypatch):
    """Against the T4-recorded runtime: existing numbers and default activity outputs are unchanged."""
    try:
        source = subprocess.run(
            ["git", "show", f"{PREVIOUS_RUNTIME}:tools/rice_capstone.py"],
            cwd=ROOT,
            check=True,
            capture_output=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError):
        pytest.skip("previous runtime not in local git history")
    path = tmp_path / "rice_capstone_previous.py"
    path.write_bytes(source)
    spec = importlib.util.spec_from_file_location("rice_capstone_previous", path)
    old = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(old)
    results = {}
    for name, mod in (("old", old), ("new", run)):
        root = tmp_path / name
        out = _fixture(root, monkeypatch, mod)
        mod.evaluate(root)
        mod.activity(root)
        results[name] = (
            run.read(out / "metrics.json"),
            run.read(out / "bootstrap.json"),
            _read_all(out, ACTIVITY_FILES + ("counts.csv",)),
        )
    old_metrics, old_boot, old_files = results["old"]
    new_metrics, new_boot, new_files = results["new"]
    assert {k: new_files[k] for k in ACTIVITY_FILES} == {k: old_files[k] for k in ACTIVITY_FILES}
    assert {k: v for k, v in new_metrics.items() if k != "training_median"} == old_metrics
    new_boot["comparisons"].pop("detector_adapted_minus_training_median")
    assert new_boot == old_boot
    old_counts, new_counts = _rows(old_files["counts.csv"]), _rows(new_files["counts.csv"])
    assert [{k: v for k, v in r.items() if k != "training_median"} for r in new_counts] == old_counts
