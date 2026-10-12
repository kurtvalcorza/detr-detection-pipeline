"""Regressions for the 2026-10-12 capstone review (RPS-M1, RPS-m1..m10).

Static notebook checks plus NumPy-only helper checks. No model is loaded; nothing here is
hosted or pretrained-inference evidence.
"""

import ast
import inspect
import json
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import build_rice_capstone as build  # noqa: E402
import rice_capstone as run  # noqa: E402
import rice_core as core  # noqa: E402

NOTEBOOK = ROOT / "tutorials" / build.NAME


def _notebook():
    return json.loads(NOTEBOOK.read_text(encoding="utf-8"))


def _markdown():
    return ["".join(c["source"]) for c in _notebook()["cells"] if c["cell_type"] == "markdown"]


def _section(number):
    cells = _markdown()
    start = next(i for i, s in enumerate(cells) if s.startswith(f"## {number}."))
    out = [cells[start]]
    for text in cells[start + 1 :]:
        if text.startswith("## "):
            break
        out.append(text)
    return "\n".join(out)


def test_rps_M1_unmatched_detections_are_not_called_detector_errors():
    section7 = _section(7)
    assert "no matching published reference box" in section7
    assert "unannotated target insect" in section7 and "reference_completeness" in section7
    section10 = _section(10)
    assert "unannotated target insects" in section10 and "completeness is unresolved" in section10
    glossary = next(s for s in _markdown() if s.startswith("### Glossary"))
    assert "not a confirmed detector error" in glossary


def test_rps_m1_section3_is_a_fixed_prompt_zero_shot_comparison():
    section3 = _section(3)
    assert "zero-shot" in section3 and "fixed common-name prompts" in section3
    assert "biodiversity representation outperform" not in section3


def test_rps_m2_evaluate_bootstraps_against_zero_count_without_moving_the_original_interval():
    source = inspect.getsource(run.evaluate)
    call = r'paired_group_bootstrap\(\s*truth, systems\["detector_adapted"\], systems\["zero_count"\]'
    assert re.search(call, source)
    original = source.index('systems["detector_adapted"], baseline')
    assert original < source.index('systems["zero_count"], groups')
    rng = np.random.default_rng(0)
    truth = rng.integers(0, 9, (12, 2))
    pred = rng.integers(0, 9, (12, 2))
    groups = [f"g{i % 6}" for i in range(12)]
    first = core.paired_group_bootstrap(truth, pred, np.full((12, 2), 3.0), groups, n_boot=200)
    core.paired_group_bootstrap(truth, pred, np.zeros((12, 2)), groups, n_boot=200)
    assert core.paired_group_bootstrap(truth, pred, np.full((12, 2), 3.0), groups, n_boot=200) == first


def test_rps_m3_policy_values_are_labelled_as_reference_crop_values():
    assert "validation reference-box crops" in _section(6)
    sources = ["".join(c["source"]) for c in _notebook()["cells"]]
    policy = next(s for s in sources if "run('policy')" in s)
    assert "Validation reference-box crops" in policy


def test_rps_m4_m6_m10_opening():
    opening = _markdown()[0]
    assert "**Learning objectives.**" in opening and opening.count("\n1. ") == 1
    assert "pending hosted measurement" not in opening
    assert re.search(r"recorded Colab T4 run.{0,200}1,140 s", opening)
    assert "> **⚠️ Notice.**" in opening and "not a field pest-surveillance" in opening
    assert opening.index("> **⚠️ Notice.**") < opening.index("**Learning objectives.**")
    assert _notebook()["metadata"]["dimer"]["notebook_kind"] == "CAPSTONE"


def test_rps_m5_section8_is_a_comparison_and_predicts_have_worked_answers():
    assert "change one thing" not in _section(8).lower()
    assert sum("<details><summary>One possible answer" in s for s in _markdown()) == 4


def test_rps_m7_record_announces_truncation(capsys, tmp_path):
    tree = ast.parse(build.BOOTSTRAP)
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "record")
    (tmp_path / "outputs").mkdir()
    (tmp_path / "outputs" / "big.json").write_text(json.dumps({"curve": list(range(9000))}), encoding="utf-8")
    (tmp_path / "outputs" / "small.json").write_text(json.dumps({"a": 1}), encoding="utf-8")
    shown = []
    namespace = {"json": json, "ROOT": tmp_path, "FileLink": str, "display": shown.append}
    exec(compile(ast.Module([node], []), "record", "exec"), namespace)
    namespace["record"]("big.json")
    assert "[truncated: showing 18,000 of" in capsys.readouterr().out
    namespace["record"]("small.json")
    assert "truncated" not in capsys.readouterr().out
    assert len(shown) == 2


def test_rps_m8_label_lineage_is_disclosed():
    section2 = _section(2)
    assert "species–sex codes" in section2 and "collapses them into two species" in section2
    assert "640×640" in section2 and "dataset_audit.json" in section2


def test_rps_m9_readable_links_use_a_commit():
    text = "\n".join(_markdown())
    refs = re.findall(r"detr-detection-pipeline/blob/([^/]+)/tools/", text)
    assert refs and all(re.fullmatch(r"[0-9a-f]{40}", ref) for ref in refs)
    assert set(refs) == {build.SOURCE_REVISION}


def test_rps_m9_source_revision_matches_the_carried_files_when_history_is_available():
    """CI checks out one commit, so this skips there; locally it catches a stale SOURCE_REVISION."""
    rev = build.SOURCE_REVISION
    try:
        subprocess.run(
            ["git", "cat-file", "-e", f"{rev}^{{commit}}"], cwd=ROOT, check=True, capture_output=True
        )
    except (OSError, subprocess.CalledProcessError):
        pytest.skip("SOURCE_REVISION not in local git history")
    for name in ("rice_capstone.py", "rice_core.py", "rice_figures.py"):
        pinned = subprocess.run(
            ["git", "show", f"{rev}:tools/{name}"], cwd=ROOT, check=True, capture_output=True
        ).stdout
        current = (ROOT / "tools" / name).read_bytes()
        assert pinned == current, f"{name} differs from {rev}; bump SOURCE_REVISION"
