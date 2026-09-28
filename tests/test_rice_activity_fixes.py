"""Regressions for the two observations from the Colab T4 run of bfa45cc (display-only activity fixes)."""

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import rice_capstone as run  # noqa: E402

NOTEBOOK = ROOT / "tutorials" / "DIMER_Philippine_Rice_Pest_Surveillance_Capstone.ipynb"
RBB, WSB = run.SPECIES


@pytest.mark.parametrize(
    ("locked", "expected"),
    [(0.9, [0.8, 0.9, 0.95]), (0.95, [0.85, 0.95]), (0.5, [0.4, 0.5, 0.6]), (0.05, [0.05, 0.15])],
)
def test_display_thresholds_stay_inside_the_validation_grid(locked, expected):
    thresholds = run.display_thresholds(locked)
    assert thresholds == expected
    assert max(thresholds) < 1.0, "a threshold of 1.0 drops every box"


def _fixture():
    """One held-out image: confident correct, uncertain wrong-species and uncertain spurious boxes."""
    record = {
        "image_id": "img",
        "split": "test",
        "objects": [
            {"object_id": "a", "bbox_xyxy": [0, 0, 10, 10], "species": RBB, "ignore": False},
            {"object_id": "b", "bbox_xyxy": [20, 20, 30, 30], "species": WSB, "ignore": False},
        ],
    }
    prediction = {
        "boxes": [[0, 0, 10, 10], [20, 20, 30, 30], [40, 40, 50, 50]],
        "scores": [0.95, 0.95, 0.95],
        "adapted": [[0.95, 0.05], [0.55, 0.45], [0.4, 0.6]],
    }
    curve = [
        {"threshold": 0.0, "coverage": 1.0, "accuracy": 0.9},
        {"threshold": 0.01, "coverage": 1.0, "accuracy": 0.9},
        {"threshold": 0.15, "coverage": 0.9, "accuracy": 0.95},
        {"threshold": 0.3, "coverage": 0.8, "accuracy": 0.97},
        {"threshold": 0.9, "coverage": 0.1, "accuracy": 1.0},
    ]
    policy = {
        "detector_threshold": 0.9,
        "review_margin": 0.01,
        "review_validation": {"coverage": 1.0, "accuracy": 0.9, "curve": curve},
    }
    return record, prediction, policy


def test_review_margin_rows_show_what_stricter_margins_would_refer():
    record, prediction, policy = _fixture()
    rows = run.review_margin_rows([record], [prediction], policy)
    roles = ["canonical", "illustrative ~90% coverage", "illustrative ~80% coverage"]
    assert [r["role"] for r in rows] == roles
    assert [r["review_margin"] for r in rows] == [0.01, 0.15, 0.3]
    canonical, ninety, eighty = rows
    assert canonical["test_referred"] == 0 and canonical["test_retained"] == 3
    # Margin 0.15 refers the wrong-species box (margin 0.1); 0.3 also refers the spurious box (0.2).
    fields = ("referred_wrong_species", "referred_spurious", "referred_correct")
    assert tuple(ninety[f] for f in fields) == (1, 0, 0)
    assert tuple(eighty[f] for f in fields) == (1, 1, 0)
    assert policy["review_margin"] == 0.01, "the locked margin is never changed"


def test_review_margin_rows_without_a_curve_keep_only_the_locked_margin():
    record, prediction, policy = _fixture()
    rows = run.review_margin_rows([record], [prediction], {**policy, "review_validation": {}})
    assert [r["role"] for r in rows] == ["canonical"]
    assert rows[0]["validation_coverage"] == ""


def test_notebook_explains_zero_referral_and_shows_the_margin_table():
    cells = json.loads(NOTEBOOK.read_text(encoding="utf-8"))["cells"]
    sources = ["".join(c["source"]) for c in cells]
    policy_cell = next(s for s in sources if "run('policy')" in s)
    assert "review_validation_coverage'] >= 1.0" in policy_cell and "refers nothing" in policy_cell
    activity_cell = next(s for s in sources if "run('activity')" in s)
    assert "table('activity_review_margins.csv')" in activity_cell
    assert any("0.05–0.95" in s and "zero-count baseline" in s for s in sources)
