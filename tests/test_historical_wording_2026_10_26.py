"""Run-derived prose is a record of named runs, never an expectation (Kurt's ruling, 2026-10-26).

Runs: Kaggle T4 of blob a98705ed (2026-09-25); Colab T4 of blobs 6b2e5fc3 (2026-10-04),
da6b14bd and 712d22d5 (2026-10-25).
"""
import json
from pathlib import Path

NB = Path(__file__).resolve().parents[1] / "tutorials" / "detr_detection_colab.ipynb"


def _markdown():
    nb = json.loads(NB.read_text(encoding="utf-8"))
    return "\n".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "markdown")


def test_run_status_and_expectations_are_gone():
    md = _markdown()
    for stale in ("Hosted runs of this revision are pending", "so expect your count to differ",
                  "ended between 0.5602 and 0.6103", "in both Colab runs"):
        assert stale not in md, stale


def test_confirming_run_of_712d22d5_is_recorded():
    md = _markdown()
    assert "(0.6103, 0.6078 and 0.6244 in the three Colab runs)" in md
    assert "between 136.6 s and 171.5 s of wall time" in md
    assert "`ap` \nto" not in md and "0.2983, 0.2993 and 0.3093" in md
    assert "**1, 2 and 2 of the 5** reference signs" in md
