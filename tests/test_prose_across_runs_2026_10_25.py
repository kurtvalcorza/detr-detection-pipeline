"""Prose numbers stay true across every recorded hosted run (Kurt's ruling, 2026-10-25).

Runs: Kaggle T4 of blob a98705ed (2026-09-25);
Colab T4 of blobs 6b2e5fc3 (2026-10-04) and da6b14bd (2026-10-25).
"""
import json
from pathlib import Path

NB = Path(__file__).resolve().parents[1] / "tutorials" / "detr_detection_colab.ipynb"


def _markdown():
    nb = json.loads(NB.read_text(encoding="utf-8"))
    return "\n".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "markdown")


def test_single_run_values_are_gone():
    md = _markdown()
    for stale in ("At the adapted threshold the same run found",
                  "With every score packed between about 0.15 and 0.31",
                  "earlier revisions gave 0.27–0.32 and 0.22–0.29", "**What the recorded run shows.**",
                  "precision and recall all 0.5538), and at that threshold 15 held-out boxes came back"):
        assert stale not in md, stale


def test_values_that_moved_are_given_per_run():
    md = _markdown()
    assert "`ap50` rose from 0.0172 to 0.3851, 0.3594 and 0.3617" in md
    assert "**1, 2 and 2 of the 5** reference signs" in md
    assert "0.2232 in the second (F1 0.629, precision 0.661, recall 0.6)" in md
    assert "ended between 0.5602 and 0.6244" in md
