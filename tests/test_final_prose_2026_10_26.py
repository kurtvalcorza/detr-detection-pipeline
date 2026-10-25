"""Final prose pass (Kurt's ruling, 2026-10-26): the flagged expectation sentences are removed or
rewritten as records of named runs. detr: Section 7 baseline and Section 14 worked answer.
Fails on PR head a5ed14a.
"""
import json
from pathlib import Path

NB = Path(__file__).resolve().parents[1] / 'tutorials/detr_detection_colab.ipynb'


def _markdown():
    nb = json.loads(NB.read_text(encoding="utf-8"))
    return "\n".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "markdown")


GONE = ['expected to be near zero', 'Near zero is the expected floor', 'Expect `returned_at_threshold` to stay 0']
PRESENT = ['**The baseline is the floor.**']


def test_flagged_expectations_are_gone():
    md = _markdown()
    for text in GONE:
        assert text not in md, text


def test_replacement_records_are_present():
    md = _markdown()
    for text in PRESENT:
        assert text in md, text
