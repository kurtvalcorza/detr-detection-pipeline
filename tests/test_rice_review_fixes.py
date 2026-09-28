"""Regressions for the PR #2 capstone review (M1, M2, m1-m4).

CPU only. The DETR used here is tiny and randomly initialised: it exercises the pinned
Transformers postprocessor and the runtime's validation boundary, not pretrained inference.
"""

import csv
import json
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import rice_capstone as run  # noqa: E402

NOTEBOOK = ROOT / "tutorials" / "DIMER_Philippine_Rice_Pest_Surveillance_Capstone.ipynb"


def _record(root, image_id="img", size=(48, 40), objects=()):
    data = root / "data"
    data.mkdir(parents=True, exist_ok=True)
    path = data / f"{image_id}.png"
    rng = np.random.default_rng(len(image_id))
    Image.fromarray(rng.integers(0, 255, (size[1], size[0], 3), dtype=np.uint8)).save(path)
    return {
        "image_id": image_id,
        "relative_path": path.name,
        "bytes": path.stat().st_size,
        "sha256": run.core.sha256(path),
        "width": size[0],
        "height": size[1],
        "split": "test",
        "capture_group_id": image_id,
        "attribution": "synthetic fixture",
        "objects": [
            {"object_id": str(k), "bbox_xyxy": box, "species": species, "ignore": False}
            for k, (box, species) in enumerate(objects)
        ],
    }


def _read_csv(path):
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


# --- M1: raw detector outputs are validated before threshold-based postprocessing ---------


@pytest.fixture(scope="module")
def tiny_detr():
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")

    torch.manual_seed(0)
    backbone = transformers.ResNetConfig(
        num_channels=3,
        embedding_size=8,
        hidden_sizes=[8, 16],
        depths=[1, 1],
        layer_type="basic",
        out_features=["stage2"],
    )
    config = transformers.DetrConfig(
        use_timm_backbone=False,
        use_pretrained_backbone=False,
        backbone=None,
        backbone_config=backbone,
        d_model=16,
        encoder_layers=1,
        decoder_layers=1,
        encoder_attention_heads=2,
        decoder_attention_heads=2,
        encoder_ffn_dim=16,
        decoder_ffn_dim=16,
        num_queries=6,
        num_labels=1,
        id2label={0: "target_pest"},
        label2id={"target_pest": 0},
    )
    model = transformers.DetrForObjectDetection(config).eval()
    processor = transformers.DetrImageProcessor(size={"shortest_edge": 64, "longest_edge": 64})
    return model, processor


class _Tampered:
    """Wrap the tiny model and rewrite its raw outputs, as a numerically broken model would."""

    def __init__(self, model, logits=None, boxes=None):
        self.model, self.config = model, model.config
        self.logits, self.boxes = logits, boxes

    def __call__(self, **inputs):
        output = self.model(**inputs)
        if self.logits:
            output.logits = self.logits(output.logits.clone())
        if self.boxes:
            output.pred_boxes = self.boxes(output.pred_boxes.clone())
        return output


def _nan_query(tensor):
    tensor[0, 2, 0] = float("nan")
    return tensor


def _inf_logits(tensor):
    tensor[..., 0] = float("inf")
    return tensor


def _nan_boxes(boxes):
    boxes[..., 1] = float("nan")
    return boxes


@pytest.fixture
def cpu(monkeypatch):
    monkeypatch.setattr(run, "DEVICE", "cpu")


def test_pinned_postprocessor_silently_drops_nan_scores(tiny_detr, tmp_path):
    """Documents the hazard: the upstream filter removes NaN rows instead of failing."""
    import torch

    model, processor = tiny_detr
    record = _record(tmp_path)
    inputs = processor(images=run.image(tmp_path, record), return_tensors="pt")
    with torch.inference_mode():
        output = model(**inputs)
    output.logits = output.logits.clone()
    output.logits[0, 2, 0] = float("nan")
    post = processor.post_process_object_detection(output, threshold=0.0, target_sizes=[(40, 48)])[0]
    assert len(post["scores"]) == 5


def test_finite_detections_keep_every_query(tiny_detr, tmp_path, cpu):
    model, processor = tiny_detr
    record = _record(tmp_path)
    [row] = run.detections(tmp_path, model, processor, [record])
    assert len(row["scores"]) == len(row["boxes"]) == 6
    assert np.isfinite(row["scores"]).all() and np.isfinite(row["boxes"]).all()


@pytest.mark.parametrize(
    "logits, boxes, message",
    [
        (_nan_query, None, "Nonfinite"),
        (lambda t: t * float("nan"), None, "Nonfinite"),
        (_inf_logits, None, "Nonfinite"),
        (None, _nan_boxes, "Nonfinite"),
        (lambda t: t[:, :5], lambda b: b[:, :5], "Malformed"),
        (lambda t: t[..., :1], None, "Malformed"),
        (None, lambda b: b + 2, "outside"),
    ],
)
def test_invalid_raw_outputs_fail_before_postprocessing(tiny_detr, tmp_path, cpu, logits, boxes, message):
    model, processor = tiny_detr
    record = _record(tmp_path)
    with pytest.raises(ValueError, match=message):
        run.detections(tmp_path, _Tampered(model, logits, boxes), processor, [record])


def test_all_below_threshold_is_a_valid_zero_count(tiny_detr, tmp_path, cpu):
    """A finite detector that finds nothing is a result, not an execution failure."""
    model, processor = tiny_detr

    def no_object(tensor):
        tensor[..., 0], tensor[..., 1] = -20.0, 20.0
        return tensor

    record = _record(tmp_path)
    [row] = run.detections(tmp_path, _Tampered(model, no_object), processor, [record])
    assert len(row["scores"]) == 6 and max(row["scores"]) < 1e-6
    row["adapted"] = [[0.5, 0.5]] * 6
    assert run.count_predictions([row], 0.5).tolist() == [[0, 0]]


def test_compose_refuses_nonfinite_species_scores(tiny_detr, tmp_path, cpu, monkeypatch):
    """Final composition (validation, test and reload all use it) fails instead of counting."""
    torch = pytest.importorskip("torch")
    model, processor = tiny_detr

    class FakeBioclip:
        logit_scale = torch.tensor(np.log(10.0))

        def encode_text(self, tokens, normalize=True):
            return torch.eye(2, 768)

    head = {"weight": np.eye(2, 768, dtype=np.float32), "bias": np.zeros(2, np.float32)}
    monkeypatch.setattr(run, "torch_runtime", lambda: torch)
    monkeypatch.setattr(run, "detr_model", lambda root, adapted=False: (model, processor))
    monkeypatch.setattr(run, "bioclip", lambda root: (FakeBioclip(), None, lambda text: torch.zeros(2, 4)))
    monkeypatch.setattr(run, "load_head", lambda root: head)
    record = _record(tmp_path)

    def vectors(value):
        return lambda model, transform, images: np.full((len(images), 768), value, np.float32)

    monkeypatch.setattr(run, "embed", vectors(0.01))
    [row] = run.compose(tmp_path, [record])
    run.validate_species_scores(row["adapted"], len(row["boxes"]))
    monkeypatch.setattr(run, "embed", vectors(np.nan))
    with pytest.raises(ValueError, match="embeddings"):
        run.compose(tmp_path, [record])
    monkeypatch.setattr(run, "embed", lambda model, transform, images: np.zeros((1, 768), np.float32))
    with pytest.raises(ValueError, match="embeddings"):
        run.compose(tmp_path, [record])


def test_species_score_contract():
    run.validate_species_scores([[0.6, 0.4]], 1)
    run.validate_species_scores(np.empty((0, 2)), 0)
    for bad, message in (
        ([[np.nan, 0.5]], "Nonfinite"),
        ([[0.6, 0.6]], "probability"),
        ([[0.6, 0.4, 0.0]], "disagree"),
    ):
        with pytest.raises(ValueError, match=message):
            run.validate_species_scores(bad, 1)


# --- M2: diagnosis evidence is visible and reconciles with the counts ------------------------


def _policy():
    return {"detector_threshold": 0.5, "review_margin": 0.3}


def _prediction(image_id, rows):
    return {
        "image_id": image_id,
        "boxes": [box for box, _, _ in rows],
        "scores": [score for _, score, _ in rows],
        "adapted": [species for _, _, species in rows],
    }


def test_wrong_species_outcome_names_both_species_and_both_scores(tmp_path):
    record = _record(tmp_path, objects=[([4, 4, 20, 20], "rice_black_bug")])
    prediction = _prediction("img", [([4, 4, 20, 20], 0.9, [0.4, 0.6])])
    objects = run.chosen_objects(prediction, _policy())
    [row] = run.object_outcomes(record, objects)
    assert row["outcome"] == "wrong_species"
    assert (row["reference_species"], row["predicted_species"]) == ("rice_black_bug", "white_stemborer")
    assert (row["detector_score"], row["species_score"], row["review"]) == (0.9, 0.6, "REVIEW")
    counts = run.species_reconciliation("img", [row], objects)
    assert [(c["reference"], c["raw"], c["wrong_out"], c["wrong_in"]) for c in counts] == [
        (1, 0, 1, 0),
        (0, 1, 0, 1),
    ]


def test_panels_export_labelled_examples_and_report_absent_categories(tmp_path):
    (tmp_path / "outputs").mkdir()
    records = [
        _record(tmp_path, "wrong", objects=[([4, 4, 20, 20], "rice_black_bug")]),
        _record(tmp_path, "empty", objects=[([4, 4, 20, 20], "white_stemborer")]),
        _record(
            tmp_path,
            "dense",
            size=(200, 200),
            objects=[
                ([x, y, x + 8, y + 8], "rice_black_bug") for x in range(0, 190, 10) for y in (0, 50, 100)
            ],
        ),
    ]
    predictions = [
        _prediction("wrong", [([4, 4, 20, 20], 0.9, [0.2, 0.8])]),
        _prediction("empty", []),
        _prediction(
            "dense",
            [([x, 0, x + 8, 8], 0.8, [0.9, 0.1]) for x in range(0, 190, 10)]
            + [([150, 170, 170, 190], 0.7, [0.9, 0.1])],
        ),
    ]
    run.panels(tmp_path, records, predictions, _policy())
    out = tmp_path / "outputs"
    inventory = run.read(out / "error_panel_inventory.json")
    assert inventory["selected"] == {"wrong_species": "wrong", "misses": "empty", "spurious": "dense"}
    assert inventory["absent_categories"] == ["success"]
    assert sorted(p.name for p in (out / "local_figures").glob("*.png")) == [
        "dense.png",
        "empty.png",
        "wrong.png",
    ]
    rows = _read_csv(out / "error_examples.csv")
    wrong = next(r for r in rows if r["image_id"] == "wrong")
    assert wrong["reference_species"] == "rice_black_bug" and wrong["predicted_species"] == "white_stemborer"
    assert float(wrong["detector_score"]) == 0.9 and float(wrong["species_score"]) == 0.8
    assert {r["outcome"] for r in rows if r["image_id"] == "empty"} == {"missed"}
    counts = _read_csv(out / "error_example_counts.csv")
    dense = [r for r in counts if r["image_id"] == "dense" and r["species"] == "rice_black_bug"][0]
    assert (dense["reference"], dense["raw"], dense["correct"], dense["missed"], dense["spurious"]) == (
        "57",
        "20",
        "19",
        "38",
        "1",
    )


def test_error_summary_and_confusion_reconcile(tmp_path):
    (tmp_path / "outputs").mkdir()
    record = _record(
        tmp_path,
        objects=[([4, 4, 20, 20], "rice_black_bug"), ([24, 4, 40, 20], "white_stemborer")],
    )
    prediction = _prediction("img", [([4, 4, 20, 20], 0.9, [0.1, 0.9]), ([30, 24, 44, 38], 0.9, [0.8, 0.2])])
    run.error_summaries(tmp_path, [record], [prediction], _policy())
    summary = {r["species"]: r for r in _read_csv(tmp_path / "outputs" / "error_summary.csv")}
    assert summary["all"]["reference"] == "2" and summary["all"]["raw"] == "2"
    assert summary["rice_black_bug"]["wrong_out"] == "1" and summary["white_stemborer"]["missed"] == "1"
    confusion = _read_csv(tmp_path / "outputs" / "matched_species_confusion.csv")
    assert confusion[0]["predicted white_stemborer"] == "1"
    assert confusion[1]["not detected (missed)"] == "1"
    assert confusion[2]["predicted rice_black_bug"] == "1"


# --- m1: review fallback follows the specification's accuracy-first rule -------------------


def test_review_fallback_is_accuracy_first_over_all_nonempty_sets():
    y = [0, 0, 0, 0]
    # margins 0.9 (right), 0.7 (wrong), 0.3 (right), 0.1 (wrong)
    probabilities = [[0.95, 0.05], [0.15, 0.85], [0.65, 0.35], [0.45, 0.55]]
    selected = run.review_policy(y, probabilities)
    assert not selected["target_met"] and selected["selection_rule"].startswith("fallback")
    assert (selected["threshold"], selected["coverage"], selected["accuracy"]) == pytest.approx(
        (0.9, 0.25, 1.0)
    )
    met = run.review_policy([0, 0, 0, 1], [[0.9, 0.1], [0.8, 0.2], [0.7, 0.3], [0.2, 0.8]])
    assert met["target_met"] and met["coverage"] == 1 and met["selection_rule"].startswith("target")


# --- m2 / M2 / m3: learner-facing notebook structure ---------------------------------------


def test_notebook_surfaces_diagnosis_and_hides_carrier():
    cells = json.loads(NOTEBOOK.read_text(encoding="utf-8"))["cells"]
    sources = ["".join(c["source"]) for c in cells]
    carrier = next(c for c, s in zip(cells, sources, strict=True) if s.startswith("# @title Carrier"))
    assert carrier["metadata"]["cellView"] == "form" and carrier["metadata"]["jupyter"]["source_hidden"]
    assert any(s.startswith("### Glossary") for s in sources)
    features = next(s for s in sources if "run('features')" in s)
    assert features.index("figures('crops')") < features.index("run('features')")
    evaluate = next(s for s in sources if "run('evaluate')" in s)
    for name in (
        "error_summary.csv",
        "matched_species_confusion.csv",
        "error_panel_inventory.json",
        "error_examples.csv",
        "error_example_counts.csv",
    ):
        assert name in evaluate
    assert "activity_paired_counts.csv" in next(s for s in sources if "run('activity')" in s)
    assert "review_selection_rule" in next(s for s in sources if "run('policy')" in s)
    prepare = next(s for s in sources if "run('prepare')" in s)
    assert "sample_summary.csv" in prepare and "audit_gates.csv" in prepare
