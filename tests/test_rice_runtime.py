"""CPU scientific/software integration; no model checkpoint is downloaded."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import rice_capstone as run  # noqa: E402


def test_head_training_includes_epoch_zero_and_selects_validation_minimum():
    pytest.importorskip("torch")
    x = np.eye(768, dtype=np.float32)[[0, 1, 0, 1, 0, 1]]
    text = np.eye(768, dtype=np.float32)[:2]
    tensors, selected, history = run.fit_head(
        x, np.array([0, 1, 0, 1, 0, 1]), text, 2.0, [0, 1, 2, 3], [4, 5], epochs=3
    )
    assert len(history) == 4 and history[0]["epoch"] == 0
    assert selected == min(history, key=lambda r: r["validation_cross_entropy"])["epoch"]
    assert tensors["weight"].shape == (2, 768)
    assert all(np.isfinite(r["training_cross_entropy"]) for r in history)
    zero, selected, _ = run.fit_head(
        x, np.array([0, 1, 0, 1, 0, 1]), text, 2.0, [0, 1, 2, 3], [4, 5], epochs=0
    )
    assert selected == 0
    np.testing.assert_allclose(zero["weight"].numpy(), text * 2)


def test_coco_evaluator_perfect_empty_and_no_references():
    pytest.importorskip("pycocotools")
    records = [
        {
            "image_id": "a",
            "width": 32,
            "height": 32,
            "objects": [{"bbox_xyxy": [2, 2, 20, 20], "ignore": False}],
        }
    ]
    perfect = [{"image_id": "a", "boxes": [[2, 2, 20, 20]], "scores": [0.9]}]
    assert run.coco_metrics(records, perfect)["ap50"] == pytest.approx(1)
    assert run.coco_metrics(records, [{"image_id": "a", "boxes": [], "scores": []}])["ap50"] == 0
    assert run.coco_metrics([{**records[0], "objects": []}], perfect)["ap50"] is None


def test_head_loader_refuses_class_order_and_tensor_tampering(tmp_path):
    from safetensors.numpy import save_file

    folder = tmp_path / "outputs" / "classifier_adapter"
    folder.mkdir(parents=True)
    save_file(
        {"weight": np.zeros((2, 768), np.float32), "bias": np.zeros(2, np.float32)},
        str(folder / "head.safetensors"),
    )
    manifest = {
        "format": "dimer-rice-frozen-head-v1",
        "classes": list(run.SPECIES),
        "base": list(run.PINS["bioclip"]),
        "preprocessing": "pinned-bioclIP-config-normalized-features",
        "sha256": run.core.sha256(folder / "head.safetensors"),
    }
    run.write(folder / "manifest.json", manifest)
    assert run.load_head(tmp_path)["weight"].shape == (2, 768)
    run.write(folder / "manifest.json", {**manifest, "classes": list(reversed(run.SPECIES))})
    with pytest.raises(ValueError, match="identity"):
        run.load_head(tmp_path)
    run.write(folder / "manifest.json", manifest)
    (folder / "head.safetensors").write_bytes(b"changed")
    with pytest.raises(ValueError, match="digest"):
        run.load_head(tmp_path)


def test_count_threshold_keeps_referred_detections_in_raw_count():
    predictions = [{"scores": [0.9, 0.8, 0.1], "adapted": [[0.51, 0.49], [0.1, 0.9], [0.9, 0.1]]}]
    assert run.count_predictions(predictions, 0.5).tolist() == [[1, 1]]
    assert run.count_predictions(predictions, 0.95).tolist() == [[0, 0]]


def test_manifest_audit_binds_exact_bytes(tmp_path):
    root = Path(__file__).resolve().parents[1]
    for target, source in (
        ("data_manifest.json", "rice_data.json"),
        ("dataset_audit.json", "rice_dataset_audit.json"),
    ):
        (tmp_path / target).write_bytes((root / "tools" / source).read_bytes())
    manifest, records = run.context(tmp_path)
    assert len(records) == 200 and manifest["scope"] == "exploratory_published_annotations"
    path = tmp_path / "data_manifest.json"
    path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(ValueError, match="exact data manifest"):
        run.context(tmp_path)


def test_synthetic_evaluation_activity_report_and_csv_tamper(tmp_path, monkeypatch):
    """Exercise exports with artificial predictions; this is not model evidence."""
    import csv
    import zipfile

    out = tmp_path / "outputs"
    out.mkdir()
    records = [
        {
            "image_id": str(i),
            "width": 32,
            "height": 32,
            "split": "train" if i < 2 else "test",
            "capture_group_id": str(i),
            "objects": [
                {
                    "object_id": str(i),
                    "bbox_xyxy": [2, 2, 20, 20],
                    "species": run.SPECIES[i % 2],
                    "ignore": False,
                }
            ],
        }
        for i in range(4)
    ]
    lineage = [
        {"crop_id": str(i), "parent_image_id": str(i), "species": run.SPECIES[i % 2], "split": r["split"]}
        for i, r in enumerate(records)
    ]
    manifest = {"scope": "exploratory_published_annotations", "records": records}
    for name in ("data_manifest.json", "dataset_audit.json", "model_manifest.json"):
        run.write(tmp_path / name, manifest if name == "data_manifest.json" else {"test_fixture": True})
    monkeypatch.setattr(run, "context", lambda root: (manifest, records))
    policy = {"detector_threshold": 0.5, "review_margin": 0.2, "digest": "test_fixture"}
    monkeypatch.setattr(run, "locked_policy", lambda root: policy)
    predictions = [
        {
            "image_id": str(i),
            "boxes": [[2, 2, 20, 20]],
            "scores": [0.9],
            "adapted": [[0.9, 0.1] if i % 2 == 0 else [0.1, 0.9]],
            "zero_shot": [[0.9, 0.1] if i % 2 == 0 else [0.1, 0.9]],
        }
        for i in (2, 3)
    ]
    monkeypatch.setattr(run, "compose", lambda root, rows: predictions)
    monkeypatch.setattr(run, "panels", lambda *args: None)
    run.write(out / "lineage.json", lineage)
    scores = np.array([[0.9, 0.1], [0.1, 0.9], [0.9, 0.1], [0.1, 0.9]])
    np.savez(out / "features.npz", zero_shot=scores)
    for name in ("adapted", "siglip", "rgb", "majority"):
        np.save(out / f"{name}.npy", scores)
    run.evaluate(tmp_path)
    assert run.read(out / "metrics.json")["detector_adapted"]["mae"] == 0
    with (out / "error_summary.csv").open(encoding="utf-8", newline="") as handle:
        summary = {row["species"]: row for row in csv.DictReader(handle)}
    assert summary["all"]["correct"] == "2" and summary["all"]["missed"] == "0"
    run.activity(tmp_path)
    assert policy["detector_threshold"] == 0.5
    with (out / "activity_thresholds.csv").open(encoding="utf-8", newline="") as handle:
        activity_rows = list(csv.DictReader(handle))
    assert [r["role"] for r in activity_rows] == ["lower", "canonical", "higher"]
    assert all(r["wrong_species"] == "0" for r in activity_rows)
    with (out / "activity_paired_counts.csv").open(encoding="utf-8", newline="") as handle:
        paired = list(csv.DictReader(handle))
    assert len(paired) == 6 and {r["image_id"] for r in paired} == {"2"}
    # Metric charts join the archive; photo previews never do.
    (out / "figures").mkdir()
    for name in ("evaluate.png", "prepare.png", "crops.png"):
        (out / "figures" / name).write_bytes(b"png fixture")
    # A clearly marked fixture, not a substitute for actual fresh-process inference.
    run.write(out / "verification.json", {"fresh_process": True, "test_fixture": True})
    run.report(tmp_path)
    assert run.read(out / "run_summary.json")["verification"]["csv_metric_parity"]
    with zipfile.ZipFile(out / "results.zip") as archive:
        assert archive.testzip() is None and "DATA_LICENSE.txt" in archive.namelist()
        assert "figures/evaluate.png" in archive.namelist()
        assert not {"figures/prepare.png", "figures/crops.png"} & set(archive.namelist())
    assert run.read(out / "run_summary.json")["charts_included"] == ["figures/evaluate.png"]
    with (out / "counts.csv").open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    rows[0]["raw"] = 999
    run.csv_write(out / "counts.csv", rows)
    with pytest.raises(ValueError, match="parity"):
        run.report(tmp_path)
