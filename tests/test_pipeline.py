import hashlib
import json
import re
from pathlib import Path
from typing import Any

import pytest
from PIL import Image

from detr_detection_pipeline import (
    ARTIFACT_FORMAT,
    DEFAULT_WEIGHTS_DIR,
    DETECTION_THRESHOLD,
    LABELS,
    MAX_DETECTIONS,
    MODEL_ID,
    MODEL_KEY,
    SIGN_CLASSES,
    UNANNOTATED_LABEL_IDS,
    DetrDetectionPipeline,
    average_precision,
    box_iou,
    is_pinned,
    read_detection_records,
    sign_dataset,
    split_dataset,
    stage_missing_files,
    validate_dataset,
    verify_snapshot,
)
from detr_detection_pipeline import pipeline as pipeline_module

HEX40 = re.compile(r"^[0-9a-f]{40}$")
REPO = Path(__file__).resolve().parents[1]
SNAPSHOT = REPO / "weights" / MODEL_KEY


def test_identity_constants_agree_with_the_committed_manifest():
    manifest = json.loads((SNAPSHOT / "dimer-base-manifest.json").read_text(encoding="utf-8"))
    assert MODEL_ID == "facebook/detr-resnet-50" == manifest["modelId"]
    assert manifest["revision"] == pipeline_module.MODEL_REVISION
    assert is_pinned() == bool(HEX40.match(pipeline_module.MODEL_REVISION))
    assert DEFAULT_WEIGHTS_DIR == SNAPSHOT
    assert ARTIFACT_FORMAT == "detr-adapter-v1"
    assert manifest["totalBytes"] == sum(entry["bytes"] for entry in manifest["files"])
    assert {entry["path"] for entry in manifest["files"]} == {
        "README.md",
        "config.json",
        "model.safetensors",
        "preprocessor_config.json",
    }
    if not is_pinned():
        assert all(entry["sha256"] is None for entry in manifest["files"])


def test_labels_equal_the_snapshot_id2label_order():
    config = json.loads((SNAPSHOT / "config.json").read_text(encoding="utf-8"))
    id2label = {int(k): v for k, v in config["id2label"].items()}
    assert list(LABELS) == [id2label[i] for i in range(len(id2label))]
    assert len(LABELS) == 91 and config["num_queries"] == MAX_DETECTIONS == 100
    assert [LABELS[i] for i in UNANNOTATED_LABEL_IDS] == [
        "N/A",
        "street sign",
        "hat",
        "shoe",
        "eye glasses",
        "plate",
        "mirror",
        "window",
        "desk",
        "door",
        "blender",
    ]
    assert len(LABELS) - len(UNANNOTATED_LABEL_IDS) == 80
    assert 0 < DETECTION_THRESHOLD < 1


def test_unpinned_package_refuses_every_weight_operation(tmp_path, monkeypatch):
    monkeypatch.setattr(pipeline_module, "MODEL_REVISION", "unpinned")
    for call in (
        lambda: verify_snapshot(tmp_path),
        lambda: stage_missing_files(tmp_path, allow_download=True, downloader=lambda *_: None),
        lambda: DetrDetectionPipeline.from_pretrained(weights_dir=tmp_path),
    ):
        with pytest.raises(RuntimeError, match="pin_snapshot.py"):
            call()


def _write_snapshot(
    root: Path, revision: str, content: bytes, sha: str | None = None, size: int | None = None
) -> None:
    (root / "config.json").write_bytes(content)
    manifest = {
        "modelId": MODEL_ID,
        "revision": revision,
        "files": [
            {
                "path": "config.json",
                "bytes": len(content) if size is None else size,
                "sha256": hashlib.sha256(content).hexdigest() if sha is None else sha,
            }
        ],
        "totalBytes": len(content),
    }
    (root / "dimer-base-manifest.json").write_text(json.dumps(manifest), encoding="utf-8")


def test_verify_snapshot_accepts_matching_manifest(tmp_path, pinned):
    _write_snapshot(tmp_path, pinned, b'{"model_type": "detr"}')
    info = verify_snapshot(tmp_path)
    assert info["revision"] == pinned and info["files"] == 1


def test_verify_snapshot_rejects_tampered_digest(tmp_path, pinned):
    content = b'{"model_type": "detr"}'
    good = hashlib.sha256(content).hexdigest()
    flipped = ("0" if good[0] != "0" else "1") + good[1:]
    _write_snapshot(tmp_path, pinned, content, sha=flipped)
    with pytest.raises(ValueError, match="sha256"):
        verify_snapshot(tmp_path)


def test_verify_snapshot_rejects_missing_digest_size_file_and_revision(tmp_path, pinned):
    _write_snapshot(tmp_path, pinned, b"abc")
    manifest = json.loads((tmp_path / "dimer-base-manifest.json").read_text())
    manifest["files"][0]["sha256"] = None
    (tmp_path / "dimer-base-manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="no sha256"):
        verify_snapshot(tmp_path)
    _write_snapshot(tmp_path, pinned, b"abc", size=99)
    with pytest.raises(ValueError, match="size"):
        verify_snapshot(tmp_path)
    _write_snapshot(tmp_path, "f" * 40, b"abc")
    with pytest.raises(ValueError, match="revision"):
        verify_snapshot(tmp_path)
    _write_snapshot(tmp_path, pinned, b"abc")
    (tmp_path / "config.json").unlink()
    with pytest.raises(FileNotFoundError):
        verify_snapshot(tmp_path)


def test_stage_missing_files_fetches_only_absent_entries_then_verifies(tmp_path, pinned):
    payload = b"weights-bytes"
    (tmp_path / "config.json").write_bytes(b"{}")
    manifest = {
        "modelId": MODEL_ID,
        "revision": pinned,
        "files": [
            {"path": "config.json", "bytes": 2, "sha256": hashlib.sha256(b"{}").hexdigest()},
            {
                "path": "model.safetensors",
                "bytes": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest(),
            },
        ],
    }
    (tmp_path / "dimer-base-manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(FileNotFoundError, match="allow_download=True"):
        stage_missing_files(tmp_path)
    fetched = []

    def fake_download(relative_path, root):
        fetched.append(relative_path)
        (root / relative_path).write_bytes(payload)

    assert stage_missing_files(tmp_path, allow_download=True, downloader=fake_download) == [
        "model.safetensors"
    ]
    assert fetched == ["model.safetensors"]
    assert verify_snapshot(tmp_path)["files"] == 2
    assert stage_missing_files(tmp_path, allow_download=True, downloader=fake_download) == []


def test_stage_missing_files_refuses_foreign_manifest(tmp_path, pinned):
    manifest = {"modelId": "someone/else", "revision": pinned, "files": []}
    (tmp_path / "dimer-base-manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="refusing to stage"):
        stage_missing_files(tmp_path, allow_download=True, downloader=lambda *_: None)


class MockPipeline(DetrDetectionPipeline):
    """Pipeline whose backend is a fixed function, for fast offline tests of the public contract."""

    def __init__(self, runner=None, class_names=LABELS):
        self.runner = runner
        super().__init__(
            model=None, processor=None, device="cpu", class_names=tuple(class_names), source="mock"
        )

    def _run(self, image: Image.Image, threshold: float) -> list[dict[str, Any]]:
        if self.runner:
            return self.runner(image, threshold)
        return [
            {"box": [1.0, 2.0, 10.0, 20.0], "label": "clock", "score": 0.91},
            {"box": [5.0, 5.0, 30.0, 30.0], "label": "stop sign", "score": 0.97},
        ]


def test_detect_sorts_by_score_and_records_identity():
    result = MockPipeline().detect(Image.new("RGB", (64, 48)), threshold=0.5)
    assert [d["label"] for d in result["detections"]] == ["stop sign", "clock"]
    assert (result["width"], result["height"], result["threshold"]) == (64, 48, 0.5)
    assert result["model_id"] == MODEL_ID and result["adapted"] is False


def test_detect_rejects_malformed_backend_output():
    bad = MockPipeline(runner=lambda *_: [{"box": [0, 0, 1], "label": "clock", "score": 0.5}])
    with pytest.raises(RuntimeError, match="malformed"):
        bad.detect(Image.new("RGB", (64, 64)))
    unknown = MockPipeline(runner=lambda *_: [{"box": [0, 0, 1, 1], "label": "forklift", "score": 0.5}])
    with pytest.raises(RuntimeError, match="malformed"):
        unknown.detect(Image.new("RGB", (64, 64)))
    flood = MockPipeline(runner=lambda *_: [{"box": [0, 0, 1, 1], "label": "clock", "score": 0.5}] * 101)
    with pytest.raises(RuntimeError, match="num_queries"):
        flood.detect(Image.new("RGB", (64, 64)))


def test_save_artifact_refuses_an_unadapted_pipeline(tmp_path):
    with pytest.raises(RuntimeError, match="not been fine-tuned"):
        MockPipeline().save_artifact(tmp_path / "adapter.safetensors")


def test_box_iou_and_average_precision():
    assert box_iou([0, 0, 10, 10], [0, 0, 10, 10]) == pytest.approx(1.0)
    assert box_iou([0, 0, 10, 10], [5, 0, 15, 10]) == pytest.approx(50 / 150)
    assert box_iou([0, 0, 1, 1], [2, 2, 3, 3]) == 0.0
    with pytest.raises(ValueError):
        box_iou([0, 0, 1], [0, 0, 1, 1])
    references = [
        {"boxes": [[0, 0, 10, 10]], "labels": ["a"]},
        {"boxes": [[20, 20, 30, 30]], "labels": ["b"]},
    ]
    perfect = [
        [{"box": [0, 0, 10, 10], "label": "a", "score": 0.9}],
        [{"box": [20, 20, 30, 30], "label": "b", "score": 0.8}],
    ]
    assert average_precision(perfect, references, ["a", "b"])["ap"] == pytest.approx(1.0)
    empty = average_precision([[], []], references, ["a", "b"])
    assert empty["ap"] == 0.0 and empty["n_references"] == 2
    wrong_label = [[{"box": [0, 0, 10, 10], "label": "b", "score": 0.9}], []]
    assert average_precision(wrong_label, references, ["a", "b"])["ap50"] == 0.0


def test_validate_dataset_accepts_samples_and_reports_absent_classes():
    records = sign_dataset(6, seed=3)
    manifest = validate_dataset(records, SIGN_CLASSES, epochs=2)
    assert manifest["verdict"] == "accepted" and manifest["n_records"] == 6
    assert sum(manifest["boxes_per_class"].values()) == manifest["n_boxes"]
    manifest = validate_dataset(records, (*SIGN_CLASSES, "never-drawn"), epochs=2)
    assert any("never-drawn" in finding for finding in manifest["findings"])


@pytest.mark.parametrize(
    ("record", "message"),
    [
        ({"image": "x", "boxes": [], "labels": []}, "PIL.Image.Image"),
        (
            {"image": Image.new("RGB", (64, 64)), "boxes": [[0, 0, 1, 1]], "labels": []},
            "1 boxes but 0 labels",
        ),
        (
            {"image": Image.new("RGB", (64, 64)), "boxes": [[0, 0, 99, 10]], "labels": ["stop-sign"]},
            "outside image",
        ),
        ({"image": Image.new("RGB", (64, 64)), "boxes": [[5, 5, 5, 9]], "labels": ["stop-sign"]}, "empty"),
        ({"image": Image.new("RGB", (64, 64)), "boxes": [[0, 0, 5, 5]], "labels": ["cat"]}, "unknown class"),
        ({"image": Image.new("RGB", (64, 64))}, "must be a mapping"),
    ],
)
def test_validate_dataset_names_the_failed_rule(record, message):
    with pytest.raises((TypeError, ValueError), match=message):
        validate_dataset([record], SIGN_CLASSES, epochs=1)


def test_validate_dataset_rejects_bad_vocabulary_and_epochs():
    records = sign_dataset(2)
    with pytest.raises(ValueError, match="duplicates"):
        validate_dataset(records, ["a", "a"], epochs=1)
    with pytest.raises(ValueError, match="epochs"):
        validate_dataset(records, SIGN_CLASSES, epochs=0)
    with pytest.raises(ValueError, match="at least one record"):
        validate_dataset([], SIGN_CLASSES)


def test_split_dataset_is_deterministic_and_disjoint():
    records = sign_dataset(8, seed=1)
    train, held = split_dataset(records, train_fraction=0.75, seed=4)
    again_train, _ = split_dataset(records, train_fraction=0.75, seed=4)
    assert [r["id"] for r in train] == [r["id"] for r in again_train]
    assert len(train) == 6 and len(held) == 2
    assert not {r["id"] for r in train} & {r["id"] for r in held}
    assert len(split_dataset(records[:2], train_fraction=0.99)[1]) == 1


def _write_byod(root: Path, entries: list[dict]) -> None:
    root.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (64, 64), "white").save(root / "a.png")
    (root / "annotations.json").write_text(json.dumps(entries), encoding="utf-8")


def test_read_detection_records_reads_relative_files(tmp_path):
    _write_byod(tmp_path, [{"file": "a.png", "boxes": [[1, 1, 20, 20]], "labels": ["thing"]}])
    records = read_detection_records(tmp_path)
    assert records[0]["image"].size == (64, 64) and records[0]["labels"] == ["thing"]
    assert validate_dataset(records, ["thing"], epochs=1)["verdict"] == "accepted"


@pytest.mark.parametrize("name", ["../a.png", "/etc/passwd"])
def test_read_detection_records_refuses_paths_outside_the_directory(tmp_path, name):
    _write_byod(tmp_path / "set", [{"file": name, "boxes": [], "labels": []}])
    with pytest.raises(ValueError, match="relative|outside"):
        read_detection_records(tmp_path / "set")


def test_read_detection_records_requires_an_index(tmp_path):
    with pytest.raises(FileNotFoundError, match="annotations.json"):
        read_detection_records(tmp_path)
