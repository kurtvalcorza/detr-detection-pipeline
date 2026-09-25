"""tools/pin_snapshot.py against a fake Hub: no network, no weights."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import shutil
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
COMMIT = "89abcdef0123456789abcdef0123456789abcdef"


def _load_tool():
    spec = importlib.util.spec_from_file_location("pin_snapshot", ROOT / "tools" / "pin_snapshot.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


pin_snapshot = _load_tool()


def _copy_repo(tmp_path: Path) -> Path:
    for relative in (
        "tools/notebook_template.py",
        "README.md",
        "MODEL_CARD.md",
        "STATUS.md",
        "docs/WEIGHTS.md",
    ):
        (tmp_path / relative).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(ROOT / relative, tmp_path / relative)
    shutil.copytree(ROOT / "src", tmp_path / "src")
    shutil.copytree(ROOT / "weights", tmp_path / "weights", ignore=shutil.ignore_patterns("*.safetensors"))
    return tmp_path


def _fake_hub(files: dict[str, bytes], lfs_override: dict[str, str] | None = None):
    lfs_override = lfs_override or {}

    def model_info(model_id, revision, files_metadata):
        assert files_metadata is True
        siblings = []
        for name, data in files.items():
            lfs = None
            if name.endswith(".safetensors"):
                lfs = SimpleNamespace(sha256=lfs_override.get(name, hashlib.sha256(data).hexdigest()))
            siblings.append(SimpleNamespace(rfilename=name, lfs=lfs))
        return SimpleNamespace(sha=COMMIT, siblings=siblings)

    calls = []

    def download(model_id, filename, revision, local_dir):
        calls.append((filename, revision))
        target = Path(local_dir) / filename
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(files[filename])

    return model_info, download, calls


FILES = {
    "README.md": b"# card\n",
    "config.json": b'{"model_type": "detr"}',
    "model.safetensors": b"\x00" * 64,
    "preprocessor_config.json": b"{}",
}


def test_pin_writes_commit_digests_and_module_revision(tmp_path):
    root = _copy_repo(tmp_path)
    model_info, download, calls = _fake_hub(FILES)
    assert pin_snapshot.pin(root, model_info=model_info, download=download) == 0
    assert {revision for _name, revision in calls} == {COMMIT}
    manifest = json.loads((root / "weights/detr-resnet-50/dimer-base-manifest.json").read_text())
    assert manifest["revision"] == COMMIT
    assert {f["path"]: f["sha256"] for f in manifest["files"]} == {
        name: hashlib.sha256(data).hexdigest() for name, data in FILES.items()
    }
    assert manifest["totalBytes"] == sum(len(data) for data in FILES.values())
    module = (root / "src/detr_detection_pipeline/pipeline.py").read_text()
    assert f'MODEL_REVISION = "{COMMIT}"' in module


def test_pin_refuses_a_digest_that_disagrees_with_the_hub(tmp_path):
    root = _copy_repo(tmp_path)
    before = (root / "weights/detr-resnet-50/dimer-base-manifest.json").read_text()
    module_before = (root / "src/detr_detection_pipeline/pipeline.py").read_text()
    model_info, download, _calls = _fake_hub(FILES, lfs_override={"model.safetensors": "f" * 64})
    assert pin_snapshot.pin(root, model_info=model_info, download=download) == 1
    assert (root / "weights/detr-resnet-50/dimer-base-manifest.json").read_text() == before
    assert (root / "src/detr_detection_pipeline/pipeline.py").read_text() == module_before


def test_pin_refuses_when_a_manifest_file_is_absent_upstream(tmp_path):
    root = _copy_repo(tmp_path)
    partial = {name: data for name, data in FILES.items() if name != "README.md"}
    model_info, download, calls = _fake_hub(partial)
    assert pin_snapshot.pin(root, model_info=model_info, download=download) == 1
    assert calls == []


def test_dry_run_writes_nothing(tmp_path):
    root = _copy_repo(tmp_path)
    before = (root / "weights/detr-resnet-50/dimer-base-manifest.json").read_text()
    model_info, download, _calls = _fake_hub(FILES)
    assert pin_snapshot.pin(root, dry_run=True, model_info=model_info, download=download) == 0
    assert (root / "weights/detr-resnet-50/dimer-base-manifest.json").read_text() == before
