# ruff: noqa: E501  -- assertion lines name the exported fields in full
"""Regression tests for the 2026-10-02 review of detr_detection_colab (DTR-M1..M4, DTR-m1..m3).

The notebook's own learner cells (Sections 4-13) are executed from the committed .ipynb in a namespace built from
the package, with a tiny random-weight DETR standing in for the pinned checkpoint (the same classes and
code paths as `tests/test_tiny_model.py`). The tiny model says nothing about detection quality; these tests check
control flow, the exported fields and the BYOD file handling. The real-weights checks are recorded in
docs/release-verification.md.
"""

from __future__ import annotations

import contextlib
import io
import json
import re
import sys
import types
import zipfile
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")  # imported before any NumPy linear algebra (Windows DLL load order)
transformers = pytest.importorskip("transformers")
pytest.importorskip("scipy")

import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

from detr_detection_pipeline import pipeline as pipeline_module  # noqa: E402
from detr_detection_pipeline import samples as samples_module  # noqa: E402
from detr_detection_pipeline.pipeline import (  # noqa: E402
    DEFAULT_SEED,
    LABELS,
    MAX_DETECTIONS,
    DetrDetectionPipeline,
)

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "tutorials" / "detr_detection_colab.ipynb"


def _tiny(class_names, seed: int) -> DetrDetectionPipeline:
    from transformers import (
        DetrConfig,
        DetrForObjectDetection,
        DetrImageProcessor,
        ResNetConfig,
    )

    torch.manual_seed(0)  # the "snapshot": every tensor except the class head is the same for every build
    backbone = ResNetConfig(
        num_channels=3, embedding_size=8, hidden_sizes=[8, 16], depths=[1, 1], layer_type="basic", out_features=["stage2"]
    )
    config = DetrConfig(
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
        num_labels=len(class_names),
        id2label=dict(enumerate(class_names)),
        label2id={name: i for i, name in enumerate(class_names)},
    )
    model = DetrForObjectDetection(config).eval()
    torch.manual_seed(seed)  # the re-headed class layer depends on the seed, as in from_pretrained
    model.class_labels_classifier.reset_parameters()
    processor = DetrImageProcessor(size={"shortest_edge": 64, "longest_edge": 64})
    return DetrDetectionPipeline(
        model=model, processor=processor, device="cpu", class_names=tuple(class_names), source="tiny"
    )


class TinyPipeline(DetrDetectionPipeline):
    """Stand-in for the pinned checkpoint: `from_pretrained` builds the tiny model; `load_artifact` is inherited."""

    @classmethod
    def from_pretrained(cls, device=None, weights_dir=None, allow_download=False, class_names=None, seed=DEFAULT_SEED):
        tiny = _tiny(tuple(class_names) if class_names is not None else LABELS, seed)
        return cls(**{f: getattr(tiny, f) for f in ("model", "processor", "device", "class_names", "source")})


def _notebook() -> dict:
    return json.loads(NOTEBOOK.read_text(encoding="utf-8"))


def _source(cell: dict) -> str:
    return "".join(cell["source"]) if isinstance(cell["source"], list) else cell["source"]


def _section_code(nb: dict, number: int) -> str:
    cells = nb["cells"]
    for index, cell in enumerate(cells):
        if cell["cell_type"] == "markdown" and f"## {number}. " in _source(cell):
            return next(_source(c) for c in cells[index + 1 :] if c["cell_type"] == "code")
    raise KeyError(number)


def _set(source: str, name: str, value: str) -> str:
    new, n = re.subn(rf"^{name} = .*?(  # @param.*)$", lambda m: f"{name} = {value}{m.group(1)}", source, flags=re.M)
    assert n == 1, name
    return new


@pytest.fixture
def notebook_run(tmp_path, monkeypatch):
    """Run Sections 4-13 of the notebook at small sizes (8 images, 1 epoch); returns (namespace, run, notebook)."""
    monkeypatch.chdir(tmp_path)
    nb = _notebook()
    ns: dict = {"__name__": "__main__"}
    ns.update({k: v for k, v in vars(samples_module).items() if not k.startswith("__")})
    ns.update({k: v for k, v in vars(pipeline_module).items() if not k.startswith("__")})
    ns.update(
        DetrDetectionPipeline=TinyPipeline,
        WEIGHTS_DIR=tmp_path / "weights",
        NOTEBOOK_SOURCE={"repository_revision": "test"},
        np=np,
        Image=Image,
        torch=torch,
        transformers=transformers,
        timm=types.SimpleNamespace(__version__="stand-in"),  # timm itself is not needed by the tiny model
        platform=__import__("platform"),
    )
    ns["pipe"] = TinyPipeline.from_pretrained()

    def run(number: int, **fields: str) -> str:
        source = _section_code(nb, number)
        for name, value in fields.items():
            source = _set(source, name, value)
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            exec(compile(source, f"section{number}", "exec"), ns)
        return buffer.getvalue()

    for number in (4, 5):
        run(number)
    run(6, N_IMAGES="8", EPOCHS="1")
    for number in range(7, 14):
        run(number)
    return ns, run, nb


# ---------------------------------------------------------------- DTR-M2: the detection cap is one box per query


def test_noise_probe_prose_names_the_cap():
    nb = _notebook()
    markdown = "\n".join(_source(c) for c in nb["cells"] if c["cell_type"] == "markdown")
    assert MAX_DETECTIONS == 100
    assert "**The count is capped.**" in markdown and "`MAX_DETECTIONS` (100)" in markdown


# ---------------------------------------------------------------- DTR-M1: uv isolated runtime, no in-kernel install


def test_only_the_uv_install_and_router_cells_run_in_the_kernel():
    nb = _notebook()
    code = [_source(c) for c in nb["cells"] if c["cell_type"] == "code"]
    kernel = [i for i, src in enumerate(code) if "# dimer: kernel cell" in src]
    assert kernel == [0, 1]
    install = code[0]
    assert '"--require-hashes", "--only-binary", ":all:"' in install and '"--managed-python"' in install
    assert "sys.executable" not in install.split("SKIP_INSTALL = ", 1)[1]
    lock = (ROOT / "tutorials" / "requirements-colab.lock.txt").read_text(encoding="utf-8")
    for pin in ("torch==2.14.0", "transformers==4.57.6", "timm==1.0.29", "scipy==1.18.1", "numpy==2.5.3"):
        assert f"\n{pin} \\" in lock


# ---------------------------------------------------------------- DTR-M2: honest new-data claim and the decision view


def test_new_data_claim_is_gone_and_the_threshold_distinction_is_taught():
    nb = _notebook()
    markdown = "\n".join(_source(c) for c in nb["cells"] if c["cell_type"] == "markdown")
    assert "The adapted pipeline detects the new sign classes" not in markdown
    assert "**AP is computed at the evaluation threshold.**" in markdown
    assert "returned **no** boxes" in markdown


def test_default_run_exports_the_operating_threshold_view(notebook_run):
    ns, _run, _nb = notebook_run
    check = ns["operating_check"]
    assert set(check) == {"threshold", "returned", "correct", "references", "precision", "recall", "highest_score"}
    assert check["threshold"] == 0.9 and check["references"] == sum(len(r["boxes"]) for r in ns["held_out"])
    row = ns["new_data_rows"][0]
    assert {"highest_score", "best_same_label_at_evaluation_threshold", "same_label_iou"} <= set(row)
    assert len(row["best_same_label_at_evaluation_threshold"]) == len(row["truth"])
    result = json.loads(Path("outputs/detr_detection_result.json").read_text(encoding="utf-8"))
    assert result["evaluation_threshold"] == pipeline_module.EVAL_DETECTION_THRESHOLD
    assert result["adaptation"]["held_out_at_operating_threshold"] == check
    assert len(result["adaptation"]["run_history"]) == 1


# ---------------------------------------------------------------- DTR-M3: every fine-tune starts from the re-headed model


def test_rerunning_the_finetune_starts_from_the_reheaded_model(notebook_run):
    ns, run, _nb = notebook_run
    first = list(ns["run"]["epoch_losses"])
    out = run(8)  # same settings, re-run as a learner would after Section 9
    assert "rebuilt_from_verified_snapshot" in out
    assert ns["run"]["epoch_losses"] == first  # identical start, not continued training
    run(9)
    unfrozen_out = run(8, FREEZE_BACKBONE="False")
    assert "rebuilt_from_verified_snapshot" in unfrozen_out
    assert ns["run"]["freeze_backbone"] is False and ns["run"]["frozen_prefixes"] == []
    assert abs(ns["run"]["epoch_losses"][0] - first[0]) <= 0.1 * first[0]  # the review's acceptance check
    run(9)
    history = ns["RUN_HISTORY"]
    assert [row["freeze_backbone"] for row in history] == [True, True, False]


# ---------------------------------------------------------------- DTR-m2 / DTR-m3: BYOD results, zip, uploads


def _byod_files(folder: Path, sub: str = "") -> None:
    (folder / sub).mkdir(parents=True, exist_ok=True)
    records = samples_module.sign_dataset(4, seed=7)
    entries = []
    for k, record in enumerate(records):
        name = f"{sub}{k:03d}.png"
        record["image"].save(folder / name)
        entries.append({"file": name, "boxes": record["boxes"], "labels": record["labels"]})
    (folder / "annotations.json").write_text(json.dumps(entries), encoding="utf-8")


def test_byod_dataset_from_a_zip_writes_results_and_compares_the_reload(notebook_run, tmp_path):
    _ns, run, _nb = notebook_run
    source = tmp_path / "src"
    _byod_files(source, sub="images/")
    archive = tmp_path / "mine.zip"
    with zipfile.ZipFile(archive, "w") as bundle:
        for path in sorted(source.rglob("*")):
            if path.is_file():
                bundle.write(path, "top/" + path.relative_to(source).as_posix())
    out = run(13, USE_BYOD_DATASET="True", BYOD_DATASET_DIR=repr(str(archive)))
    assert "BYOD adapter exported, reloaded and compared" in out
    result = json.loads(Path("outputs/byod_detr_result.json").read_text(encoding="utf-8"))
    assert result["reload_check"]["equivalent"] is True and result["reload_check"]["tolerance"] == 1e-3
    assert result["reload_check"]["detections_compared"] > 0  # never a vacuous comparison
    assert {"annotations_sha256", "split", "finetune", "baseline", "adapted", "held_out_at_operating_threshold"} <= set(result)
    assert Path("outputs/byod_detr_detections.csv").read_text(encoding="utf-8").startswith("image,rank,label")


def test_byod_refuses_an_unsafe_zip_and_explains_a_missing_file(notebook_run, tmp_path):
    _ns, run, _nb = notebook_run
    evil = tmp_path / "evil.zip"
    with zipfile.ZipFile(evil, "w") as bundle:
        bundle.writestr("annotations.json", "[]")
        bundle.writestr("../escape.png", b"x")
    with pytest.raises(ValueError, match="unsafe member path"):
        run(13, USE_BYOD_DATASET="True", BYOD_DATASET_DIR=repr(str(evil)))
    assert not (tmp_path / "outputs" / "escape.png").exists()
    missing = tmp_path / "missing"
    missing.mkdir()
    (missing / "annotations.json").write_text(json.dumps([{"file": "images/0001.png", "boxes": [], "labels": []}] * 2))
    with pytest.raises(FileNotFoundError, match="flat"):
        run(13, USE_BYOD_DATASET="True", BYOD_DATASET_DIR=repr(str(missing)))


def test_a_second_image_upload_replaces_the_first(notebook_run, monkeypatch):
    ns, run, _nb = notebook_run
    payloads = []
    for colour in ("red", "blue"):
        buffer = io.BytesIO()
        Image.new("RGB", (64, 48), colour).save(buffer, format="PNG")
        payloads.append(buffer.getvalue())
    queue = iter([{"a_first.png": payloads[0]}, {"z_second.png": payloads[1]}])
    google = types.ModuleType("google")
    google.__path__ = []
    colab = types.ModuleType("google.colab")
    files = types.ModuleType("google.colab.files")
    files.upload = lambda: next(queue)
    colab.files = files
    google.colab = colab
    for name, module in (("google", google), ("google.colab", colab), ("google.colab.files", files)):
        monkeypatch.setitem(sys.modules, name, module)
    run(13, USE_BYOD_IMAGE="True")
    assert ns["image_path"].name == "a_first.png"
    run(13, USE_BYOD_IMAGE="True")
    assert ns["image_path"].name == "z_second.png"
    assert sorted(p.name for p in ns["image_path"].parent.iterdir()) == ["z_second.png"]


# ---------------------------------------------------------------- DTR-M2 (Kurt, as CDT-M2 option B): adapted threshold on the training split


def test_adapted_threshold_is_chosen_on_the_training_split_only(notebook_run, monkeypatch):
    ns, _run, nb = notebook_run
    assert "adapted_threshold = select_adapted_threshold(adapter, train_records)" in _section_code(nb, 9)
    chosen = ns["adapted_threshold"]
    assert chosen["selected_on"] == "training split" and ns["ADAPTED_THRESHOLD"] == chosen["threshold"]
    seen = []
    original = TinyPipeline.detect

    def spy(self, image, **kwargs):
        seen.append(id(image))
        return original(self, image, **kwargs)

    monkeypatch.setattr(TinyPipeline, "detect", spy)
    again = ns["select_adapted_threshold"](ns["adapter"], ns["train_records"])
    assert again == chosen  # deterministic, and reproducible from the training split alone
    train_ids = {id(r["image"]) for r in ns["train_records"]}
    assert seen and set(seen) <= train_ids  # never a held-out or unseen image
    assert not set(seen) & {id(r["image"]) for r in ns["held_out"] + ns["new_records"]}


def test_adapted_threshold_is_reported_next_to_the_coco_view(notebook_run):
    ns, _run, _nb = notebook_run
    row = ns["new_data_rows"][0]
    assert row["threshold"] == 0.9 and row["adapted_threshold"] == ns["ADAPTED_THRESHOLD"]
    assert len(row["same_label_iou_at_adapted_threshold"]) == len(row["truth"])
    result = json.loads(Path("outputs/detr_detection_result.json").read_text(encoding="utf-8"))
    assert result["adaptation"]["adapted_threshold"] == ns["adapted_threshold"]
    assert result["adaptation"]["held_out_at_adapted_threshold"]["threshold"] == ns["ADAPTED_THRESHOLD"]
    header = Path("outputs/detr_detection_detections.csv").read_text(encoding="utf-8").splitlines()[0]
    assert header.endswith(",threshold")


# ---------------------------------------------------------------- FIX_PACKET addendum: google.colab stubs carry a spec


@pytest.mark.parametrize("real_google", [False, True])
def test_worker_colab_stubs_have_specs(monkeypatch, real_google):
    """Colab only: accelerate calls importlib.util.find_spec("google.colab"), which raised on the spec-less stub."""
    import ast
    import importlib.util

    nb = _notebook()
    router = [_source(c) for c in nb["cells"] if c["cell_type"] == "code"][1]
    worker = next(
        node.value.value
        for node in ast.parse(router).body
        if isinstance(node, ast.Assign) and getattr(node.targets[0], "id", "") == "_WORKER_SOURCE"
    )
    start = worker.index('if os.environ.get("DIMER_KERNEL_IS_COLAB") == "1":')
    shim = worker[start : worker.index('_main = types.ModuleType("__main__")', start)]
    fake_google = types.ModuleType("google")
    fake_google.__path__ = []
    monkeypatch.setitem(sys.modules, "google", fake_google if real_google else None)
    monkeypatch.delitem(sys.modules, "google.colab", raising=False)
    monkeypatch.delitem(sys.modules, "google.colab.files", raising=False)
    monkeypatch.setenv("DIMER_KERNEL_IS_COLAB", "1")
    try:
        exec(compile(shim, "worker-colab-shim", "exec"), {"os": __import__("os"), "sys": sys, "types": types, "_send": None, "_recv": None})
        for name in ("google.colab", "google.colab.files"):
            spec = importlib.util.find_spec(name)  # raised ValueError before the fix
            assert spec is not None and spec.name == name
        assert sys.modules["google.colab"].__path__ == [] and callable(sys.modules["google.colab.files"].upload)
        if not real_google:
            assert importlib.util.find_spec("google") is not None
    finally:
        for name in ("google", "google.colab", "google.colab.files"):
            sys.modules.pop(name, None)  # monkeypatch then restores whatever was there before
