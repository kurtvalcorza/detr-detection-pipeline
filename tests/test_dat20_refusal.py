"""DAT19/DAT20: a refused BYOD input shows one `Refused: <file>: <reason>` line and no traceback.

Two halves, both read from the generated notebook:
- the router (Section 1, template `dat20-router-refusal/1` from ml-worker
  `integrations/dimer/fleet-specs/notebook-router/`) relays an exception carrying `dimer_refusal` as one
  line, while every other exception keeps its traceback;
- the BYOD cells turn every user-input problem into `BYODRefusal` and remove the earlier run's results and
  `outputs/` files before they stop.
"""

# ruff: noqa: E501  -- refusal messages and parametrized cases read best on one line
from __future__ import annotations

import hashlib
import io
import json
import os
import re
import signal
import subprocess
import sys
import types
import zipfile
from multiprocessing.connection import Connection
from pathlib import Path

import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "tutorials/detr_detection_colab.ipynb"
BYOD_CELL = "detr_detection-33"
CARRIED = (
    "class BYODRefusal(ValueError):\n"
    '    """A refused BYOD input: ``BYODRefusal(file_name, reason_and_fix)``."""\n\n'
    "    dimer_refusal = True\n\n"
    "    def __init__(self, file_name, reason):\n"
    '        super().__init__(f"{file_name}: {reason}")\n'
    "        self.file_name = file_name\n"
    "        self.reason = reason\n\n"
    "    def _render_traceback_(self):\n"
    '        return [f"Refused: {self}"]\n'
)


def _code_cells(path: Path = NOTEBOOK) -> list[dict]:
    return [c for c in json.loads(path.read_text(encoding="utf-8"))["cells"] if c["cell_type"] == "code"]


def _source(cell: dict) -> str:
    return "".join(cell["source"])


def _cell(cell_id: str, path: Path = NOTEBOOK) -> str:
    return next(_source(c) for c in _code_cells(path) if c.get("id") == cell_id)


# --- router, kernel side (portable) -------------------------------------------------------------------------------


def _router_namespace(start: str) -> dict:
    source = next(_source(c) for c in _code_cells() if '_WORKER_SOURCE = r"""' in _source(c))
    chunk = source[source.index(start) : source.index("def _is_user_cell():")]
    namespace = {"os": os, "sys": sys, "signal": signal, "subprocess": subprocess, "Connection": Connection, "__name__": "router"}
    exec(compile(chunk, "<router>", "exec"), namespace)
    return namespace


class _Pipe:
    def __init__(self, messages=()):
        self.messages, self.sent = list(messages), []

    def recv(self):
        return self.messages.pop(0)

    def send(self, message):
        self.sent.append(message)


def _kernel_runtime(messages):
    namespace = _router_namespace("class IsolatedCellError(RuntimeError):")
    runtime = object.__new__(namespace["IsolatedRuntime"])
    runtime._send, runtime._recv, runtime._display = _Pipe(), _Pipe(messages), lambda *a, **k: None
    return namespace, runtime


def test_router_relays_a_refusal_as_one_line_without_a_traceback(capsys):
    """Acceptance: fails on main, whose router has no refusal channel."""
    namespace, runtime = _kernel_runtime([("refusal", "hostname: not readable")])
    with pytest.raises(namespace["IsolatedCellRefusal"]) as caught:
        runtime.run("x = 1\n")
    assert caught.value._render_traceback_() == ["Refused: hostname: not readable"]
    assert caught.value.__cause__ is None and caught.value.__suppress_context__
    assert capsys.readouterr().err == ""
    assert runtime._send.sent == [("run", "x = 1\n")]


def test_router_keeps_the_traceback_of_a_genuine_error(capsys):
    namespace, runtime = _kernel_runtime([("error", "Traceback (most recent call last):\nRuntimeError: bug\n", "RuntimeError: bug")])
    with pytest.raises(namespace["IsolatedCellError"], match="RuntimeError: bug"):
        runtime.run("x = 1\n")
    assert "Traceback (most recent call last)" in capsys.readouterr().err


def test_router_normal_cell_output_is_unchanged(capsys):
    _, runtime = _kernel_runtime([("stream", "stdout", "hello\n"), ("done",)])
    assert runtime.run("print('hello')\n") is None
    assert capsys.readouterr() == ("hello\n", "")


def test_router_template_is_applied():
    source = next(_source(c) for c in _code_cells() if '_WORKER_SOURCE = r"""' in _source(c))
    assert source.index('elif kind == "refusal":') < source.index('elif kind == "error":')
    assert 'if getattr(exc, "dimer_refusal", False):' in source
    header = "Its router refusal channel (DAT20) is applied from ml-worker integrations/dimer/fleet-specs/notebook-router/."
    assert header in (ROOT / "tools" / "build_notebook.py").read_text(encoding="utf-8")


@pytest.mark.skipif(os.name == "nt", reason="the isolated-cell router is POSIX-only (pass_fds, start_new_session)")
def test_router_worker_end_to_end(capsys):
    namespace = _router_namespace('_WORKER_SOURCE = r"""')
    runtime = namespace["IsolatedRuntime"](sys.executable, display=lambda *a, **k: None)
    try:
        runtime.run("class R(ValueError):\n    dimer_refusal = True\nkept = 41\n")
        with pytest.raises(namespace["IsolatedCellRefusal"]) as caught:
            runtime.run("raise R('hostname: not readable')\n")
        assert caught.value._render_traceback_() == ["Refused: hostname: not readable"]
        assert "Traceback" not in capsys.readouterr().err
        runtime.run("print(kept + 1)\n")  # the worker namespace survives a refusal
        assert capsys.readouterr().out == "42\n"
        with pytest.raises(namespace["IsolatedCellError"], match="RuntimeError: bug"):
            runtime.run("raise RuntimeError('bug')\n")
        assert "Traceback (most recent call last)" in capsys.readouterr().err
    finally:
        runtime.close()


def test_byod_refusal_is_carried_verbatim():
    assert CARRIED in _cell(BYOD_CELL)


def _refusal_line(exc: BaseException) -> str:
    assert getattr(exc, "dimer_refusal", False), f"not a refusal: {exc!r}"
    (line,) = exc._render_traceback_()
    assert "\n" not in line and "Traceback" not in line
    return line


SOURCE = _cell(BYOD_CELL)
PREFIX = re.search(r"'outputs/(byod_[a-z_]+_)\*'", SOURCE).group(1)
PIPELINE_CLASS = re.search(r"byod_pipe = (\w+)\.from_pretrained", SOURCE).group(1)
EARLIER = {"byod_image": "old", "byod_result": "old", "byod_manifest": "old", "byod_adapted": "old", "byod_export": "old"}


def _png(size=(32, 32)) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", size, (120, 30, 30)).save(buffer, format="PNG")
    return buffer.getvalue()


def _seed_outputs(tmp_path: Path) -> None:
    out = tmp_path / "outputs"
    out.mkdir(exist_ok=True)
    for name in ("result.json", "detections.csv", "adapter.safetensors"):
        (out / f"{PREFIX}{name}").write_text("{}", encoding="utf-8")
    (out / "sample_result.json").write_text("{}", encoding="utf-8")


def _validate_inputs(obj, threshold=None, names=None):
    if isinstance(obj, str):
        raise TypeError(f"expected a PIL image, got {type(obj).__name__}")
    if min(obj.size) < 16 or max(obj.size) > 4096:
        raise ValueError(f"image side {min(obj.size)}x{max(obj.size)} is outside [16, 4096] px")
    return {"verdict": "accepted"}


def _validate_dataset(records, names, epochs=None):
    if len(records) < 2:
        raise ValueError(f"need at least 2 records, got {len(records)}")
    for record in records:
        width, height = record["image"].size
        for box in record["boxes"]:
            if box[2] > width or box[3] > height:
                raise ValueError(f"box {box} lies outside the {width}x{height} image")
    return {"schema": "s", "records": len(records)}


def _read_detection_records(folder):
    records = []
    for i, item in enumerate(json.loads((Path(folder) / "annotations.json").read_text(encoding="utf-8"))):
        path = Path(folder) / item["file"]
        if not path.is_file():
            raise FileNotFoundError(f"{item['file']} is named in annotations.json but missing")
        records.append({"id": str(i), "image": Image.open(path).convert("RGB"), "boxes": item["boxes"], "labels": item["labels"]})
    return records


class _Pipeline:
    @classmethod
    def from_pretrained(cls, **kwargs):
        raise RuntimeError("internal: weights not staged")


def _run(monkeypatch, tmp_path, *, image=None, dataset=None, use_image=False, use_dataset=False, colab=None, pipe=None) -> dict:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setitem(sys.modules, "google.colab", colab)
    _seed_outputs(tmp_path)
    source = SOURCE.replace("USE_BYOD_IMAGE = False", f"USE_BYOD_IMAGE = {use_image}", 1).replace("USE_BYOD_DATASET = False", f"USE_BYOD_DATASET = {use_dataset}", 1)
    if image is not None:
        source = source.replace('BYOD_IMAGE_PATH = ""', f"BYOD_IMAGE_PATH = {image!r}", 1)
    if dataset is not None:
        source = source.replace('BYOD_DATASET_DIR = ""', f"BYOD_DATASET_DIR = {dataset!r}", 1)
    ns: dict = {
        "Path": Path, "Image": Image, "json": json, "hashlib": hashlib, "OUTPUTS": Path("outputs"), "MAX_RECORDS": 5000,
        "validate_inputs": _validate_inputs, "validate_dataset": _validate_dataset, "read_detection_records": _read_detection_records,
        "blank_scene": lambda: Image.new("RGB", (64, 64)), "SIGN_CLASSES": ["stop"], "threshold": 0.7, "EPOCHS": 1, "WEIGHTS_DIR": Path("weights"), "HOLDOUT": 0.25, "SEED": 0,
        "split_dataset": lambda records, train_fraction, seed: (records[:-1], records[-1:]), PIPELINE_CLASS: _Pipeline,
        "pipe": pipe or types.SimpleNamespace(detect=lambda image, threshold: {"detections": []}), "evaluation_report": lambda *a, **k: {"verdict": "not-measurable"},
        **EARLIER,
    }
    exec(compile(source, "<section 13>", "exec"), ns)
    return ns


def _colab(uploaded: dict):
    module = types.ModuleType("google.colab")
    module.files = types.SimpleNamespace(upload=lambda: uploaded)
    return module


def _assert_refused(caught, file_name, rule, tmp_path):
    line = _refusal_line(caught.value)
    assert line.startswith(f"Refused: {file_name}: ") and rule in line, line
    assert "again" in line and "earlier results of this session and their outputs/ files were removed" in line
    left = sorted(p.name for p in (tmp_path / "outputs").iterdir() if p.is_file())
    assert left == ["sample_result.json"], left  # this input's earlier BYOD results are gone; the sample's stay


# --- image branch -------------------------------------------------------------------------------------------------


def _image_cases(tmp_path: Path) -> dict:
    (tmp_path / "hostname").write_text("runtime-host\n", encoding="utf-8")
    (tmp_path / "folder").mkdir()
    (tmp_path / "tiny.png").write_bytes(_png((8, 8)))
    return {
        "text file (the scripted /etc/hostname run)": ("hostname", "PIL cannot open this file as an image"),
        "folder": ("folder", "does not exist or is not a file"),
        "missing path": ("nope.png", "does not exist or is not a file"),
        "too small": ("tiny.png", "ValueError: image side 8x8 is outside [16, 4096] px"),
    }


@pytest.mark.parametrize("case", ["text file (the scripted /etc/hostname run)", "folder", "missing path", "too small"])
def test_image_problems_are_one_refused_line_and_remove_earlier_byod_results(monkeypatch, tmp_path, case):
    file_name, rule = _image_cases(tmp_path)[case]
    with pytest.raises(ValueError) as caught:
        _run(monkeypatch, tmp_path, image=str(tmp_path / file_name), use_image=True)
    _assert_refused(caught, file_name, rule, tmp_path)


def test_image_refusal_clears_the_session(monkeypatch, tmp_path):
    (tmp_path / "notes.txt").write_text("text", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setitem(sys.modules, "google.colab", None)
    source = SOURCE.replace("USE_BYOD_IMAGE = False", "USE_BYOD_IMAGE = True", 1).replace('BYOD_IMAGE_PATH = ""', "BYOD_IMAGE_PATH = 'notes.txt'", 1)
    ns = {"Path": Path, "Image": Image, "json": json, "hashlib": hashlib, "OUTPUTS": Path("outputs"), "MAX_RECORDS": 5000, "validate_inputs": _validate_inputs,
          "validate_dataset": _validate_dataset, "blank_scene": lambda: Image.new("RGB", (64, 64)), "SIGN_CLASSES": ["stop"], "threshold": 0.7, "pipe": "kept", **EARLIER}
    with pytest.raises(ValueError, match="^notes.txt: PIL cannot open this file"):
        exec(compile(source, "<section 13>", "exec"), ns)
    assert not set(EARLIER) & set(ns), "an earlier BYOD run's results survived the refusal"
    assert ns["pipe"] == "kept"


def test_upload_outside_colab_is_one_refused_line(monkeypatch, tmp_path):
    with pytest.raises(ValueError) as caught:
        _run(monkeypatch, tmp_path, use_image=True, colab=None)
    _assert_refused(caught, "(no file)", "there is no upload dialog outside Google Colab", tmp_path)


@pytest.mark.parametrize(("uploaded", "file_name", "rule"), [({}, "(no file)", "nothing was uploaded"), ({"a.png": _png(), "b.png": _png()}, "a.png, b.png", "upload exactly one image for the image branch, got 2")])
def test_cancelled_or_multiple_image_upload_is_one_refused_line(monkeypatch, tmp_path, uploaded, file_name, rule):
    with pytest.raises(ValueError) as caught:
        _run(monkeypatch, tmp_path, use_image=True, colab=_colab(uploaded))
    _assert_refused(caught, file_name, rule, tmp_path)


def test_good_image_still_runs_detection(monkeypatch, tmp_path, capsys):
    (tmp_path / "ok.png").write_bytes(_png())
    ns = _run(monkeypatch, tmp_path, image=str(tmp_path / "ok.png"), use_image=True)
    assert ns["byod_image"].size == (32, 32) and ns["byod_result"] == {"detections": []}
    assert "accepted" in capsys.readouterr().out


def test_image_genuine_internal_error_keeps_its_traceback(monkeypatch, tmp_path):
    (tmp_path / "ok.png").write_bytes(_png())

    def broken(image, threshold):
        raise RuntimeError("internal bug")

    with pytest.raises(RuntimeError, match="internal bug") as caught:
        _run(monkeypatch, tmp_path, image=str(tmp_path / "ok.png"), use_image=True, pipe=types.SimpleNamespace(detect=broken))
    assert not getattr(caught.value, "dimer_refusal", False)
    assert (tmp_path / "outputs" / f"{PREFIX}result.json").exists()  # nothing is removed for an internal error


@pytest.mark.skipif(not Path("/etc/hostname").is_file(), reason="needs a POSIX /etc/hostname, as in a fresh Colab runtime")
def test_scripted_refusal_input_is_the_real_etc_hostname(monkeypatch, tmp_path):
    """The hosted refusal run sets USE_BYOD_IMAGE=True;BYOD_IMAGE_PATH='/etc/hostname': a file present in every fresh runtime."""
    with pytest.raises(ValueError) as caught:
        _run(monkeypatch, tmp_path, image="/etc/hostname", use_image=True)
    _assert_refused(caught, "hostname", "PIL cannot open this file as an image", tmp_path)


def test_default_run_is_unchanged(monkeypatch, tmp_path, capsys):
    ns = _run(monkeypatch, tmp_path)
    out = capsys.readouterr().out
    assert "BYOD image branch is off" in out and "BYOD dataset branch is off" in out
    assert ns["byod_export"] == "old"


# --- dataset branch -----------------------------------------------------------------------------------------------


def _zip(entries: dict) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        for name, data in entries.items():
            archive.writestr(name, data)
    return buffer.getvalue()


def _annotations(*items) -> bytes:
    return json.dumps([{"file": f, "boxes": [box], "labels": ["stop"]} for f, box in items]).encode("utf-8")


def _dataset_cases(tmp_path: Path) -> dict:
    (tmp_path / "empty").mkdir()
    (tmp_path / "fake.zip").write_bytes(b"not a zip")
    (tmp_path / "unsafe.zip").write_bytes(_zip({"../evil.txt": b"x"}))
    (tmp_path / "missing_image.zip").write_bytes(_zip({"annotations.json": _annotations(("a.png", [0, 0, 4, 4]), ("b.png", [0, 0, 4, 4])), "a.png": _png()}))
    (tmp_path / "bad_box.zip").write_bytes(_zip({"annotations.json": _annotations(("a.png", [0, 0, 99, 4]), ("b.png", [0, 0, 4, 4])), "a.png": _png(), "b.png": _png()}))
    (tmp_path / "bad_json.zip").write_bytes(_zip({"annotations.json": b"{not json", "a.png": _png()}))
    return {
        "missing folder": ("nope", "does not exist"),
        "no annotations.json": ("empty", "expected one annotations.json"),
        "corrupt zip": ("fake.zip", "BadZipFile"),
        "unsafe member": ("unsafe.zip", "unsafe member path"),
        "image named but missing": ("missing_image.zip", "b.png is named in annotations.json but missing"),
        "box outside the image": ("bad_box.zip", "lies outside the 32x32 image"),
        "annotations not JSON": ("bad_json.zip", "JSONDecodeError"),
    }


@pytest.mark.parametrize("case", ["missing folder", "no annotations.json", "corrupt zip", "unsafe member", "image named but missing", "box outside the image", "annotations not JSON"])
def test_dataset_problems_are_one_refused_line_and_remove_earlier_byod_results(monkeypatch, tmp_path, case):
    file_name, rule = _dataset_cases(tmp_path)[case]
    with pytest.raises(ValueError) as caught:
        _run(monkeypatch, tmp_path, dataset=str(tmp_path / file_name), use_dataset=True)
    _assert_refused(caught, file_name, rule, tmp_path)


def test_cancelled_dataset_upload_is_one_refused_line(monkeypatch, tmp_path):
    with pytest.raises(ValueError) as caught:
        _run(monkeypatch, tmp_path, use_dataset=True, colab=_colab({}))
    _assert_refused(caught, "(no file)", "nothing was uploaded", tmp_path)


def test_valid_dataset_reaches_the_pipeline_and_a_genuine_error_keeps_its_traceback(monkeypatch, tmp_path):
    (tmp_path / "ok.zip").write_bytes(_zip({"annotations.json": _annotations(("a.png", [0, 0, 4, 4]), ("b.png", [1, 1, 5, 5])), "a.png": _png(), "b.png": _png()}))
    with pytest.raises(RuntimeError, match="internal: weights not staged") as caught:
        _run(monkeypatch, tmp_path, dataset=str(tmp_path / "ok.zip"), use_dataset=True)
    assert not getattr(caught.value, "dimer_refusal", False)
    assert (tmp_path / "outputs" / f"{PREFIX}result.json").exists()
