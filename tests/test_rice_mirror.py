"""Offline tests for the hash-pinned sample mirror and the rate-limit-aware Zenodo fallback."""

import base64
import email.message
import hashlib
import io
import json
import sys
import urllib.error
import zipfile
import zlib
from pathlib import Path

import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import rice_assets as assets  # noqa: E402

NOTEBOOK = ROOT / "tutorials" / "DIMER_Philippine_Rice_Pest_Surveillance_Capstone.ipynb"
MIRROR_URL = "https://github.com/example/repo/releases/download/v1/sample.zip"
ZENODO_URL = "https://zenodo.org/api/records/1/files/datasets.zip/content"


def _png(colour):
    stream = io.BytesIO()
    Image.new("RGB", (24, 16), colour).save(stream, format="PNG")
    return stream.getvalue()


IMAGES = {"images/a.png": _png("red"), "images/b.png": _png("blue")}


def _source_archive():
    """A Zenodo-like DEFLATE archive plus a manifest whose records pin its members."""
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w", zipfile.ZIP_DEFLATED) as bundle:
        bundle.writestr("readme.txt", "padding member")
        for name, data in IMAGES.items():
            bundle.writestr("datasets/" + name, data)
    raw = stream.getvalue()
    records = []
    with zipfile.ZipFile(io.BytesIO(raw)) as bundle:
        for name, data in IMAGES.items():
            info = bundle.getinfo("datasets/" + name)
            records.append(
                {
                    "relative_path": name,
                    "bytes": len(data),
                    "sha256": hashlib.sha256(data).hexdigest(),
                    "width": 24,
                    "height": 16,
                    "zip_member": {
                        "name": info.filename,
                        "offset": info.header_offset,
                        "compressed": info.compress_size,
                        "size": info.file_size,
                        "crc32": f"{info.CRC:08x}",
                        "method": info.compress_type,
                    },
                }
            )
    manifest = {"archive": {"url": ZENODO_URL, "bytes": len(raw)}, "records": records}
    return raw, manifest


def _bundle(members):
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w", zipfile.ZIP_STORED) as bundle:
        for name in sorted(members):
            bundle.writestr(zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0)), members[name])
    return stream.getvalue()


def _mirror(bundle, **overrides):
    digest = hashlib.sha256(bundle).hexdigest()
    spec = {"url": MIRROR_URL, "bytes": len(bundle), "sha256": digest, "members": 2}
    return {**spec, **overrides}


class _Response:
    def __init__(self, body, url=MIRROR_URL, status=200, headers=None):
        self.body, self.url, self.status, self.headers = body, url, status, headers or {}

    def read(self, limit):
        return self.body[:limit]

    def geturl(self):
        return self.url

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def _serve_mirror(monkeypatch, bundle):
    monkeypatch.setattr(assets.urllib.request, "urlopen", lambda request, timeout: _Response(bundle))


def _serve_zenodo(monkeypatch, raw):
    calls = []

    def read_range(url, start, length, size):
        calls.append((start, length))
        return raw[start : start + length]

    monkeypatch.setattr(assets, "read_range", read_range)
    return calls


def _no_zenodo(monkeypatch):
    def refuse(*args):
        raise AssertionError("Zenodo must not be contacted when the mirror verifies")

    monkeypatch.setattr(assets, "read_range", refuse)


def _http_error(code, retry_after=None):
    headers = email.message.Message()
    if retry_after is not None:
        headers["Retry-After"] = str(retry_after)
    return urllib.error.HTTPError(ZENODO_URL, code, "throttled", headers, None)


def _assert_images(root):
    for name, data in IMAGES.items():
        assert (root / "data" / name).read_bytes() == data


def test_mirror_success_verifies_every_image_without_zenodo(tmp_path, monkeypatch):
    _, manifest = _source_archive()
    bundle = _bundle(IMAGES)
    _serve_mirror(monkeypatch, bundle)
    _no_zenodo(monkeypatch)
    summary = assets.prepare_images(tmp_path, manifest, _mirror(bundle))
    assert summary == {"images": 2, "bytes": sum(map(len, IMAGES.values())), "sha256_verified": True,
                       "source": "mirror"}
    _assert_images(tmp_path)


def test_bundle_hash_mismatch_falls_back_to_zenodo(tmp_path, monkeypatch, capsys):
    raw, manifest = _source_archive()
    bundle = _bundle(IMAGES)
    _serve_mirror(monkeypatch, bundle)
    calls = _serve_zenodo(monkeypatch, raw)
    summary = assets.prepare_images(tmp_path, manifest, _mirror(bundle, sha256="0" * 64))
    assert summary["source"] == "zenodo"
    assert "falling back to Zenodo" in capsys.readouterr().out
    assert len(calls) == 2, "one bounded range request per member"
    _assert_images(tmp_path)


def test_tampered_member_in_valid_hash_bundle_is_refused(tmp_path, monkeypatch):
    raw, manifest = _source_archive()
    original = IMAGES["images/b.png"]
    tampered = original[:-1] + bytes([original[-1] ^ 1])
    bundle = _bundle({**IMAGES, "images/b.png": tampered})
    with pytest.raises(ValueError, match="Mirror member SHA-256"):
        assets.extract_mirror(tmp_path, bundle, manifest["records"])
    _serve_mirror(monkeypatch, bundle)
    _serve_zenodo(monkeypatch, raw)
    assert assets.prepare_images(tmp_path / "run", manifest, _mirror(bundle))["source"] == "zenodo"
    _assert_images(tmp_path / "run")


@pytest.mark.parametrize("name", ["../escape.png", "images/../../escape.png", "/abs.png"])
def test_path_traversal_in_bundle_is_refused(tmp_path, name):
    _, manifest = _source_archive()
    bundle = _bundle({**IMAGES, name: b"x"})
    with pytest.raises(ValueError, match="Unsafe|escapes"):
        assets.extract_mirror(tmp_path / "root", bundle, manifest["records"])
    assert not (tmp_path / "escape.png").exists()
    assert not (tmp_path / "root" / "data").exists()


def test_mirror_host_is_validated_separately_from_zenodo():
    with pytest.raises(ValueError, match="allowed HTTPS host"):
        assets.download_mirror({"url": "https://zenodo.org/sample.zip", "bytes": 1, "sha256": "0" * 64})
    with pytest.raises(ValueError, match="HTTPS"):
        assets.read_range(MIRROR_URL, 0, 30, 999)


def test_429_then_206_honours_retry_after(monkeypatch):
    body = b"0123456789"
    responses = [_http_error(429, retry_after=7)]
    sleeps = []

    def urlopen(request, timeout):
        if responses:
            raise responses.pop()
        headers = {"Content-Range": "bytes 5-14/100"}
        return _Response(body, url=ZENODO_URL, status=206, headers=headers)

    monkeypatch.setattr(assets.urllib.request, "urlopen", urlopen)
    monkeypatch.setattr(assets.time, "sleep", sleeps.append)
    assert assets.read_range(ZENODO_URL, 5, 10, 100) == body
    assert sleeps == [7.0]


def test_persistent_429_raises_after_capped_waits(monkeypatch):
    sleeps = []

    def urlopen(request, timeout):
        raise _http_error(429, retry_after=500)

    monkeypatch.setattr(assets.urllib.request, "urlopen", urlopen)
    monkeypatch.setattr(assets.time, "sleep", sleeps.append)
    with pytest.raises(urllib.error.HTTPError):
        assets.read_range(ZENODO_URL, 0, 30, 100)
    assert sleeps == [assets.RETRY_AFTER_CAP] * (assets.ATTEMPTS - 1)


def test_client_errors_are_not_retried(monkeypatch):
    sleeps = []

    def urlopen(request, timeout):
        raise _http_error(404)

    monkeypatch.setattr(assets.urllib.request, "urlopen", urlopen)
    monkeypatch.setattr(assets.time, "sleep", sleeps.append)
    with pytest.raises(urllib.error.HTTPError):
        assets.read_range(ZENODO_URL, 0, 30, 100)
    assert sleeps == []


def test_section_2_cell_surfaces_helper_stderr_and_carries_mirror():
    cells = json.loads(NOTEBOOK.read_text(encoding="utf-8"))["cells"]
    sources = ["".join(c["source"]) for c in cells if c["cell_type"] == "code"]
    prepare = next(s for s in sources if "run('prepare')" in s)
    assert "checked([PYTHON" in prepare and "subprocess.run" not in prepare
    assert "load('mirror.json')" in prepare
    carrier = next(s for s in sources if "def checked" in s)
    assert "capture_output=True" in carrier and "result.stderr" in carrier
    assert "raise RuntimeError" in carrier.split("def checked", 1)[1].split("\ndef ", 1)[0]
    assert max(len(line) for line in carrier.splitlines()) <= 1000, "long lines freeze Colab's editor"
    namespace = {"base64": base64, "hashlib": hashlib, "zlib": zlib, "json": json}
    exec(carrier.split("\nfor name, source", 1)[0], namespace)
    mirror_text = (ROOT / "tools" / "rice_mirror.json").read_text(encoding="utf-8")
    assert namespace["FILES"]["mirror.json"] == mirror_text
    mirror = json.loads(mirror_text)
    assert mirror["members"] == 200 and len(mirror["sha256"]) == 64
    assert assets.urllib.parse.urlsplit(mirror["url"]).hostname in assets.MIRROR_HOSTS
