"""Offline ZIP/member integrity tests; no public network needed."""

import hashlib
import io
import struct
import sys
import zipfile
from pathlib import Path

import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import rice_assets as assets  # noqa: E402


def fixture():
    photo = io.BytesIO()
    Image.new("RGB", (32, 32), "red").save(photo, format="PNG")
    content = photo.getvalue()
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w", zipfile.ZIP_DEFLATED) as bundle:
        bundle.writestr("images/red.png", content)
    with zipfile.ZipFile(stream) as bundle:
        info = bundle.getinfo("images/red.png")
        member = {
            "name": info.filename,
            "offset": 0,
            "compressed": info.compress_size,
            "size": info.file_size,
            "crc32": f"{info.CRC:08x}",
            "method": info.compress_type,
        }
    raw = stream.getvalue()
    fields = struct.unpack("<4s5H3L2H", raw[:30])
    end = 30 + fields[-2] + fields[-1] + member["compressed"]
    return content, raw[:30], raw[30:end], member


def test_exact_member_and_corruption_refusal():
    content, header, body, member = fixture()
    assert assets.unpack_member(header, body, member) == content
    with pytest.raises(ValueError, match="identity"):
        assets.unpack_member(header, body, {**member, "name": "different.png"})
    with pytest.raises(ValueError, match="CRC"):
        assets.unpack_member(header, body, {**member, "crc32": "00000000"})
    with pytest.raises(ValueError, match="ceiling"):
        assets.unpack_member(header, body, {**member, "size": 26 * 1024**2})


def test_frozen_fetch_and_cache_tampering(tmp_path, monkeypatch):
    content, header, body, member = fixture()
    monkeypatch.setattr(assets, "read_range", lambda url, start, length, size: header if start == 0 else body)
    record = {
        "relative_path": "images/red.png",
        "zip_member": member,
        "bytes": len(content),
        "sha256": hashlib.sha256(content).hexdigest(),
        "width": 32,
        "height": 32,
    }
    path = assets.fetch_image(tmp_path, record, {"url": "fixture", "bytes": 999})
    assert path.read_bytes() == content
    path.write_bytes(b"changed")
    with pytest.raises(ValueError, match="cache"):
        assets.fetch_image(tmp_path, record, {"url": "fixture", "bytes": 999})


def test_range_host_and_size_refusals():
    with pytest.raises(ValueError, match="HTTPS"):
        assets.read_range("http://example.com/a.zip", 0, 30, 999)
    with pytest.raises(ValueError, match="range"):
        assets.read_range("https://zenodo.org/a.zip", 900, 200, 999)
