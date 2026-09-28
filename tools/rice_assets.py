"""Fetch only frozen ZIP members, checking HTTP ranges, ZIP metadata and image digests."""

from __future__ import annotations

import hashlib
import struct
import time
import urllib.error
import urllib.parse
import urllib.request
import zlib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import rice_core as core
from PIL import Image


def read_range(url: str, start: int, length: int, archive_bytes: int) -> bytes:
    """Bound every network read; never accept a server silently returning the whole archive."""
    parsed = urllib.parse.urlsplit(url)
    if parsed.scheme != "https" or parsed.hostname != "zenodo.org":
        raise ValueError("Only the pinned Zenodo HTTPS archive is supported")
    if start < 0 or length < 1 or length > 26 * 1024**2 or start + length > archive_bytes:
        raise ValueError("Invalid archive byte range")
    end = start + length - 1
    for attempt in range(3):
        request = urllib.request.Request(
            url, headers={"Range": f"bytes={start}-{end}", "User-Agent": "DIMER-Rice-Capstone/1"}
        )
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                if response.status != 206:
                    raise ValueError("Archive server did not honour the bounded range")
                if response.headers.get("Content-Range") != f"bytes {start}-{end}/{archive_bytes}":
                    raise ValueError("Archive Content-Range mismatch")
                payload = response.read(length + 1)
            if len(payload) != length:
                raise ValueError("Archive range length mismatch")
            return payload
        except (urllib.error.URLError, TimeoutError, ConnectionError):
            if attempt == 2:
                raise
            time.sleep(2**attempt)
    raise RuntimeError("Unreachable range retry state")


def unpack_member(header: bytes, remainder: bytes, member: dict) -> bytes:
    """Decode a pinned local ZIP member without extracting arbitrary paths."""
    if len(header) != 30:
        raise ValueError("Invalid local ZIP header length")
    fields = struct.unpack("<4s5H3L2H", header)
    signature, _, flags, method = fields[:4]
    name_size, extra_size = fields[-2:]
    if signature != b"PK\x03\x04" or flags & 1 or method != member["method"]:
        raise ValueError("Local ZIP signature/encryption/compression mismatch")
    name = remainder[:name_size].decode("utf-8" if flags & 0x800 else "cp437")
    if name != member["name"]:
        raise ValueError("ZIP member identity mismatch")
    core.safe_path(Path("."), name)
    expected = name_size + extra_size + member["compressed"]
    if len(remainder) != expected or member["size"] > 25 * 1024**2:
        raise ValueError("ZIP member byte ceiling or length mismatch")
    payload = remainder[name_size + extra_size :]
    if method == 8:
        decoder = zlib.decompressobj(-15)
        payload = decoder.decompress(payload, member["size"] + 1)
        if not decoder.eof or decoder.unconsumed_tail or decoder.unused_data:
            raise ValueError("ZIP deflate stream exceeds declared size or contains extra bytes")
    elif method != 0:
        raise ValueError("Unsupported ZIP compression")
    if len(payload) != member["size"] or f"{zlib.crc32(payload):08x}" != member["crc32"]:
        raise ValueError("ZIP member size/CRC mismatch")
    return payload


def fetch_image(root: Path, record: dict, archive: dict) -> Path:
    """Validate existing caches as strictly as freshly downloaded members."""
    target = core.safe_path(root / "data", record["relative_path"])
    if not target.exists():
        member = record["zip_member"]
        if member["size"] != record["bytes"] or member["compressed"] > 25 * 1024**2:
            raise ValueError("Manifest ZIP member size mismatch")
        header = read_range(archive["url"], member["offset"], 30, archive["bytes"])
        fields = struct.unpack("<4s5H3L2H", header)
        length = fields[-2] + fields[-1] + member["compressed"]
        remainder = read_range(archive["url"], member["offset"] + 30, length, archive["bytes"])
        payload = unpack_member(header, remainder, member)
        if hashlib.sha256(payload).hexdigest() != record["sha256"]:
            raise ValueError("Downloaded image SHA-256 mismatch")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(payload)
    if target.stat().st_size != record["bytes"] or core.sha256(target) != record["sha256"]:
        raise ValueError("Image cache byte/hash mismatch")
    with Image.open(target) as image:
        if image.format not in {"JPEG", "PNG", "WEBP"}:
            raise ValueError("Unsupported image format")
        if image.size != (record["width"], record["height"]):
            raise ValueError("Image dimensions mismatch")
        image.load()
    return target


def prepare_images(root: Path, manifest: dict) -> dict:
    """Fetch a frozen sample with bounded concurrency, preserving manifest order."""
    archive = manifest["archive"]
    records = manifest["records"]
    if len(records) > 1000 or sum(r["bytes"] for r in records) > 500 * 1024**2:
        raise ValueError("Frozen sample ceiling exceeded")
    with ThreadPoolExecutor(max_workers=4) as pool:
        paths = list(pool.map(lambda row: fetch_image(root, row, archive), records))
    return {"images": len(paths), "bytes": sum(p.stat().st_size for p in paths), "sha256_verified": True}
