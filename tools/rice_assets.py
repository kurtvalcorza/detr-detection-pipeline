"""Fetch only frozen ZIP members, checking HTTP ranges, ZIP metadata and image digests."""

from __future__ import annotations

import hashlib
import io
import struct
import time
import urllib.error
import urllib.parse
import urllib.request
import zipfile
import zlib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import rice_core as core
from PIL import Image

ATTEMPTS = 6
RETRY_AFTER_CAP = 90.0
EXTRA_FIELD_CAP = 1024
MIRROR_HOSTS = {"github.com"}
MIRROR_CEILING = 64 * 1024**2


def retry_delay(error: Exception, attempt: int) -> float:
    """Honour a server's Retry-After on 429/503; otherwise back off exponentially."""
    headers = getattr(error, "headers", None)
    value = headers.get("Retry-After") if headers is not None else None
    try:
        delay = float(value) if value is not None else float(2**attempt)
    except ValueError:
        delay = float(2**attempt)
    return max(1.0, min(delay, RETRY_AFTER_CAP))


def open_with_retry(request: urllib.request.Request, reader):
    """Retry throttling (429), server errors (5xx) and dropped connections; raise other HTTP errors."""
    for attempt in range(ATTEMPTS):
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                return reader(response)
        except urllib.error.HTTPError as error:
            if (error.code != 429 and error.code < 500) or attempt == ATTEMPTS - 1:
                raise
            time.sleep(retry_delay(error, attempt))
        except (urllib.error.URLError, TimeoutError, ConnectionError) as error:
            if attempt == ATTEMPTS - 1:
                raise
            time.sleep(retry_delay(error, attempt))
    raise RuntimeError("Unreachable retry state")


def read_range(url: str, start: int, length: int, archive_bytes: int) -> bytes:
    """Bound every network read; never accept a server silently returning the whole archive."""
    parsed = urllib.parse.urlsplit(url)
    if parsed.scheme != "https" or parsed.hostname != "zenodo.org":
        raise ValueError("Only the pinned Zenodo HTTPS archive is supported")
    if start < 0 or length < 1 or length > 26 * 1024**2 or start + length > archive_bytes:
        raise ValueError("Invalid archive byte range")
    end = start + length - 1
    request = urllib.request.Request(
        url, headers={"Range": f"bytes={start}-{end}", "User-Agent": "DIMER-Rice-Capstone/1"}
    )

    def reader(response):
        if response.status != 206:
            raise ValueError("Archive server did not honour the bounded range")
        if response.headers.get("Content-Range") != f"bytes {start}-{end}/{archive_bytes}":
            raise ValueError("Archive Content-Range mismatch")
        return response.read(length + 1)

    payload = open_with_retry(request, reader)
    if len(payload) != length:
        raise ValueError("Archive range length mismatch")
    return payload


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


def fetch_member(archive: dict, member: dict) -> tuple[bytes, bytes]:
    """Read header and member in one bounded request; a second only for an unusually large extra field."""
    offset = member["offset"]
    guess = 30 + len(member["name"].encode("utf-8")) + EXTRA_FIELD_CAP + member["compressed"]
    data = read_range(archive["url"], offset, min(guess, archive["bytes"] - offset), archive["bytes"])
    header = data[:30]
    fields = struct.unpack("<4s5H3L2H", header)
    length = fields[-2] + fields[-1] + member["compressed"]
    if len(data) >= 30 + length:
        return header, data[30 : 30 + length]
    return header, read_range(archive["url"], offset + 30, length, archive["bytes"])


def fetch_image(root: Path, record: dict, archive: dict) -> Path:
    """Validate existing caches as strictly as freshly downloaded members."""
    target = core.safe_path(root / "data", record["relative_path"])
    if not target.exists():
        member = record["zip_member"]
        if member["size"] != record["bytes"] or member["compressed"] > 25 * 1024**2:
            raise ValueError("Manifest ZIP member size mismatch")
        header, remainder = fetch_member(archive, member)
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


def download_mirror(mirror: dict) -> bytes:
    """One bounded HTTPS download of the sample bundle, accepted only at its pinned size and SHA-256."""
    parsed = urllib.parse.urlsplit(mirror["url"])
    if parsed.scheme != "https" or parsed.hostname not in MIRROR_HOSTS:
        raise ValueError("Mirror URL is not an allowed HTTPS host")
    size = mirror["bytes"]
    if not 0 < size <= MIRROR_CEILING:
        raise ValueError("Mirror bundle size ceiling exceeded")
    request = urllib.request.Request(mirror["url"], headers={"User-Agent": "DIMER-Rice-Capstone/1"})

    def reader(response):
        if urllib.parse.urlsplit(response.geturl()).scheme != "https":
            raise ValueError("Mirror redirected away from HTTPS")
        return response.read(size + 1)

    payload = open_with_retry(request, reader)
    if len(payload) != size or hashlib.sha256(payload).hexdigest() != mirror["sha256"]:
        raise ValueError("Mirror bundle size/SHA-256 mismatch")
    return payload


def extract_mirror(root: Path, bundle: bytes, records: list[dict]) -> int:
    """Write only the manifest's own paths, each re-checked against its record before it touches disk."""
    wanted = {r["relative_path"]: r for r in records}
    with zipfile.ZipFile(io.BytesIO(bundle)) as archive:
        infos = archive.infolist()
        for info in infos:
            core.safe_path(root / "data", info.filename)
        if len(infos) != len(wanted) or sorted(i.filename for i in infos) != sorted(wanted):
            raise ValueError("Mirror bundle members do not match the manifest")
        written = 0
        for info in infos:
            record = wanted[info.filename]
            if info.file_size != record["bytes"]:
                raise ValueError("Mirror member size mismatch")
            payload = archive.read(info)
            if len(payload) != record["bytes"] or hashlib.sha256(payload).hexdigest() != record["sha256"]:
                raise ValueError("Mirror member SHA-256 mismatch")
            target = core.safe_path(root / "data", info.filename)
            if not target.exists():
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(payload)
                written += 1
    return written


def prepare_images(root: Path, manifest: dict, mirror: dict | None = None) -> dict:
    """Try the hash-pinned mirror, fall back to rate-limit-aware Zenodo ranges, then verify every image."""
    archive = manifest["archive"]
    records = manifest["records"]
    if len(records) > 1000 or sum(r["bytes"] for r in records) > 500 * 1024**2:
        raise ValueError("Frozen sample ceiling exceeded")
    source = "zenodo"
    if mirror is not None:
        try:
            if mirror["members"] != len(records):
                raise ValueError("Mirror member count does not match the manifest")
            extract_mirror(root, download_mirror(mirror), records)
            source = "mirror"
        except Exception as error:  # noqa: BLE001 - any mirror failure falls back to the source of record
            reason = f"{type(error).__name__}: {error}"
            print(f"Mirror unavailable ({reason}); falling back to Zenodo.", flush=True)
    with ThreadPoolExecutor(max_workers=2) as pool:
        paths = list(pool.map(lambda row: fetch_image(root, row, archive), records))
    return {
        "images": len(paths),
        "bytes": sum(p.stat().st_size for p in paths),
        "sha256_verified": True,
        "source": source,
    }
