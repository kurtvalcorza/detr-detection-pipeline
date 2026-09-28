"""Inspect the rice-pest source ZIP without treating filenames as ground truth.

The directory mode accepts a byte range beginning at the central directory.
It inventories source members; it does not qualify annotations or independence.
"""

from __future__ import annotations

import argparse
import json
import struct
import zipfile
import zlib
from collections import Counter
from pathlib import Path


def directory_entries(data: bytes) -> list[dict]:
    """Read ZIP central-directory entries, including ZIP64 size/offset fields."""
    entries = []
    pos = 0
    while data[pos : pos + 4] == b"PK\x01\x02":
        fields = struct.unpack_from("<4s6H3L5H2L", data, pos)
        flags, method = fields[3:5]
        crc, compressed, size = fields[7:10]
        name_len, extra_len, comment_len = fields[10:13]
        offset = fields[16]
        name_start = pos + 46
        name = data[name_start : name_start + name_len].decode("utf-8" if flags & 0x800 else "cp437")
        extra = data[name_start + name_len : name_start + name_len + extra_len]
        cursor = 0
        while cursor + 4 <= len(extra):
            kind, length = struct.unpack_from("<HH", extra, cursor)
            value = extra[cursor + 4 : cursor + 4 + length]
            if kind == 1:
                values = iter(struct.unpack("<" + "Q" * (len(value) // 8), value))
                if size == 0xFFFFFFFF:
                    size = next(values)
                if compressed == 0xFFFFFFFF:
                    compressed = next(values)
                if offset == 0xFFFFFFFF:
                    offset = next(values)
            cursor += 4 + length
        entries.append(
            dict(
                name=name, size=size, compressed=compressed, offset=offset, crc32=f"{crc:08x}", method=method
            )
        )
        pos = name_start + name_len + extra_len + comment_len
    if not entries:
        raise ValueError("No central directory entries found")
    return entries


def summarize(entries: list[dict]) -> dict:
    """Return measured filename-level facts, never inferred annotation quality."""
    files = [r for r in entries if not r["name"].endswith("/")]
    return {
        "entry_count": len(entries),
        "file_count": len(files),
        "uncompressed_bytes": sum(r["size"] for r in files),
        "extensions": dict(Counter(Path(r["name"]).suffix.lower() for r in files)),
        "directories": dict(Counter(str(Path(r["name"]).parent) for r in files)),
        "limitation": (
            "Filename inventory does not establish human annotation, capture provenance, "
            "or independent groups."
        ),
    }


def extract_range(data: bytes, start: int, entries: list[dict], output: Path) -> list[str]:
    """Extract complete members in a cached HTTP range, checking size and CRC."""
    extracted = []
    for row in entries:
        name = row["name"]
        if name.endswith("/") or row["offset"] < start:
            continue
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts or "\\" in name or ":" in name:
            raise ValueError("Unsafe archive path")
        pos = row["offset"] - start
        if pos + 30 > len(data):
            continue
        header = struct.unpack_from("<4s5H3L2H", data, pos)
        if header[0] != b"PK\x03\x04":
            raise ValueError("Local header mismatch")
        begin = pos + 30 + header[-2] + header[-1]
        end = begin + row["compressed"]
        if end > len(data):
            continue
        payload = data[begin:end]
        if row["method"] == 8:
            payload = zlib.decompress(payload, -15)
        elif row["method"] != 0:
            raise ValueError("Unsupported compression")
        if len(payload) != row["size"] or f"{zlib.crc32(payload):08x}" != row["crc32"]:
            raise ValueError("Member integrity mismatch")
        target = output / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(payload)
        extracted.append(name)
    return extracted


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("--directory-range", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.directory_range:
        entries = directory_entries(args.source.read_bytes())
    else:
        with zipfile.ZipFile(args.source) as archive:
            entries = [
                dict(
                    name=i.filename,
                    size=i.file_size,
                    compressed=i.compress_size,
                    offset=i.header_offset,
                    crc32=f"{i.CRC:08x}",
                    method=i.compress_type,
                )
                for i in archive.infolist()
            ]
    args.output.mkdir(parents=True, exist_ok=True)
    for name, data in (("archive_index.json", entries), ("inventory.json", summarize(entries))):
        (args.output / name).write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summarize(entries), indent=2))


if __name__ == "__main__":
    main()
