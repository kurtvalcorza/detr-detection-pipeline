#!/usr/bin/env python3
"""Pin the upstream Hugging Face snapshot: record the immutable commit and every file's size and SHA-256.

The committed manifest lists the files the pipeline loads, with the byte sizes the Hub reported when the
repository was built, but no revision and no digests (`"revision": "unpinned"`, `"sha256": null`). Until
this tool has run, the package refuses to stage, verify or load weights.

What it does, in order:

1. resolves ``--revision`` (default ``main``) to a 40-hex commit with ``HfApi.model_info``;
2. downloads every manifest-listed file **at that commit** into ``weights/<key>/``, replacing any
   committed copy, so every digest is computed from bytes fetched at the pinned commit;
3. computes each file's SHA-256 and byte size, and cross-checks LFS files against the SHA-256 the Hub
   records for them — a disagreement stops the tool without writing anything;
4. writes the manifest (revision, bytes, sha256, totalBytes) and replaces ``MODEL_REVISION = "unpinned"``
   in ``src/<package>/pipeline.py`` with the commit.

It then prints what is left to do by hand: regenerate the notebook, update the prose that says the
snapshot is not yet pinned, and run the validator. Needs network access to huggingface.co and the pinned
``huggingface-hub``; it imports nothing from this repository's package.

Usage (from the repository root):
    python tools/pin_snapshot.py                     # pin whatever `main` resolves to now
    python tools/pin_snapshot.py --revision <commit> # pin a specific commit
    python tools/pin_snapshot.py --dry-run           # resolve and hash, write nothing
"""

# ruff: noqa: E501  -- printed guidance is kept on one line per message
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import re
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SHA40 = re.compile(r"^[0-9a-f]{40}$")


def _template(root: Path) -> dict:
    spec = importlib.util.spec_from_file_location(
        "notebook_template", root / "tools" / "notebook_template.py"
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.TEMPLATE


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def pin(
    root: Path = ROOT,
    revision: str = "main",
    *,
    dry_run: bool = False,
    model_info: Callable[..., Any] | None = None,
    download: Callable[..., Any] | None = None,
) -> int:
    """Pin the snapshot under ``root``; ``model_info``/``download`` default to the huggingface_hub calls."""
    if model_info is None or download is None:
        from huggingface_hub import HfApi, hf_hub_download

        model_info = model_info or HfApi().model_info
        download = download or hf_hub_download

    template = _template(root)
    key = template["weights_key"]
    weights_dir = root / "weights" / key
    manifest_path = weights_dir / "dimer-base-manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    model_id = manifest["modelId"]

    info = model_info(model_id, revision=revision, files_metadata=True)
    commit = info.sha
    if not commit or not SHA40.match(commit):
        print(f"could not resolve {model_id}@{revision} to a commit (got {commit!r})", file=sys.stderr)
        return 1
    siblings = {s.rfilename: s for s in (info.siblings or [])}
    print(f"{model_id}@{revision} -> {commit}")

    missing = [entry["path"] for entry in manifest["files"] if entry["path"] not in siblings]
    if missing:
        print(f"manifest files absent at {commit}: {missing}", file=sys.stderr)
        return 1

    staging = weights_dir if not dry_run else root / "outputs" / "pin-dry-run" / key
    staging.mkdir(parents=True, exist_ok=True)
    pinned_files = []
    for entry in manifest["files"]:
        relative = entry["path"]
        target = staging / relative
        if target.exists():
            target.unlink()
        download(model_id, relative, revision=commit, local_dir=str(staging))
        size = target.stat().st_size
        digest = _sha256(target)
        lfs = getattr(siblings[relative], "lfs", None)
        lfs_sha = getattr(lfs, "sha256", None) if lfs is not None else None
        if lfs_sha and lfs_sha != digest:
            print(
                f"{relative}: downloaded sha256 {digest} != Hub LFS sha256 {lfs_sha}; nothing written",
                file=sys.stderr,
            )
            return 1
        if entry.get("bytes") not in (None, size):
            print(
                f"note: {relative} is {size:,} bytes at {commit}; the manifest recorded {entry['bytes']:,} (updated)"
            )
        pinned_files.append({"path": relative, "bytes": size, "sha256": digest})
        print(
            f"  {relative}: {size:,} bytes  sha256 {digest}"
            + ("  (matches Hub LFS record)" if lfs_sha else "")
        )

    pinned = {
        **manifest,
        "revision": commit,
        "files": pinned_files,
        "totalBytes": sum(f["bytes"] for f in pinned_files),
    }
    if dry_run:
        print(json.dumps(pinned, indent=2))
        return 0

    manifest_path.write_text(json.dumps(pinned, indent=2) + "\n", encoding="utf-8", newline="\n")
    module_path = root / "src" / template["package"] / template.get("entry_module", "pipeline.py")
    text = module_path.read_text(encoding="utf-8")
    new_text, n = re.subn(
        r'^MODEL_REVISION = "[^"]*"$', f'MODEL_REVISION = "{commit}"', text, count=1, flags=re.M
    )
    if n != 1:
        print(
            f"{module_path}: MODEL_REVISION constant not found; manifest written, module unchanged",
            file=sys.stderr,
        )
        return 1
    module_path.write_text(new_text, encoding="utf-8", newline="\n")
    print(f"wrote {manifest_path.relative_to(root)} and MODEL_REVISION in {module_path.relative_to(root)}")

    leftovers = [
        name
        for name in (
            "README.md",
            "MODEL_CARD.md",
            "STATUS.md",
            "docs/WEIGHTS.md",
            "tutorials/README.md",
            "docs/release-verification.md",
        )
        if (root / name).exists()
        and (
            "not yet pinned" in (root / name).read_text(encoding="utf-8")
            or "unpinned" in (root / name).read_text(encoding="utf-8")
        )
    ]
    print("next:")
    print("  1. commit, then run `python tools/build_notebook.py` and commit the regenerated notebook")
    if leftovers:
        print(
            f"  2. replace the 'not yet pinned' statements in: {', '.join(leftovers)} (cite the commit and digests)"
        )
    print("  3. run `python tools/validate_release_assets.py` and `pytest`")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--revision", default="main", help="branch, tag or commit to resolve (default: main)")
    parser.add_argument("--dry-run", action="store_true", help="resolve and hash only; write nothing")
    args = parser.parse_args(argv)
    return pin(ROOT, args.revision, dry_run=args.dry_run)


if __name__ == "__main__":
    raise SystemExit(main())
