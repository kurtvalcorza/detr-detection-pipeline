"""NOTEBOOK_SPEC 2.1 parity tests (PAR1–PAR3) for the standalone tutorial notebook.

The notebook carries `src/<package>/*.py` verbatim; these tests fail whenever a carried cell, the inline
manifest, or the inline pins diverge from the repository at HEAD.
"""

from __future__ import annotations

import importlib.util
import json
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, TOOLS / f"{name}.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


build = _load("build_notebook")
TEMPLATE = _load("notebook_template").TEMPLATE
NOTEBOOK = ROOT / "tutorials" / TEMPLATE["notebook_name"]
MANIFEST = ROOT / "weights" / TEMPLATE["weights_key"] / "dimer-base-manifest.json"
REWRITES = TEMPLATE.get("rewrites", build.DEFAULT_REWRITES)
PKG_DIR = ROOT / "src" / TEMPLATE["package"]


@pytest.fixture(scope="module")
def notebook() -> dict:
    if not NOTEBOOK.exists():
        pytest.skip(f"{NOTEBOOK.name} not generated yet")
    return json.loads(NOTEBOOK.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def context() -> dict:
    return build.load_context(ROOT, TEMPLATE)


def _cells(notebook: dict, cell_type: str) -> list[dict]:
    return [c for c in notebook["cells"] if c["cell_type"] == cell_type]


def _source(cell: dict) -> str:
    src = cell["source"]
    return "".join(src) if isinstance(src, list) else src


def _tagged(notebook: dict) -> list[dict]:
    return [
        c for c in _cells(notebook, "code") if c.get("metadata", {}).get("dimer", {}).get("embedded_module")
    ]


def test_par1_every_module_is_carried_in_dependency_order(notebook: dict, context: dict) -> None:
    names = [c["metadata"]["dimer"]["embedded_module"] for c in _tagged(notebook)]
    assert names == context["module_rels"], "carried modules differ from the generator's dependency order"
    assert len(names) == len(TEMPLATE["modules"])


def test_par1_each_carried_cell_equals_its_repository_module(notebook: dict, context: dict) -> None:
    texts = {m: (PKG_DIR / m).read_text(encoding="utf-8") for m in context["modules"]}
    expected = build.apply_rewrites(texts, REWRITES)
    for cell, module in zip(_tagged(notebook), context["modules"], strict=True):
        assert _source(cell).rstrip("\n") + "\n" == expected[module], (
            f"embedded {module} drifted from src/; regenerate"
        )


def test_par1_each_carried_cell_records_its_own_module_digest(notebook: dict, context: dict) -> None:
    for cell in _tagged(notebook):
        meta = cell["metadata"]["dimer"]
        assert meta["module_sha256"] == context["per_module_sha256"][meta["embedded_module"]]


def test_par1_rewrite_rules_are_the_only_difference(context: dict) -> None:
    texts = {m: (PKG_DIR / m).read_text(encoding="utf-8") for m in context["modules"]}
    rewritten = build.apply_rewrites(dict(texts), REWRITES)
    changed: list[tuple[str, str]] = []
    for module, original in texts.items():
        for a, b in zip(original.splitlines(), rewritten[module].splitlines(), strict=False):
            if a != b:
                changed.append((a, b))
    rewrite_lines = [(a, b) for a, b in changed if "standalone rewrite" in b and "__file__" in a]
    import_lines = [(a, b) for a, b in changed if a.lstrip().startswith("from .")]
    assert len(rewrite_lines) == len(REWRITES)
    assert changed and len(rewrite_lines) + len(import_lines) == len(changed), changed


def test_par2_inline_manifest_and_pins_match_repository(notebook: dict) -> None:
    code = "\n".join(_source(c) for c in _cells(notebook, "code"))
    inline = re.search(r"^MANIFEST = (\{.*?^\})$", code, re.M | re.S)
    assert inline, "model cell must carry MANIFEST = {...}"
    assert json.loads(inline.group(1)) == json.loads(MANIFEST.read_text(encoding="utf-8"))
    pins_block = re.search(r"^PINS = \[(.*?)^\]", code, re.M | re.S)
    assert pins_block, "install cell must carry PINS = [...]"
    assert re.findall(r"'([^']+)'", pins_block.group(1)) == build._pins(ROOT)
    meta = notebook["metadata"]["dimer"]
    assert meta["standalone"] is True
    assert meta["notebook_spec"] == build.NOTEBOOK_SPEC == "2.1"
    assert meta["generated_from"]["module"] == f"src/{TEMPLATE['package']}/pipeline.py"
    assert meta["generated_from"]["module_sha256"] == build.load_context(ROOT, TEMPLATE)["module_sha256"]


def test_par3_generator_check_is_clean(notebook: dict) -> None:
    # The recorded revision is a provenance label carried through the check (see build_notebook.py
    # --check); content drift is what fails this comparison.
    recorded = notebook["metadata"]["dimer"]["generated_from"]["revision"]
    rendered = build.to_bytes(build.render(ROOT, TEMPLATE, recorded))
    current = NOTEBOOK.read_bytes().replace(b"\r\n", b"\n")  # autocrlf checkouts are CRLF
    assert current == rendered, "notebook is stale; run python tools/build_notebook.py"


def test_st1_primary_path_has_no_repository_dependency(notebook: dict) -> None:
    code = "\n".join(_source(c) for c in _cells(notebook, "code"))
    assert "git" not in re.findall(r"subprocess\.run\(\[([^\]]*)\]", code).__str__()
    assert f"import {TEMPLATE['package']}" not in code
    assert f"from {TEMPLATE['package']}" not in code
    assert "github.com" not in code
