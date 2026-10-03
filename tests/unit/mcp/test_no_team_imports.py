"""Import-isolation guards for the MCP package and the runtime-neutral core.

Two static invariants are enforced by AST-walking the source tree:

* ``cs_copilot.mcp`` never imports the Agno team, factories, registry,
  configured model backend, or the Chainlit app. The only agents submodule it
  may import is :mod:`cs_copilot.agents.instructions` (served verbatim as MCP
  prompts). The opt-in delegation helpers reach the team through explicitly
  listed dynamic imports.
* The runtime-neutral core (routing, workflows, skills, storage, tracking and
  the shared capability/execution modules) imports neither runtime: no
  ``agno``, no ``cs_copilot.agents``, no ``cs_copilot.mcp``.

Relative imports are resolved against each file's package, ``from X import Y``
is checked as ``X.Y``, literal ``__import__``/``import_module`` targets and
``factory("module:Class")`` toolkit targets are checked like static imports,
and prefixes only match on dot boundaries. The runtime counterpart of this
guard, which inspects ``sys.modules`` in a fresh interpreter, lives in
``tests/unit/test_runtime_import_isolation.py``.
"""

from __future__ import annotations

import ast
import importlib.util
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import pytest

SRC_ROOT = Path(__file__).resolve().parents[3] / "src"
PACKAGE_ROOT = SRC_ROOT / "cs_copilot"

MCP_FORBIDDEN = ("cs_copilot.agents", "cs_copilot.model_config", "chainlit_app")
MCP_ALLOWED = ("cs_copilot.agents.instructions",)
# Dynamic imports that deliberately reach the Agno runtime from opt-in MCP paths.
MCP_DYNAMIC_EXCEPTIONS = {
    "mcp/agno_delegate.py": frozenset(
        {
            "cs_copilot.agents.execution_binding",
            "cs_copilot.agents.session_runs",
            "cs_copilot.agents.teams",
        }
    ),
    "mcp/llm/agno_model.py": frozenset({"cs_copilot.model_config"}),
}

NEUTRAL_FORBIDDEN = (
    "agno",
    "cs_copilot.agents",
    "cs_copilot.mcp",
    "cs_copilot.model_config",
    "chainlit_app",
)
NEUTRAL_GLOBS = (
    "routing.py",
    "capabilities.py",
    "workflows/**/*.py",
    "skills/**/*.py",
    "storage/**/*.py",
    "tracking/**/*.py",
    "execution/**/*.py",
)
# Explicit, reviewed exceptions to the neutral-core rule (currently none).
NEUTRAL_EXCEPTIONS: dict[str, frozenset[str]] = {}

# Modules whose non-constant dynamic imports are reviewed by other means. The
# MCP tool-spec factory resolves ``factory("module:Class")`` literals, which
# are collected and checked as imports below.
NON_CONSTANT_DYNAMIC_IMPORTS = frozenset({"mcp/tool_specs/common.py"})

_DYNAMIC_IMPORT_FUNCTIONS = frozenset({"__import__", "import_module"})


@dataclass(frozen=True)
class ImportRef:
    lineno: int
    name: str
    kind: str  # "static" | "dynamic" | "factory" | "dynamic-non-constant"


def _module_name(path: Path) -> str:
    parts = list(path.relative_to(SRC_ROOT).with_suffix("").parts)
    if parts[-1] == "__init__":
        parts.pop()
    return ".".join(parts)


def _package_name(path: Path) -> str:
    module = _module_name(path)
    return module if path.name == "__init__.py" else module.rpartition(".")[0]


def _call_name(node: ast.Call) -> str | None:
    if isinstance(node.func, ast.Name):
        return node.func.id
    if isinstance(node.func, ast.Attribute):
        return node.func.attr
    return None


def _string_constant(node: ast.AST | None) -> str | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    return None


def iter_import_refs(
    source: str, *, package: str, filename: str = "<unknown>"
) -> Iterator[ImportRef]:
    """Yield every module reference made by ``source``, fully qualified."""

    tree = ast.parse(source, filename=filename)
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                yield ImportRef(node.lineno, alias.name, "static")
        elif isinstance(node, ast.ImportFrom):
            base = node.module or ""
            if node.level:
                base = importlib.util.resolve_name("." * node.level + base, package)
            for alias in node.names:
                name = base if alias.name == "*" else f"{base}.{alias.name}"
                yield ImportRef(node.lineno, name, "static")
        elif isinstance(node, ast.Call):
            called = _call_name(node)
            first = node.args[0] if node.args else None
            if called in _DYNAMIC_IMPORT_FUNCTIONS:
                target = _string_constant(first)
                if target is None:
                    yield ImportRef(node.lineno, "<non-constant>", "dynamic-non-constant")
                    continue
                if target.startswith("."):
                    anchor = next(
                        (
                            _string_constant(keyword.value)
                            for keyword in node.keywords
                            if keyword.arg == "package"
                        ),
                        None,
                    )
                    target = importlib.util.resolve_name(target, anchor or package)
                yield ImportRef(node.lineno, target, "dynamic")
            elif called == "factory":
                target = _string_constant(first)
                if target and ":" in target:
                    yield ImportRef(node.lineno, target.partition(":")[0], "factory")


def _matches(name: str, prefix: str) -> bool:
    return name == prefix or name.startswith(prefix + ".")


def _refs_for(path: Path) -> list[ImportRef]:
    return list(
        iter_import_refs(
            path.read_text(encoding="utf-8"),
            package=_package_name(path),
            filename=str(path),
        )
    )


def _rel(path: Path) -> str:
    return path.relative_to(PACKAGE_ROOT).as_posix()


def _mcp_files() -> list[Path]:
    return sorted((PACKAGE_ROOT / "mcp").rglob("*.py"))


def _neutral_files() -> list[Path]:
    files: set[Path] = set()
    for pattern in NEUTRAL_GLOBS:
        files.update(path for path in PACKAGE_ROOT.glob(pattern) if path.is_file())
    return sorted(files)


def _mcp_violations(path: Path) -> list[str]:
    rel = _rel(path)
    violations = []
    for ref in _refs_for(path):
        if ref.kind == "dynamic-non-constant":
            if rel not in NON_CONSTANT_DYNAMIC_IMPORTS:
                violations.append(f"line {ref.lineno}: non-constant dynamic import")
            continue
        if not any(_matches(ref.name, prefix) for prefix in MCP_FORBIDDEN):
            continue
        if any(_matches(ref.name, allowed) for allowed in MCP_ALLOWED):
            continue
        if ref.kind == "dynamic" and ref.name in MCP_DYNAMIC_EXCEPTIONS.get(rel, frozenset()):
            continue
        violations.append(f"line {ref.lineno}: {ref.kind} import of {ref.name!r}")
    return violations


def _neutral_violations(path: Path) -> list[str]:
    rel = _rel(path)
    violations = []
    for ref in _refs_for(path):
        if ref.kind == "dynamic-non-constant":
            violations.append(f"line {ref.lineno}: non-constant dynamic import")
            continue
        if not any(_matches(ref.name, prefix) for prefix in NEUTRAL_FORBIDDEN):
            continue
        if ref.name in NEUTRAL_EXCEPTIONS.get(rel, frozenset()):
            continue
        violations.append(f"line {ref.lineno}: {ref.kind} import of {ref.name!r}")
    return violations


@pytest.mark.parametrize("path", _mcp_files(), ids=_rel)
def test_no_forbidden_imports(path: Path):
    violations = _mcp_violations(path)
    assert not violations, f"{_rel(path)} breaks MCP import isolation: {violations}"


@pytest.mark.parametrize("path", _neutral_files(), ids=_rel)
def test_neutral_core_has_no_runtime_imports(path: Path):
    violations = _neutral_violations(path)
    assert not violations, f"{_rel(path)} imports a runtime from the neutral core: {violations}"


# The execution kernel is imported by the MCP adapter at module level, before
# the server applies SESSION_ID. Storage reads SESSION_ID at import time, so the
# kernel may only reach these dependencies from inside functions.
EXECUTION_LAZY_ONLY = (
    "cs_copilot.storage",
    "cs_copilot.workflows",
    "cs_copilot.tools",
    "pandas",
    "numpy",
    "fsspec",
)


def _is_type_checking_guard(test: ast.expr) -> bool:
    return (isinstance(test, ast.Name) and test.id == "TYPE_CHECKING") or (
        isinstance(test, ast.Attribute) and test.attr == "TYPE_CHECKING"
    )


def _module_level_imports(path: Path) -> list[tuple[int, str]]:
    """Return fully qualified imports executed when ``path`` is imported."""

    return _module_level_imports_in(
        path.read_text(encoding="utf-8"),
        package=_package_name(path),
        filename=str(path),
    )


def _module_level_imports_in(
    source: str, *, package: str, filename: str = "<unknown>"
) -> list[tuple[int, str]]:
    found: list[tuple[int, str]] = []

    def visit(statements: list[ast.stmt]) -> None:
        for node in statements:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            if isinstance(node, ast.If) and _is_type_checking_guard(node.test):
                visit(node.orelse)
                continue
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                source = ast.unparse(node)
                found.extend(
                    (ref.lineno + node.lineno - 1, ref.name)
                    for ref in iter_import_refs(source, package=package)
                )
                continue
            for field_name in ("body", "orelse", "finalbody", "handlers"):
                nested = getattr(node, field_name, None)
                if isinstance(nested, list):
                    visit([item for item in nested if isinstance(item, ast.stmt)])
                    for handler in nested:
                        if isinstance(handler, ast.ExceptHandler):
                            visit(handler.body)

    visit(ast.parse(source, filename=filename).body)
    return found


def _execution_files() -> list[Path]:
    return sorted((PACKAGE_ROOT / "execution").rglob("*.py"))


@pytest.mark.parametrize("path", _execution_files(), ids=_rel)
def test_execution_kernel_imports_heavy_dependencies_lazily(path: Path):
    eager = [
        f"line {lineno}: {name!r}"
        for lineno, name in _module_level_imports(path)
        if any(_matches(name, prefix) for prefix in EXECUTION_LAZY_ONLY)
    ]
    assert not eager, f"{_rel(path)} imports heavy dependencies at module level: {eager}"


def test_execution_kernel_has_no_packaging_excluded_directories():
    excluded = {"runs", "events", "artifacts", "sessions", ".staging"}
    directories = {path.name for path in (PACKAGE_ROOT / "execution").rglob("*") if path.is_dir()}
    assert not directories & excluded


def test_module_level_import_scan_skips_functions_and_type_checking():
    assert _module_level_imports(PACKAGE_ROOT / "execution" / "__init__.py") == []
    source = (
        "from typing import TYPE_CHECKING\n"
        "import pandas\n"
        "if TYPE_CHECKING:\n"
        "    import numpy\n"
        "def lazy():\n"
        "    import fsspec\n"
        "try:\n"
        "    from cs_copilot.storage import S3\n"
        "except ImportError:\n"
        "    import cs_copilot.workflows\n"
    )
    names = {name for _, name in _module_level_imports_in(source, package="cs_copilot.execution")}
    assert {"pandas", "cs_copilot.storage.S3", "cs_copilot.workflows"} <= names
    assert "numpy" not in names and "fsspec" not in names


@pytest.mark.parametrize(
    ("exceptions", "kind"),
    [(MCP_DYNAMIC_EXCEPTIONS, "dynamic"), (NEUTRAL_EXCEPTIONS, None)],
    ids=["mcp-dynamic", "neutral"],
)
def test_import_exceptions_are_still_needed(exceptions, kind):
    for rel, names in exceptions.items():
        path = PACKAGE_ROOT / rel
        assert path.is_file(), f"stale import exception for missing file {rel}"
        used = {ref.name for ref in _refs_for(path) if kind is None or ref.kind == kind}
        stale = sorted(names - used)
        assert not stale, f"stale import exceptions for {rel}: {stale}"


def test_non_constant_dynamic_import_allowances_are_still_needed():
    for rel in NON_CONSTANT_DYNAMIC_IMPORTS:
        kinds = {ref.kind for ref in _refs_for(PACKAGE_ROOT / rel)}
        assert "dynamic-non-constant" in kinds, f"stale non-constant allowance for {rel}"


def test_guard_scopes_are_not_empty():
    assert _mcp_files()
    neutral = {_rel(path) for path in _neutral_files()}
    assert {"routing.py", "workflows/runtime.py", "storage/client.py"} <= neutral


@pytest.mark.parametrize(
    ("source", "package", "expected"),
    [
        ("from ..agents import teams", "cs_copilot.mcp", "cs_copilot.agents.teams"),
        ("from . import instructions", "cs_copilot.agents", "cs_copilot.agents.instructions"),
        ("from cs_copilot import agents", "cs_copilot.mcp", "cs_copilot.agents"),
        ("import cs_copilot.model_config as mc", "cs_copilot.mcp", "cs_copilot.model_config"),
        ("__import__('cs_copilot.agents.teams')", "cs_copilot.mcp", "cs_copilot.agents.teams"),
        (
            "importlib.import_module('..agents.registry', package='cs_copilot.mcp')",
            "cs_copilot.mcp",
            "cs_copilot.agents.registry",
        ),
        ("factory('chainlit_app:App')", "cs_copilot.mcp", "chainlit_app"),
    ],
)
def test_import_refs_are_fully_qualified(source, package, expected):
    names = {ref.name for ref in iter_import_refs(source, package=package)}
    assert expected in names


@pytest.mark.parametrize(
    "name",
    ["cs_copilot.agents.teams", "cs_copilot.agents", "cs_copilot.model_config", "chainlit_app"],
)
def test_forbidden_prefixes_match(name):
    assert any(_matches(name, prefix) for prefix in MCP_FORBIDDEN)
    assert not any(_matches(name, allowed) for allowed in MCP_ALLOWED)


def test_prefixes_match_only_on_dot_boundaries():
    assert not _matches("cs_copilot.agentsx", "cs_copilot.agents")
    assert not _matches("agnostic", "agno")
    assert _matches("agno.agent.Agent", "agno")
    assert _matches("cs_copilot.agents.instructions", "cs_copilot.agents.instructions")
