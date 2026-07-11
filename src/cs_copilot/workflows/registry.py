"""Pure-Python loader for reusable cs_copilot workflow contracts."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

WORKFLOW_MD = "WORKFLOW.md"
WORKFLOWS_ENV = "CS_COPILOT_WORKFLOWS_DIR"
# Required top-level frontmatter fields (Agent-Skills style: name + description).
_REQUIRED_FIELDS = {"name", "description"}


@dataclass(frozen=True)
class WorkflowSpec:
    """Metadata and content for one reusable workflow contract."""

    slug: str
    title: str
    summary: str
    status: str
    tags: tuple[str, ...]
    keywords: tuple[str, ...]
    preflight_tools: tuple[str, ...]
    required_tools: tuple[str, ...]
    optional_tools: tuple[str, ...]
    expected_artifacts: tuple[str, ...]
    recommended_prompt: str | None
    workflow_md: str
    path: Path

    def as_dict(self, *, include_content: bool = False) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "slug": self.slug,
            "title": self.title,
            "summary": self.summary,
            "status": self.status,
            "tags": list(self.tags),
            "keywords": list(self.keywords),
            "preflight_tools": list(self.preflight_tools),
            "required_tools": list(self.required_tools),
            "optional_tools": list(self.optional_tools),
            "expected_artifacts": list(self.expected_artifacts),
            "recommended_prompt": self.recommended_prompt,
        }
        if include_content:
            payload["workflow_md"] = self.workflow_md
        return payload


class WorkflowRegistry:
    """Discover and serve local workflow contracts."""

    def __init__(self, root: str | os.PathLike[str] | None = None) -> None:
        self.root = Path(root).resolve() if root is not None else discover_workflow_root()
        self._workflows: dict[str, WorkflowSpec] | None = None

    def list_workflows(self) -> list[WorkflowSpec]:
        return list(self._load().values())

    def get_workflow(self, slug: str) -> WorkflowSpec:
        normalized = _normalize_slug(slug)
        try:
            return self._load()[normalized]
        except KeyError as exc:
            available = ", ".join(sorted(self._load()))
            raise KeyError(f"Unknown workflow '{slug}'. Available workflows: {available}") from exc

    def search_workflows(self, query: str, *, limit: int = 10) -> list[WorkflowSpec]:
        terms = _tokenize(query)
        query_lc = (query or "").lower()
        scored: list[tuple[int, WorkflowSpec]] = []
        for index, spec in enumerate(self.list_workflows()):
            haystack = _search_text(spec)
            if not terms:
                score = max(1, 100 - index)
            else:
                score = sum(_term_score(term, haystack, spec) for term in terms)
                score += _keyword_score(query_lc, spec.keywords)
            if score > 0:
                scored.append((score, spec))
        scored.sort(key=lambda item: (-item[0], item[1].slug))
        return [spec for _score, spec in scored[:limit]]

    def _load(self) -> dict[str, WorkflowSpec]:
        if self._workflows is not None:
            return self._workflows
        if not self.root.exists():
            raise FileNotFoundError(f"Workflow catalog root does not exist: {self.root}")
        workflows: dict[str, WorkflowSpec] = {}
        for path in sorted(item for item in self.root.iterdir() if item.is_dir()):
            spec = _load_workflow(path)
            if spec.slug in workflows:
                raise ValueError(f"Duplicate workflow slug: {spec.slug}")
            workflows[spec.slug] = spec
        self._workflows = workflows
        return workflows


_DEFAULT_REGISTRY: WorkflowRegistry | None = None


def list_workflows() -> list[WorkflowSpec]:
    return _registry().list_workflows()


def get_workflow(slug: str) -> WorkflowSpec:
    return _registry().get_workflow(slug)


def search_workflows(query: str, *, limit: int = 10) -> list[WorkflowSpec]:
    return _registry().search_workflows(query, limit=limit)


def discover_workflow_root() -> Path:
    env_root = os.getenv(WORKFLOWS_ENV)
    candidates: list[Path] = []
    if env_root:
        candidates.append(Path(env_root))

    here = Path(__file__).resolve()
    for parent in here.parents:
        candidates.append(parent / "workflow_catalog")
    candidates.append(here.parent / "catalog")

    for candidate in candidates:
        if _looks_like_workflow_root(candidate):
            return candidate.resolve()
    return candidates[0].resolve() if candidates else Path("workflow_catalog").resolve()


def _registry() -> WorkflowRegistry:
    global _DEFAULT_REGISTRY
    if _DEFAULT_REGISTRY is None:
        _DEFAULT_REGISTRY = WorkflowRegistry()
    return _DEFAULT_REGISTRY


def _looks_like_workflow_root(path: Path) -> bool:
    return path.is_dir() and any((child / WORKFLOW_MD).is_file() for child in path.iterdir())


def _load_workflow(path: Path) -> WorkflowSpec:
    md_path = path / WORKFLOW_MD
    if not md_path.is_file():
        raise FileNotFoundError(f"Missing {WORKFLOW_MD}: {md_path}")

    front, body = _split_frontmatter(md_path.read_text(encoding="utf-8"), md_path)
    data = _load_yaml(front, md_path)
    metadata = data.get("metadata") or {}
    if not isinstance(metadata, dict):
        raise ValueError(f"{md_path} frontmatter 'metadata' must be a mapping")

    missing = sorted(field for field in _REQUIRED_FIELDS if not data.get(field))
    if missing:
        raise ValueError(f"{md_path} frontmatter is missing required fields: {', '.join(missing)}")

    slug = _normalize_slug(str(data["name"]))
    if slug != path.name:
        raise ValueError(f"{md_path} name '{slug}' must match directory name '{path.name}'")

    recommended_prompt = metadata.get("recommended_prompt")
    return WorkflowSpec(
        slug=slug,
        title=str(metadata.get("title") or _title_from_slug(slug)).strip(),
        summary=str(data["description"]).strip(),
        status=str(metadata.get("status") or "stable").strip(),
        tags=_as_tuple(metadata.get("tags")),
        keywords=_as_tuple(metadata.get("keywords")),
        preflight_tools=_as_tuple(metadata.get("preflight_tools")),
        required_tools=_as_tuple(metadata.get("required_tools")),
        optional_tools=_as_tuple(metadata.get("optional_tools")),
        expected_artifacts=_as_tuple(metadata.get("expected_artifacts")),
        recommended_prompt=str(recommended_prompt).strip() if recommended_prompt else None,
        workflow_md=body,
        path=path,
    )


def _split_frontmatter(text: str, path: Path) -> tuple[str, str]:
    """Split leading ``---`` YAML frontmatter from the Markdown body."""
    if not text.startswith("---"):
        raise ValueError(f"{path} is missing YAML frontmatter (expected '---' on the first line)")
    lines = text.splitlines()
    end = next((idx for idx in range(1, len(lines)) if lines[idx].strip() == "---"), None)
    if end is None:
        raise ValueError(f"{path} frontmatter is not terminated with '---'")
    return "\n".join(lines[1:end]), "\n".join(lines[end + 1 :]).strip()


def _load_yaml(text: str, path: Path) -> dict[str, Any]:
    try:
        import yaml  # type: ignore

        loaded = yaml.safe_load(text)
    except ModuleNotFoundError:
        return _parse_yaml_block(text.splitlines(), path)
    if not isinstance(loaded, dict):
        raise ValueError(f"{path} frontmatter must contain a mapping")
    return loaded


def _indent(line: str) -> int:
    return len(line) - len(line.lstrip(" "))


def _parse_yaml_block(lines: Sequence[str], path: Path) -> dict[str, Any]:
    """Indentation-aware YAML mapping parser (stdlib fallback, no PyYAML).

    Supports scalars, ``- item`` lists, and nested mappings — enough for the
    WORKFLOW.md frontmatter shape (``name``/``description`` plus a ``metadata:``
    block). Not a general YAML parser.
    """
    rows = [ln for ln in lines if ln.strip() and not ln.strip().startswith("#")]
    if not rows:
        return {}
    base = _indent(rows[0])
    data: dict[str, Any] = {}
    index = 0
    while index < len(rows):
        line = rows[index]
        if _indent(line) != base:
            raise ValueError(f"Unexpected indentation in {path}: {line!r}")
        key, sep, value = line.strip().partition(":")
        if not sep:
            raise ValueError(f"Unsupported YAML line in {path}: {line!r}")
        key = key.strip()
        value = value.strip()
        nxt = index + 1
        while nxt < len(rows) and _indent(rows[nxt]) > base:
            nxt += 1
        block = rows[index + 1 : nxt]
        if value:
            data[key] = _clean_scalar(value)
        elif block and block[0].strip().startswith("- "):
            items: list[str] = []
            for item in block:
                stripped = item.strip()
                if not stripped.startswith("- "):
                    raise ValueError(f"Unsupported list item in {path}: {item!r}")
                items.append(_clean_scalar(stripped[2:]))
            data[key] = items
        elif block:
            data[key] = _parse_yaml_block(block, path)
        else:
            data[key] = ""
        index = nxt
    return data


def _title_from_slug(slug: str) -> str:
    return slug.replace("-", " ").replace("_", " ").strip().title()


def _clean_scalar(value: str) -> str:
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        return value[1:-1]
    return value


def _normalize_slug(value: str) -> str:
    slug = str(value).strip().lower()
    slug = re.sub(r"[^a-z0-9_.-]+", "-", slug).strip("-")
    if not slug:
        raise ValueError("workflow slug cannot be empty")
    return slug


def _as_tuple(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,) if value.strip() else ()
    if isinstance(value, Sequence):
        return tuple(str(item).strip() for item in value if str(item).strip())
    raise TypeError(f"Expected string or sequence, got {type(value).__name__}")


def _tokenize(query: str) -> list[str]:
    return [token.lower() for token in re.findall(r"[a-zA-Z0-9_:+.-]+", query or "")]


def _search_text(spec: WorkflowSpec) -> str:
    return " ".join(
        (
            spec.slug,
            spec.title,
            spec.summary,
            spec.status,
            " ".join(spec.tags),
            " ".join(spec.keywords),
            " ".join(spec.preflight_tools),
            " ".join(spec.required_tools),
            " ".join(spec.optional_tools),
            " ".join(spec.expected_artifacts),
            spec.recommended_prompt or "",
        )
    ).lower()


# Weight for a routing keyword that appears as a whole-word phrase in the query.
# Sits between a tag hit (20) and a slug-substring hit (40): keywords are a
# strong, deliberate routing signal but should not outrank an exact slug match.
_KEYWORD_PHRASE_SCORE = 35


def _keyword_score(query_lc: str, keywords: tuple[str, ...]) -> int:
    """Score routing keywords (incl. multi-word phrases) against the full query.

    The per-term scorer in :func:`_term_score` only sees tokenized words, so it
    cannot match multi-word keywords such as "amino acid". This scans the raw
    lowercased query for each keyword as a whole-word phrase instead.
    """

    if not query_lc or not keywords:
        return 0
    total = 0
    for keyword in keywords:
        kw = keyword.strip().lower()
        # Whole-word (so "amp" never matches "example") but plural-tolerant
        # ("candidate" matches "candidates", "analog" matches "analogs").
        if kw and re.search(rf"\b{re.escape(kw)}(?:s|es)?\b", query_lc):
            total += _KEYWORD_PHRASE_SCORE
    return total


def _term_score(term: str, haystack: str, spec: WorkflowSpec) -> int:
    if term == spec.slug:
        return 100
    if term in spec.preflight_tools:
        return 50
    if term in spec.required_tools:
        return 45
    if term in spec.optional_tools:
        return 30
    if term in spec.slug:
        return 40
    if term in spec.title.lower():
        return 25
    if term in spec.tags:
        return 20
    if term in spec.expected_artifacts:
        return 15
    if term in haystack:
        return 5
    return 0
