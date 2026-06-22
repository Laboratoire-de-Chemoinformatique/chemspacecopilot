"""Pure-Python loader for reusable cs_copilot workflow contracts."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

WORKFLOW_MD = "WORKFLOW.md"
WORKFLOW_YAML = "workflow.yaml"
WORKFLOWS_ENV = "CS_COPILOT_WORKFLOWS_DIR"
_LIST_FIELDS = {
    "tags",
    "keywords",
    "preflight_tools",
    "required_tools",
    "optional_tools",
    "expected_artifacts",
}
_REQUIRED_FIELDS = {"slug", "title", "summary"}


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
    yaml_path = path / WORKFLOW_YAML
    md_path = path / WORKFLOW_MD
    if not yaml_path.is_file():
        raise FileNotFoundError(f"Missing {WORKFLOW_YAML}: {yaml_path}")
    if not md_path.is_file():
        raise FileNotFoundError(f"Missing {WORKFLOW_MD}: {md_path}")

    data = _load_yaml(yaml_path)
    missing = sorted(field for field in _REQUIRED_FIELDS if not data.get(field))
    if missing:
        raise ValueError(f"{yaml_path} is missing required fields: {', '.join(missing)}")

    slug = _normalize_slug(str(data["slug"]))
    if slug != path.name:
        raise ValueError(f"{yaml_path} slug '{slug}' must match directory name '{path.name}'")

    recommended_prompt = data.get("recommended_prompt")
    return WorkflowSpec(
        slug=slug,
        title=str(data["title"]).strip(),
        summary=str(data["summary"]).strip(),
        status=str(data.get("status") or "stable").strip(),
        tags=_as_tuple(data.get("tags")),
        keywords=_as_tuple(data.get("keywords")),
        preflight_tools=_as_tuple(data.get("preflight_tools")),
        required_tools=_as_tuple(data.get("required_tools")),
        optional_tools=_as_tuple(data.get("optional_tools")),
        expected_artifacts=_as_tuple(data.get("expected_artifacts")),
        recommended_prompt=str(recommended_prompt).strip() if recommended_prompt else None,
        workflow_md=md_path.read_text(encoding="utf-8").strip(),
        path=path,
    )


def _load_yaml(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    try:
        import yaml  # type: ignore

        loaded = yaml.safe_load(text)
        if not isinstance(loaded, dict):
            raise ValueError(f"{path} must contain a mapping")
        return loaded
    except ModuleNotFoundError:
        return _parse_simple_yaml(text, path)


def _parse_simple_yaml(text: str, path: Path) -> dict[str, Any]:
    data: dict[str, Any] = {}
    current_key: str | None = None
    for raw_line in text.splitlines():
        line = raw_line.rstrip()
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if line.startswith((" ", "\t")):
            if current_key is None or not stripped.startswith("- "):
                raise ValueError(f"Unsupported YAML shape in {path}: {raw_line!r}")
            data.setdefault(current_key, [])
            data[current_key].append(_clean_scalar(stripped[2:]))
            continue
        key, sep, value = stripped.partition(":")
        if not sep:
            raise ValueError(f"Unsupported YAML line in {path}: {raw_line!r}")
        current_key = key.strip()
        if current_key in _LIST_FIELDS and not value.strip():
            data[current_key] = []
        else:
            data[current_key] = _clean_scalar(value.strip())
    return data


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
