"""Pure-Python loader for ChemSpace workflow skills."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

SKILL_MD = "SKILL.md"
SKILLS_ENV = "CS_COPILOT_SKILLS_DIR"
# Required top-level frontmatter fields (Agent-Skills style: name + description).
_REQUIRED_FIELDS = {"name", "description"}


@dataclass(frozen=True)
class SkillSpec:
    """Metadata and content for one reusable ChemSpace skill."""

    slug: str
    title: str
    summary: str
    status: str
    tags: tuple[str, ...]
    keywords: tuple[str, ...]
    required_tools: tuple[str, ...]
    optional_tools: tuple[str, ...]
    artifact_outputs: tuple[str, ...]
    example_prompts: tuple[str, ...]
    skill_md: str
    path: Path

    def as_dict(self, *, include_content: bool = False) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "slug": self.slug,
            "title": self.title,
            "summary": self.summary,
            "status": self.status,
            "tags": list(self.tags),
            "keywords": list(self.keywords),
            "required_tools": list(self.required_tools),
            "optional_tools": list(self.optional_tools),
            "artifact_outputs": list(self.artifact_outputs),
            "example_prompts": list(self.example_prompts),
        }
        if include_content:
            payload["skill_md"] = self.skill_md
        return payload


class SkillRegistry:
    """Discover and serve local ChemSpace skills."""

    def __init__(self, root: str | os.PathLike[str] | None = None) -> None:
        self.root = Path(root).resolve() if root is not None else discover_skill_root()
        self._skills: dict[str, SkillSpec] | None = None

    def list_skills(self) -> list[SkillSpec]:
        return list(self._load().values())

    def get_skill(self, slug: str) -> SkillSpec:
        normalized = _normalize_slug(slug)
        try:
            return self._load()[normalized]
        except KeyError as exc:
            available = ", ".join(sorted(self._load()))
            raise KeyError(f"Unknown skill '{slug}'. Available skills: {available}") from exc

    def search_skills(self, query: str, *, limit: int = 10) -> list[SkillSpec]:
        terms = _tokenize(query)
        query_lc = (query or "").lower()
        scored: list[tuple[int, SkillSpec]] = []
        for index, spec in enumerate(self.list_skills()):
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

    def _load(self) -> dict[str, SkillSpec]:
        if self._skills is not None:
            return self._skills
        if not self.root.exists():
            raise FileNotFoundError(f"Skill catalog root does not exist: {self.root}")
        skills: dict[str, SkillSpec] = {}
        for path in sorted(item for item in self.root.iterdir() if item.is_dir()):
            spec = _load_skill(path)
            if spec.slug in skills:
                raise ValueError(f"Duplicate skill slug: {spec.slug}")
            skills[spec.slug] = spec
        self._skills = skills
        return skills


_DEFAULT_REGISTRY: SkillRegistry | None = None


def list_skills() -> list[SkillSpec]:
    return _registry().list_skills()


def get_skill(slug: str) -> SkillSpec:
    return _registry().get_skill(slug)


def search_skills(query: str, *, limit: int = 10) -> list[SkillSpec]:
    return _registry().search_skills(query, limit=limit)


def discover_skill_root() -> Path:
    env_root = os.getenv(SKILLS_ENV)
    candidates: list[Path] = []
    if env_root:
        candidates.append(Path(env_root))

    here = Path(__file__).resolve()
    for parent in here.parents:
        candidates.append(parent / "skills")
    candidates.append(here.parent / "catalog")

    for candidate in candidates:
        if _looks_like_skill_root(candidate):
            return candidate.resolve()
    return candidates[0].resolve() if candidates else Path("skills").resolve()


def _registry() -> SkillRegistry:
    global _DEFAULT_REGISTRY
    if _DEFAULT_REGISTRY is None:
        _DEFAULT_REGISTRY = SkillRegistry()
    return _DEFAULT_REGISTRY


def _looks_like_skill_root(path: Path) -> bool:
    return path.is_dir() and any((child / SKILL_MD).is_file() for child in path.iterdir())


def _load_skill(path: Path) -> SkillSpec:
    md_path = path / SKILL_MD
    if not md_path.is_file():
        raise FileNotFoundError(f"Missing {SKILL_MD}: {md_path}")

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

    return SkillSpec(
        slug=slug,
        title=str(metadata.get("title") or _title_from_slug(slug)).strip(),
        summary=str(data["description"]).strip(),
        status=str(metadata.get("status") or "stable").strip(),
        tags=_as_tuple(metadata.get("tags")),
        keywords=_as_tuple(metadata.get("keywords")),
        required_tools=_as_tuple(metadata.get("required_tools")),
        optional_tools=_as_tuple(metadata.get("optional_tools")),
        artifact_outputs=_as_tuple(metadata.get("artifact_outputs")),
        example_prompts=_as_tuple(metadata.get("example_prompts")),
        skill_md=body,
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
    SKILL.md frontmatter shape (``name``/``description`` plus a ``metadata:``
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
        raise ValueError("skill slug cannot be empty")
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


def _search_text(spec: SkillSpec) -> str:
    return " ".join(
        (
            spec.slug,
            spec.title,
            spec.summary,
            spec.status,
            " ".join(spec.tags),
            " ".join(spec.keywords),
            " ".join(spec.required_tools),
            " ".join(spec.optional_tools),
            " ".join(spec.artifact_outputs),
            " ".join(spec.example_prompts),
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


def _term_score(term: str, haystack: str, spec: SkillSpec) -> int:
    if term == spec.slug:
        return 100
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
    if term in spec.artifact_outputs:
        return 15
    if term in haystack:
        return 5
    return 0
