"""Pure-Python loader for ChemSpace workflow skills."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

SKILL_MD = "SKILL.md"
SKILLS_ENV = "CS_COPILOT_SKILLS_DIR"
# Required top-level frontmatter fields (Agent-Skills style: name + description).
_REQUIRED_FIELDS = {"name", "description"}
_REQUIRED_METADATA_FIELDS = {
    "version",
    "depends_on",
    "profiles",
    "permissions",
    "input_artifacts",
    "output_artifacts",
}
_SEMVER_RE = re.compile(
    r"^(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)"
    r"(?:-([0-9A-Za-z-]+(?:\.[0-9A-Za-z-]+)*))?"
    r"(?:\+([0-9A-Za-z-]+(?:\.[0-9A-Za-z-]+)*))?$"
)


@dataclass(frozen=True)
class ArtifactContract:
    """A named artifact accepted or produced by a catalog procedure."""

    name: str
    kind: str
    required: bool = True
    description: str | None = None
    media_type: str | None = None

    def as_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "name": self.name,
            "kind": self.kind,
            "required": self.required,
        }
        if self.description:
            payload["description"] = self.description
        if self.media_type:
            payload["media_type"] = self.media_type
        return payload


@dataclass(frozen=True)
class SkillSpec:
    """Metadata and content for one reusable ChemSpace skill."""

    slug: str
    title: str
    summary: str
    version: str
    status: str
    tags: tuple[str, ...]
    keywords: tuple[str, ...]
    depends_on: tuple[str, ...]
    profiles: tuple[str, ...]
    permissions: tuple[str, ...]
    input_artifacts: tuple[ArtifactContract, ...]
    output_artifacts: tuple[ArtifactContract, ...]
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
            "version": self.version,
            "status": self.status,
            "tags": list(self.tags),
            "keywords": list(self.keywords),
            "depends_on": list(self.depends_on),
            "profiles": list(self.profiles),
            "permissions": list(self.permissions),
            "input_artifacts": [contract.as_dict() for contract in self.input_artifacts],
            "output_artifacts": [contract.as_dict() for contract in self.output_artifacts],
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
        _validate_catalog_dependencies(skills)
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
    if env_root:
        # An explicit catalog path is a policy boundary. A missing or malformed
        # override must fail visibly instead of silently selecting bundled logic.
        return Path(env_root).resolve()

    candidates: list[Path] = []

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

    missing_metadata = sorted(field for field in _REQUIRED_METADATA_FIELDS if field not in metadata)
    if missing_metadata:
        raise ValueError(
            f"{md_path} metadata is missing required fields: {', '.join(missing_metadata)}"
        )
    version = _validate_semver(metadata.get("version"), md_path)
    profiles = _as_tuple(metadata.get("profiles"))
    permissions = _as_tuple(metadata.get("permissions"))
    if not profiles or not permissions:
        raise ValueError(f"{md_path} metadata profiles and permissions cannot be empty")
    input_artifacts = _as_artifact_contracts(metadata.get("input_artifacts"), md_path)
    output_artifacts = _as_artifact_contracts(metadata.get("output_artifacts"), md_path)
    _validate_unique_artifacts(input_artifacts, output_artifacts, md_path)

    return SkillSpec(
        slug=slug,
        title=str(metadata.get("title") or _title_from_slug(slug)).strip(),
        summary=str(data["description"]).strip(),
        version=version,
        status=str(metadata.get("status") or "stable").strip(),
        tags=_as_tuple(metadata.get("tags")),
        keywords=_as_tuple(metadata.get("keywords")),
        depends_on=_as_tuple(metadata.get("depends_on")),
        profiles=profiles,
        permissions=permissions,
        input_artifacts=input_artifacts,
        output_artifacts=output_artifacts,
        required_tools=_as_tuple(metadata.get("required_tools")),
        optional_tools=_as_tuple(metadata.get("optional_tools")),
        artifact_outputs=tuple(contract.name for contract in output_artifacts),
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

    Supports scalars, nested mappings, scalar lists, and lists of mappings.
    This intentionally covers the repository's catalog contract shape without
    introducing PyYAML as a mandatory runtime dependency. It is not a general
    YAML parser.
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
        elif block:
            data[key] = _parse_yaml_node(block, path)
        else:
            data[key] = ""
        index = nxt
    return data


def _parse_yaml_node(lines: Sequence[str], path: Path) -> Any:
    """Parse one fallback YAML node, including a list of mapping records."""
    rows = [ln for ln in lines if ln.strip() and not ln.strip().startswith("#")]
    if not rows:
        return {}
    base = _indent(rows[0])
    if not rows[0].strip().startswith("- "):
        return _parse_yaml_block(rows, path)

    items: list[Any] = []
    index = 0
    while index < len(rows):
        line = rows[index]
        if _indent(line) != base or not line.strip().startswith("- "):
            raise ValueError(f"Unsupported YAML list item in {path}: {line!r}")
        nxt = index + 1
        while nxt < len(rows):
            if _indent(rows[nxt]) == base and rows[nxt].strip().startswith("- "):
                break
            nxt += 1
        payload = line.strip()[2:].strip()
        block = rows[index + 1 : nxt]
        if not payload:
            if not block:
                raise ValueError(f"Empty YAML list item in {path}: {line!r}")
            items.append(_parse_yaml_node(block, path))
        elif re.match(r"^[^:]+:(?:\s|$)", payload):
            synthetic = " " * (base + 2) + payload
            items.append(_parse_yaml_block([synthetic, *block], path))
        else:
            if block:
                raise ValueError(f"Scalar YAML list item cannot have a block in {path}: {line!r}")
            items.append(_clean_scalar(payload))
        index = nxt
    return items


def _title_from_slug(slug: str) -> str:
    return slug.replace("-", " ").replace("_", " ").strip().title()


def _clean_scalar(value: str) -> Any:
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        return value[1:-1]
    lowered = value.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    if lowered in {"null", "none", "~"}:
        return None
    if value == "[]":
        return []
    if value == "{}":
        return {}
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


def _validate_semver(value: Any, path: Path) -> str:
    version = str(value or "").strip()
    match = _SEMVER_RE.fullmatch(version)
    prerelease = match.group(4) if match else None
    invalid_numeric_identifier = prerelease and any(
        part.isdigit() and len(part) > 1 and part.startswith("0") for part in prerelease.split(".")
    )
    if match is None or invalid_numeric_identifier:
        raise ValueError(
            f"{path} metadata version '{version}' is not valid semantic versioning "
            "(expected MAJOR.MINOR.PATCH)"
        )
    return version


def _as_artifact_contracts(value: Any, path: Path) -> tuple[ArtifactContract, ...]:
    if value is None:
        return ()
    if isinstance(value, (str, Mapping)):
        rows: Sequence[Any] = (value,)
    elif isinstance(value, Sequence):
        rows = value
    else:
        raise TypeError(
            f"{path} artifact contracts must be a mapping or sequence, "
            f"got {type(value).__name__}"
        )

    contracts: list[ArtifactContract] = []
    for row in rows:
        if isinstance(row, str):
            name = row.strip()
            kind = "artifact"
            required = True
            description = None
            media_type = None
        elif isinstance(row, Mapping):
            name = str(row.get("name") or "").strip()
            kind = str(row.get("kind") or "artifact").strip()
            required = row.get("required", True)
            if not isinstance(required, bool):
                raise TypeError(f"{path} artifact '{name}' required must be a boolean")
            description_value = row.get("description")
            media_type_value = row.get("media_type")
            description = str(description_value).strip() if description_value is not None else None
            media_type = str(media_type_value).strip() if media_type_value is not None else None
        else:
            raise TypeError(
                f"{path} artifact contract must be a string or mapping, "
                f"got {type(row).__name__}"
            )
        if not name:
            raise ValueError(f"{path} artifact contract is missing a name")
        if not kind:
            raise ValueError(f"{path} artifact '{name}' is missing a kind")
        contracts.append(
            ArtifactContract(
                name=name,
                kind=kind,
                required=required,
                description=description,
                media_type=media_type,
            )
        )
    return tuple(contracts)


def _validate_unique_artifacts(
    inputs: tuple[ArtifactContract, ...],
    outputs: tuple[ArtifactContract, ...],
    path: Path,
) -> None:
    for label, contracts in (("input", inputs), ("output", outputs)):
        names = [contract.name for contract in contracts]
        duplicates = sorted({name for name in names if names.count(name) > 1})
        if duplicates:
            raise ValueError(
                f"{path} has duplicate {label} artifact contracts: {', '.join(duplicates)}"
            )


def _dependency_slug(reference: str, path: Path) -> str:
    raw = str(reference).strip()
    if "@" in raw:
        raise ValueError(
            f"{path} dependency '{raw}' uses a version constraint; v2 catalog dependencies "
            "must reference an exact catalog slug"
        )
    return _normalize_slug(raw)


def _validate_catalog_dependencies(skills: Mapping[str, SkillSpec]) -> None:
    graph: dict[str, tuple[str, ...]] = {}
    for slug, spec in skills.items():
        dependencies = tuple(
            _dependency_slug(item, spec.path / SKILL_MD) for item in spec.depends_on
        )
        missing = sorted(dependency for dependency in dependencies if dependency not in skills)
        if missing:
            raise ValueError(
                f"{spec.path / SKILL_MD} references unknown skill dependencies: "
                f"{', '.join(missing)}"
            )
        graph[slug] = dependencies

    visiting: list[str] = []
    visited: set[str] = set()

    def visit(slug: str) -> None:
        if slug in visited:
            return
        if slug in visiting:
            cycle = visiting[visiting.index(slug) :] + [slug]
            raise ValueError(f"Skill dependency cycle detected: {' -> '.join(cycle)}")
        visiting.append(slug)
        for dependency in graph[slug]:
            visit(dependency)
        visiting.pop()
        visited.add(slug)

    for slug in graph:
        visit(slug)


def _tokenize(query: str) -> list[str]:
    return [token.lower() for token in re.findall(r"[a-zA-Z0-9_:+.-]+", query or "")]


def _search_text(spec: SkillSpec) -> str:
    return " ".join(
        (
            spec.slug,
            spec.title,
            spec.summary,
            spec.status,
            spec.version,
            " ".join(spec.tags),
            " ".join(spec.keywords),
            " ".join(spec.depends_on),
            " ".join(spec.profiles),
            " ".join(spec.permissions),
            " ".join(contract.name for contract in spec.input_artifacts),
            " ".join(contract.name for contract in spec.output_artifacts),
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
