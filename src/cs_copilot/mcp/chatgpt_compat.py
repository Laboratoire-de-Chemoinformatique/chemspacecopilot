"""ChatGPT-compatible ``search`` / ``fetch`` tools for the MCP server.

ChatGPT developer mode can call arbitrary MCP tools, but data-only apps,
company knowledge, and deep research rely on the conventional read-only
``search`` and ``fetch`` pair. These helpers expose ChemSpace's MCP tool
catalog, prompt catalog, and current session artifacts through that interface
without importing the optional MCP SDK.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Iterable, List, Literal, Optional

from pydantic import BaseModel

MAX_SEARCH_RESULTS = 20


class SearchResult(BaseModel):
    """One ChatGPT-compatible search result."""

    id: str
    title: str
    url: str


class SearchOutput(BaseModel):
    """Return shape required by ChatGPT-compatible MCP search."""

    results: List[SearchResult]


class FetchOutput(BaseModel):
    """Return shape required by ChatGPT-compatible MCP fetch."""

    id: str
    title: str
    text: str
    url: str
    metadata: Optional[dict[str, Any]] = None


@dataclass(frozen=True)
class _Document:
    """Internal searchable document descriptor."""

    id: str
    title: str
    url: str
    kind: Literal["catalog", "tool", "prompt", "skill", "resource"]
    summary: str
    metadata: dict[str, Any]


def search(query: str) -> SearchOutput:
    """Search ChemSpace MCP tools, skills, prompts, and session artifacts."""

    scored = _rank_documents(_iter_documents(), query)
    return SearchOutput(
        results=[
            SearchResult(id=doc.id, title=doc.title, url=doc.url)
            for doc, _score in scored[:MAX_SEARCH_RESULTS]
        ]
    )


def fetch(id: str) -> FetchOutput:
    """Fetch one ChemSpace MCP catalog entry or session artifact by id."""

    for doc in _iter_documents():
        if doc.id == id:
            return _fetch_document(doc)
    raise ValueError(f"Unknown ChemSpace MCP search result id: {id!r}")


def _iter_documents() -> Iterable[_Document]:
    yield from _catalog_documents()
    yield from _tool_documents()
    yield from _prompt_documents()
    yield from _skill_documents()
    yield from _resource_documents()


def _catalog_documents() -> Iterable[_Document]:
    yield _Document(
        id="catalog:overview",
        title="ChemSpace Copilot MCP overview",
        url="cscopilot://mcp/overview",
        kind="catalog",
        summary=(
            "Overview of the ChemSpace MCP server, remote transports, tool "
            "catalog, prompt catalog, and session artifact resources."
        ),
        metadata={},
    )
    yield _Document(
        id="catalog:tools",
        title="ChemSpace MCP tool catalog",
        url="cscopilot://mcp/tools",
        kind="catalog",
        summary="List of callable ChemSpace MCP tools for full MCP clients.",
        metadata={},
    )
    yield _Document(
        id="catalog:prompts",
        title="ChemSpace MCP prompt catalog",
        url="cscopilot://mcp/prompts",
        kind="catalog",
        summary="List of ChemSpace workflow and agent prompts exposed over MCP.",
        metadata={},
    )
    yield _Document(
        id="catalog:skills",
        title="ChemSpace skill catalog",
        url="cscopilot://mcp/skills",
        kind="catalog",
        summary="List of reusable ChemSpace workflow skills and their required tools.",
        metadata={},
    )
    yield _Document(
        id="catalog:session",
        title="ChemSpace session artifacts",
        url="cscopilot://session/manifest.json",
        kind="catalog",
        summary="Files, reports, plots, and datasets written by the active session.",
        metadata={},
    )


def _tool_documents() -> Iterable[_Document]:
    from .tools_registry import iter_specs

    for spec in iter_specs():
        yield _Document(
            id=f"tool:{spec.mcp_name}",
            title=f"Tool: {spec.mcp_name}",
            url=f"cscopilot://mcp/tools/{spec.mcp_name}",
            kind="tool",
            summary=spec.summary,
            metadata={
                "name": spec.mcp_name,
                "method": spec.method,
                "group": spec.group,
                "forced_arguments": sorted(spec.forces),
            },
        )


def _prompt_documents() -> Iterable[_Document]:
    from .prompts_registry import iter_specs

    for spec in iter_specs():
        yield _Document(
            id=f"prompt:{spec.mcp_name}",
            title=f"Prompt: {spec.mcp_name}",
            url=f"cscopilot://mcp/prompts/{spec.mcp_name}",
            kind="prompt",
            summary=spec.summary,
            metadata={
                "name": spec.mcp_name,
                "arguments": list(spec.arguments),
            },
        )


def _skill_documents() -> Iterable[_Document]:
    from cs_copilot.skills import list_skills

    for spec in list_skills():
        yield _Document(
            id=f"skill:{spec.slug}",
            title=f"Skill: {spec.title}",
            url=f"cscopilot://mcp/skills/{spec.slug}",
            kind="skill",
            summary=spec.summary,
            metadata={
                "slug": spec.slug,
                "status": spec.status,
                "tags": list(spec.tags),
                "required_tools": list(spec.required_tools),
                "optional_tools": list(spec.optional_tools),
                "artifact_outputs": list(spec.artifact_outputs),
            },
        )


def _resource_documents() -> Iterable[_Document]:
    from .resources import list_entries

    for entry in list_entries():
        rel_path = entry.uri.rsplit("/", 1)[-1]
        if entry.uri.startswith("cscopilot://session/"):
            rel_path = entry.uri[len("cscopilot://session/") :]
        yield _Document(
            id=f"resource:{rel_path}",
            title=f"Session artifact: {rel_path}",
            url=entry.uri,
            kind="resource",
            summary=f"{entry.mime_type} session artifact {rel_path}",
            metadata={
                "uri": entry.uri,
                "mime_type": entry.mime_type,
                "size": entry.size,
            },
        )


def _rank_documents(docs: Iterable[_Document], query: str) -> list[tuple[_Document, int]]:
    terms = _tokenize(query)
    ranked: list[tuple[_Document, int]] = []
    for index, doc in enumerate(docs):
        metadata_text = " ".join(str(value) for value in doc.metadata.values())
        haystack = f"{doc.id} {doc.title} {doc.summary} {doc.kind} {metadata_text}".lower()
        if not terms:
            score = max(1, 100 - index)
        else:
            score = sum(_term_score(term, haystack, doc) for term in terms)
        if score > 0:
            ranked.append((doc, score))
    ranked.sort(key=lambda item: (-item[1], item[0].kind, item[0].title))
    return ranked


def _tokenize(query: str) -> list[str]:
    return [token.lower() for token in re.findall(r"[a-zA-Z0-9_:+.-]+", query)]


def _term_score(term: str, haystack: str, doc: _Document) -> int:
    if term == doc.id.lower():
        return 100
    if term in doc.id.lower():
        return 30
    if term in doc.title.lower():
        return 20
    if term in haystack:
        return 5
    return 0


def _fetch_document(doc: _Document) -> FetchOutput:
    if doc.kind == "catalog":
        text = _render_catalog(doc.id)
    elif doc.kind == "tool":
        text = _render_tool(doc)
    elif doc.kind == "prompt":
        text = _render_prompt(doc)
    elif doc.kind == "skill":
        text = _render_skill(doc)
    elif doc.kind == "resource":
        text = _render_resource(doc)
    else:  # pragma: no cover - exhaustive for type checkers
        text = doc.summary

    return FetchOutput(
        id=doc.id,
        title=doc.title,
        text=text,
        url=doc.url,
        metadata={"kind": doc.kind, **doc.metadata},
    )


def _render_catalog(doc_id: str) -> str:
    if doc_id == "catalog:tools":
        rows = []
        for doc in _tool_documents():
            rows.append(f"- {doc.metadata['name']}: {doc.summary}")
        return "ChemSpace MCP tools\n\n" + "\n".join(rows)

    if doc_id == "catalog:prompts":
        rows = []
        for doc in _prompt_documents():
            args = doc.metadata.get("arguments") or []
            suffix = ""
            if args:
                suffix = " Arguments: " + ", ".join(str(arg.get("name")) for arg in args)
            rows.append(f"- {doc.metadata['name']}: {doc.summary}{suffix}")
        return "ChemSpace MCP prompts\n\n" + "\n".join(rows)

    if doc_id == "catalog:skills":
        rows = []
        for doc in _skill_documents():
            tools = doc.metadata.get("required_tools") or []
            suffix = ""
            if tools:
                suffix = " Required tools: " + ", ".join(str(tool) for tool in tools)
            rows.append(f"- {doc.metadata['slug']}: {doc.summary}{suffix}")
        return "ChemSpace skills\n\n" + "\n".join(rows)

    if doc_id == "catalog:session":
        rows = []
        for doc in _resource_documents():
            rows.append(f"- {doc.url}: {doc.summary}")
        return "ChemSpace session artifacts\n\n" + "\n".join(rows)

    return (
        "ChemSpace Copilot MCP server\n\n"
        "Use full ChatGPT developer-mode MCP access to call ChemSpace tools "
        "directly. Use this search/fetch interface for read-only discovery, "
        "deep research, company knowledge, and citations over the tool catalog, "
        "prompt catalog, and session artifacts."
    )


def _render_tool(doc: _Document) -> str:
    forced = doc.metadata.get("forced_arguments") or []
    forced_text = ", ".join(forced) if forced else "none"
    group_text = doc.metadata.get("group") or "uncategorized"
    return (
        f"Tool: {doc.metadata['name']}\n\n"
        f"Group: {group_text}\n"
        f"Summary: {doc.summary}\n"
        f"Backed by toolkit method: {doc.metadata['method']}\n"
        f"Server-forced arguments hidden from clients: {forced_text}\n\n"
        "In full MCP developer mode, call this tool by name with arguments "
        "matching the tool schema returned by list_tools. In data-only "
        "ChatGPT modes, this entry is documentation only."
    )


def _render_prompt(doc: _Document) -> str:
    from .prompts_registry import iter_specs

    name = doc.metadata["name"]
    spec = next((item for item in iter_specs() if item.mcp_name == name), None)
    if spec is None:  # pragma: no cover - guarded by document construction
        return doc.summary

    if spec.arguments:
        arg_rows = [f"- {arg.get('name')}: {arg.get('description', '')}" for arg in spec.arguments]
        return (
            f"Prompt: {name}\n\n"
            f"{spec.summary}\n\n"
            "This prompt is parameterized. Fetch it through the MCP prompts "
            "interface with these arguments:\n" + "\n".join(arg_rows)
        )

    return f"Prompt: {name}\n\n{spec.summary}\n\n{spec.render()}"


def _render_skill(doc: _Document) -> str:
    from cs_copilot.skills import get_skill

    spec = get_skill(str(doc.metadata["slug"]))
    required = ", ".join(spec.required_tools) or "none"
    optional = ", ".join(spec.optional_tools) or "none"
    artifacts = ", ".join(spec.artifact_outputs) or "none"
    return (
        f"Skill: {spec.title}\n\n"
        f"Slug: {spec.slug}\n"
        f"Status: {spec.status}\n"
        f"Summary: {spec.summary}\n"
        f"Required tools: {required}\n"
        f"Optional tools: {optional}\n"
        f"Expected artifacts: {artifacts}\n\n"
        f"{spec.skill_md}"
    )


def _render_resource(doc: _Document) -> str:
    from . import resources

    uri = doc.metadata["uri"]
    mime_type = doc.metadata["mime_type"]
    if resources.is_text_resource(uri) or mime_type == "application/json":
        return resources.read_text(uri)
    size = doc.metadata.get("size")
    size_text = f", {size} bytes" if size is not None else ""
    return (
        f"Binary session artifact: {uri}\n"
        f"MIME type: {mime_type}{size_text}\n\n"
        "This artifact is available through the MCP resource interface, but "
        "the ChatGPT-compatible fetch text field only contains metadata for "
        "non-text resources."
    )
