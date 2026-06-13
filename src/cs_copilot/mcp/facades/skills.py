"""Skill catalog facade for MCP tool registration."""

from __future__ import annotations

import functools
from typing import Any, List


class SkillFacade:
    """Direct MCP access to the pure-Python cs_copilot skill catalog."""

    def list(self, include_content: bool = False) -> List[dict[str, Any]]:
        """List reusable cs_copilot workflow skills."""
        from cs_copilot.skills import list_skills

        return [spec.as_dict(include_content=include_content) for spec in list_skills()]

    def search(
        self,
        query: str,
        limit: int = 10,
        include_content: bool = False,
    ) -> List[dict[str, Any]]:
        """Search reusable cs_copilot workflow skills by metadata or tool names."""
        from cs_copilot.skills import search_skills

        return [
            spec.as_dict(include_content=include_content)
            for spec in search_skills(query, limit=limit)
        ]

    def fetch(self, slug: str, include_content: bool = True) -> dict[str, Any]:
        """Fetch one reusable cs_copilot workflow skill by slug."""
        from cs_copilot.skills import get_skill

        return get_skill(slug).as_dict(include_content=include_content)


@functools.lru_cache(maxsize=1)
def skill_facade() -> SkillFacade:
    return SkillFacade()
