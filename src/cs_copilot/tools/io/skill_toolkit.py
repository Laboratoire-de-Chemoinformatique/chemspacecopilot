"""Agno toolkit for consulting ChemSpace reusable skills."""

from __future__ import annotations

from typing import Any, Dict, List

from agno.tools.toolkit import Toolkit

from cs_copilot.skills import get_skill, list_skills, search_skills


class SkillToolkit(Toolkit):
    """Read-only access to the ChemSpace skill catalog."""

    def __init__(self) -> None:
        super().__init__("skills")
        self.register(self.list_skills)
        self.register(self.search_skills)
        self.register(self.fetch_skill)

    def list_skills(self) -> List[Dict[str, Any]]:
        """List all reusable ChemSpace workflow skills."""

        return [skill.as_dict(include_content=False) for skill in list_skills()]

    def search_skills(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Search ChemSpace skills by topic, tool, artifact, or workflow name."""

        return [
            skill.as_dict(include_content=False)
            for skill in search_skills(query, limit=max(1, int(limit)))
        ]

    def fetch_skill(self, slug: str) -> Dict[str, Any]:
        """Fetch one ChemSpace skill, including its full SKILL.md content."""

        return get_skill(slug).as_dict(include_content=True)
