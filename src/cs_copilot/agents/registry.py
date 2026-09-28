#!/usr/bin/env python
# coding: utf-8
"""
Agent registry system for managing and creating agents dynamically.
Provides the main public API for agent creation.
"""

import inspect
import logging
from typing import Any, Dict, List

from agno.agent import Agent
from agno.models.base import Model

from . import factories as factory_module
from .factories import BaseAgentFactory


class AgentRegistry:
    """Registry for managing agent factories and configurations."""

    def __init__(self):
        self._factories: Dict[str, BaseAgentFactory] = {}
        self._aliases: Dict[str, str] = {}  # Alias -> canonical agent_type mapping
        self.logger = logging.getLogger(__name__)

    def register(
        self, agent_type: str, factory: BaseAgentFactory, aliases: List[str] = None
    ) -> None:
        """Register an agent factory with optional aliases.

        Args:
            agent_type: Canonical agent type name
            factory: Factory instance
            aliases: Optional list of alias names that redirect to this agent
        """
        if agent_type in self._factories:
            self.logger.warning(f"Overriding existing factory for agent type: {agent_type}")
        self._factories[agent_type] = factory
        self.logger.info(f"Registered factory for agent type: {agent_type}")

        # Register aliases
        if aliases:
            for alias in aliases:
                self._aliases[alias] = agent_type
                self.logger.info(f"Registered alias '{alias}' -> '{agent_type}'")

    def create_agent(self, agent_type: str, model: Model, **kwargs: Any) -> Agent:
        """Create an agent by type (supports aliases).

        Args:
            agent_type: Agent type or alias
            model: LLM model instance
            **kwargs: Additional arguments for agent creation

        Returns:
            Agent instance

        Raises:
            ValueError: If agent_type/alias is not registered
        """
        # Resolve alias if provided
        resolved_type = self._aliases.get(agent_type, agent_type)

        if resolved_type not in self._factories:
            available_types = list(self._factories.keys())
            available_aliases = list(self._aliases.keys())
            raise ValueError(
                f"Unknown agent type: {agent_type}. "
                f"Available types: {available_types}. "
                f"Available aliases: {available_aliases}"
            )

        factory = self._factories[resolved_type]
        return factory.create_agent(model, **kwargs)

    def list_agent_types(self) -> List[str]:
        """List all registered agent types."""
        return list(self._factories.keys())

    def auto_register(self) -> None:
        """Automatically discover and register all available factories."""
        for _, cls in inspect.getmembers(factory_module, inspect.isclass):
            if (
                issubclass(cls, BaseAgentFactory)
                and cls is not BaseAgentFactory
                and getattr(cls, "agent_type", None)
            ):
                # Get optional aliases from factory class
                aliases = getattr(cls, "aliases", None)
                self.register(cls.agent_type, cls(), aliases=aliases)


# Global agent registry instance with automatic discovery
_agent_registry = AgentRegistry()
_agent_registry.auto_register()


def create_agent(agent_type: str, model: Model, **kwargs: Any) -> Agent:
    """
    Create an agent by type using the global registry.

    Args:
        agent_type: The type of agent to create
        model: The language model to use
        **kwargs: Additional arguments passed to the agent factory

    Returns:
        Agent: The created agent instance

    Raises:
        ValueError: If agent_type is not registered
        AgentCreationError: If agent creation fails
    """
    return _agent_registry.create_agent(agent_type, model, **kwargs)


def list_available_agent_types() -> List[str]:
    """List all available agent types."""
    return _agent_registry.list_agent_types()


def get_registry() -> AgentRegistry:
    """Get the global agent registry instance."""
    return _agent_registry
