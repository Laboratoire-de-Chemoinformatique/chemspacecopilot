"""Shared helpers for MCP tool specification modules."""

from __future__ import annotations

from typing import Any, Callable


def factory(import_path: str) -> Callable[[], Any]:
    """Return a factory that lazily imports and freshly instantiates a toolkit."""

    def _build() -> Any:
        module_name, _, class_name = import_path.rpartition(":")
        if not module_name or not class_name:
            raise ValueError(f"Invalid factory path: {import_path!r}")
        module = __import__(module_name, fromlist=[class_name])
        cls = getattr(module, class_name)
        return cls()

    return _build
