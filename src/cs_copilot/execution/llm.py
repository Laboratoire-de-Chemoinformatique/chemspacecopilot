"""Runtime-neutral access to an in-process language model for toolkits.

Some toolkit steps (the ChEMBL short-keyword judges, LLM design engines) need a
language model. Toolkits never need to know which runtime drives them: they ask
this port for the model attached to the ``agent`` argument they were given.

The shared policy vocabulary:

* ``external``: LLM steps become tasks for an external reasoner (the MCP
  server's default; no model is attached).
* ``agno-model``: the configured Agno model runs toolkit LLM steps in-process.
* ``disabled``: LLM-dependent work is rejected.

A host that declares no policy at all (a real Agno ``Agent``, which always
carries its model) is treated as in-process. This module only depends on the
standard library.
"""

from __future__ import annotations

from typing import Any, Literal

LLMPolicy = Literal["external", "agno-model", "disabled"]
LLM_POLICIES: tuple[LLMPolicy, ...] = ("external", "agno-model", "disabled")
DEFAULT_LLM_POLICY: LLMPolicy = "external"


def normalize_llm_policy(value: str | None) -> LLMPolicy:
    """Normalize and validate an LLM policy value."""

    normalized = str(value or DEFAULT_LLM_POLICY).strip().lower().replace("_", "-")
    if normalized not in LLM_POLICIES:
        raise ValueError(
            f"Unsupported LLM policy {value!r}. " f"Use one of: {', '.join(LLM_POLICIES)}."
        )
    return normalized  # type: ignore[return-value]


class LLMUnavailableError(RuntimeError):
    """Raised when a toolkit step needs a model that the runtime did not provide.

    Subclasses :class:`RuntimeError` so existing broad fallbacks (for example the
    ChEMBL judges marking themselves ``unavailable``) keep working.
    """

    def __init__(self, purpose: str, *, policy: str | None = None) -> None:
        self.purpose = purpose
        self.policy = policy
        if policy == "disabled":
            message = f"{purpose} requires a language model, but LLM use is disabled."
        else:
            message = (
                f"{purpose} requires an agent with a model, and the active runtime "
                "did not provide one."
            )
        super().__init__(message)


def llm_policy_of(agent: Any) -> LLMPolicy | None:
    """Return the normalized policy ``agent`` declares, or ``None`` if it declares none.

    Only a string declares a policy; any other attribute value (an arbitrary
    host object or a test double) counts as undeclared.
    """

    value = getattr(agent, "llm_policy", None)
    return normalize_llm_policy(value) if isinstance(value, str) else None


def resolve_model(agent: Any, purpose: str | None = None) -> Any | None:
    """Return the in-process model for ``agent``, or ``None`` when none may be used."""

    if agent is None or llm_policy_of(agent) == "disabled":
        return None
    return getattr(agent, "model", None)


def require_model(agent: Any, purpose: str) -> Any:
    """Return the in-process model for ``agent`` or raise :class:`LLMUnavailableError`."""

    model = resolve_model(agent, purpose)
    if model is None:
        raise LLMUnavailableError(purpose, policy=llm_policy_of(agent))
    return model
