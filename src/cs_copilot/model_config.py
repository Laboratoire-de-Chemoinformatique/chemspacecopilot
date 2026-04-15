#!/usr/bin/env python
# coding: utf-8
"""
Centralized model configuration for Cs_copilot.

Reads the `.modelconf` file (simple key=value format) and returns
the appropriate agno Model instance.  Environment variables
MODEL_PROVIDER, MODEL_ID and OLLAMA_HOST take precedence over
the config file, making it easy to override in Docker / CI.

Also provides retry wrappers (`run_with_retry`, `arun_with_retry`)
for resilient agent execution — especially useful with local Ollama
models that may produce malformed tool-call JSON.
"""

import asyncio
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Sentinel used when the project root cannot be determined.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent  # src/cs_copilot -> src -> repo root

DEFAULT_PROVIDER = "deepseek"
DEFAULT_MODEL_ID = "deepseek-chat"
DEFAULT_OLLAMA_HOST = "http://localhost:11434"

VALID_PROVIDERS = ("deepseek", "ollama", "openrouter")


# ---------------------------------------------------------------------------
# Config file parser
# ---------------------------------------------------------------------------


def _find_modelconf(explicit_path: Optional[str] = None) -> Optional[Path]:
    """Locate the .modelconf file.

    Search order:
    1. *explicit_path* argument
    2. ``MODELCONF_PATH`` environment variable
    3. Project root (relative to this source file)
    4. Current working directory
    """
    candidates = []

    if explicit_path:
        candidates.append(Path(explicit_path))

    env_path = os.getenv("MODELCONF_PATH")
    if env_path:
        candidates.append(Path(env_path))

    candidates.append(_PROJECT_ROOT / ".modelconf")
    candidates.append(Path.cwd() / ".modelconf")

    for p in candidates:
        if p.is_file():
            return p

    return None


def parse_modelconf(config_path: Optional[str] = None) -> Dict[str, str]:
    """Parse a `.modelconf` file into a dict.

    The file uses a trivial ``key=value`` format.  Lines starting with
    ``#`` and blank lines are ignored.  Environment variables take
    precedence over file values.

    Returns a dict with at least ``provider`` and ``model_id`` keys.
    """
    conf: Dict[str, str] = {}

    path = _find_modelconf(config_path)
    if path is not None:
        logger.info("Loading model config from %s", path)
        for line in path.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" not in line:
                continue
            key, _, value = line.partition("=")
            conf[key.strip()] = value.strip()
    else:
        logger.info("No .modelconf file found – using defaults / environment overrides")

    # Environment variable overrides (always win)
    env_provider = os.getenv("MODEL_PROVIDER")
    if env_provider:
        conf["provider"] = env_provider

    env_model_id = os.getenv("MODEL_ID")
    if env_model_id:
        conf["model_id"] = env_model_id

    env_ollama_host = os.getenv("OLLAMA_HOST")
    if env_ollama_host:
        conf["ollama_host"] = env_ollama_host

    # Apply defaults for missing keys
    conf.setdefault("provider", DEFAULT_PROVIDER)
    conf.setdefault("model_id", DEFAULT_MODEL_ID)

    # Normalise provider name
    conf["provider"] = conf["provider"].lower().strip()

    if conf["provider"] not in VALID_PROVIDERS:
        raise ValueError(
            f"Invalid model provider '{conf['provider']}' in .modelconf. "
            f"Must be one of: {', '.join(VALID_PROVIDERS)}"
        )

    return conf


# ---------------------------------------------------------------------------
# Model factory
# ---------------------------------------------------------------------------


def load_model_from_config(config_path: Optional[str] = None) -> Any:
    """Return an agno ``Model`` instance based on ``.modelconf``.

    Parameters
    ----------
    config_path : str, optional
        Explicit path to a ``.modelconf`` file.  When *None* the
        standard search order is used (see :func:`parse_modelconf`).

    Returns
    -------
    agno.models.base.Model
        A configured ``DeepSeek`` or ``Ollama`` model instance.
    """
    conf = parse_modelconf(config_path)
    provider = conf["provider"]
    model_id = conf["model_id"]

    if provider == "ollama":
        from agno.models.ollama import Ollama

        host = conf.get("ollama_host", DEFAULT_OLLAMA_HOST)
        logger.info(f"Using Ollama model '{model_id}' at {host}")
        return Ollama(id=model_id, host=host)

    if provider == "openrouter":
        from agno.models.openrouter import OpenRouter

        api_key = os.getenv("OPENROUTER_API_KEY")
        logger.info(f"Using OpenRouter model '{model_id}'")
        return OpenRouter(id=model_id, api_key=api_key)

    # provider == "deepseek"
    from agno.models.deepseek import DeepSeek

    api_key = os.getenv("DEEPSEEK_API_KEY")
    logger.info(f"Using DeepSeek model '{model_id}'")
    return DeepSeek(id=model_id, api_key=api_key)


def get_model_provider(config_path: Optional[str] = None) -> str:
    """Return the configured provider name without loading the model.

    Useful for conditional logic (e.g. skipping API-key checks for Ollama).
    """
    return parse_modelconf(config_path)["provider"]


# ---------------------------------------------------------------------------
# Retry helpers for resilient agent execution
# ---------------------------------------------------------------------------

# Error substrings that indicate a transient / retriable failure.
# These are common with local Ollama models generating malformed tool-call
# JSON or experiencing intermittent connection issues.
RETRIABLE_ERROR_PATTERNS: List[str] = [
    "error parsing tool call",
    "unexpected end of JSON input",
    "connection refused",
    "connection reset",
    "internal server error",
]


def _is_retriable(exc: Exception) -> bool:
    """Return *True* if *exc* matches a known transient error pattern."""
    error_str = str(exc).lower()
    return any(pat in error_str for pat in RETRIABLE_ERROR_PATTERNS)


def run_with_retry(
    agent_or_team,
    prompt: str,
    *,
    max_retries: int = 3,
    base_delay: float = 2.0,
    stream: bool = False,
    **run_kwargs,
):
    """Run an agent/team synchronously with retry on transient errors.

    Uses exponential back-off: ``base_delay * 2 ** attempt`` seconds
    between retries.

    Parameters
    ----------
    agent_or_team:
        Any object with a ``.run(prompt, stream=..., **kwargs)`` method
        (typically an ``agno.agent.Agent`` or ``agno.team.Team``).
    prompt:
        The user prompt to send.
    max_retries:
        Maximum number of retry attempts (default 3).
    base_delay:
        Initial delay in seconds before the first retry (default 2.0).
    stream:
        Passed through to ``.run()``.
    **run_kwargs:
        Extra keyword arguments forwarded to ``.run()``.

    Returns
    -------
    The response object from a successful ``.run()`` call.

    Raises
    ------
    Exception
        The last exception if all retries are exhausted, or the original
        exception if it is not considered retriable.
    """
    last_exception: Optional[Exception] = None

    for attempt in range(max_retries + 1):
        try:
            return agent_or_team.run(prompt, stream=stream, **run_kwargs)
        except Exception as e:
            last_exception = e
            if _is_retriable(e) and attempt < max_retries:
                delay = base_delay * (2**attempt)
                logger.warning(
                    "Retriable error on attempt %d/%d: %s – retrying in %.1fs …",
                    attempt + 1,
                    max_retries + 1,
                    e,
                    delay,
                )
                time.sleep(delay)
                continue
            # Non-retriable or final attempt – propagate immediately
            raise

    # Safety net (should never be reached)
    raise last_exception  # type: ignore[misc]


async def arun_with_retry(
    agent_or_team,
    prompt: str,
    *,
    max_retries: int = 3,
    base_delay: float = 2.0,
    stream: bool = True,
    **run_kwargs,
):
    """Run an agent/team **asynchronously** with retry on transient errors.

    When ``stream=True`` the function returns an async iterator.  If an
    error occurs *during* iteration (mid-stream), the whole call is
    retried from scratch so the caller always gets a clean stream.

    Parameters
    ----------
    agent_or_team:
        Any object with an ``.arun(prompt, stream=..., **kwargs)`` method.
    prompt:
        The user prompt to send.
    max_retries:
        Maximum number of retry attempts (default 3).
    base_delay:
        Initial delay in seconds before the first retry (default 2.0).
    stream:
        Passed through to ``.arun()`` (default ``True``).
    **run_kwargs:
        Extra keyword arguments forwarded to ``.arun()``.

    Returns
    -------
    The response (or async iterator) from a successful ``.arun()`` call.
    """
    if not stream:
        # Non-streaming: simple retry around the awaited call.
        last_exception: Optional[Exception] = None
        for attempt in range(max_retries + 1):
            try:
                return await agent_or_team.arun(prompt, stream=False, **run_kwargs)
            except Exception as e:
                last_exception = e
                if _is_retriable(e) and attempt < max_retries:
                    delay = base_delay * (2**attempt)
                    logger.warning(
                        "Retriable error on async attempt %d/%d: %s – retrying in %.1fs …",
                        attempt + 1,
                        max_retries + 1,
                        e,
                        delay,
                    )
                    await asyncio.sleep(delay)
                    continue
                raise
        raise last_exception  # type: ignore[misc]

    # Streaming: return a resilient async generator that retries the
    # entire .arun() call when a retriable error occurs mid-stream.
    async def _resilient_stream():
        last_exception: Optional[Exception] = None
        for attempt in range(max_retries + 1):
            try:
                raw_stream = agent_or_team.arun(prompt, stream=True, **run_kwargs)
                async for chunk in raw_stream:
                    yield chunk
                return  # stream completed successfully
            except Exception as e:
                last_exception = e
                if _is_retriable(e) and attempt < max_retries:
                    delay = base_delay * (2**attempt)
                    logger.warning(
                        "Retriable streaming error on attempt %d/%d: %s " "– retrying in %.1fs …",
                        attempt + 1,
                        max_retries + 1,
                        e,
                        delay,
                    )
                    await asyncio.sleep(delay)
                    continue
                raise
        raise last_exception  # type: ignore[misc]

    return _resilient_stream()
