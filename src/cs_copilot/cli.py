#!/usr/bin/env python
# coding: utf-8
"""
Interactive CLI agent for ChemSpace Copilot.

Launch with::

    uv run cscopilot
"""

import asyncio
import json
import logging
import os
import signal
import uuid

from dotenv import load_dotenv
from prompt_toolkit import PromptSession
from prompt_toolkit.history import FileHistory
from rich.console import Console
from rich.live import Live
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table

load_dotenv()

# Suppress verbose startup logs emitted at import time by the agent registry
# and gtm_operations.setup_logging().
logging.disable(logging.CRITICAL)
from cs_copilot.agents.teams import get_cs_copilot_agent_team  # noqa: E402
from cs_copilot.model_config import (  # noqa: E402
    _is_retriable,
    arun_with_retry,
    load_model_from_config,
    parse_modelconf,
)
from cs_copilot.storage import S3  # noqa: E402

logging.disable(logging.NOTSET)

logger = logging.getLogger(__name__)

APP_NAME = "ChemSpace Copilot"
APP_VERSION = "0.1.0"
HISTORY_FILE = os.path.expanduser("~/.cscopilot_history")


# ---------------------------------------------------------------------------
# CLI Renderer
# ---------------------------------------------------------------------------


class CLIRenderer:
    """Handles all terminal output via Rich Console."""

    def __init__(self, console: Console):
        self.console = console

    def print_banner(self, session_id: str, provider: str, model_id: str):
        info = (
            f"[bold cyan]{APP_NAME}[/bold cyan] CLI v{APP_VERSION}\n"
            f"AI-powered chemical space analysis\n\n"
            f"[dim]Session:[/dim]  {session_id}\n"
            f"[dim]Model:[/dim]    {provider}/{model_id}\n"
            f"[dim]Storage:[/dim]  {S3.prefix}"
        )
        self.console.print(Panel(info, border_style="cyan", expand=False))
        self.console.print("[dim]Type /help for commands, /quit to exit.[/dim]\n")

    def print_help(self):
        table = Table(title="Commands", show_header=True, border_style="dim")
        table.add_column("Command", style="bold")
        table.add_column("Description")
        table.add_row("/help", "Show this help message")
        table.add_row("/quit, /exit", "Exit the CLI")
        table.add_row("/clear", "Clear the screen")
        table.add_row("/session", "Show session information")
        table.add_row("/debug", "Toggle debug mode")
        self.console.print(table)

    def print_tool_start(self, name: str, args):
        args_str = ""
        if args:
            try:
                args_str = json.dumps(args, ensure_ascii=False, default=str)
            except (TypeError, ValueError):
                args_str = str(args)
            if len(args_str) > 200:
                args_str = args_str[:200] + "\u2026"
        label = f"[bold]{name}[/bold]"
        if args_str:
            label += f"({args_str})"
        self.console.print(
            Panel(
                label,
                title="[yellow]Tool Call[/yellow]",
                border_style="yellow",
                expand=False,
            )
        )

    def print_tool_end(self, result):
        display = str(result)
        if len(display) > 300:
            display = display[:300] + "\u2026"
        self.console.print(f"  [dim]{display}[/dim]")

    def print_error(self, msg: str):
        self.console.print(f"[bold red]Error:[/bold red] {msg}")

    def print_info(self, msg: str):
        self.console.print(f"[dim]{msg}[/dim]")


# ---------------------------------------------------------------------------
# Stream response (terminal equivalent of chainlit_app.py relay())
# ---------------------------------------------------------------------------


async def stream_response(stream, renderer: CLIRenderer):
    """Consume an agent stream and render to the terminal.

    Text chunks are progressively rendered as Markdown via ``rich.live.Live``.
    Tool-call events appear as panels printed above the live area.
    """
    buffer = ""
    live = Live(Markdown(""), console=renderer.console, refresh_per_second=8)
    live.start()
    try:
        async for chunk in stream:
            # ── tool events ──────────────────────────────────────────────
            ev = getattr(chunk, "event", None)

            if ev == "ToolCallStarted":
                t = chunk.tool
                name = getattr(t, "tool_name", None) or getattr(t, "name", "tool")
                args = getattr(t, "tool_args", None) or getattr(t, "arguments", {})
                renderer.print_tool_start(name, args)
                continue

            if ev and ev.endswith("Completed"):
                result = getattr(chunk, "content", "") or "done"
                renderer.print_tool_end(result)
                continue

            # ── plain text ───────────────────────────────────────────────
            text = (
                chunk
                if isinstance(chunk, str)
                else getattr(chunk, "content", "") or getattr(chunk, "text", "")
            )
            if not text:
                continue

            buffer += text
            live.update(Markdown(buffer))
    finally:
        live.stop()


# ---------------------------------------------------------------------------
# Slash commands
# ---------------------------------------------------------------------------


def handle_slash_command(cmd, renderer, session_id, provider, model_id, team):
    """Handle a slash command.

    Returns ``True`` if handled, ``None`` to quit, ``False`` if unrecognised.
    """
    command = cmd.strip().split()[0].lower()

    if command in ("/quit", "/exit"):
        renderer.console.print("[dim]Goodbye![/dim]")
        return None

    if command == "/help":
        renderer.print_help()
        return True

    if command == "/clear":
        renderer.console.clear()
        renderer.print_banner(session_id, provider, model_id)
        return True

    if command == "/session":
        renderer.print_info(
            f"Session: {session_id}\n"
            f"Model:   {provider}/{model_id}\n"
            f"Storage: {S3.prefix}\n"
            f"Debug:   {team.debug_mode}"
        )
        return True

    if command == "/debug":
        team.debug_mode = not team.debug_mode
        level = logging.DEBUG if team.debug_mode else logging.WARNING
        logging.getLogger().setLevel(level)
        state = "ON" if team.debug_mode else "OFF"
        renderer.print_info(f"Debug mode: {state}")
        return True

    return False


# ---------------------------------------------------------------------------
# Main REPL
# ---------------------------------------------------------------------------


async def run_repl():
    """Interactive read-eval-print loop."""
    console = Console()
    renderer = CLIRenderer(console)

    # Suppress verbose logs; /debug toggles back to DEBUG.
    # Must run here (not at module level) because gtm_operations.setup_logging()
    # overrides logging.basicConfig during import.
    logging.basicConfig(level=logging.WARNING, format="%(message)s", force=True)

    # Model
    conf = parse_modelconf()
    provider = conf["provider"]
    model_id = conf["model_id"]
    model = load_model_from_config()

    # Session
    session_id = uuid.uuid4().hex[:12]
    S3.prefix = f"sessions/{session_id}"

    # Agent team
    renderer.print_info("Initializing agent team\u2026")
    try:
        team = get_cs_copilot_agent_team(model, show_members_responses=False)
    except Exception as e:
        renderer.print_error(f"Failed to initialize agent team: {e}")
        return

    renderer.print_banner(session_id, provider, model_id)

    # Input session with persistent history
    prompt_session: PromptSession = PromptSession(history=FileHistory(HISTORY_FILE))

    # Ctrl+C handling — first press cancels generation, second exits
    cancelled = asyncio.Event()
    generating = False
    loop = asyncio.get_event_loop()

    def _sigint_handler():
        nonlocal generating
        if generating:
            cancelled.set()
        else:
            # Re-raise as KeyboardInterrupt so prompt_toolkit exits cleanly
            loop.call_soon(lambda: (_ for _ in ()).throw(KeyboardInterrupt))

    try:
        loop.add_signal_handler(signal.SIGINT, _sigint_handler)
        has_signal_handler = True
    except NotImplementedError:
        # Windows does not support add_signal_handler
        has_signal_handler = False

    # ── REPL ─────────────────────────────────────────────────────────────
    while True:
        cancelled.clear()
        try:
            user_input = await prompt_session.prompt_async("you> ")
        except (EOFError, KeyboardInterrupt):
            console.print("\n[dim]Goodbye![/dim]")
            break

        user_input = user_input.strip()
        if not user_input:
            continue

        # Slash commands
        if user_input.startswith("/"):
            result = handle_slash_command(
                user_input, renderer, session_id, provider, model_id, team
            )
            if result is None:
                break
            if result:
                continue
            renderer.print_info(
                f"Unknown command: {user_input.split()[0]}. Type /help for commands."
            )
            continue

        # ── agent call with double-retry ─────────────────────────────────
        generating = True
        max_retries = 3
        base_delay = 2.0
        try:
            for attempt in range(max_retries + 1):
                try:
                    stream = await arun_with_retry(
                        team,
                        user_input,
                        stream=True,
                        session_id=session_id,
                        max_retries=1,  # light inner retry; outer loop is primary
                    )
                    await stream_response(stream, renderer)
                    console.print()  # blank line after response
                    break
                except (KeyboardInterrupt, asyncio.CancelledError):
                    console.print()
                    renderer.print_info("Generation cancelled.")
                    break
                except Exception as e:
                    if cancelled.is_set():
                        cancelled.clear()
                        console.print()
                        renderer.print_info("Generation cancelled.")
                        break
                    if _is_retriable(e) and attempt < max_retries:
                        delay = base_delay * (2**attempt)
                        renderer.print_info(
                            f"Transient error, retrying "
                            f"({attempt + 2}/{max_retries + 1}) in {delay:.0f}s\u2026"
                        )
                        await asyncio.sleep(delay)
                        continue
                    renderer.print_error(str(e))
                    break
        finally:
            generating = False

    # Cleanup
    if has_signal_handler:
        loop.remove_signal_handler(signal.SIGINT)


def main():
    """Entry point for ``uv run cscopilot``."""
    try:
        asyncio.run(run_repl())
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
