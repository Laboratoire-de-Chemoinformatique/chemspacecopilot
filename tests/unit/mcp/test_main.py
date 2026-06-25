"""Tests for the MCP command-line entry point."""

from __future__ import annotations


def test_main_loads_dotenv_before_bootstrap(monkeypatch):
    from cs_copilot.mcp import __main__ as entrypoint
    from cs_copilot.mcp import server as server_module

    calls: list[str] = []

    class FakeServer:
        def run(self, transport: str, **kwargs) -> None:
            calls.append(f"run:{transport}")

    def fake_build_server(*args, **kwargs):
        calls.append("build_server")
        return FakeServer()

    monkeypatch.setattr(entrypoint, "load_dotenv", lambda: calls.append("load_dotenv"))
    monkeypatch.setattr(entrypoint, "require_mcp", lambda: calls.append("require_mcp"))
    monkeypatch.setattr(
        entrypoint,
        "configure_logging",
        lambda log_level: calls.append(f"configure_logging:{log_level}"),
    )
    monkeypatch.setattr(
        entrypoint,
        "apply_session_id",
        lambda session_id: calls.append(f"apply_session_id:{session_id}"),
    )
    monkeypatch.setattr(entrypoint, "bootstrap", lambda config: calls.append("bootstrap") or {})
    monkeypatch.setattr(server_module, "build_server", fake_build_server)

    entrypoint.main(
        [
            "--session-id",
            "dotenv-test",
            "--log-level",
            "warning",
            "--no-tools",
            "--no-prompts",
            "--no-resources",
            "--no-chatgpt-compat",
        ]
    )

    assert calls[:5] == [
        "load_dotenv",
        "require_mcp",
        "configure_logging:warning",
        "apply_session_id:dotenv-test",
        "bootstrap",
    ]
    assert calls[-2:] == ["build_server", "run:stdio"]
