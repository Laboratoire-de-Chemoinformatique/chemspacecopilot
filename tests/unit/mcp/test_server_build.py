"""End-to-end smoke test for FastMCP server assembly.

Builds the server in-process (no stdio) and asserts that every registered
tool, prompt, and the resource manifest are present.
"""

from __future__ import annotations

import asyncio
import os

import pytest

mcp = pytest.importorskip("mcp")

from cs_copilot.mcp.session import (  # noqa: E402  (import-after-skip)
    BootstrapConfig,
    apply_session_id,
    bootstrap,
)


@pytest.fixture(scope="module")
def server(tmp_path_factory):
    work = tmp_path_factory.mktemp("mcp-server")
    os.chdir(work)
    apply_session_id("mcp-server-test")
    ctx = bootstrap(BootstrapConfig(session_id="mcp-server-test", workflow_slug="smoke"))
    from cs_copilot.mcp.server import build_server

    return build_server(ctx)


def test_server_instructions_are_chatgpt_orchestration_contract(server):
    from cs_copilot.mcp.server import SERVER_INSTRUCTIONS

    assert server.instructions == SERVER_INSTRUCTIONS
    assert server._mcp_server.instructions == SERVER_INSTRUCTIONS
    assert len(SERVER_INSTRUCTIONS) <= 512
    assert "external MCP client is the reasoning layer" in SERVER_INSTRUCTIONS
    assert "Do not invoke the Agno team" in SERVER_INSTRUCTIONS
    assert "chemspace_workflow" in SERVER_INSTRUCTIONS
    assert "chembl_retrieval_judge" in SERVER_INSTRUCTIONS


def test_tools_registered(server):
    names = sorted(t.name for t in server._tool_manager.list_tools())
    assert "search" in names
    assert "fetch" in names
    assert "chembl_fetch_compounds" in names
    assert "gtm_optimization" in names
    assert "report_save_markdown" in names


def test_tool_annotations_for_chatgpt_approval(server):
    tools = {tool.name: tool for tool in server._tool_manager.list_tools()}

    for name, tool in tools.items():
        assert tool.annotations is not None, name
        assert tool.annotations.destructiveHint is False, name
        assert tool.annotations.openWorldHint is False, name

    assert tools["search"].annotations.readOnlyHint is True
    assert tools["fetch"].annotations.readOnlyHint is True
    assert tools["chem_calculate_tanimoto_similarity"].annotations.readOnlyHint is True
    assert tools["chembl_describe_dataset"].annotations.readOnlyHint is True
    assert tools["gtm_get_density_summary"].annotations.readOnlyHint is True
    assert tools["session_resolve_candidate_set"].annotations.readOnlyHint is True

    assert tools["chembl_fetch_compounds"].annotations.readOnlyHint is False
    assert tools["gtm_optimization"].annotations.readOnlyHint is False
    assert tools["gtm_sample_nodes"].annotations.readOnlyHint is False
    assert tools["session_select_session_object"].annotations.readOnlyHint is False
    assert tools["session_summarize_session_memory"].annotations.readOnlyHint is False
    assert tools["report_save_markdown"].annotations.readOnlyHint is False


def test_prompts_registered(server):
    names = sorted(p.name for p in server._prompt_manager.list_prompts())
    assert "chemspace_workflow" in names
    assert "chembl_retrieval_judge" in names


def test_resources_include_manifest(server):
    entries = asyncio.run(server.list_resources())
    uris = {str(e.uri) for e in entries}
    assert "cscopilot://session/manifest.json" in uris


def test_transport_security_adds_public_proxy_hosts(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    apply_session_id("mcp-transport-security-test")
    ctx = bootstrap(
        BootstrapConfig(session_id="mcp-transport-security-test", workflow_slug="smoke")
    )
    from cs_copilot.mcp.server import build_server

    server = build_server(
        ctx,
        allowed_hosts=["mcp.example.com"],
        allowed_origins=["https://chatgpt.com"],
    )

    security = server.settings.transport_security
    assert security.enable_dns_rebinding_protection is True
    assert "127.0.0.1:*" in security.allowed_hosts
    assert "localhost:*" in security.allowed_hosts
    assert "mcp.example.com" in security.allowed_hosts
    assert "http://127.0.0.1:*" in security.allowed_origins
    assert "https://chatgpt.com" in security.allowed_origins


def test_transport_security_can_be_disabled(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    apply_session_id("mcp-transport-security-disabled-test")
    ctx = bootstrap(
        BootstrapConfig(
            session_id="mcp-transport-security-disabled-test",
            workflow_slug="smoke",
        )
    )
    from cs_copilot.mcp.server import build_server

    server = build_server(ctx, disable_dns_rebinding_protection=True)

    security = server.settings.transport_security
    assert security.enable_dns_rebinding_protection is False


def test_transport_security_rejects_unlisted_host(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    apply_session_id("mcp-transport-security-http-test")
    ctx = bootstrap(
        BootstrapConfig(session_id="mcp-transport-security-http-test", workflow_slug="smoke")
    )
    from cs_copilot.mcp.server import build_server

    server = build_server(ctx)

    async def request_with_public_host():
        import httpx

        app = server.streamable_http_app()
        async with server.session_manager.run():
            transport = httpx.ASGITransport(app=app)
            async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
                return await client.post(
                    "/mcp",
                    json={},
                    headers={"Host": "mcp.example.com"},
                )

    response = asyncio.run(request_with_public_host())

    assert response.status_code == 421


def test_transport_security_allows_configured_host(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    apply_session_id("mcp-transport-security-http-allow-test")
    ctx = bootstrap(
        BootstrapConfig(
            session_id="mcp-transport-security-http-allow-test",
            workflow_slug="smoke",
        )
    )
    from cs_copilot.mcp.server import build_server

    server = build_server(ctx, allowed_hosts=["mcp.example.com"])

    async def request_with_public_host():
        import httpx

        app = server.streamable_http_app()
        async with server.session_manager.run():
            transport = httpx.ASGITransport(app=app)
            async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
                return await client.post(
                    "/mcp",
                    json={},
                    headers={"Host": "mcp.example.com"},
                )

    response = asyncio.run(request_with_public_host())

    assert response.status_code != 421


def test_bearer_auth_configured(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    apply_session_id("mcp-auth-test")
    ctx = bootstrap(BootstrapConfig(session_id="mcp-auth-test", workflow_slug="smoke"))
    from cs_copilot.mcp.server import build_server

    protected = "https://mcp.example.com/mcp"
    server = build_server(
        ctx,
        auth_token="secret",
        auth_token_client_id="client-1",
        auth_token_scopes=["mcp:read"],
        auth_issuer_url=protected,
        auth_resource_url=protected,
    )

    assert str(server.settings.auth.issuer_url) == protected
    assert str(server.settings.auth.resource_server_url) == protected
    assert server.settings.auth.required_scopes == ["mcp:read"]
    token = asyncio.run(server._token_verifier.verify_token("secret"))
    assert token.client_id == "client-1"


def test_bearer_auth_rejects_missing_header(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    apply_session_id("mcp-auth-http-test")
    ctx = bootstrap(BootstrapConfig(session_id="mcp-auth-http-test", workflow_slug="smoke"))
    from cs_copilot.mcp.server import build_server

    server = build_server(
        ctx,
        auth_token="secret",
        auth_issuer_url="https://mcp.example.com/mcp",
        auth_resource_url="https://mcp.example.com/mcp",
    )

    async def request_without_auth():
        import httpx

        transport = httpx.ASGITransport(app=server.streamable_http_app())
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            return await client.post("/mcp", json={})

    response = asyncio.run(request_without_auth())

    assert response.status_code == 401
    assert response.json()["error"] == "invalid_token"
    assert response.headers["www-authenticate"].startswith("Bearer")
