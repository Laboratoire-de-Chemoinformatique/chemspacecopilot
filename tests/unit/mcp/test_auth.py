"""Tests for MCP remote auth helpers."""

from __future__ import annotations

import asyncio

import pytest

pytest.importorskip("mcp")

from cs_copilot.mcp.auth import StaticBearerTokenVerifier, build_auth_settings


def test_static_bearer_verifier_accepts_expected_token():
    verifier = StaticBearerTokenVerifier(
        expected_token="secret",
        client_id="client-1",
        scopes=("mcp:read",),
    )

    token = asyncio.run(verifier.verify_token("secret"))

    assert token is not None
    assert token.client_id == "client-1"
    assert token.scopes == ["mcp:read"]


def test_static_bearer_verifier_rejects_other_token():
    verifier = StaticBearerTokenVerifier(expected_token="secret")

    token = asyncio.run(verifier.verify_token("wrong"))

    assert token is None


def test_build_auth_settings():
    settings = build_auth_settings(
        issuer_url="https://mcp.example.com/mcp",
        resource_server_url="https://mcp.example.com/mcp",
        required_scopes=("mcp:read",),
    )

    assert str(settings.issuer_url) == "https://mcp.example.com/mcp"
    assert str(settings.resource_server_url) == "https://mcp.example.com/mcp"
    assert settings.required_scopes == ["mcp:read"]
