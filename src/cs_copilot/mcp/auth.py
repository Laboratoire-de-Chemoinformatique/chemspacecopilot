"""Authentication helpers for remote MCP transports."""

from __future__ import annotations

import hmac
from dataclasses import dataclass, field
from typing import Sequence

DEFAULT_AUTH_TOKEN_ENV = "CS_COPILOT_MCP_AUTH_TOKEN"
DEFAULT_AUTH_CLIENT_ID = "chemspace-mcp-client"


@dataclass(frozen=True)
class StaticBearerTokenVerifier:
    """MCP SDK token verifier for a single shared bearer token."""

    expected_token: str
    client_id: str = DEFAULT_AUTH_CLIENT_ID
    scopes: Sequence[str] = field(default_factory=tuple)

    async def verify_token(self, token: str):
        """Return MCP access information when ``token`` matches exactly."""

        if not hmac.compare_digest(token, self.expected_token):
            return None

        from mcp.server.auth.provider import AccessToken

        return AccessToken(
            token=token,
            client_id=self.client_id,
            scopes=list(self.scopes),
        )


def build_auth_settings(
    *,
    issuer_url: str,
    resource_server_url: str,
    required_scopes: Sequence[str] = (),
):
    """Build MCP SDK auth settings for a protected resource server."""

    from mcp.server.auth.settings import AuthSettings

    return AuthSettings(
        issuer_url=issuer_url,
        resource_server_url=resource_server_url,
        required_scopes=list(required_scopes) or None,
    )
