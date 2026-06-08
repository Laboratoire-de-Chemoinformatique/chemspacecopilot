"""Local readiness check for remote cs_copilot MCP clients."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import socket
import sys
from dataclasses import asdict, dataclass
from typing import Sequence

from .auth import DEFAULT_AUTH_CLIENT_ID, DEFAULT_AUTH_TOKEN_ENV
from .lazy import require_mcp

DEFAULT_REQUIRED_TOOLS = (
    "search",
    "fetch",
    "chembl_prepare_retrieval",
    "chembl_fetch_compounds",
    "chemspace_plan_analysis",
    "gtm_optimization",
    "report_save_markdown",
    "skill_list",
    "pandas_create_dataframe",
    "mol_validate_design_candidates",
    "peptide_validate_design_candidates",
    "synplanner_identify_input",
)

EXPECTED_READ_ONLY_HINTS = {
    "search": True,
    "fetch": True,
    "chembl_prepare_retrieval": True,
    "chembl_fetch_compounds": False,
    "chemspace_plan_analysis": True,
    "gtm_optimization": False,
    "report_save_markdown": False,
    "skill_list": True,
    "pandas_create_dataframe": False,
    "mol_validate_design_candidates": True,
    "peptide_validate_design_candidates": True,
    "synplanner_identify_input": True,
}

EXPECTED_INSTRUCTION_SNIPPETS = (
    "external MCP client is the reasoning layer",
    "Do not invoke the Agno team",
    "cs_copilot_workflow",
    "chembl_prepare_retrieval",
    "chemspace_plan_analysis",
    "session_*",
    "chembl_retrieval_judge",
    "Review write actions",
)

CHATGPT_CONNECTOR_NAME = "cs_copilot"
CHATGPT_CONNECTOR_DESCRIPTION = (
    "Chemistry and chemography MCP tools for ChEMBL retrieval, GTM "
    "chemical-space modeling, chemoinformatics analysis, molecular design, "
    "peptide design, session artifacts, and report generation."
)
CHATGPT_SMOKE_PROMPT = (
    "Use cs_copilot. Do not use built-in browsing or other tools. "
    "Use only the cs_copilot connector. Search the MCP catalog for "
    "ChEMBL retrieval tools, fetch prompt:cs_copilot_workflow, and explain "
    "which preflight and retrieval tools would handle CDK2 inhibitor activity "
    "data. Do not run long ChEMBL or GTM jobs yet."
)
CHATGPT_EXPECTED_EVIDENCE = (
    "ChatGPT selects the cs_copilot connector / Developer Mode app.",
    "The transcript shows a `search` tool call against the cs_copilot MCP catalog.",
    "The transcript shows a `fetch` tool call for `prompt:cs_copilot_workflow`.",
    "The answer names `chembl_prepare_retrieval` as the ChEMBL preflight tool.",
    "The answer names `chembl_fetch_compounds` as the ChEMBL retrieval tool.",
)


class CheckError(RuntimeError):
    """Raised when the readiness check cannot prove MCP connectivity."""


@dataclass(frozen=True)
class CheckReport:
    """Result from a successful remote MCP readiness check."""

    endpoint_url: str
    session_id: str | None
    tool_count: int
    prompt_count: int
    resource_count: int
    required_tools: tuple[str, ...]
    fetch_id: str
    workflow_prompt_id: str
    skill_id: str
    auth_enabled: bool
    mode: str
    annotated_tool_count: int
    read_only_tool_count: int
    write_tool_count: int
    instructions_length: int


def _positive_float(value: str) -> float:
    parsed = float(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return parsed


def _path(value: str) -> str:
    if not value.startswith("/"):
        raise argparse.ArgumentTypeError("path must start with '/'")
    return value


def _url(value: str) -> str:
    if not value.startswith(("http://", "https://")):
        raise argparse.ArgumentTypeError("URL must start with http:// or https://")
    return value


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="cscopilot-mcp-check",
        description=(
            "Verify ChatGPT-ready cs_copilot MCP capabilities by probing an "
            "existing streamable HTTP endpoint or by starting a temporary "
            "local MCP server."
        ),
    )
    parser.add_argument(
        "--url",
        type=_url,
        default=None,
        help=(
            "Existing streamable HTTP MCP endpoint to probe instead of "
            "starting a temporary local server, for example https://host/mcp."
        ),
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Host for the temporary HTTP server.",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=0,
        help="Port for the temporary HTTP server. Use 0 to pick a free port.",
    )
    parser.add_argument(
        "--path",
        type=_path,
        default="/mcp",
        help="Streamable HTTP endpoint path to probe.",
    )
    parser.add_argument(
        "--session-id",
        default="mcp-check",
        help="Session id for the temporary server.",
    )
    parser.add_argument(
        "--workflow-slug",
        default="smoke",
        help="Workflow slug for the temporary server.",
    )
    parser.add_argument(
        "--timeout",
        type=_positive_float,
        default=30.0,
        help="Seconds to wait for the temporary server to become reachable.",
    )
    parser.add_argument(
        "--log-level",
        choices=("debug", "info", "warning", "error"),
        default="error",
        help="Log level passed to the temporary MCP server.",
    )
    parser.add_argument(
        "--use-s3",
        action="store_true",
        help="Use the current S3 environment instead of forcing local storage.",
    )
    parser.add_argument(
        "--required-tool",
        action="append",
        default=[],
        help=("Additional MCP tool name that must be present. Can be supplied " "multiple times."),
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print a machine-readable JSON readiness report.",
    )
    parser.add_argument(
        "--auth-token-env",
        default=DEFAULT_AUTH_TOKEN_ENV,
        help=(
            "Environment variable containing a bearer token for the HTTP "
            "endpoint. Leave unset for no built-in bearer auth."
        ),
    )
    parser.add_argument(
        "--auth-token",
        default=None,
        help=(
            "Bearer token for the HTTP endpoint. Prefer "
            "--auth-token-env so the token is not visible in process listings."
        ),
    )
    parser.add_argument(
        "--auth-client-id",
        default=DEFAULT_AUTH_CLIENT_ID,
        help="Client id exposed to the MCP SDK after bearer-token verification.",
    )
    parser.add_argument(
        "--auth-scope",
        action="append",
        default=[],
        help="Required bearer-token scope. Can be supplied multiple times.",
    )
    return parser.parse_args(argv)


def _free_port(host: str) -> int:
    bind_host = "127.0.0.1" if host in {"0.0.0.0", "::"} else host
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind((bind_host, 0))
        return int(sock.getsockname()[1])


def _connect_host(host: str) -> str:
    if host in {"0.0.0.0", "::"}:
        return "127.0.0.1"
    return host


def _endpoint_url(host: str, port: int, path: str) -> str:
    return f"http://{_connect_host(host)}:{port}{path}"


def _resolve_auth_token(args: argparse.Namespace) -> str | None:
    return args.auth_token or os.getenv(args.auth_token_env)


def _server_env(*, use_s3: bool, auth_token_env: str, auth_token: str | None) -> dict[str, str]:
    env = dict(os.environ)
    env.setdefault("AGNO_TELEMETRY", "false")
    if not use_s3:
        env["USE_S3"] = "false"
    if auth_token:
        env[auth_token_env] = auth_token
    return env


async def _terminate_process(proc: asyncio.subprocess.Process) -> tuple[str, str]:
    if proc.returncode is None:
        proc.terminate()
        try:
            await asyncio.wait_for(proc.wait(), timeout=5)
        except asyncio.TimeoutError:  # pragma: no cover - defensive cleanup
            proc.kill()
            await proc.wait()
    stdout, stderr = await proc.communicate()
    return stdout.decode(errors="replace"), stderr.decode(errors="replace")


async def _open_streamable_http_client(endpoint_url: str, auth_token: str | None):
    from contextlib import AsyncExitStack

    import httpx

    from mcp.client.streamable_http import streamable_http_client

    stack = AsyncExitStack()
    await stack.__aenter__()
    try:
        http_client = None
        if auth_token:
            http_client = await stack.enter_async_context(
                httpx.AsyncClient(headers={"Authorization": f"Bearer {auth_token}"})
            )
        client = await stack.enter_async_context(
            streamable_http_client(endpoint_url, http_client=http_client)
        )
        return stack, client
    except Exception:
        await stack.__aexit__(*sys.exc_info())
        raise


def _validate_server_instructions(instructions: str | None) -> int:
    if not instructions:
        raise CheckError("MCP endpoint did not expose server instructions")
    missing = [snippet for snippet in EXPECTED_INSTRUCTION_SNIPPETS if snippet not in instructions]
    if missing:
        raise CheckError(
            "MCP endpoint exposed incomplete server instructions; missing: " + ", ".join(missing)
        )
    if len(instructions) > 512:
        raise CheckError(
            "MCP endpoint server instructions exceed ChatGPT's recommended "
            f"self-contained 512-character window: {len(instructions)} characters"
        )
    return len(instructions)


def _annotation_value(tool: object, field: str) -> object:
    annotations = getattr(tool, "annotations", None)
    if annotations is None:
        return None
    return getattr(annotations, field, None)


def _validate_tool_annotations(tools: Sequence[object]) -> tuple[int, int, int]:
    missing = sorted(
        str(getattr(tool, "name", "<unnamed>"))
        for tool in tools
        if getattr(tool, "annotations", None) is None
    )
    if missing:
        raise CheckError("MCP endpoint exposed tools without annotations: " + ", ".join(missing))

    read_only = 0
    write = 0
    bad_read_only = []
    bad_destructive = []
    bad_open_world = []
    tools_by_name = {str(getattr(tool, "name", "")): tool for tool in tools}

    for tool in tools:
        name = str(getattr(tool, "name", "<unnamed>"))
        read_only_hint = _annotation_value(tool, "readOnlyHint")
        if read_only_hint is True:
            read_only += 1
        elif read_only_hint is False:
            write += 1
        else:
            bad_read_only.append(name)

        if _annotation_value(tool, "destructiveHint") is not False:
            bad_destructive.append(name)
        if _annotation_value(tool, "openWorldHint") is not False:
            bad_open_world.append(name)

    if bad_read_only:
        raise CheckError(
            "MCP endpoint exposed tools without boolean readOnlyHint: "
            + ", ".join(sorted(bad_read_only))
        )
    if bad_destructive:
        raise CheckError(
            "MCP endpoint exposed unexpected destructive tools: "
            + ", ".join(sorted(bad_destructive))
        )
    if bad_open_world:
        raise CheckError(
            "MCP endpoint exposed unexpected open-world tools: " + ", ".join(sorted(bad_open_world))
        )

    mismatched = []
    for name, expected in EXPECTED_READ_ONLY_HINTS.items():
        tool = tools_by_name.get(name)
        if tool is None:
            continue
        actual = _annotation_value(tool, "readOnlyHint")
        if actual is not expected:
            mismatched.append(f"{name}={actual!r}, expected {expected!r}")
    if mismatched:
        raise CheckError(
            "MCP endpoint exposed incorrect readOnlyHint values: " + "; ".join(mismatched)
        )

    return len(tools), read_only, write


async def _verify_search_fetch(
    session: object,
    *,
    query: str,
    expected_id: str | None = None,
    required_text: Sequence[str] = (),
) -> str:
    search_result = await session.call_tool("search", {"query": query})
    structured = search_result.structuredContent
    if not structured or not structured.get("results"):
        raise CheckError(f"search tool returned no structured results for query: {query}")

    result_ids = [str(item.get("id")) for item in structured["results"]]
    fetch_id = expected_id or result_ids[0]
    if fetch_id not in result_ids:
        raise CheckError(
            f"search query {query!r} did not return required result {fetch_id!r}; "
            f"got: {', '.join(result_ids)}"
        )

    fetch_result = await session.call_tool("fetch", {"id": fetch_id})
    fetch_payload = fetch_result.structuredContent
    if not fetch_payload or fetch_payload.get("id") != fetch_id:
        raise CheckError(f"fetch tool did not return the requested result: {fetch_id}")

    text = str(fetch_payload.get("text") or "")
    missing_text = [snippet for snippet in required_text if snippet not in text]
    if missing_text:
        raise CheckError(
            f"fetch result {fetch_id!r} was missing required text: " + ", ".join(missing_text)
        )
    return fetch_id


async def _probe_server(
    *,
    endpoint_url: str,
    proc: asyncio.subprocess.Process | None,
    timeout: float,
    required_tools: tuple[str, ...],
    auth_token: str | None,
    mode: str,
) -> CheckReport:
    from mcp import ClientSession

    deadline = asyncio.get_running_loop().time() + timeout
    last_error: Exception | None = None

    while asyncio.get_running_loop().time() < deadline:
        if proc is not None and proc.returncode is not None:
            raise CheckError(f"temporary MCP server exited early with code {proc.returncode}")
        try:
            stack, client = await _open_streamable_http_client(endpoint_url, auth_token)
            try:
                read, write, get_session_id = client
                async with ClientSession(read, write) as session:
                    initialize_result = await session.initialize()
                    instructions_length = _validate_server_instructions(
                        initialize_result.instructions
                    )

                    tools = await session.list_tools()
                    tool_names = {tool.name for tool in tools.tools}
                    missing = sorted(set(required_tools) - tool_names)
                    if missing:
                        raise CheckError(
                            "MCP endpoint is missing required tools: " + ", ".join(missing)
                        )
                    annotated_tool_count, read_only_tool_count, write_tool_count = (
                        _validate_tool_annotations(tools.tools)
                    )

                    prompts = await session.list_prompts()
                    resources = await session.list_resources()
                    resource_uris = {str(resource.uri) for resource in resources.resources}
                    if "cscopilot://session/manifest.json" not in resource_uris:
                        raise CheckError("session manifest resource was not exposed")

                    fetch_id = await _verify_search_fetch(
                        session,
                        query="chembl fetch compounds",
                        expected_id="tool:chembl_fetch_compounds",
                        required_text=("Tool: chembl_fetch_compounds", "LLM-as-judge"),
                    )
                    workflow_prompt_id = await _verify_search_fetch(
                        session,
                        query="cs_copilot workflow prompt",
                        expected_id="prompt:cs_copilot_workflow",
                        required_text=(
                            "Prompt: cs_copilot_workflow",
                            "external reasoner",
                            "agent selection",
                            "session_state",
                        ),
                    )

                    skill_id = await _verify_search_fetch(
                        session,
                        query="gtm activity landscape skill",
                        expected_id="skill:gtm-activity-landscape",
                        required_text=(
                            "Skill: GTM activity landscape",
                            "gtm_create_activity_landscapes",
                            "Required tools",
                        ),
                    )
                    await _verify_search_fetch(
                        session,
                        query="molecular design",
                        expected_id="tool:mol_validate_design_candidates",
                        required_text=("Tool: mol_validate_design_candidates",),
                    )
                    await _verify_search_fetch(
                        session,
                        query="peptide design",
                        expected_id="tool:peptide_validate_design_candidates",
                        required_text=("Tool: peptide_validate_design_candidates",),
                    )
                    await _verify_search_fetch(
                        session,
                        query="synplanner",
                        expected_id="tool:synplanner_identify_input",
                        required_text=("Tool: synplanner_identify_input",),
                    )

                    return CheckReport(
                        endpoint_url=endpoint_url,
                        session_id=get_session_id(),
                        tool_count=len(tools.tools),
                        prompt_count=len(prompts.prompts),
                        resource_count=len(resources.resources),
                        required_tools=required_tools,
                        fetch_id=fetch_id,
                        workflow_prompt_id=workflow_prompt_id,
                        skill_id=skill_id,
                        auth_enabled=bool(auth_token),
                        mode=mode,
                        annotated_tool_count=annotated_tool_count,
                        read_only_tool_count=read_only_tool_count,
                        write_tool_count=write_tool_count,
                        instructions_length=instructions_length,
                    )
            finally:
                await stack.aclose()
        except CheckError:
            raise
        except Exception as exc:  # noqa: BLE001 - retry while the server boots
            last_error = exc
            await asyncio.sleep(0.25)

    raise CheckError(f"timed out connecting to {endpoint_url}: {last_error}")


async def run_check(args: argparse.Namespace) -> CheckReport:
    """Run the remote MCP readiness check and return the successful report."""

    require_mcp()

    required_tools = tuple(dict.fromkeys((*DEFAULT_REQUIRED_TOOLS, *args.required_tool)))
    auth_token = _resolve_auth_token(args)

    if args.url:
        return await _probe_server(
            endpoint_url=args.url,
            proc=None,
            timeout=args.timeout,
            required_tools=required_tools,
            auth_token=auth_token,
            mode="existing-url",
        )

    port = args.port or _free_port(args.host)
    endpoint_url = _endpoint_url(args.host, port, args.path)

    server_args = [
        sys.executable,
        "-m",
        "cs_copilot.mcp",
        "--transport",
        "streamable-http",
        "--session-id",
        args.session_id,
        "--workflow-slug",
        args.workflow_slug,
        "--host",
        args.host,
        "--port",
        str(port),
        "--streamable-http-path",
        args.path,
        "--log-level",
        args.log_level,
    ]
    if auth_token:
        server_args.extend(
            [
                "--auth-token-env",
                args.auth_token_env,
                "--auth-client-id",
                args.auth_client_id,
                "--auth-issuer-url",
                endpoint_url,
                "--auth-resource-url",
                endpoint_url,
            ]
        )
        for scope in args.auth_scope:
            server_args.extend(["--auth-scope", scope])

    proc = await asyncio.create_subprocess_exec(
        *server_args,
        env=_server_env(
            use_s3=args.use_s3,
            auth_token_env=args.auth_token_env,
            auth_token=auth_token,
        ),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )

    try:
        report = await _probe_server(
            endpoint_url=endpoint_url,
            proc=proc,
            timeout=args.timeout,
            required_tools=required_tools,
            auth_token=auth_token,
            mode="temporary-server",
        )
    except Exception as exc:
        stdout, stderr = await _terminate_process(proc)
        raise CheckError(
            f"{exc}\n\nserver stdout:\n{stdout or '<empty>'}\n\n"
            f"server stderr:\n{stderr or '<empty>'}"
        ) from exc

    stdout, stderr = await _terminate_process(proc)
    if proc.returncode not in (0, -15):
        raise CheckError(
            f"temporary MCP server exited with code {proc.returncode}\n\n"
            f"server stdout:\n{stdout or '<empty>'}\n\nserver stderr:\n{stderr or '<empty>'}"
        )
    return report


def _report_payload(report: CheckReport) -> dict[str, object]:
    payload = asdict(report)
    payload["required_tools"] = list(report.required_tools)
    payload["status"] = "passed"
    payload["auth"] = "bearer-token" if report.auth_enabled else "none"
    payload["chatgpt_next"] = (
        "Expose this endpoint as HTTPS, or connect it through OpenAI Secure "
        "MCP Tunnel / another trusted tunnel, then create the ChatGPT app "
        "from that remote MCP URL."
    )
    payload["chatgpt_connector_name"] = CHATGPT_CONNECTOR_NAME
    payload["chatgpt_connector_description"] = CHATGPT_CONNECTOR_DESCRIPTION
    payload["chatgpt_smoke_prompt"] = CHATGPT_SMOKE_PROMPT
    payload["chatgpt_expected_evidence"] = list(CHATGPT_EXPECTED_EVIDENCE)
    return payload


def _print_report(report: CheckReport, *, json_output: bool = False) -> None:
    if json_output:
        print(json.dumps(_report_payload(report), indent=2, sort_keys=True))
        return

    print("cs_copilot MCP readiness check passed")
    print(f"endpoint_url: {report.endpoint_url}")
    print(f"mcp_session_id: {report.session_id or '<none>'}")
    print(f"tools: {report.tool_count}")
    print(f"prompts: {report.prompt_count}")
    print(f"resources: {report.resource_count}")
    print(f"mode: {report.mode}")
    print(f"auth: {'bearer-token' if report.auth_enabled else 'none'}")
    print(f"required_tools: {', '.join(report.required_tools)}")
    print(f"server_instructions: ok ({report.instructions_length} chars)")
    print(
        "tool_annotations: ok "
        f"({report.annotated_tool_count} annotated, "
        f"{report.read_only_tool_count} read-only, "
        f"{report.write_tool_count} write-action)"
    )
    payload = _report_payload(report)
    print(f"workflow_prompt: ok ({report.workflow_prompt_id})")
    print(f"skill_fetch: ok ({report.skill_id})")
    print(f"search_fetch: ok ({report.fetch_id})")
    print(f"chatgpt_connector_name: {payload['chatgpt_connector_name']}")
    print(f"chatgpt_smoke_prompt: {payload['chatgpt_smoke_prompt']}")
    print(payload["chatgpt_next"])


def main(argv: Sequence[str] | None = None) -> int:
    """Console entry point for ``cscopilot-mcp-check``."""

    args = _parse_args(argv)
    try:
        report = asyncio.run(run_check(args))
    except Exception as exc:  # noqa: BLE001 - CLI should show actionable failure text
        print(f"cs_copilot MCP readiness check failed: {exc}", file=sys.stderr)
        return 1
    _print_report(report, json_output=args.json)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
