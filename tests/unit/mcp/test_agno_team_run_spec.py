"""``agno_team_run`` is an opt-in, kernel-governed MCP tool spec."""

from __future__ import annotations

import pytest

from cs_copilot.execution.spec import ToolSpec
from cs_copilot.mcp.__main__ import _parse_args
from cs_copilot.mcp.tools_registry import (
    OPT_IN_GROUPS,
    all_specs,
    required_permissions_for_spec,
)


def _team_run_spec() -> ToolSpec:
    [spec] = [spec for spec in all_specs(opt_in_groups=OPT_IN_GROUPS) if spec.group == "agno"]
    return spec


def test_team_run_is_only_listed_when_opted_in():
    assert "agno_team_run" not in {spec.mcp_name for spec in all_specs()}
    assert _team_run_spec().mcp_name == "agno_team_run"
    reporting = all_specs("reporting", opt_in_groups=OPT_IN_GROUPS)
    assert "agno_team_run" not in {spec.mcp_name for spec in reporting}
    with pytest.raises(ValueError, match="Unknown opt-in tool groups: chembl"):
        all_specs(opt_in_groups={"chembl"})


def test_team_run_contract():
    spec = _team_run_spec()

    assert spec.roles == ("supervisor",)
    assert spec.profiles == ("standard",)
    assert spec.delegates_execution
    assert spec.write_scope == "none" and not spec.read_only
    assert spec.open_world and spec.requires_network and spec.risk == "high"
    assert not spec.idempotent and spec.max_retries == 0
    assert spec.timeout_s == 1800
    assert spec.agno_bindings == ()
    assert {"artifact:write", "network:read"} <= required_permissions_for_spec(spec)


@pytest.mark.parametrize(
    "overrides",
    [
        {"read_only": True},
        {"write_scope": "session"},
        {"result_artifact_type": "report"},
        {"run_in_worker_process": True},
    ],
)
def test_a_delegating_tool_cannot_declare_its_own_writes(overrides):
    with pytest.raises(ValueError, match="delegating tool"):
        ToolSpec(
            mcp_name="agno_probe",
            toolkit_factory=object,
            method="run",
            summary="Probe.",
            delegates_execution=True,
            **overrides,
        )


def test_cli_requires_the_agno_model_policy_and_standard_profile(capsys, monkeypatch):
    monkeypatch.delenv("CS_COPILOT_MCP_LLM_POLICY", raising=False)
    monkeypatch.delenv("CS_COPILOT_MCP_PROFILE", raising=False)
    with pytest.raises(SystemExit):
        _parse_args(["--enable-agno-team-tool"])
    assert "requires --llm-policy agno-model" in capsys.readouterr().err

    with pytest.raises(SystemExit):
        _parse_args(
            ["--enable-agno-team-tool", "--llm-policy", "agno-model", "--profile", "reporting"]
        )
    assert "only available with --profile standard" in capsys.readouterr().err

    args = _parse_args(["--enable-agno-team-tool", "--llm-policy", "agno-model"])
    assert args.enable_agno_team_tool and args.llm_policy == "agno-model"
