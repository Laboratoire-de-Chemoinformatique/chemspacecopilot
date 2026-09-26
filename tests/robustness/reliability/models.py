"""Normalized schemas for reliability benchmark outputs."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

RELIABILITY_SCHEMA_VERSION = "1.1"


@dataclass
class ToolCallRecord:
    """Compact, JSON-safe record of one executed tool call."""

    sequence: int
    tool_name: str
    agent_name: Optional[str] = None
    tool_args: Dict[str, Any] = field(default_factory=dict)
    created_at: Optional[int] = None
    duration_seconds: Optional[float] = None
    error: bool = False
    result_preview: Optional[str] = None
    result_sha256: Optional[str] = None
    child_run_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ValidationResult:
    """One machine-verifiable acceptance check."""

    name: str
    passed: bool
    evidence: str
    category: Optional[str] = None
    severity: str = "required"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ReliabilityRunRecord:
    """Publication-facing normalized record for one prompt execution."""

    benchmark_run_id: str
    case_name: str
    run_id: str
    session_id: str
    system_under_test: str
    tier: str
    prompt_variant: int
    repetition: int
    prompt: str
    response_path: Optional[str]
    execution_status: str
    task_success: bool
    started_at: str
    finished_at: str
    wall_time_seconds: float
    model_provider: Optional[str] = None
    model_id: Optional[str] = None
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    total_tokens: Optional[int] = None
    reasoning_tokens: Optional[int] = None
    cache_read_tokens: Optional[int] = None
    cache_write_tokens: Optional[int] = None
    llm_duration_seconds: Optional[float] = None
    estimated_cost: Optional[float] = None
    tool_call_count: Optional[int] = None
    failed_tool_call_count: Optional[int] = None
    telemetry_status: str = "unavailable"
    token_metrics_status: str = "unavailable"
    tool_calls: List[Dict[str, Any]] = field(default_factory=list)
    validations: List[Dict[str, Any]] = field(default_factory=list)
    failure_categories: List[str] = field(default_factory=list)
    generated_files: Dict[str, str] = field(default_factory=dict)
    scientific_outcome: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None
    schema_version: str = RELIABILITY_SCHEMA_VERSION

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
