from cs_copilot.tracking.replay import (
    GoldenTrajectory,
    evaluate_trajectory,
    golden_for_workflow,
    log_trajectory_report,
)


def _event(kind, **payload):
    return {"event_type": kind, "payload": payload}


def test_golden_trajectory_accepts_semantic_workflow():
    events = [
        _event("run_status_changed", status="running"),
        _event(
            "tool_call_recorded",
            tool_name="chembl_prepare_retrieval",
            role="chembl_downloader",
            status="success",
            duration_ms=2,
        ),
        _event(
            "tool_call_recorded",
            tool_name="chembl_fetch_compounds",
            role="chembl_downloader",
            status="success",
            duration_ms=8,
        ),
        _event("handoff_recorded", receiver_role="gtm_agent"),
        _event("artifact_registered", artifact={"artifact_type": "clean_dataset_path"}),
        _event("run_status_changed", status="completed"),
    ]
    golden = GoldenTrajectory(
        required_event_types=("handoff_recorded",),
        required_tool_order=("chembl_prepare_retrieval", "chembl_fetch_compounds"),
        required_artifact_types=("clean_dataset_path",),
        preflight_requirements={
            "chembl_fetch_compounds": ("chembl_prepare_retrieval",),
        },
        role_tool_allowlists={
            "chembl_downloader": frozenset({"chembl_prepare_retrieval", "chembl_fetch_compounds"})
        },
    )

    report = evaluate_trajectory(events, golden)

    assert report.passed
    assert report.tool_call_count == 2
    assert report.handoff_count == 1
    assert report.total_tool_duration_ms == 10


def test_golden_trajectory_reports_order_permissions_duplicates_and_artifacts():
    events = [
        _event(
            "tool_call_recorded",
            tool_name="chembl_fetch_compounds",
            role="report_generator",
            status="success",
            arguments={"target": "EGFR"},
        ),
        _event(
            "tool_call_recorded",
            tool_name="chembl_fetch_compounds",
            role="report_generator",
            status="success",
            arguments={"target": "EGFR"},
        ),
        _event("run_status_changed", status="partial"),
    ]
    golden = GoldenTrajectory(
        required_tool_order=("chembl_prepare_retrieval", "chembl_fetch_compounds"),
        required_artifact_types=("clean_dataset_path",),
        preflight_requirements={
            "chembl_fetch_compounds": ("chembl_prepare_retrieval",),
        },
        role_tool_allowlists={"report_generator": frozenset({"report_save_rich"})},
    )

    report = evaluate_trajectory(events, golden)

    assert not report.passed
    assert report.duplicate_call_count == 1
    assert any("disallowed tool" in item for item in report.violations)
    assert any("before preflight" in item for item in report.violations)
    assert any("missing artifact" in item for item in report.violations)
    assert any("terminal status" in item for item in report.violations)


def test_pilot_golden_trajectory_is_derived_from_catalog_contracts():
    golden = golden_for_workflow("chembl-to-gtm-report")

    assert golden.required_tool_order[0] == "chembl_prepare_retrieval"
    assert golden.required_tool_order[-1] == "report_save_rich"
    assert "clean_dataset_path" in golden.required_artifact_types
    assert "html_report_path" in golden.required_artifact_types
    assert golden.preflight_requirements["chembl_fetch_compounds"] == ("chembl_prepare_retrieval",)
    assert "chemspace_plan_analysis" in golden.preflight_requirements["gtm_optimization"]
    assert golden.role_tool_allowlists["report_generator"] == frozenset(
        {
            "llm_get_task",
            "llm_submit_task_result",
            "report_save_markdown",
            "report_save_rich",
        }
    )
    assert golden.required_event_types == (
        "handoff_recorded",
        "artifact_registered",
    )


def test_trajectory_report_exports_metrics_and_diagnostics_to_tracker():
    report = evaluate_trajectory(
        [_event("run_status_changed", status="completed")],
        GoldenTrajectory(),
    )

    class Tracker:
        def __init__(self):
            self.metrics = None
            self.artifact = None

        def log_metrics(self, metrics):
            self.metrics = metrics

        def log_dict(self, payload, path):
            self.artifact = (payload, path)

    tracker = Tracker()
    log_trajectory_report(report, tracker)

    assert tracker.metrics["trajectory_passed"] == 1.0
    assert tracker.artifact[1] == "trajectory_report.json"
    assert tracker.artifact[0]["passed"] is True


def test_failed_calls_do_not_satisfy_order_or_preflight_requirements():
    report = evaluate_trajectory(
        [
            _event(
                "tool_call_recorded",
                tool_name="chembl_prepare_retrieval",
                role="chembl_downloader",
                status="error",
            ),
            _event(
                "tool_call_recorded",
                tool_name="chembl_fetch_compounds",
                role="chembl_downloader",
                status="success",
            ),
            _event("run_status_changed", status="completed"),
        ],
        GoldenTrajectory(
            required_tool_order=(
                "chembl_prepare_retrieval",
                "chembl_fetch_compounds",
            ),
            preflight_requirements={
                "chembl_fetch_compounds": ("chembl_prepare_retrieval",),
            },
            role_tool_allowlists={
                "chembl_downloader": frozenset(
                    {"chembl_prepare_retrieval", "chembl_fetch_compounds"}
                )
            },
        ),
    )

    assert not report.passed
    assert any("missing or out-of-order" in item for item in report.violations)
    assert any("before preflight" in item for item in report.violations)


def test_cache_hits_are_not_duplicate_executions_and_attempts_count_retries():
    report = evaluate_trajectory(
        [
            _event(
                "tool_call_recorded",
                tool_name="chembl_fetch_compounds",
                role="chembl_downloader",
                status="success",
                arguments={"target": "EGFR"},
                attempts=3,
            ),
            _event(
                "tool_call_recorded",
                tool_name="chembl_fetch_compounds",
                role="chembl_downloader",
                status="success",
                arguments={"target": "EGFR"},
                cached=True,
            ),
            _event("run_status_changed", status="completed"),
        ],
        GoldenTrajectory(
            role_tool_allowlists={"chembl_downloader": frozenset({"chembl_fetch_compounds"})},
        ),
    )

    assert report.passed
    assert report.duplicate_call_count == 0
    assert report.retry_count == 2


def test_workflow_tool_call_without_role_fails_the_allowlist_check():
    report = evaluate_trajectory(
        [
            _event(
                "tool_call_recorded",
                tool_name="chembl_fetch_compounds",
                status="success",
            ),
            _event("run_status_changed", status="completed"),
        ],
        GoldenTrajectory(
            role_tool_allowlists={"chembl_downloader": frozenset({"chembl_fetch_compounds"})},
        ),
    )

    assert not report.passed
    assert any("missing its executing role" in item for item in report.violations)
