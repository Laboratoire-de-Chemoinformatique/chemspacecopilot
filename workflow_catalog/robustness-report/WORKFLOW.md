---
name: robustness-report
description: Load robustness results, identify failures and trends, and persist an analysis report.
metadata:
  title: Robustness report
  status: stable
  tags:
    - robustness
    - evaluation
    - report
  keywords:
    - robustness
    - prompt variation
    - test run
  required_tools:
    - robustness_load_test_results
    - robustness_analyze_score_distribution
    - robustness_identify_failing_prompts
    - robustness_generate_insights
    - robustness_export_analysis_report
  optional_tools:
    - robustness_load_test_summary_csv
    - robustness_compare_test_runs
    - robustness_analyze_temporal_trends
  expected_artifacts:
    - robustness_report_path
    - failing_prompts_summary
    - score_distribution_summary
  recommended_prompt: robustness_evaluation
---

# Robustness Report

Use this workflow to turn robustness test outputs into a compact report.

1. Load raw robustness results and optional summary CSV data.
2. Analyze score distribution and identify prompts above the failure threshold.
3. Compare runs or analyze temporal trends when multiple runs are relevant.
4. Generate textual insights from the analysis.
5. Persist the final report with `robustness_export_analysis_report`.
