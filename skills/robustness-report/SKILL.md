---
name: robustness-report
description: Load robustness results, identify failures and trends, and persist an analysis report.
metadata:
  title: Robustness report
  status: stable
  version: 2.0.0
  depends_on: []
  profiles:
    - robustness
  permissions:
    - artifact:read
    - artifact:write
  input_artifacts:
    - name: robustness_results
      kind: evaluation-results
      required: true
  output_artifacts:
    - name: robustness_report_path
      kind: report
      required: true
    - name: failing_prompts_summary
      kind: evaluation-summary
      required: true
    - name: score_distribution_summary
      kind: evaluation-summary
      required: true
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
  example_prompts:
    - Analyze the latest robustness run, list failing prompts, and save a report.
---

# Robustness Report

Use this skill when the user wants a compact report from robustness test outputs.

## Procedure

1. Identify the requested robustness run. If no timestamp is specified, use `robustness_load_test_results` with the intended default/latest behavior or inspect available runs when that tool path is available.
2. Load raw results with `robustness_load_test_results` and optionally load summary CSV data with `robustness_load_test_summary_csv`.
3. Analyze score distribution with `robustness_analyze_score_distribution` and classify the overall run quality.
4. Identify failing prompts with `robustness_identify_failing_prompts`, prioritizing low scores, validation errors, tool errors, timeouts, and process inconsistencies.
5. Compare runs with `robustness_compare_test_runs` or analyze temporal trends with `robustness_analyze_temporal_trends` when multiple runs are relevant.
6. Generate insights with `robustness_generate_insights`, linking recommendations to score components or concrete failures.
7. Persist the report with `robustness_export_analysis_report` and return the saved path.

## Expected Outputs

- Robustness analysis report.
- Failing prompt summary.
- Score distribution and trend summary when available.

## Details

- **Rating bands**: Excellent ≥0.90, Good ≥0.80, Acceptable ≥0.70, Concerning <0.70. Also report the success rate from total/passed/failed.
- **Failing-prompt threshold**: 0.70. Group failures by type (timeouts, validation errors, tool errors, low scores) and surface the most critical first.
- **Component metrics** (identify the lowest as the biggest weakness and prioritize accordingly): `data_similarity` (data fetch/filter inconsistency), `semantic_similarity` (LLM response variation), `process_consistency` (tool-call sequence variation), `visual_similarity` (plotting parameter variation).
- **Clarification vs immediate**: split by the `requires_clarification` column and report a significant gap (>10% success-rate difference).
- **Temporal trends**: improvement = score increase >0.05, regression = decrease >0.05; overall trend Improving / Declining / Stable; emphasize regressions as critical.
- For data-focused tests (e.g. chembl_download), check dataset-name / row-count consistency across prompt variations, and compare tool-call patterns between successful and failed runs.
- **Recommendation priorities**: Critical (score <0.70) / Important (regressions) / Nice-to-have (improvements); keep recommendations specific and actionable, linked to the weakest component.
