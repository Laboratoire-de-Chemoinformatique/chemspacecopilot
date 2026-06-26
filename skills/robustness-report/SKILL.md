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
