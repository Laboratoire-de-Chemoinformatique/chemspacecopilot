# Robustness Report

Use this skill when the user wants a compact report from robustness test outputs.

## Procedure

1. Load raw results with `robustness_load_test_results` and optionally load summary CSV data.
2. Analyze score distribution with `robustness_analyze_score_distribution`.
3. Identify failing prompts with `robustness_identify_failing_prompts`.
4. Compare runs or analyze temporal trends when multiple runs are relevant.
5. Generate insights with `robustness_generate_insights`.
6. Persist the report with `robustness_export_analysis_report`.

## Expected Outputs

- Robustness analysis report.
- Failing prompt summary.
- Score distribution and trend summary when available.
