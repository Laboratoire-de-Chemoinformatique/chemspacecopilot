# GTM Activity Landscape

Use this skill for GTM-based activity/SAR maps, active-region sampling, and report-ready activity landscape outputs. For density maps and compound-distribution analysis, use the `gtm-density-landscape` skill.

## Procedure

1. Call `chemspace_plan_analysis` with the user's request and available session summary.
2. If preflight returns `needs_clarification=true`, ask the returned clarification questions before calling mutating GTM tools.
3. Resolve the active clean dataset from session memory or from an explicit user path. Prefer `clean_dataset_path`; `dataset_path` is only a legacy clean-data alias.
4. Respect map mode when available. `default_map` means project onto the pretrained default map unless the user explicitly asks to build/train a new map. `new_map` or missing means use the session-local GTM behavior.
5. Reuse an existing GTM map from session state when available and appropriate. Load it with `gtm_load_model_only` and prepare the active dataset with `gtm_load_and_prep_data`.
6. Build a new GTM only when no suitable map exists or the user asks for one. Default to a low optimization strategy unless the user explicitly requests medium/high/thorough search. If building, call `gtm_optimization`, then persist the model and projected data with `gtm_save_model_and_data`.
7. If the user supplies new data or generated candidates for an existing map, materialize the candidate set if needed and use `gtm_project_data` before landscape analysis.
8. Create activity landscapes with `gtm_create_activity_landscapes`. When both report-ready renderers are required, produce the discrete Altair landscape and the smooth Plotly landscape; if re-rendering saved landscape CSVs, use `gtm_save_landscape_plot` for the needed renderer outputs.
9. Inspect the result with `gtm_get_activity_landscape_summary` and sample relevant nodes with `gtm_sample_activity_landscape_nodes` or `gtm_sample_top_activity_molecules` only when sampling is part of the request.
10. Never call compounds or nodes "top active", "most potent", or assign pIC50/pChEMBL ranks unless the claim is backed by loaded activity values from the landscape, DataFrame, or tool output.
11. Save or reference static report figures when preparing a final report. Prefer artifact paths over inline tables for large results.

## Expected Outputs

- Fitted or loaded GTM model.
- Projected source dataset.
- Activity landscape CSV.
- Static or interactive plot artifacts.
- Summary of regions, nodes, and activity patterns suitable for downstream report generation.
