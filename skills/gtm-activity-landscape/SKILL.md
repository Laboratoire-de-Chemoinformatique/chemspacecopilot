# GTM Activity Landscape

Use this skill for GTM-based chemical-space analysis, density/activity maps, SAR inspection, and report-ready landscape outputs.

## Procedure

1. Resolve the active clean dataset from session memory or from an explicit user path.
2. Reuse an existing GTM map from session state when available and appropriate. Build a new GTM only when no suitable map exists or the user asks for one.
3. If building a map, call `gtm_optimization`, then persist the model and projected data with `gtm_save_model_and_data`.
4. Create activity landscapes with `gtm_create_activity_landscapes`.
5. Inspect the result with `gtm_get_activity_landscape_summary` and sample relevant nodes if needed.
6. Save or reference static report figures when preparing a final report. Prefer artifact paths over inline tables for large results.

## Expected Outputs

- Fitted or loaded GTM model.
- Projected source dataset.
- Activity landscape CSV.
- Static or interactive plot artifacts.
- Summary of regions, nodes, and activity patterns suitable for downstream report generation.
