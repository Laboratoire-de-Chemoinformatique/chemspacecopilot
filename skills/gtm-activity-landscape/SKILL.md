# GTM Activity Landscape

Use this skill for GTM-based chemical-space analysis, density/activity maps, SAR inspection, and report-ready landscape outputs.

## Procedure

1. Call `chemspace_plan_analysis` with the user's request and available session summary.
2. If preflight returns `needs_clarification=true`, ask the returned clarification questions before calling mutating GTM tools.
3. Resolve the active clean dataset from session memory or from an explicit user path.
4. Reuse an existing GTM map from session state when available and appropriate. Build a new GTM only when no suitable map exists or the user asks for one.
5. If building a map, call `gtm_optimization`, then persist the model and projected data with `gtm_save_model_and_data`.
6. Create activity landscapes with `gtm_create_activity_landscapes`.
7. Inspect the result with `gtm_get_activity_landscape_summary` and sample relevant nodes if needed.
8. Save or reference static report figures when preparing a final report. Prefer artifact paths over inline tables for large results.

## Expected Outputs

- Fitted or loaded GTM model.
- Projected source dataset.
- Activity landscape CSV.
- Static or interactive plot artifacts.
- Summary of regions, nodes, and activity patterns suitable for downstream report generation.
