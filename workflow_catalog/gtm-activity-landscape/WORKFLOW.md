# GTM Activity Landscape

Use this workflow for density/activity maps, SAR inspection, active-region sampling, and report-ready GTM artifacts.

1. Run `chemspace_plan_analysis` with the request and session summary.
2. Resolve the active clean dataset from session memory or an explicit path.
3. Respect map mode: project onto the default map unless the user explicitly asks to build/train a new map; otherwise use session-local GTM behavior.
4. Reuse a suitable loaded GTM with `gtm_load_model_only` / `gtm_load_and_prep_data`, or build a new map with `gtm_optimization` followed by `gtm_save_model_and_data`.
5. For new datasets on an existing model, use `gtm_project_data`.
6. Create activity landscapes with `gtm_create_activity_landscapes`; write plot artifacts required by the user/report.
7. Inspect `gtm_get_activity_landscape_summary` and sample relevant nodes as needed.
8. Save or reference report figures with `report_save_rich` when a report artifact is requested.
