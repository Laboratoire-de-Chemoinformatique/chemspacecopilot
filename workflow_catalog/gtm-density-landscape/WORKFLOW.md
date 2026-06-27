# GTM Density Landscape

Use this workflow for density maps, compound-distribution inspection, dense-node sampling, and report-ready GTM density artifacts.

1. Run `chemspace_plan_analysis` with the request and session summary.
2. Resolve the active clean dataset from session memory or an explicit path.
3. Respect map mode: project onto the default map unless the user explicitly asks to build/train a new map; otherwise use session-local GTM behavior.
4. Reuse a suitable loaded GTM with `gtm_load_model_only` / `gtm_load_and_prep_data`, or build a new map with `gtm_optimization` followed by `gtm_save_model_and_data`.
5. For new datasets on an existing model, use `gtm_project_data`.
6. Create density visualizations with `gtm_save_density_plot`; use `gtm_load_density_matrix` only when a density table is needed.
7. Inspect `gtm_get_density_summary` and sample dense nodes with `gtm_sample_dense_nodes` only when requested.
8. Save or reference density report figures with `report_save_rich` when a report artifact is requested.
