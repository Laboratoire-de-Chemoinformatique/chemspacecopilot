---
name: gtm-density-landscape
description: Build or reuse a GTM map, create density landscapes, inspect compound distributions, and prepare report-ready density artifacts.
metadata:
  title: GTM density landscape
  status: stable
  tags:
    - gtm
    - chemography
    - density-landscape
    - compound-distribution
  keywords:
    - gtm
    - density
    - density map
    - density landscape
    - compound distribution
    - densest nodes
  required_tools:
    - chemspace_plan_analysis
    - gtm_save_density_plot
    - gtm_get_density_summary
  optional_tools:
    - gtm_optimization
    - gtm_save_model_and_data
    - gtm_load_density_matrix
    - gtm_load_model_only
    - gtm_load_and_prep_data
    - gtm_sample_dense_nodes
    - gtm_project_data
    - report_save_rich
  artifact_outputs:
    - gtm_model_path
    - projected_dataset_path
    - density_matrix_path
    - density_plot_path
  example_prompts:
    - Show the GTM density landscape for the current clean dataset and summarize the densest nodes.
---

# GTM Density Landscape

Use this skill for GTM density maps, compound distribution analysis, dense-node inspection, and report-ready density artifacts.

## Procedure

1. Call `chemspace_plan_analysis` with the user's request and available session summary.
2. If preflight returns `needs_clarification=true`, ask the returned clarification questions before calling mutating GTM tools.
3. Resolve the active clean dataset from session memory or from an explicit user path. Prefer `clean_dataset_path`; `dataset_path` is only a legacy clean-data alias.
4. Respect map mode when available. `default_map` means project onto the pretrained default map unless the user explicitly asks to build/train a new map. `new_map` or missing means use the session-local GTM behavior.
5. Reuse an existing GTM map from session state when available and appropriate. Load it with `gtm_load_model_only` and prepare the active dataset with `gtm_load_and_prep_data`.
6. Build a new GTM only when no suitable map exists or the user asks for one. Default to a low optimization strategy unless the user explicitly requests medium/high/thorough search. If building, call `gtm_optimization`, then persist the model and projected data with `gtm_save_model_and_data`.
7. If the user supplies new data or generated candidates for an existing map, materialize the candidate set if needed and use `gtm_project_data` before density analysis.
8. For density visualizations in MCP mode, call `gtm_save_density_plot`; `gtm_load_density_matrix` returns density tables but does not itself write plots.
9. Summarize dense regions with `gtm_get_density_summary`. Sample compounds from dense nodes with `gtm_sample_dense_nodes` only when sampling is part of the request.
10. Do not describe dense nodes as active or potent unless measured activity values from a loaded table or tool output support that claim.
11. Save or reference density plot artifacts when preparing a report. Put density figures directly after density analysis in report sections.

## Expected Outputs

- Fitted or loaded GTM model.
- Projected source dataset.
- Density matrix or density summary table.
- Static density plot artifact.
- Summary of dense regions and compound distributions suitable for downstream report generation.

## Details

- **Density table** columns are `x`, `y`, `nodes`, `filtered_density`. Report global stats (max / min / mean / median density) and identify the top-5 densest and top-5 sparsest nodes.
- **Neighborhood-preservation table** columns are `x`, `y`, `nodes`, `density`, `neighborhood score`; report preservation quality and flag well-preserved vs poorly-preserved regions.
- Describe spatial patterns in compass/quadrant terms (e.g. "dense band across the center", "sparse south-west corner").
- Close with a 3-bullet executive summary of the density structure.
- **Optimization strategy** (when building): `gtm_optimization(strategy=...)` levels are low (fast heuristic, ~9 combinations), medium (~108 combinations), high (Bayesian, ~50 trials). Always low by default; for datasets >5,000 molecules always low and tell the user medium/high are available.
