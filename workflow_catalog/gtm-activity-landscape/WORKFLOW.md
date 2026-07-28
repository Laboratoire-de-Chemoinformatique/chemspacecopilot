---
name: gtm-activity-landscape
description: Build or reuse a GTM map, create activity/SAR landscapes, sample active regions, and prepare report-ready activity outputs.
metadata:
  title: GTM activity landscape
  status: stable
  version: 2.0.0
  depends_on: []
  profiles:
    - gtm-analysis
  permissions:
    - network:read
    - compute:execute
    - artifact:read
    - artifact:write
  input_artifacts:
    - name: clean_dataset_path
      kind: dataset
      required: true
    - name: gtm_model_path
      kind: model
      required: false
  output_artifacts:
    - name: gtm_model_path
      kind: model
      required: true
    - name: projected_dataset_path
      kind: dataset
      required: true
    - name: activity_landscape_csv
      kind: analysis-table
      required: true
    - name: landscape_plot_path
      kind: visualization
      required: false
  tags:
    - gtm
    - activity-landscape
    - sar
  keywords:
    - gtm
    - activity landscape
    - activity map
    - sar landscape
    - active regions
    - project
    - projection
  preflight_tools:
    - chemspace_plan_analysis
  required_tools:
    - gtm_create_activity_landscapes
    - gtm_get_activity_landscape_summary
  optional_tools:
    - gtm_optimization
    - gtm_save_model_and_data
    - gtm_load_model_only
    - gtm_load_and_prep_data
    - gtm_sample_activity_landscape_nodes
    - gtm_project_data
    - report_save_rich
  recommended_prompt: gtm_agent
---

# GTM Activity Landscape

Use this workflow for activity/SAR maps, active-region sampling, and report-ready GTM activity artifacts. For density maps and compound-distribution analysis, use the `gtm-density-landscape` workflow.

1. Run `chemspace_plan_analysis` with the request and session summary.
2. Resolve the active clean dataset from session memory or an explicit path.
3. Respect map mode: project onto the default map unless the user explicitly asks to build/train a new map; otherwise use session-local GTM behavior.
4. Reuse a suitable loaded GTM with `gtm_load_model_only` / `gtm_load_and_prep_data`, or build a new map with `gtm_optimization` followed by `gtm_save_model_and_data`.
5. For new datasets on an existing model, use `gtm_project_data`.
6. Create activity landscapes with `gtm_create_activity_landscapes`; write plot artifacts required by the user/report.
7. Inspect `gtm_get_activity_landscape_summary` and sample relevant nodes as needed.
8. Save or reference report figures with `report_save_rich` when a report artifact is requested.
