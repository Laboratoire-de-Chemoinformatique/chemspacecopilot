---
name: chembl-to-gtm-report
description: Retrieve ChEMBL data, create GTM density/activity landscapes, and write a report artifact.
metadata:
  title: ChEMBL to GTM report
  status: stable
  version: 2.0.0
  depends_on:
    - chembl-target-retrieval
    - gtm-density-landscape
    - gtm-activity-landscape
  profiles:
    - standard
  permissions:
    - network:read
    - compute:execute
    - artifact:read
    - artifact:write
  input_artifacts:
    - name: retrieval_request
      kind: request
      required: true
  output_artifacts:
    - name: retrieval_plan
      kind: task-plan
      required: true
    - name: raw_dataset_path
      kind: dataset
      required: true
    - name: clean_dataset_path
      kind: dataset
      required: true
    - name: descriptor_parquet_path
      kind: descriptor-table
      required: false
    - name: filtered_rows_path
      kind: dataset
      required: false
    - name: standardization_report_path
      kind: report
      required: true
    - name: gtm_plan
      kind: task-plan
      required: true
    - name: gtm_model_path
      kind: model
      required: true
    - name: projected_dataset_path
      kind: dataset
      required: true
    - name: density_plot_path
      kind: visualization
      required: false
    - name: density_summary
      kind: analysis-summary
      required: false
    - name: activity_landscape_csv
      kind: analysis-table
      required: false
    - name: activity_plot_path
      kind: visualization
      required: false
    - name: activity_summary
      kind: analysis-summary
      required: false
    - name: html_report_path
      kind: report
      required: true
    - name: pdf_report_path
      kind: report
      required: false
    - name: markdown_report_path
      kind: report
      required: false
  tasks:
    - id: chembl-preflight
      role: chembl_downloader
      profile: chembl-retrieval
      depends_on: []
      required_tools:
        - chembl_prepare_retrieval
      input_artifacts:
        - retrieval_request
      output_artifacts:
        - retrieval_plan
      acceptance_criteria:
        - Preflight either permits retrieval or records all required clarification questions.
    - id: chembl-retrieval
      role: chembl_downloader
      profile: chembl-retrieval
      depends_on:
        - chembl-preflight
      required_tools:
        - chembl_convert_to_chembl_query
        - chembl_fetch_compounds
        - chembl_describe_dataset
      input_artifacts:
        - retrieval_request
        - retrieval_plan
      output_artifacts:
        - raw_dataset_path
        - clean_dataset_path
        - descriptor_parquet_path
        - filtered_rows_path
        - standardization_report_path
      acceptance_criteria:
        - Clean data matches the clarified target and retains retrieval provenance.
    - id: gtm-preflight
      role: gtm_agent
      profile: gtm-analysis
      depends_on:
        - chembl-retrieval
      required_tools:
        - chemspace_plan_analysis
      input_artifacts:
        - clean_dataset_path
      output_artifacts:
        - gtm_plan
      acceptance_criteria:
        - Map mode and requested density or activity analyses are explicit.
    - id: gtm-model
      role: gtm_agent
      profile: gtm-analysis
      depends_on:
        - gtm-preflight
      required_tools:
        - gtm_optimization
        - gtm_save_model_and_data
      input_artifacts:
        - clean_dataset_path
        - gtm_plan
      output_artifacts:
        - gtm_model_path
        - projected_dataset_path
      acceptance_criteria:
        - A reusable GTM model and its projected source dataset are registered.
    - id: gtm-landscapes
      role: gtm_agent
      profile: gtm-analysis
      depends_on:
        - gtm-model
      required_tools:
        - gtm_save_density_plot
        - gtm_get_density_summary
        - gtm_create_activity_landscapes
        - gtm_get_activity_landscape_summary
      input_artifacts:
        - clean_dataset_path
        - gtm_model_path
        - projected_dataset_path
      output_artifacts:
        - density_plot_path
        - density_summary
        - activity_landscape_csv
        - activity_plot_path
        - activity_summary
      acceptance_criteria:
        - Requested density and activity outputs are present and supported by measured data.
    - id: report
      role: report_generator
      profile: reporting
      depends_on:
        - chembl-retrieval
        - gtm-landscapes
      required_tools:
        - report_save_rich
      input_artifacts:
        - raw_dataset_path
        - clean_dataset_path
        - filtered_rows_path
        - standardization_report_path
        - gtm_model_path
        - projected_dataset_path
        - density_plot_path
        - density_summary
        - activity_landscape_csv
        - activity_plot_path
        - activity_summary
      output_artifacts:
        - html_report_path
        - pdf_report_path
        - markdown_report_path
      acceptance_criteria:
        - The report cites every scientific source and registered artifact used in its conclusions.
  tags:
    - chembl
    - gtm
    - report
  keywords:
    - chembl
    - gtm
    - report
    - landscape
  preflight_tools:
    - chembl_prepare_retrieval
    - chemspace_plan_analysis
  required_tools:
    - chembl_convert_to_chembl_query
    - chembl_fetch_compounds
    - chembl_describe_dataset
    - gtm_optimization
    - gtm_save_model_and_data
    - gtm_save_density_plot
    - gtm_create_activity_landscapes
    - gtm_get_density_summary
    - gtm_get_activity_landscape_summary
    - report_save_rich
  optional_tools:
    - chembl_create_external_judge_task
    - chembl_submit_external_judge_result
    - llm_get_task
    - llm_submit_task_result
    - gtm_load_model_only
    - gtm_load_and_prep_data
    - gtm_sample_activity_landscape_nodes
    - gtm_sample_dense_nodes
    - report_save_markdown
  recommended_prompt: cs_copilot_workflow
---

# ChEMBL To GTM Report

Use this workflow for the common end-to-end path from target retrieval to report-ready chemical-space analysis, including density and/or activity landscapes.

1. Run ChEMBL retrieval preflight and collect any required clarifications.
2. Convert the target query and retrieve ChEMBL data after preflight succeeds.
3. Treat the returned `clean_dataset_path` as the downstream dataset.
4. Run chemical-space preflight, then reuse or build the GTM map.
5. Create and summarize density artifacts with `gtm_save_density_plot` / `gtm_get_density_summary` when distribution analysis is requested.
6. Create and summarize activity landscapes with `gtm_create_activity_landscapes` / `gtm_get_activity_landscape_summary` when activity/SAR analysis is requested.
7. Save a rich report with `report_save_rich` that references raw data, clean data, descriptor Parquet, GTM outputs, density/activity plots, and standardization artifacts.
