---
name: chembl-to-gtm-report
description: Retrieve ChEMBL data, create GTM density/activity landscapes, and write a report artifact.
metadata:
  title: ChEMBL to GTM report
  status: stable
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
    - gtm_load_model_only
    - gtm_load_and_prep_data
    - gtm_sample_activity_landscape_nodes
    - gtm_sample_dense_nodes
    - report_save_markdown
  expected_artifacts:
    - clean_dataset_path
    - gtm_model_path
    - density_plot_path
    - activity_landscape_csv
    - html_report_path
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
