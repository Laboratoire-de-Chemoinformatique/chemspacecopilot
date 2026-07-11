---
name: chembl-to-gtm-report
description: Retrieve ChEMBL data, build or reuse GTM density/activity landscapes, and save a report artifact.
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
  required_tools:
    - chembl_prepare_retrieval
    - chembl_convert_to_chembl_query
    - chembl_fetch_compounds
    - chembl_describe_dataset
    - chemspace_plan_analysis
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
  artifact_outputs:
    - clean_dataset_path
    - gtm_model_path
    - density_plot_path
    - activity_landscape_csv
    - html_report_path
  example_prompts:
    - Retrieve human CDK2 binding data, create a GTM activity landscape, and save a rich report.
---

# ChEMBL To GTM Report

Use this skill when the user wants a complete ChEMBL retrieval, GTM density/activity landscape analysis, and report artifact.

## Procedure

1. Run `chembl_prepare_retrieval` and resolve any clarification questions before retrieval.
2. Convert the clarified request with `chembl_convert_to_chembl_query`, then call `chembl_fetch_compounds`.
3. Describe the clean dataset with `chembl_describe_dataset` and use `clean_dataset_path` downstream.
4. Run `chemspace_plan_analysis` before GTM work.
5. Build a GTM with `gtm_optimization` and persist it with `gtm_save_model_and_data`, or reuse a suitable map with `gtm_load_model_only` / `gtm_load_and_prep_data`.
6. Create density artifacts with `gtm_save_density_plot` and summarize dense regions with `gtm_get_density_summary` when the request includes density or distribution analysis.
7. Create activity landscapes with `gtm_create_activity_landscapes` and summarize them with `gtm_get_activity_landscape_summary` when the request includes activity/SAR analysis.
8. Save a report with `report_save_rich`, referencing data, GTM, density/activity landscape, plot, and standardization artifacts.

## Expected Outputs

- Clean ChEMBL dataset path.
- GTM model and projected dataset artifacts.
- Density plot and/or activity landscape CSV artifacts.
- Rich report path.
