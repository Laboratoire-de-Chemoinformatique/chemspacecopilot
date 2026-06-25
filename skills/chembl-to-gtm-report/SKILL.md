# ChEMBL To GTM Report

Use this skill when the user wants a complete ChEMBL retrieval, GTM activity landscape, and report artifact.

## Procedure

1. Run `chembl_prepare_retrieval` and resolve any clarification questions before retrieval.
2. Convert the clarified request with `chembl_convert_to_chembl_query`, then call `chembl_fetch_compounds`.
3. Describe the clean dataset with `chembl_describe_dataset` and use `clean_dataset_path` downstream.
4. Run `chemspace_plan_analysis` before GTM work.
5. Build a GTM with `gtm_optimization` and persist it with `gtm_save_model_and_data`, or reuse a suitable map with `gtm_load_model_only` / `gtm_load_and_prep_data`.
6. Create landscapes with `gtm_create_activity_landscapes` and summarize them with `gtm_get_activity_landscape_summary`.
7. Save a report with `report_save_rich`, referencing data, GTM, landscape, plot, and standardization artifacts.

## Expected Outputs

- Clean ChEMBL dataset path.
- GTM model and projected dataset artifacts.
- Activity landscape CSV or plot artifacts.
- Rich report path.
