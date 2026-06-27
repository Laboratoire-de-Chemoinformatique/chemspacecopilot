# ChEMBL To GTM Report

Use this workflow for the common end-to-end path from target retrieval to report-ready chemical-space analysis, including density and/or activity landscapes.

1. Run ChEMBL retrieval preflight and collect any required clarifications.
2. Convert the target query and retrieve ChEMBL data after preflight succeeds.
3. Treat the returned `clean_dataset_path` as the downstream dataset.
4. Run chemical-space preflight, then reuse or build the GTM map.
5. Create and summarize density artifacts with `gtm_save_density_plot` / `gtm_get_density_summary` when distribution analysis is requested.
6. Create and summarize activity landscapes with `gtm_create_activity_landscapes` / `gtm_get_activity_landscape_summary` when activity/SAR analysis is requested.
7. Save a rich report with `report_save_rich` that references raw data, clean data, descriptor Parquet, GTM outputs, density/activity plots, and standardization artifacts.
