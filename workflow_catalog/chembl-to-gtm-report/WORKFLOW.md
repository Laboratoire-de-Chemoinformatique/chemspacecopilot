# ChEMBL To GTM Report

Use this workflow for the common end-to-end path from target retrieval to report-ready chemical-space analysis.

1. Run ChEMBL retrieval preflight and collect any required clarifications.
2. Convert the target query and retrieve ChEMBL data after preflight succeeds.
3. Treat the returned `clean_dataset_path` as the downstream dataset.
4. Run chemical-space preflight, then reuse or build the GTM map.
5. Create and summarize activity landscapes.
6. Save a rich report with `report_save_rich` that references raw data, clean data, GTM outputs, plots, and standardization artifacts.
