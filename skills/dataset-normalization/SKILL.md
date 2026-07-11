# Dataset Normalization

Use this skill when uploaded or session-resident tabular data must be prepared for analysis.

## Procedure

1. Inspect available data with `session_list_loadable_session_data` or `session_summarize_session_memory`.
2. Load existing session data with `pandas_load_dataframe_from_session` or create a DataFrame with `pandas_create_dataframe`.
3. Use `pandas_run_operation` for lightweight cleanup when needed.
4. Normalize the dataset with `pandas_normalize_for_analysis`.
5. Return the normalized path or DataFrame name for downstream GTM or report workflows.

## Expected Outputs

- Analysis-ready normalized dataset.
- Identified SMILES, activity, and optional cluster columns.
- Downstream path or registry name.

## Details

- **Input resolution order**: session GTM `source_mols` (use its `node_index` as `cluster_id`) → `clean_dataset_path` → `dataset_path` (legacy clean alias) → ask the user for a path.
- **Activity-column detection**: raw potency with units (IC50, EC50, Ki, Kd, MIC), p-scale potency (pIC50, pKi, pChEMBL), or labels (activity, label, class). Normalization records `activity_mapping` (source column) and `final_activity_mapping` (final merged activity); preserve both for downstream SAR.
- **Required column**: a SMILES column (`smiles` / `SMILES` / `canonical_smiles`); `cluster_id` and an activity column are optional but enable clustering and SAR downstream.
