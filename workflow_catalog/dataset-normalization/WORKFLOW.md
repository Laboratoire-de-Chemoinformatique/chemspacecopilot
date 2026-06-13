# Dataset Normalization

Use this workflow when uploaded or session-resident tabular data must be prepared for analysis.

1. Inspect session data with `session_list_loadable_session_data` or summarize memory.
2. Load an existing session artifact with `pandas_load_dataframe_from_session`, or create a DataFrame with `pandas_create_dataframe`.
3. Use `pandas_run_operation` for lightweight cleanup when needed.
4. Normalize columns and activity fields with `pandas_normalize_for_analysis`.
5. Return the normalized path or DataFrame name for downstream GTM/report workflows.
