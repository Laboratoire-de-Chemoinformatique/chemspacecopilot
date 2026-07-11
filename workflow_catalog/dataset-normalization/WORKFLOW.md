---
name: dataset-normalization
description: Load uploaded or session tabular data, normalize it for analysis, and prepare it for GTM or reporting.
metadata:
  title: Dataset normalization
  status: stable
  tags:
    - dataset
    - pandas
    - normalization
  keywords:
    - normalize
    - normalise
    - standardize dataset
    - uploaded
    - normalization
  required_tools:
    - pandas_normalize_for_analysis
  optional_tools:
    - pandas_load_dataframe_from_session
    - pandas_create_dataframe
    - pandas_run_operation
    - session_list_loadable_session_data
    - session_summarize_session_memory
  expected_artifacts:
    - normalized_dataset_path
    - analysis_ready_dataframe
  recommended_prompt: chemoinformatician_agent
---

# Dataset Normalization

Use this workflow when uploaded or session-resident tabular data must be prepared for analysis.

1. Inspect session data with `session_list_loadable_session_data` or summarize memory.
2. Load an existing session artifact with `pandas_load_dataframe_from_session`, or create a DataFrame with `pandas_create_dataframe`.
3. Use `pandas_run_operation` for lightweight cleanup when needed.
4. Normalize columns and activity fields with `pandas_normalize_for_analysis`.
5. Return the normalized path or DataFrame name for downstream GTM/report workflows.
