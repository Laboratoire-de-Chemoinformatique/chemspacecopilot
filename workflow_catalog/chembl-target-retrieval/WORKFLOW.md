---
name: chembl-target-retrieval
description: Retrieve, validate, standardize, and persist ChEMBL bioactivity data for a specific biological target.
metadata:
  title: ChEMBL target retrieval
  status: stable
  tags:
    - chembl
    - bioactivity
    - retrieval
  keywords:
    - chembl
    - bioactivity
    - assay
    - activity data
  preflight_tools:
    - chembl_prepare_retrieval
  required_tools:
    - chembl_convert_to_chembl_query
    - chembl_fetch_compounds
    - chembl_describe_dataset
  optional_tools:
    - session_summarize_session_memory
  expected_artifacts:
    - raw_dataset_path
    - clean_dataset_path
    - descriptor_parquet_path
    - standardization_report_path
  recommended_prompt: chembl_agent
---

# ChEMBL Target Retrieval

Use this workflow when the user needs ChEMBL bioactivity data for a specific target, organism, assay type, and mechanism preference.

1. Run `chembl_prepare_retrieval` with the user request and available session context.
2. Ask for any returned clarifications before calling mutating retrieval tools.
3. Enforce target specificity, abbreviation confirmation, organism, assay type, and mechanism preference through preflight/user answers. Do not infer missing values.
4. Convert clarified natural language to ChEMBL keyword form with `chembl_convert_to_chembl_query`.
5. Fetch only after `can_proceed=true` using `chembl_fetch_compounds`.
6. If ambiguous rows need judge-style filtering, use the `chembl_retrieval_judge` and `chembl_metadata_judge` prompts with the external MCP client's reasoning.
7. Summarize the clean dataset with `chembl_describe_dataset` and return raw, clean, descriptor, filtered-row, and standardization artifact paths.
