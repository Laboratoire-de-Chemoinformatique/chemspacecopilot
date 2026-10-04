---
name: chembl-target-retrieval
description: Retrieve, validate, standardize, and persist ChEMBL bioactivity data for a specific biological target.
metadata:
  title: ChEMBL target retrieval
  status: stable
  version: 2.0.0
  depends_on: []
  profiles:
    - chembl-retrieval
  permissions:
    - network:read
    - artifact:read
    - artifact:write
  input_artifacts:
    - name: retrieval_request
      kind: request
      required: true
  output_artifacts:
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
    - chembl_create_external_judge_task
    - chembl_submit_external_judge_result
    - llm_get_task
    - llm_submit_task_result
    - session_summarize_session_memory
  recommended_prompt: chembl_agent
---

# ChEMBL Target Retrieval

Use this workflow when the user needs ChEMBL bioactivity data for a specific target, organism, assay type, and mechanism preference.

1. Run `chembl_prepare_retrieval` with the user request and available session context.
2. Ask for any returned clarifications before calling mutating retrieval tools.
3. Enforce target specificity, abbreviation confirmation, organism, assay type, and mechanism preference through preflight/user answers. Do not infer missing values.
4. Convert clarified natural language to ChEMBL keyword form with `chembl_convert_to_chembl_query`.
5. Before every retrieval, show the exact comma-separated keywords, including expanded synonyms, and organism, assay-type, and mechanism filters in user-facing chat. State when a filter is unrestricted and briefly explain how the terms match the requested target. Tool logs or internal specialist messages are insufficient; coordinators must relay these details to the user. Show revised queries before retries; do not silently broaden the target or remove user-selected filters.
6. If a keyword, synonym, target mapping, or filter has uncertain relevance, show the questionable terms, explain the uncertainty, and ask the user to choose or correct them. Combine related questions and wait for the user's answer before retrieval. A complete preflight is not proof of relevance. When relevance is clear, proceed after displaying the query without routine approval.
7. Fetch only after `can_proceed=true` and relevance questions have been resolved, using the displayed query and filters in `chembl_fetch_compounds`.
8. If ambiguous rows need judge-style filtering, use the `chembl_retrieval_judge` and `chembl_metadata_judge` prompts with the external MCP client's reasoning. If target metadata leaves query relevance uncertain, ask the user before retrying or using the dataset downstream.
9. Summarize the clean dataset with `chembl_describe_dataset` and return raw, clean, descriptor, filtered-row, and standardization artifact paths. Include the queries and filters actually used in the final user-facing response, including when no data was found.
