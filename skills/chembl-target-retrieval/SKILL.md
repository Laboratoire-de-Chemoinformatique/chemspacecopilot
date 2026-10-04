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
    - standardization
  keywords:
    - chembl
    - bioactivity
    - assay
    - activity data
  required_tools:
    - chembl_prepare_retrieval
    - chembl_convert_to_chembl_query
    - chembl_fetch_compounds
    - chembl_describe_dataset
  optional_tools:
    - chembl_create_external_judge_task
    - chembl_submit_external_judge_result
    - llm_get_task
    - llm_submit_task_result
    - session_list_session_objects
    - session_summarize_session_memory
  example_prompts:
    - Retrieve human CDK2 inhibitor binding data from ChEMBL and prepare it for downstream GTM analysis.
---

# ChEMBL Target Retrieval

Use this skill when the user needs ChEMBL bioactivity data for a target, organism, assay type, and optional mechanism filter.

## Procedure

1. Call `chembl_prepare_retrieval` with the user's request and available session summary before any mutating retrieval tool.
2. If preflight returns `needs_clarification=true`, ask all returned clarification questions in one message and stop. Do not call retrieval tools until the user explicitly answers target specificity, organism, assay type, and mechanism requirements.
3. Reject bare family names or family-plus-index targets such as "kinase 2", "receptor 5", "phosphatase", or "GPCR". Ask for a recognized gene symbol or full canonical protein name. Confirm abbreviations such as CDK2, EGFR, PDE4, BRAF, and JAK2 unless preflight has already confirmed the target.
4. Never default organism, assay type, or mechanism. A mechanism answer of "unspecified", "any", or "no preference" is valid and means no mechanism filter.
5. Use `chembl_convert_to_chembl_query` when the clarified target is natural language and needs canonical ChEMBL keyword form. Use the generated semantic keywords rather than adding punctuation variants manually; retrieval handles hyphen/space variants internally.
6. Before every retrieval, show the exact comma-separated search keywords in a user-facing chat message, including all expanded synonyms, plus the organism, assay-type, and mechanism filters. State explicitly when a filter is unrestricted. Briefly explain how the terms match the requested target. Tool calls, logs, and internal specialist messages alone do not count; the coordinator must relay the query to the user. Show revised queries before retries as well.
7. If any keyword, synonym, target mapping, or filter has uncertain relevance, identify the questionable terms, explain the uncertainty, and ask the user to choose or correct them (for example, provide the intended gene symbol or full protein name). Combine related questions in one message and wait for the user's answer before retrieval. A complete preflight is not proof that generated synonyms are relevant. When relevance is clear and requirements are satisfied, proceed after displaying the query without routine approval.
8. Call `chembl_fetch_compounds` only after preflight returns `can_proceed=true` and relevance questions have been resolved. Pass the displayed query and explicit organism, assay type, and mechanism values from the user or preflight. Omit the mechanism argument when the user selected no mechanism filter.
9. The ChEMBL LLM-as-judge is **delegated, not skipped**. Under the default MCP `llm_policy="external"`, `chembl_fetch_compounds` auto-creates pending judge task(s) for ambiguous rows; you (the outer agent) reason over them and submit decisions via `chembl_submit_external_judge_result` (or `llm_submit_task_result`), and may create one explicitly with `chembl_create_external_judge_task` using the `chembl_retrieval_judge` / `chembl_metadata_judge` prompts. Under `llm_policy="agno-model"`, and in the Agno team, the judge runs in-process automatically. If returned target metadata leaves query relevance uncertain, ask the user before retrying or using the dataset downstream.
10. Use `chembl_describe_dataset` on the clean dataset path returned by retrieval. Verify the dataset covers the intended target and requested assay categories.
11. Include the queries and filters actually used in the final user-facing response, including when no data was found. Report raw, clean, filtered rows if present, descriptor Parquet, and standardization report paths. Summarize invalid rows, duplicates, raw-to-final SMILES collapses, stereochemistry handling, and activity merge policy.
12. Treat `clean_dataset_path` as the downstream dataset. `dataset_path` is only a backward-compatible alias for the clean dataset.

## Expected Outputs

- Raw retrieval dataset retaining ChEMBL provenance.
- Clean standardized dataset for downstream analysis.
- Descriptor Parquet when generated by the retrieval pipeline.
- Standardization report describing invalid rows, duplicates, SMILES collapse, stereochemistry handling, and activity merge policy.

## Details

- **Target type**: classify the request as a *protein* target (e.g. CDK2, BRAF) or an *organism-level* target (e.g. HIV-1, E. coli). For organism-level queries keep the exact organism string and include it as one of the search keywords so assays are constrained to that species/strain.
- **Assay-type codes**: map the user's choice to binding→B, functional→F, ADMET→A. Never apply a default combination.
- **Mechanism filter**: applied as a case-insensitive substring match against each assay description; pass the user's mechanism verbatim, or omit it entirely for "unspecified"/"any"/"no preference".
- **Dataset-quality checks** after fetch: confirm SMILES were mapped; expected columns are present (`activity_id`, `molecule_chembl_id`, `canonical_smiles`, `standard_value`); the `assay_type` column contains the requested B/F/A categories; note how many duplicates were removed during merging.
- **Fetch-failure troubleshooting**: verify ChEMBL connectivity (works for both SQL and REST backends) and back off on rate limiting. Show any revised query before retrying. Do not silently broaden the target or remove user-selected filters; ask for help when an alternative's relevance is uncertain.
- **Artifact shape**: the clean CSV is one row per standardized achiral compound with merged IDs and final processed activity values; descriptors are written separately to `descriptor_parquet_path` (which also carries the final activity values), never embedded in the clean CSV.
- **DataFrame hygiene**: use in-place operations and avoid printing whole DataFrames to the console (context-window safety).
