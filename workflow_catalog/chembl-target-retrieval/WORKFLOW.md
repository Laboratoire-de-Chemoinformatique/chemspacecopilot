# ChEMBL Target Retrieval

Use this workflow when the user needs ChEMBL bioactivity data for a specific target, organism, assay type, and mechanism preference.

1. Run `chembl_prepare_retrieval` with the user request and available session context.
2. Ask for any returned clarifications before calling mutating retrieval tools.
3. Convert clarified natural language to ChEMBL keyword form with `chembl_convert_to_chembl_query`.
4. Fetch only after `can_proceed=true` using `chembl_fetch_compounds`.
5. If ambiguous rows need judge-style filtering, use the `chembl_retrieval_judge` and `chembl_metadata_judge` prompts with the external MCP client's reasoning.
6. Summarize the clean dataset with `chembl_describe_dataset` and return artifact paths.
