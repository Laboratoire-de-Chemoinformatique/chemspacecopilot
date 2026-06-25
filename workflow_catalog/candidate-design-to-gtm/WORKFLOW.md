# Candidate Design To GTM

Use this workflow to generate small-molecule or peptide candidates, validate and rank them, then prepare them for GTM projection.

1. Determine whether the request is small-molecule or peptide design.
2. Use `mol_*` tools for small molecules or `peptide_*` tools for peptides.
3. In default MCP, use `engine="autoencoder"` for molecules or `engine="wae"` for peptides; LLM engines require explicit trusted Agno delegation.
4. Validate and rank candidates before presenting final structures or sequences.
5. Register the candidate set and materialize it with `session_materialize_candidate_set_dataset`.
6. Project the materialized dataset with `gtm_project_data` when a GTM map is available or requested.
