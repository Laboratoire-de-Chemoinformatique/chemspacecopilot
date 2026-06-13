# Candidate Design To GTM

Use this skill when the user wants generated small-molecule or peptide candidates prepared for GTM projection.

## Procedure

1. Determine whether the task is small-molecule or peptide design.
2. For molecules, use `mol_design_molecules` or `mol_generate_analogs`, then `mol_validate_design_candidates`, `mol_rank_design_candidates`, and `mol_register_design_candidates`.
3. For peptides, use `peptide_design_peptides` or `peptide_generate_analogs`, then `peptide_validate_design_candidates` and `peptide_rank_design_candidates`.
4. In default MCP, use `engine="autoencoder"` for molecules or `engine="wae"` for peptides. LLM engines require trusted Agno delegation.
5. Materialize the registered candidate set with `session_materialize_candidate_set_dataset`.
6. Use `gtm_project_data` only when a suitable GTM model is available or the user requested projection.

## Expected Outputs

- Registered candidate set.
- Candidate dataset CSV.
- Optional GTM projection artifact.
- Validation and ranking summary.
