---
name: candidate-design-to-gtm
description: Generate validated molecule or peptide candidates, materialize them, and prepare GTM projection.
metadata:
  title: Candidate design to GTM
  status: stable
  version: 2.0.0
  depends_on: []
  profiles:
    - standard
  permissions:
    - model:execute
    - artifact:read
    - artifact:write
  input_artifacts:
    - name: design_request
      kind: request
      required: true
    - name: seed_candidates
      kind: candidate-set
      required: false
  output_artifacts:
    - name: candidate_set_artifact
      kind: candidate-set
      required: true
    - name: candidate_dataset_csv
      kind: dataset
      required: true
    - name: projected_dataset_path
      kind: dataset
      required: false
  tags:
    - design
    - candidates
    - gtm
  keywords:
    - candidate
    - analog
    - analogue
    - design
  required_tools:
    - session_materialize_candidate_set_dataset
  optional_tools:
    - mol_design_molecules
    - mol_generate_analogs
    - mol_validate_design_candidates
    - mol_rank_design_candidates
    - mol_register_design_candidates
    - peptide_design_peptides
    - peptide_generate_analogs
    - peptide_validate_design_candidates
    - peptide_rank_design_candidates
    - gtm_project_data
  example_prompts:
    - Generate analogs for the current seed molecule and project the validated candidate set onto the active GTM map.
---

# Candidate Design To GTM

Use this skill when the user wants generated small-molecule or peptide candidates prepared for GTM projection.

## Procedure

1. Determine whether the task is small-molecule or peptide design.
2. Resolve any seed compound, peptide sequence, or prior candidate-set reference from session memory before generation.
3. For molecules, use `mol_design_molecules` or `mol_generate_analogs`, then `mol_validate_design_candidates`, `mol_rank_design_candidates`, and `mol_register_design_candidates`.
4. For peptides, use `peptide_design_peptides` or `peptide_generate_analogs`, then `peptide_validate_design_candidates`, `peptide_rank_design_candidates`, and candidate registration/materialization through the session tools available for peptide candidate sets.
5. Choose the engine: `engine="autoencoder"` (molecules) or `engine="wae"` (peptides) run in-process. `engine="llm"` is delegated to the LLM — under the default MCP `llm_policy="external"` the design tool returns a `needs_external_llm` task you (the outer agent) complete via `llm_get_task` / `llm_submit_task_result`; the Agno team uses its own model inline. Only `llm_policy="disabled"` rejects it.
6. Materialize the registered candidate set with `session_materialize_candidate_set_dataset`.
7. Use `gtm_project_data` only when a suitable GTM model is available or the user requested projection.

## Expected Outputs

- Registered candidate set.
- Candidate dataset CSV.
- Optional GTM projection artifact.
- Validation and ranking summary.
