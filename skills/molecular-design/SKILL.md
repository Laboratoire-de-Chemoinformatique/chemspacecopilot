---
name: molecular-design
description: Generate, validate, rank, and persist small-molecule candidates, optionally using a GTM-guided context.
metadata:
  title: Molecular design
  status: agno_available
  version: 2.0.0
  depends_on: []
  profiles:
    - molecular-design
  permissions:
    - model:execute
    - artifact:read
    - artifact:write
  input_artifacts:
    - name: molecular_design_request
      kind: request
      required: true
    - name: seed_molecule
      kind: molecule
      required: false
  output_artifacts:
    - name: candidate_set_artifact
      kind: candidate-set
      required: true
    - name: candidate_dataset_csv
      kind: dataset
      required: true
    - name: validation_summary
      kind: validation-summary
      required: true
  tags:
    - molecular-design
    - small-molecule
    - analog-generation
    - validation
  keywords:
    - design
    - generate
    - analog
    - analogue
    - candidate
    - smiles
    - molecule
    - small molecule
  required_tools:
    - mol_list_design_engines
    - mol_validate_design_candidates
    - mol_rank_design_candidates
    - mol_register_design_candidates
    - session_materialize_candidate_set_dataset
  optional_tools:
    - mol_design_molecules
    - mol_generate_analogs
    - mol_interpolate_molecules
    - chem_find_most_similar
    - gtm_project_data
  example_prompts:
    - Generate validated analogs for this seed molecule and register the candidate set for downstream GTM projection.
---

# Molecular Design

Use this skill when the user wants small-molecule analogs, scaffold variants, or candidate generation from a seed molecule or current dataset.

## Procedure

1. Confirm the task is small-molecule design, not peptide design.
2. Resolve the seed compound from the user prompt or session memory. For "top candidates", "generated compounds", "latest designs", or similar follow-ups, resolve the active candidate set before choosing a seed.
3. Inspect available engines with `mol_list_design_engines`.
4. Choose the engine. `engine="autoencoder"` runs in-process. `engine="llm"` is **delegated to the LLM**, not blocked: under the default MCP `llm_policy="external"`, `mol_design_molecules` / `mol_generate_analogs` return a `needs_external_llm` task that you (the outer agent) complete via `llm_get_task` / `llm_submit_task_result`; `llm_policy="agno-model"` uses a configured server-side model; in the Agno team the engine uses the team's own model inline. Only `llm_policy="disabled"` rejects it — then fall back to `engine="autoencoder"`.
5. Use `mol_generate_analogs` for seed analogs, `mol_design_molecules` for objective-driven generation or sampling, and `mol_interpolate_molecules` for endpoint interpolation.
6. For LLM-backed Agno flows, validate and rank all proposed SMILES; if final candidates were produced outside `mol_design_molecules`, call `mol_register_design_candidates` before downstream GTM, SynPlanner, or reporting.
7. Validate generated or user-provided structures with `mol_validate_design_candidates` before presenting structures as final. Deduplicate and standardize before ranking.
8. Rank validated candidates with `mol_rank_design_candidates`, using seed similarity when a seed SMILES is available.
9. Persist final candidates with `mol_register_design_candidates` and keep large candidate lists in artifacts rather than chat text.
10. Use `session_materialize_candidate_set_dataset` before GTM projection or batch retrosynthesis. Use `gtm_project_data` only after a suitable GTM map exists.

## Expected Outputs

- Registered candidate set.
- Candidate artifact with full generated/validated structures.
- Optional CSV materialization for projection, reporting, or synthesis planning.
- Validation and ranking summary.

## Details

- **Analog similarity control (`noise_scale`)**: 0.05–0.15 = close analogs (high similarity), 0.2–0.4 = moderate diversity, 0.5+ = high diversity/novelty. Default to `noise_scale=0.1` and `n_neighbors=10` for analog generation unless the user specifies otherwise; sort reported analogs by Tanimoto similarity to the seed (highest first).
- **GTM-guided design**: when designing from map regions, sample seeds with the `gtm_sample_*` tools (e.g. `gtm_sample_dense_nodes` for well-explored regions, `gtm_sample_activity_landscape_nodes` for high-scoring activity nodes, `gtm_sample_top_activity_molecules` for top measured compounds, `gtm_sample_by_coordinates` for specific map regions) using `return_format="smiles"`, then encode and explore their latent neighborhood to generate. Report the source GTM region / coordinates / node IDs for each generated molecule.
- **Large result handling**: generation defaults to `n_samples=5000` with `filter_valid_unique=True` and `return_format="summary"`; the full list is saved as a candidate-set artifact and referenced by `registered_candidate_set_id` / `artifact_path` / `session_key`. Reference sets by those IDs downstream — do not re-emit full SMILES lists inline.
- **Evidence discipline**: treat generated/LLM-proposed structures as proposed candidates only; never imply their activity, potency, safety, or synthesizability has been experimentally verified.
