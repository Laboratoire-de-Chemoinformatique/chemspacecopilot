# Molecular Design

Use this skill when the user wants small-molecule analogs, scaffold variants, or candidate generation from a seed molecule or current dataset.

## Procedure

1. Confirm the task is small-molecule design, not peptide design.
2. Resolve the seed compound from the user prompt or session memory. For "top candidates", "generated compounds", "latest designs", or similar follow-ups, resolve the active candidate set before choosing a seed.
3. Inspect available engines with `mol_list_design_engines`.
4. In default MCP, use `engine="autoencoder"` for generation. `engine="llm"` is unavailable because `MCPAgentContext.model` is `None` unless trusted `agno_team_run` delegation is explicitly enabled.
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
