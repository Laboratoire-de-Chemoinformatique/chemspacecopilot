# Peptide Design

Use this skill for amino-acid sequence generation, antimicrobial peptide workflows, peptide analogs, and peptide latent-space analysis.

## Procedure

1. Confirm the user is asking for peptides, amino-acid sequences, AMPs, or DBAASP-style antimicrobial analysis.
2. Inspect available engines with `peptide_list_design_engines`.
3. For WAE-backed generation, check model readiness with `peptide_validate_model_loaded` and use `peptide_get_model_info` when model details matter.
4. In default MCP, use `engine="wae"` for generation. `engine="llm"` is unavailable because `MCPAgentContext.model` is `None` unless trusted `agno_team_run` delegation is explicitly enabled.
5. Generate candidates with `peptide_design_peptides`, `peptide_generate_analogs`, or `peptide_design_interpolation` as appropriate.
6. Validate candidates with `peptide_validate_design_candidates` before presenting sequences as final.
7. Rank candidates with `peptide_rank_design_candidates`, using seed similarity when a seed sequence is available.
8. Store full generated lists as artifacts, use `peptide_load_design_candidates` for artifact reloads, and expose compact previews in chat.
9. For latent-space workflows, use encoding/decoding or GTM latent-space tools only after peptide embeddings or latent vectors are available.
10. Materialize candidate datasets before downstream projection, reporting, or batch analysis.

## Expected Outputs

- Registered peptide candidate set.
- Candidate artifact and optional CSV materialization.
- Validation/ranking summary.
- Optional peptide activity landscape artifacts.
