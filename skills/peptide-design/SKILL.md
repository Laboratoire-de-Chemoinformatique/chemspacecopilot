# Peptide Design

Use this skill for amino-acid sequence generation, antimicrobial peptide workflows, peptide analogs, and peptide latent-space analysis.

## Procedure

1. Confirm the user is asking for peptides, amino-acid sequences, AMPs, or DBAASP-style antimicrobial analysis.
2. Inspect available engines with `peptide_list_design_engines`.
3. For WAE-backed generation, check model readiness with `peptide_validate_model_loaded` and use `peptide_get_model_info` when model details matter.
4. In default MCP, use `engine="wae"` for generation. `engine="llm"` is unavailable because `MCPAgentContext.model` is `None` unless trusted `agno_team_run` delegation is explicitly enabled.
5. Normalize peptide inputs to space-separated single-letter amino-acid codes. Maximum supported sequence length is 25 amino acids. Convert FASTA or joined strings when the conversion is unambiguous.
6. Generate candidates with `peptide_design_peptides`, `peptide_generate_analogs`, or `peptide_design_interpolation` as appropriate. Use WAE sampling/interpolation tools directly only for diagnostics or latent-space workflows.
7. Validate candidates with `peptide_validate_design_candidates` before presenting sequences as final.
8. Rank candidates with `peptide_rank_design_candidates`, using seed similarity when a seed sequence is available.
9. Store full generated lists as artifacts, use `peptide_load_design_candidates` for artifact reloads, and expose compact previews in chat.
10. For latent-space workflows, use peptide encoding/decoding or GTM latent-space tools only after peptide embeddings or latent vectors are available. Use `gtm_train_on_latent_space` for latent GTM training and `gtm_create_peptide_activity_landscapes` for DBAASP antimicrobial activity landscapes.
11. Activity landscapes are based on DBAASP antimicrobial peptide data. State the organism scope and do not describe them as universal peptide activity maps.
12. Materialize candidate datasets before downstream projection, reporting, or batch analysis.

## Expected Outputs

- Registered peptide candidate set.
- Candidate artifact and optional CSV materialization.
- Validation/ranking summary.
- Optional peptide activity landscape artifacts.
