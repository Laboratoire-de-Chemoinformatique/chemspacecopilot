---
name: peptide-design
description: Generate, validate, rank, and analyze peptide candidates, including antimicrobial peptide workflows and latent-space GTM analysis.
metadata:
  title: Peptide design
  status: agno_available
  version: 2.0.0
  depends_on: []
  profiles:
    - peptide-design
  permissions:
    - model:execute
    - artifact:read
    - artifact:write
  input_artifacts:
    - name: peptide_design_request
      kind: request
      required: true
  output_artifacts:
    - name: peptide_candidate_artifact
      kind: candidate-set
      required: true
    - name: peptide_dataset_csv
      kind: dataset
      required: true
    - name: validation_summary
      kind: validation-summary
      required: true
    - name: peptide_landscape_csv
      kind: analysis-table
      required: false
  tags:
    - peptide
    - antimicrobial-peptide
    - latent-space
    - design
  keywords:
    - peptide
    - amp
    - antimicrobial
    - amino acid
    - dbaasp
  required_tools:
    - peptide_list_design_engines
    - peptide_validate_design_candidates
    - peptide_rank_design_candidates
    - session_materialize_candidate_set_dataset
  optional_tools:
    - peptide_validate_model_loaded
    - peptide_get_model_info
    - peptide_design_peptides
    - peptide_generate_analogs
    - peptide_design_interpolation
    - peptide_load_design_candidates
    - gtm_train_on_latent_space
    - gtm_create_peptide_activity_landscapes
  example_prompts:
    - Design antimicrobial peptide candidates and prepare the validated set for latent-space GTM analysis.
---

# Peptide Design

Use this skill for amino-acid sequence generation, antimicrobial peptide workflows, peptide analogs, and peptide latent-space analysis.

## Procedure

1. Confirm the user is asking for peptides, amino-acid sequences, AMPs, or DBAASP-style antimicrobial analysis.
2. Inspect available engines with `peptide_list_design_engines`.
3. For WAE-backed generation, check model readiness with `peptide_validate_model_loaded` and use `peptide_get_model_info` when model details matter.
4. Choose the engine. `engine="wae"` runs in-process. `engine="llm"` is **delegated to the LLM**, not blocked: under the default MCP `llm_policy="external"`, `peptide_design_peptides` / `peptide_generate_analogs` return a `needs_external_llm` task that you (the outer agent) complete via `llm_get_task` / `llm_submit_task_result`; `llm_policy="agno-model"` uses a configured server-side model; in the Agno team the engine uses the team's own model inline. Only `llm_policy="disabled"` rejects it — then fall back to `engine="wae"`.
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

## Details

- **Sequence format**: space-separated single-letter codes, max 25 amino acids. Supported residues: A, C, D, E, F, G, H, I, K, L, M, N, P, Q, R, S, T, U, V, W, Y, Z. Convert FASTA / joined strings only when unambiguous.
- **Analog/reconstruction decisions**: for analogs, `noise_scale` 0.05–0.15 = close / 0.2–0.4 = moderate / 0.5+ = diverse; default `0.1`. For reconstruction tests, prefer greedy decoding at low temperature for an accurate round-trip.
- **Latent-space GTM build**: encode sequences with the peptide encoder, save the latent vectors to CSV, then `gtm_train_on_latent_space` (Optuna optimization); report the entropy score and GTM node count.
- **GTM-guided peptide sampling**: sample with `return_format="sequences"` (`gtm_sample_dense_nodes`, `gtm_sample_activity_landscape_nodes`, `gtm_sample_by_coordinates`); load a different latent dataset onto the map with `load_latent_data_on_gtm`; chain sample → encode → explore neighborhood → decode for novel peptides.
- **DBAASP activity landscapes**: `gtm_create_peptide_activity_landscapes` builds classification (active vs inactive) landscapes from DBAASP antimicrobial data; only organisms with ≥200 data points are eligible (e.g. E. coli ~5,059 samples, S. aureus, P. aeruginosa). State the organism scope; these are AMP landscapes, not universal peptide activity maps. DBAASP data is sourced from the HuggingFace `wae_peptides` repo.
