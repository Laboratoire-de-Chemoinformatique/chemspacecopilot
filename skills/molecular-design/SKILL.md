# Molecular Design

Use this skill when the user wants small-molecule analogs, scaffold variants, or candidate generation from a seed molecule or current dataset.

## Procedure

1. Confirm the task is small-molecule design, not peptide design.
2. Resolve the seed compound from the user prompt or session memory.
3. Prefer high-level molecular design workflows in the Agno team when running inside cs_copilot. The current MCP surface exposes downstream session and GTM tools, but not the full molecular designer toolkit.
4. Validate generated candidates before presenting structures as final.
5. Register candidate sets and keep large candidate lists in artifacts rather than chat text.
6. Use `session_materialize_candidate_set_dataset` before GTM projection or batch retrosynthesis.

## Expected Outputs

- Registered candidate set.
- Candidate artifact with full generated/validated structures.
- Optional CSV materialization for projection, reporting, or synthesis planning.
- Validation and ranking summary.
