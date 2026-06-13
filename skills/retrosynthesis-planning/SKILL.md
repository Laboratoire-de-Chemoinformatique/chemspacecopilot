# Retrosynthesis Planning

Use this skill when the user asks how to synthesize a target molecule, generated candidate, or named compound.

## Procedure

1. Resolve the target molecule from SMILES, name, session memory, or a selected candidate set with `session_resolve_session_reference` when a session reference is involved.
2. Confirm the SynPlanner optional backend is installed before promising route generation.
3. Use `synplanner_identify_input` to classify the query. If the target is a name, use `synplanner_convert_name_to_smiles` before planning.
4. Run `synplanner_plan_synthesis` for the canonical target.
5. Summarize the latest plan with `synplanner_describe_plan`.
6. Generate route visualizations with `synplanner_get_route_visualizations` when the user asks for figures or report-ready artifacts.
7. Persist route plans and visualizations as artifacts and summarize route count, route depth, building blocks, and failed search profiles if relevant.

## Expected Outputs

- Canonical target SMILES.
- Synthesis plan artifact.
- Route visualization artifacts when available.
- Report-ready route summary for downstream report generation.
