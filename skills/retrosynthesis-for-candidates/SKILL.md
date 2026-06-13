# Retrosynthesis For Candidates

Use this skill when the user wants synthesis routes for a generated candidate, selected session molecule, SMILES string, or named compound.

## Procedure

1. Resolve the target from the user prompt or session memory with `session_resolve_session_reference` when needed.
2. Check the optional SynPlanner backend before promising route generation.
3. Classify input with `synplanner_identify_input`; convert names with `synplanner_convert_name_to_smiles` when needed.
4. Run `synplanner_plan_synthesis` and summarize with `synplanner_describe_plan`.
5. Save route figures with `synplanner_get_route_visualizations` when the user asks for visual artifacts.
6. Include route artifacts in `report_save_rich` if a synthesis report is requested.

## Expected Outputs

- Canonical target SMILES.
- Synthesis plan artifact.
- Optional route visualizations.
- Report-ready route summary.
