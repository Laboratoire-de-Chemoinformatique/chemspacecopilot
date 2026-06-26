# Retrosynthesis For Candidates

Use this skill when the user wants synthesis routes for a generated candidate, selected session molecule, SMILES string, or named compound.

## Procedure

1. Resolve the target from the user prompt or session memory with `session_resolve_session_reference` when needed.
2. For generated-candidate references, resolve the candidate set and pass explicit selected SMILES. Ask for a candidate ID or selection when multiple candidates match.
3. Check the optional SynPlanner backend before promising route generation.
4. Classify input with `synplanner_identify_input`; convert names with `synplanner_convert_name_to_smiles` when needed. If conversion fails, ask for SMILES.
5. Run `synplanner_plan_synthesis` and summarize with `synplanner_describe_plan`.
6. Save route figures with `synplanner_get_route_visualizations` when the user asks for visual artifacts or report-ready output.
7. If no route is found and an LLM fallback is allowed, label it explicitly as not SynPlanner-validated and do not present fallback chemistry as a SynPlanner result.
8. Include route artifacts in `report_save_rich` if a synthesis report is requested.

## Expected Outputs

- Canonical target SMILES.
- Synthesis plan artifact.
- Optional route visualizations.
- Report-ready route summary.
