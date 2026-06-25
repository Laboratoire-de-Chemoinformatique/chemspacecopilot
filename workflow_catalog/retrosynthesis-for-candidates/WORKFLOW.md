# Retrosynthesis For Candidates

Use this workflow when the user asks how to synthesize a generated candidate, selected session molecule, SMILES string, or named compound.

1. Resolve the target from session memory or the user prompt.
2. Use `synplanner_identify_input` and convert names with `synplanner_convert_name_to_smiles` when needed.
3. Confirm the optional SynPlanner backend is available before promising route generation.
4. Run `synplanner_plan_synthesis`, summarize with `synplanner_describe_plan`, and save visualizations with `synplanner_get_route_visualizations` when requested.
5. Include route artifacts in a report if the user asks for a synthesis summary.
