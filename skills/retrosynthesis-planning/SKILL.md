---
name: retrosynthesis-planning
description: Resolve a target molecule, run SynPlanner retrosynthesis, persist route artifacts, and prepare synthesis summaries.
metadata:
  title: Retrosynthesis planning
  status: optional_backend
  tags:
    - retrosynthesis
    - synplanner
    - synthesis
    - route-planning
  keywords:
    - retrosynthesis
    - synthesis
    - synthetic route
    - retrosynthetic
  required_tools:
    - session_resolve_session_reference
    - synplanner_identify_input
    - synplanner_plan_synthesis
    - synplanner_describe_plan
  optional_tools:
    - synplanner_convert_name_to_smiles
    - synplanner_get_route_visualizations
    - report_save_rich
  artifact_outputs:
    - synthesis_plan_json
    - route_visualization_svg
    - route_visualization_png
  example_prompts:
    - Plan a synthesis route for the current selected generated compound and save route visualizations.
---

# Retrosynthesis Planning

Use this skill when the user asks how to synthesize a target molecule, generated candidate, or named compound.

## Procedure

1. Resolve the target molecule from SMILES, name, session memory, or a selected candidate set with `session_resolve_session_reference` when a session reference is involved.
2. For vague generated-candidate follow-ups such as "plan synthesis for top candidates", resolve the candidate set and pass explicit SMILES to SynPlanner. Do not fall back to an older seed or dataset compound.
3. Confirm the SynPlanner optional backend is installed before promising route generation.
4. Use `synplanner_identify_input` to classify the query. If the target is a name, use `synplanner_convert_name_to_smiles` before planning. If name resolution fails, ask for a SMILES string.
5. Run `synplanner_plan_synthesis` for the canonical target. The tool may retry documented search profiles when the standard search finds no route.
6. Summarize the latest plan with `synplanner_describe_plan`, including route count, route depth, building blocks, and failed search profiles if relevant.
7. Generate route visualizations with `synplanner_get_route_visualizations` when the user asks for figures or report-ready artifacts. Present route images in order (Route 0, Route 1, …) using markdown image syntax `![Route {route_index} - {caption}](png_path)`, with the route index plus node_id/score in the caption, before the detailed prose analysis.
8. If no route is found and an LLM fallback is allowed, label it explicitly as not SynPlanner-validated and keep it separate from SynPlanner route results.
9. Persist route plans and visualizations as artifacts for downstream report generation.

## Expected Outputs

- Canonical target SMILES.
- Synthesis plan artifact.
- Route visualization artifacts when available.
- Report-ready route summary for downstream report generation.
