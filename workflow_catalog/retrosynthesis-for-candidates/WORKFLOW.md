---
name: retrosynthesis-for-candidates
description: Resolve a generated candidate or named target and create SynPlanner route artifacts.
metadata:
  title: Retrosynthesis for candidates
  status: optional_backend
  version: 2.0.0
  depends_on: []
  profiles:
    - retrosynthesis
  permissions:
    - network:read
    - model:execute
    - artifact:read
    - artifact:write
  input_artifacts:
    - name: candidate_reference
      kind: molecule
      required: true
  output_artifacts:
    - name: synthesis_plan_json
      kind: route-plan
      required: true
    - name: route_visualization_svg
      kind: visualization
      required: false
    - name: route_visualization_png
      kind: visualization
      required: false
  tags:
    - retrosynthesis
    - synplanner
    - candidates
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
  recommended_prompt: synplanner_agent
---

# Retrosynthesis For Candidates

Use this workflow when the user asks how to synthesize a generated candidate, selected session molecule, SMILES string, or named compound.

1. Resolve the target from session memory or the user prompt.
2. Use `synplanner_identify_input` and convert names with `synplanner_convert_name_to_smiles` when needed.
3. Confirm the optional SynPlanner backend is available before promising route generation.
4. Run `synplanner_plan_synthesis`, summarize with `synplanner_describe_plan`, and save visualizations with `synplanner_get_route_visualizations` when requested.
5. If no route is found and an LLM fallback is allowed, label it as not SynPlanner-validated.
6. Include route artifacts in a report if the user asks for a synthesis summary.
