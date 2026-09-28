---
name: retrosynthesis-for-candidates
description: Resolve a candidate or named target, run SynPlanner retrosynthesis, and prepare route artifacts.
metadata:
  title: Retrosynthesis for candidates
  status: stable
  version: 2.0.0
  depends_on:
    - retrosynthesis-planning
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
  example_prompts:
    - Run retrosynthesis for the selected generated candidate and save route visualizations.
---

# Retrosynthesis For Candidates

Use this skill when the user wants synthesis routes for a generated candidate, selected session molecule, SMILES string, or named compound.

## Procedure

1. Resolve the target from the user prompt or session memory with `session_resolve_session_reference` when needed.
2. For generated-candidate references, resolve the candidate set and pass explicit selected SMILES. Ask for a candidate ID or selection when multiple candidates match.
3. Check that the SynPlanner runtime data is available before promising route generation.
4. Classify input with `synplanner_identify_input`; convert names with `synplanner_convert_name_to_smiles` when needed. If conversion fails, ask for SMILES.
5. Run `synplanner_plan_synthesis` and summarize with `synplanner_describe_plan`.
6. Save route figures with `synplanner_get_route_visualizations` when the user asks for visual artifacts or report-ready output. Present them in order using `![Route {route_index} - {caption}](png_path)` with index/node_id/score in the caption.
7. If no route is found and an LLM fallback is allowed, label it explicitly as not SynPlanner-validated and do not present fallback chemistry as a SynPlanner result.
8. Include route artifacts in `report_save_rich` if a synthesis report is requested.

## Expected Outputs

- Canonical target SMILES.
- Synthesis plan artifact.
- Optional route visualizations.
- Report-ready route summary.
