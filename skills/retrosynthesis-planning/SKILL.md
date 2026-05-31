# Retrosynthesis Planning

Use this skill when the user asks how to synthesize a target molecule, generated candidate, or named compound.

## Procedure

1. Resolve the target molecule from SMILES, name, session memory, or a selected candidate set.
2. Confirm the SynPlanner optional backend is installed before promising route generation.
3. In the Agno team runtime, use the SynPlanner agent for input identification, name-to-SMILES conversion, planning, route visualization, and synthesis report data.
4. In MCP mode, treat this skill as procedural guidance unless SynPlanner MCP tools are explicitly added to the server.
5. Persist route plans and visualizations as artifacts and summarize route count, route depth, building blocks, and failed search profiles if relevant.

## Expected Outputs

- Canonical target SMILES.
- Synthesis plan artifact.
- Route visualization artifacts when available.
- Report-ready route summary for downstream report generation.
