# Skills and Workflows

ChemSpace Copilot keeps reusable scientific procedures outside agent prompts
and plugin wrappers. Two related catalogs serve different consumers.

## Skills

`skills/<slug>/SKILL.md` contains a reusable procedure that the Agno
coordinator or an MCP client can discover and fetch. Its frontmatter declares
metadata such as:

- slug, title, description, and semantic version;
- keywords and dependencies;
- required capability profiles and permissions; and
- expected input and output artifact contracts.

`cs_copilot.skills.SkillRegistry` discovers the packaged catalog, rejects
invalid or duplicate metadata, validates dependency references, and provides
list, search, and fetch operations.

## Workflow contracts

`workflow_catalog/<slug>/WORKFLOW.md` is an MCP-facing execution contract. In
addition to human-readable procedure text, it can define preflight tools,
required tool groups, recommended prompts, expected artifacts, and task
contracts.

`cs_copilot.workflows.WorkflowRegistry` validates these declarations and the
v2 runtime snapshots the selected workflow and its transitive dependencies at
run creation. Resume and replay use that pinned snapshot rather than silently
adopting later catalog edits.

Only `chembl-to-gtm-report` currently declares the strict persisted task DAG
used by the MCP pilot. Other workflows are discoverable contracts with
taskless execution; catalog presence alone does not imply deterministic task
execution.

## Ownership boundaries

| Concern | Source of truth |
|---------|-----------------|
| Agent identity and role | `src/cs_copilot/agents/descriptions.py` |
| Cross-task behavioral rules | `src/cs_copilot/agents/instructions.py` |
| Reusable scientific procedure | `skills/<slug>/SKILL.md` |
| MCP execution and artifact contract | `workflow_catalog/<slug>/WORKFLOW.md` |
| Deterministic scientific behavior | `src/cs_copilot/tools/` |
| Run lifecycle and replay | `src/cs_copilot/workflows/runtime.py` |

The wheel packages both catalogs. The repository plugin under
`plugins/chemspace-copilot/` supplies MCP configuration and a bootstrap skill,
but does not copy or redefine chemistry procedures.

## Changing a procedure

1. Update the owning skill when the scientific sequence or decision guidance
   changes.
2. Update the workflow contract when required tools, permissions, tasks, or
   artifact expectations change.
3. Bump the catalog item's semantic version and update dependent catalog items
   when their contract changes.
4. Keep agent instructions and the plugin bootstrap generic.
5. Run skill, workflow-registry, plugin-contract, and MCP profile tests.
