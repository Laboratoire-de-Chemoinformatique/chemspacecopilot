# Agent System

The system uses a **Factory Pattern + Registry** for agent creation.

**Location**: `src/cs_copilot/agents/`

## Key Files

- `factories.py` — 8 factory classes (7 runtime agents plus the robustness evaluation agent)
- `registry.py` — Dynamic agent registry with auto-discovery
- `teams.py` — Multi-agent team coordination using the Agno framework
- `prompts.py` — Agent instructions and system prompts
- `skills/` — Shared reusable workflow procedures consumed by Agno and MCP

## Runtime Team Agents

| Agent | Role |
|-------|------|
| **ChEMBL Downloader** | Downloads and filters bioactivity data from the ChEMBL database |
| **GTM Agent** | Unified GTM workflows: build, load, density, activity, projection, and GTM sampling support |
| **Chemoinformatician** | Downstream chemoinformatics analysis including scaffold, similarity, clustering, and SAR workflows |
| **Report Generator** | Formats analysis outputs into reports and visual artifacts |
| **Molecular Designer** | Small-molecule design via autoencoder and LLM engines, including standalone and GTM-guided modes |
| **Peptide Designer** | Peptide design via WAE and LLM engines, latent-space GTM workflows, and DBAASP-backed peptide activity landscapes |
| **SynPlanner** | Retrosynthetic planning and route visualization for target molecules; requires the optional `synplanner` extra |

The team coordinator also has a read-only Skills toolkit (`list_skills`, `search_skills`, `fetch_skill`) so it can consult the same reusable workflow procedures exposed through MCP.

## Why use specialized agents?

The agents do not replace deterministic scientific computation. Descriptor
calculation, standardization, GTM fitting and projection, similarity, generative
model inference, and retrosynthetic search remain ordinary toolkit calls. The
language model layer decides what the user means, which validated procedure and
tools apply, how outputs from one domain become inputs to another, and what
evidence must be returned.

Specialization is useful for open-ended or cross-domain requests because it:

- gives each role a smaller, chemically coherent tool schema and instruction set;
- isolates domain context instead of placing every tool result and instruction in
  one expanding context window;
- enforces role-specific tool allowlists and typed, artifact-referenced handoffs;
- composes retrieval, mapping, analysis, design, synthesis, and reporting only
  when the request requires them; and
- avoids same-named molecular and peptide operations competing in one tool
  namespace.

These benefits are not free. Coordination adds model calls, tokens, latency, and
another possible routing failure. The repository therefore includes a controlled
flat-agent ablation that holds the model, prompts, tools, fixtures, validators,
and inference settings constant and varies only coordinator/specialist
separation. See [Robustness Tests](../testing/robustness.md) for the runbook and
reported outcomes.

## Fixed workflows versus agentic routing

A fixed workflow is preferable when inputs, tool order, branching, and acceptance
criteria are completely specified and stable. Such procedures are easier to
replay, audit, and operate without repeated routing decisions. ChemSpace Copilot
represents these procedures in `workflow_catalog/`; the v2 runtime can pin task
contracts, record immutable events, and verify checksummed artifacts.

Agentic routing is intended for requests that are ambiguous, combine domains in
unanticipated ways, or require choosing among several valid analyses. The usual
deployment is hybrid: a coordinator selects an applicable catalog procedure,
specialists execute bounded domain steps, and deterministic tools perform the
scientific computation. The strict persisted task-DAG implementation is currently
limited to the `chembl-to-gtm-report` MCP pilot, so it should not be described as
a general deterministic executor for every catalog entry.

## Handoff durability boundary

The Agno delegation guard always validates the structured handoff schema,
receiver role, private-context exclusions, and declared budget limits. Agent
factories independently enforce each role's toolkit allowlist.

Durable validation is opt-in at team construction. When a caller supplies a v2
`RunContext`, the guard records the handoff through that runtime, which checks
it against the pinned task contract and event stream. The current Chainlit and
CLI team entry points construct an ad-hoc team without a `RunContext`; their
handoff identities and counters are process-local and are not persisted as v2
workflow events. Strict persisted task-DAG enforcement is currently the
`chembl-to-gtm-report` MCP pilot.

## Prompt / Skill Ownership

The prompt layer is split to mirror Agno's own `description` / `instructions`
separation. Agent personas in `src/cs_copilot/agents/descriptions.py` define
*who each agent is* (identity/role, fed to `Agent(description=...)`). Agent
instructions in `src/cs_copilot/agents/instructions.py` define *how it behaves*
(global routing policy, clarification rules, evidence standards, session-memory
conventions, and artifact formatting, fed to `Agent(instructions=...)`).
Neither should contain long mutable tool sequences.

Reusable task procedures live in `skills/<slug>/SKILL.md`. MCP-facing workflow
contracts live in `workflow_catalog/<slug>/WORKFLOW.md` and define preflight
tools, required tool groups, expected artifacts, and recommended prompts. When
behavior changes for a domain workflow, update the skill or workflow catalog
first and keep prompts high-level.

## Separate Evaluation Agent

| Agent | Role |
|-------|------|
| **Robustness Evaluation** | Analyzes robustness test runs, score distributions, failures, and trends |

## Adding a New Agent

1. Create a factory in `src/cs_copilot/agents/factories.py`:

```python
class MyNewAgentFactory(BaseAgentFactory):
    def create_agent(self, model, **kwargs):
        config = AgentConfig(
            name="my_new_agent",
            description="What this agent does",
            instructions="Detailed instructions here",
            tools=[MyToolkit(), ...],
            model=model,
            **kwargs
        )
        return self._create_agent(config)
```

2. The registry auto-discovers it via `AgentRegistry.auto_register()`
3. Add to the team in `teams.py` if needed
