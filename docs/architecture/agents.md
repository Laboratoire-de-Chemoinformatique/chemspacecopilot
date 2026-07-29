# Agent System

The Agno execution path uses a factory and registry pattern. Agent personas,
behavioral instructions, tool access, context construction, and delegation
contracts are separate concerns under `src/cs_copilot/agents/`.

## Components

- `factories.py` — nine registered factories and their toolkit composition
- `registry.py` — factory auto-discovery, aliases, and the public creation API
- `teams.py` — construction of the seven-member production team
- `descriptions.py` — agent identity and role descriptions
- `instructions.py` — routing, evidence, state, and output behavior
- `contracts.py` — role policies, tool allowlists, and execution budgets
- `context.py` — bounded specialist context construction
- `delegation.py` — structured handoff validation and optional runtime recording
- `single_agent.py` — flat-agent baseline for architecture ablation tests

Long scientific procedures do not belong in agent instructions. They live in
the [Skills and Workflows](catalogs.md) catalogs.

## Registered factories

The registry exposes nine factory types in three categories.

### Production team

| Agent type | Role |
|------------|------|
| `chembl_downloader` | Downloads and filters ChEMBL bioactivity data |
| `gtm_agent` | Builds, loads, projects, samples, and analyzes GTM maps |
| `chemoinformatician` | Runs scaffold, similarity, clustering, and SAR analyses |
| `report_generator` | Produces reports and visual artifacts |
| `molecular_designer` | Generates small molecules in standalone and GTM-guided modes |
| `peptide_designer` | Generates peptides and supports latent-space GTM workflows |
| `synplanner` | Plans and visualizes retrosynthetic routes; requires the optional extra |

The coordinator also receives a read-only Skills toolkit so it can discover and
fetch the same reusable procedures available through MCP.

### Separate evaluation factory

`robustness_evaluation` analyzes robustness runs, distributions, failures, and
trends. It is not a production-team member.

### Ablation baseline

`single_agent` combines the production tool surface into one flat agent. It is
used to compare coordinator/specialist separation against a controlled
single-agent baseline and is not a production-team member.

## Coordination and context

Agents select and coordinate work; deterministic tools perform the scientific
computation. Each specialist receives a chemically coherent tool schema
validated against its canonical role policy.

Member transcripts and private reasoning are not broadcast. A handoff contains
only the receiving task's objective, constraints, acceptance criteria, trace
identifiers, budget, and input/output artifact contracts. The context builder
fits these facts and limited recent context into a declared budget.

The default Chainlit and CLI team validates the handoff schema, receiver role,
private-context exclusions, and budget limits in process. If team construction
receives a v2 `RunContext`, delegation is additionally checked against the
pinned workflow task and recorded in the durable event stream. Strict persisted
task-DAG enforcement remains limited to the `chembl-to-gtm-report` MCP pilot.

## Description, instruction, and procedure ownership

The split mirrors Agno's `description` and `instructions` fields:

- `descriptions.py` defines who an agent is and the domain it owns.
- `instructions.py` defines how the agent routes, clarifies, cites evidence,
  uses session state, and formats artifacts.
- `skills/<slug>/SKILL.md` defines reusable scientific procedures.
- `workflow_catalog/<slug>/WORKFLOW.md` defines MCP-facing execution contracts.

When domain behavior changes, update the skill or workflow first. Keep agent
instructions high-level unless the behavior applies to every task handled by
that role.

## Adding an agent

1. Define a `BaseAgentFactory` subclass with a unique `agent_type` and implement
   `get_agent_config()`:

```python
class MyNewAgentFactory(BaseAgentFactory):
    agent_type = "my_new_agent"

    def get_agent_config(self) -> AgentConfig:
        return AgentConfig(
            name="My New Agent",
            description=MY_NEW_AGENT_DESCRIPTION,
            instructions=MY_NEW_AGENT_INSTRUCTIONS,
            tools=[MyToolkit()],
            role_policy=ROLE_POLICIES[self.agent_type],
        )
```

2. Add the canonical `RolePolicy` and ensure every configured toolkit is
   allowed. The registry discovers the factory through `agent_type`.
3. Add the type and display name to the appropriate team constructor if it is a
   team member. A registered evaluation or benchmark factory need not join the
   production team.
4. Add or update catalog procedures instead of embedding mutable tool sequences
   in the instructions.
5. Add factory, role-policy, delegation, and team-construction tests.
