# Agent System

The system uses a **Factory Pattern + Registry** for agent creation.

**Location**: `src/cs_copilot/agents/`

## Key Files

- `factories.py` — 8 factory classes (7 runtime agents plus the robustness evaluation agent)
- `registry.py` — Dynamic agent registry with auto-discovery
- `teams.py` — Multi-agent team coordination using the Agno framework
- `prompts.py` — Agent instructions and system prompts

## Runtime Team Agents

| Agent | Role |
|-------|------|
| **ChEMBL Downloader** | Downloads and filters bioactivity data from the ChEMBL database |
| **GTM Agent** | Unified GTM workflows: build, load, density, activity, projection, and GTM sampling support |
| **Chemoinformatician** | Downstream chemoinformatics analysis including scaffold, similarity, clustering, and SAR workflows |
| **Report Generator** | Formats analysis outputs into reports and visual artifacts |
| **Molecular Designer** | Small-molecule design via autoencoder and LLM engines, including standalone and GTM-guided modes |
| **Peptide Designer** | Peptide generation, latent-space GTM workflows, and DBAASP-backed peptide activity landscapes |
| **SynPlanner** | Retrosynthetic planning and route visualization for target molecules |

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
