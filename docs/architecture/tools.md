# Tools System

Deterministic Python functions and Agno `Toolkit` classes perform the scientific
work. Language models select tools and interpret results; they do not replace
descriptor calculation, database queries, GTM fitting, model inference, or
retrosynthetic search.

## Scientific toolkits

The main packages under `src/cs_copilot/tools/` are:

```text
tools/
├── databases/      ChEMBL REST and SQL backends
├── chemography/    GTM construction, projection, landscapes, and sampling
├── chemistry/      standardization, descriptors, similarity, design, retrosynthesis
├── analysis/       robustness result analysis
├── io/             DataFrame pointers, session memory, skills, and report export
└── constants.py    shared tool configuration
```

Toolkit classes register selected methods with Agno. Method names, signatures,
docstrings, and return values form part of the LLM-visible interface, so tool
methods should remain deterministic, validate inputs, and return explicit
artifact references where appropriate.

## MCP exposure

MCP reuses the scientific implementations through an additional contract
layer under `src/cs_copilot/mcp/`:

- tool specifications define public names, schemas, risk annotations, and
  profile membership;
- facades adapt session and workflow context to toolkit calls;
- the tool adapter enforces profile, task, timeout, idempotency, and artifact
  rules; and
- the server publishes only the selected capability profile.

A toolkit method is not automatically public through MCP. It must have an MCP
specification whose name starts with its group's prefix (`gtm_`, `chem_`, ...)
and belong to the selected profile. Role grants and profiles live in one
capability table, `src/cs_copilot/capabilities.py`: changing a role's groups
changes both the Agno specialist's allowlist and the MCP roles of every tool in
those groups, and every role must still fit inside its assigned profile.

## ChEMBL backends

`ChemblToolkit` supports REST, SQLite, PostgreSQL, and MySQL. With
`backend="auto"`, configured local databases are tried before REST and a
missing optional SQL driver falls through to the next candidate. REST requires
no local database configuration.

Local SQL deployments are useful for high-volume or offline queries. Database
connection settings are documented in the installation guide and `.env.example`.

## SynPlanner backend

The `SynPlannerToolkit` wrapper and the external SynPlanner/CGRtools stack are
installed with the default project dependencies. On first use, SynPlanner
downloads its runtime model and chemistry data when no local `synplan_data`
directory is available.

## Molecular latent engines

`AutoencoderToolkit` encodes SMILES to a latent vector and decodes latent
vectors back to molecules. The model behind it is selectable, and the toolkit
class, its nine registered Agno tools and the `generation_engine="autoencoder"`
provenance string are the same whichever engine is active.

| | deepchemography LSTM | chembed |
|---|---|---|
| Representation | SMILES | SELFIES |
| Latent width | 256 | 256 |
| Checkpoint | `axelrolov/lstm_autoencoder`, 8.8 MB | `3BioCompBio/chembed-default`, 227 MB |
| Decoding | greedy or temperature-sampled | greedy only |
| `max_length` | caller-set, default 100 | fixed at 256 by the checkpoint |
| Long molecules | fails beyond roughly 50 SMILES characters | encodes the full frameset |

chembed is the default. Select the other with the `AUTOENCODER_BACKEND`
environment variable (`chembed` or `deepchemography`), or by passing `backend=`
to `make_autoencoder_toolkit()`. Construct the toolkit through that factory
rather than directly, so the choice is honoured. Set
`AUTOENCODER_BACKEND=deepchemography` to reproduce the frozen manuscript
results.

**The variable selects the generation engine only.** Descriptor families pin
their own engine: `autoencoder` is always the deepchemography LSTM and
`chembed` is always chembed, whatever the variable says. Otherwise
`descriptor_type="autoencoder"` would emit chembed vectors into the
`autoencoder_embedding` column, and the default map, which is fitted on LSTM
embeddings, would silently receive the wrong latent.

### What chembed ignores

chembed decodes greedily. `temperature`, `decode_mode` and `max_length` are
accepted and have no effect, so **an agent must not offer a temperature or
sampling sweep when this engine is active** — `get_model_info()` reports
`supports_temperature` and `supports_stochastic_decode` as `False`. Diversity
comes only from latent noise (`noise_scale`, `latent_std`), and the scales
tuned for the LSTM do not transfer: a single encoded molecule has a latent
standard deviation near 0.72. Every generation audit records the requested
settings alongside the applied ones, plus `ignored_parameters`, so no artifact
claims a setting that was never used.

Because diversity comes only from the noise, the scale matters more than it did
for the LSTM. Measured over aspirin, lidocaine and a large peptidomimetic, 60
samples each:

| `noise_scale` | aspirin | lidocaine | large seed |
|---|---|---|---|
| 0.1 | 0 distinct | 0 distinct | 1 distinct |
| 0.25 | 1, Tanimoto 0.55 | 6, Tanimoto 0.69 | 39, Tanimoto 0.52 |
| 0.5 | 23, Tanimoto 0.55 | 35, Tanimoto 0.50 | 60, Tanimoto 0.25 |
| 1.0 | 57, Tanimoto 0.30 | 60, Tanimoto 0.21 | 60, Tanimoto 0.17 |

At the LSTM's 0.1 the decoder simply returns the parent. chembed therefore
defaults to 0.5, which yields analogues across the size range. Larger seeds
drift further at any given scale, so analogue work on big molecules wants 0.25.
Omitting `noise_scale` uses the active engine's default rather than a fixed
number, so the LSTM keeps its own 0.1.

Sampling is also slower, because the decoder is autoregressive: 5000 molecules
take roughly 35 s on a GPU and about 17 minutes on CPU. `sample_molecules`
therefore defaults to 5000 on CUDA and 500 on CPU when no count is given, and
records which default it used.

### Latent spaces are not interchangeable

Both engines emit 256-wide latents, so a GTM manifold fitted on one will accept
vectors from the other without error and return meaningless coordinates. Two
mechanisms prevent that, and both must be kept:

- The descriptor families write different columns, `autoencoder_embedding` and
  `chembed_embedding`. `_compute_descriptors` reuses an existing descriptor
  column as a cache without recording which model wrote it, so distinct names
  are what stop one engine's vectors being consumed as the other's.
- Manifolds carry a `descriptor_backend` tag, checked by
  `validate_gtm_descriptor_backend` before projection. Manifolds published
  before the tag existed pass with a warning.

How little the shape tells you: encoding aspirin with both engines gives two
`(1, 256)` vectors whose cosine similarity is 0.13. They are the same size and
otherwise unrelated, so nothing but the tag and the column name can catch a
swap. `HUGGINGFACE_GTM_REVISION` pins the published manifold for the same
reason, so a repository update cannot exchange it under results that depend
on it.

Changing the engine therefore invalidates any manifold fitted on the previous
latent space. `scripts/refit_gtm_manifold.py` refits one, reusing the published
hyperparameters rather than re-tuning them.

## Adding a tool

1. Implement or extend a toolkit in the appropriate scientific package and
   register only the methods intended for Agno.
2. Add the toolkit to the owning agent factory. A new toolkit class also joins
   its group in `src/cs_copilot/capabilities.py`, and a role that needs a new
   group gets it there.
3. If MCP should expose the operation, add its tool specification (named with
   its group's prefix), facade or adapter binding, and annotations. Roles and
   profiles are derived from the capability table; do not set them on the
   spec.
4. Update affected skills and workflow contracts when the operation changes a
   procedure, permission, or artifact contract.
5. Test deterministic behavior, role access, direct-tool/MCP parity, profile
   filtering, and artifact registration as applicable.
