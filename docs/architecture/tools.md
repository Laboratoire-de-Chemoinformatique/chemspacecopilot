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

## The molecular latent engine

`AutoencoderToolkit` encodes SMILES to a latent vector and decodes latent
vectors back to molecules. chembed, a SELFIES transformer VAE
(`3BioCompBio/chembed-default`, 227 MB), is the engine behind it. It is supplied
as an `AutoencoderEngine` object, so the toolkit class, its nine registered Agno
tools and the `generation_engine="autoencoder"` provenance string do not name a
model. Build the toolkit through `make_autoencoder_toolkit()`.

**The deepchemography LSTM autoencoder was retired after v0.4.0.** It could not
encode SMILES longer than roughly 50 characters, which is most of the potent
chemotypes, and its 256-wide latent is unrelated to chembed's. Check out the
`v0.4.0` tag to reproduce results recorded on it. Names that selected it —
`AUTOENCODER_BACKEND=deepchemography`, the robustness `latent_engine` key, and
the `autoencoder`, `ae`, `latent` and `embedding` descriptor families — now
raise rather than silently running on chembed, so no result is attributed to a
model that did not produce it.

**`deepchemography` names a package and an engine, and they are not the same
thing.** The package ships two unrelated models: `LSTMAutoencoder`, the retired
small-molecule engine, and `peptides.PeptideWAE`, which backs peptide design and
has no chembed counterpart. Retiring the engine therefore did not retire the
dependency, and peptide generation keeps needing it.

### What chembed ignores

chembed decodes greedily. `temperature`, `decode_mode` and `max_length` are
accepted and have no effect, so **an agent must not offer a temperature or
sampling sweep when this engine is active** — `get_model_info()` reports
`supports_temperature` and `supports_stochastic_decode` as `False`. Diversity
comes only from latent noise (`noise_scale`, `latent_std`): a single encoded
molecule has a latent standard deviation near 0.72. Every generation audit
records the requested
settings alongside the applied ones, plus `ignored_parameters`, so no artifact
claims a setting that was never used.

Because diversity comes only from the noise, the scale matters. Measured over
aspirin, lidocaine and a large peptidomimetic, 60
samples each:

| `noise_scale` | aspirin | lidocaine | large seed |
|---|---|---|---|
| 0.1 | 0 distinct | 0 distinct | 1 distinct |
| 0.25 | 1, Tanimoto 0.55 | 6, Tanimoto 0.69 | 39, Tanimoto 0.52 |
| 0.5 | 23, Tanimoto 0.55 | 35, Tanimoto 0.50 | 60, Tanimoto 0.25 |
| 1.0 | 57, Tanimoto 0.30 | 60, Tanimoto 0.21 | 60, Tanimoto 0.17 |

At 0.1 the decoder simply returns the parent. chembed therefore defaults to
0.5, which yields analogues across the size range. Larger seeds drift further at
any given scale, so analogue work on big molecules wants 0.25. Omitting
`noise_scale` uses the engine's own default rather than a fixed number.

Sampling is also slower, because the decoder is autoregressive: 5000 molecules
take roughly 35 s on a GPU and about 17 minutes on CPU. `sample_molecules`
therefore defaults to 5000 on CUDA and 500 on CPU when no count is given, and
records which default it used.

### A manifold belongs to one latent space

The retired LSTM also emitted 256-wide latents, so a manifold fitted on it
accepts chembed vectors without error and returns meaningless coordinates. Such
manifolds survive in users' caches and in saved sessions, so two mechanisms stay:

- The descriptor families write different columns. chembed writes
  `chembed_embedding`, never the LSTM's `autoencoder_embedding`:
  `_compute_descriptors` reuses an existing descriptor column as a cache without
  recording which model wrote it.
- Manifolds carry a `descriptor_backend` tag, checked by
  `validate_gtm_descriptor_backend` before projection. One tagged `autoencoder`
  is refused; one published before the tag existed passes with a warning.

How little the shape tells you: encoding aspirin with both engines gave two
`(1, 256)` vectors whose cosine similarity is 0.13. Same size, otherwise
unrelated, so nothing but the tag and the column name can catch a swap.

The published manifold, `axelrolov/chemspacecopilot-gtm-chembed`
(`HUGGINGFACE_GTM_CHEMBED_REPO` / `_REVISION`, cached in
`~/.cache/cs_copilot/models/gtm-chembed`), is fitted on chembed latents of a
pinned 50,000-molecule frameset. It is pinned by revision rather than tracking a
branch, so a repository update cannot exchange it under results that depend on
it. `resolve_gtm_model_path` resolves the default map to it; a session that has
pinned its own map keeps that one. `scripts/refit_gtm_manifold.py` refits a
manifold, reusing the published hyperparameters rather than re-tuning them.

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
