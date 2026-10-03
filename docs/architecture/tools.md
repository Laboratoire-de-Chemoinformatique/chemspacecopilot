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
