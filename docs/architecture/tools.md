# Tools System

Tools are organized as **Toolkit classes** that inherit from `Toolkit` (Agno framework).

**Location**: `src/cs_copilot/tools/`

## Directory Structure

```
tools/
├── databases/          Database integrations
│   ├── base.py        BaseDatabaseToolkit (abstract)
│   ├── chembl.py      ChemblToolkit (REST API + MySQL backends)
│   ├── chembl_fetcher.py  RestChemblFetcher / SqlChemblFetcher strategies
│   └── types.py       Query types and configurations
│
├── chemography/       Dimensionality reduction
│   ├── gtm.py         GTMToolkit (high-level interface)
│   └── gtm_operations.py  Core GTM implementations
│
├── chemistry/         Molecular operations
│   ├── similarity_toolkit.py      Similarity calculations
│   ├── autoencoder_toolkit.py     LSTM autoencoder operations
│   └── descriptors.py             Molecular descriptors
│
├── io/                I/O and formatting
│   ├── pointer_pandas_tools.py   DataFrame ops + S3 integration
│   └── formatting.py              SMILES → images, markdown
│
└── constants.py       Configuration constants
```

Each toolkit registers methods as tools via `self.register(method)`. Agents call these tools via the Agno tool-calling mechanism.

## ChEMBL Backends

The `ChemblToolkit` supports two pluggable data backends via a strategy pattern (`chembl_fetcher.py`):

| Backend | Trigger | Dependency | Use case |
|---------|---------|------------|----------|
| **REST API** | Default (no config needed) | `chembl_webresource_client` (included) | Quick setup, always-on access |
| **MySQL** | Set `CHEMBL_MYSQL_HOST` env var | `pymysql` (included in `uv sync`) | Faster queries, offline use, full SQL |

Backend is auto-detected: MySQL when `CHEMBL_MYSQL_HOST` is present, REST otherwise. The REST API is always reported as available regardless of active backend.

Download the MySQL dump from the [ChEMBL downloads page](https://chembl.gitbook.io/chembl-interface-documentation/downloads) or the [EBI FTP](https://ftp.ebi.ac.uk/pub/databases/chembl/ChEMBLdb/latest/).

## Adding a New Tool

1. Create a toolkit in `src/cs_copilot/tools/`:

```python
from agno import Toolkit

class MyNewToolkit(Toolkit):
    def __init__(self):
        super().__init__(name="my_new_toolkit")
        self.register(self.my_tool_function)

    def my_tool_function(self, param: str) -> str:
        """Tool description for LLM."""
        return f"Result: {param}"
```

2. Import and pass to the agent factory's `tools` parameter
