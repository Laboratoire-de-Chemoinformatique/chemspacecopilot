# Installation

## Prerequisites

- Python 3.11
- [uv](https://docs.astral.sh/uv/) package manager
- A [DeepSeek](https://platform.deepseek.com/) API key for the default cloud backend, or a local [Ollama](https://ollama.com/) instance

## Install Dependencies

```bash
uv sync
```

## Environment Configuration

For file-based configuration, copy `.env.example` to `.env` in the project root:

```bash
# Required only for the default DeepSeek provider
DEEPSEEK_API_KEY=your-api-key-here

# Optional model overrides (otherwise .modelconf is used)
# MODEL_PROVIDER=deepseek
# MODEL_ID=deepseek-chat
# OLLAMA_HOST=http://localhost:11434

# Optional — S3/MinIO storage (disable with USE_S3=false)
USE_S3=true
S3_ENDPOINT_URL=http://localhost:9000
MINIO_ACCESS_KEY=cs_copilot
MINIO_SECRET_KEY=chempwd123
ASSETS_BUCKET=chatbot-assets

# Optional — ChEMBL local MySQL (faster queries, offline use)
# Download dump: https://chembl.gitbook.io/chembl-interface-documentation/downloads
# CHEMBL_MYSQL_HOST=localhost
# CHEMBL_MYSQL_PORT=3306
# CHEMBL_MYSQL_USER=chembl
# CHEMBL_MYSQL_PASSWORD=
# CHEMBL_MYSQL_DATABASE=chembl_36
```

The repository also includes a tracked `.modelconf` file. Edit it if you want to switch from the default DeepSeek backend to a local Ollama model.

## Running the Application

### Chainlit App

```bash
uv run chainlit run chainlit_app.py -w
```

Access the application at **http://localhost:8000**.

Notes:

- The bundled `chainlit.toml` currently has `[persistence] enabled = false`.
- The app sets a per-thread title from your first message; you can rename it in the UI.

### Jupyter Notebook

An example workflow is available in `notebooks/cs_copilot.ipynb`.

## Optional Services

### S3/MinIO Storage

```bash
# Run the interactive setup script
python scripts/setup_s3.py

# Or start MinIO manually
docker run -d --name minio \
  -p 9000:9000 -p 9001:9001 \
  -v /mnt/data:/data \
  -e MINIO_ROOT_USER=cs_copilot \
  -e MINIO_ROOT_PASSWORD=chempwd123 \
  minio/minio server /data --console-address ":9001"
```

If the container already exists: `docker start minio`

### Optional Chainlit Persistence

Chainlit persistence is disabled by default in `chainlit.toml`. Only set up PostgreSQL if you plan to enable Chainlit persistence manually.

```bash
docker run --name chainlit-pg -p 5432:5432 -d \
  -e POSTGRES_PASSWORD=postgres \
  -e POSTGRES_USER=postgres \
  -e POSTGRES_DB=chainlit \
  postgres:16

export DATABASE_URL="postgresql://postgres:postgres@localhost:5432/chainlit"
```

If the container already exists: `docker start chainlit-pg`

### ChEMBL Local Database (Optional)

By default the ChEMBL Downloader agent queries the [ChEMBL REST API](https://www.ebi.ac.uk/chembl/api/data). For faster queries and offline use, you can point it at a local MySQL copy of ChEMBL instead.

**Setup:**

1. Download the MySQL dump from the [ChEMBL downloads page](https://chembl.gitbook.io/chembl-interface-documentation/downloads) or directly from the [EBI FTP](https://ftp.ebi.ac.uk/pub/databases/chembl/ChEMBLdb/latest/).
2. Load the dump into a MySQL 8+ server.
3. Install the optional MySQL driver:
   ```bash
   uv sync --extra mysql
   ```
4. Set the environment variables:
   ```bash
   CHEMBL_MYSQL_HOST=localhost
   CHEMBL_MYSQL_PORT=3306
   CHEMBL_MYSQL_USER=chembl
   CHEMBL_MYSQL_PASSWORD=your-password
   CHEMBL_MYSQL_DATABASE=chembl_36
   ```

When `CHEMBL_MYSQL_HOST` is set, the agent automatically uses MySQL. The REST API remains available as a fallback. Unset the variable to revert to REST-only mode.
