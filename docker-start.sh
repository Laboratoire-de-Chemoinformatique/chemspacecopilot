#!/bin/bash
# Quick start script for Cs_copilot Docker setup

set -e

echo "🧪 Cs_copilot Docker Setup"
echo "=============================="
echo ""

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "❌ Error: Docker is not installed"
    echo "Please install Docker from https://docs.docker.com/get-docker/"
    exit 1
fi

# Check if Docker Compose is installed (V1: docker-compose, or V2: docker compose)
COMPOSE_CMD=""
if command -v docker-compose &> /dev/null; then
    COMPOSE_CMD="docker-compose"
elif docker compose version &> /dev/null; then
    COMPOSE_CMD="docker compose"
else
    echo "❌ Error: Docker Compose is not installed"
    echo "Please install Docker Compose from https://docs.docker.com/compose/install/"
    exit 1
fi

# Load .env if present (optional; user can create from .env.example when they want file-based config)
if [ -f .env ]; then
    set -a
    source .env
    set +a
    echo "📄 Loaded config from .env"
else
    echo "📄 No .env file (optional). Copy .env.example to .env for file-based config."
fi

# Detect host architecture and export TARGETARCH so docker-compose.yml can
# forward it to the Dockerfile build arg. On DGX Spark (aarch64) this
# triggers the NVIDIA NGC PyTorch base with a CUDA-enabled torch build.
case "$(uname -m)" in
    aarch64|arm64)
        export TARGETARCH=arm64
        ;;
    x86_64|amd64)
        export TARGETARCH=amd64
        ;;
    *)
        export TARGETARCH=amd64
        echo "⚠️  Unknown host arch $(uname -m); defaulting TARGETARCH=amd64"
        ;;
esac
echo "📌 Build target arch: $TARGETARCH"

# Agno telemetry status
if [ "${AGNO_TELEMETRY:-false}" = "true" ]; then
    echo "📡 Agno telemetry: ENABLED"
else
    echo "📡 Agno telemetry: DISABLED"
fi
echo ""

# Detect model provider from .modelconf, MODEL_PROVIDER env var, or default to deepseek
MODEL_PROVIDER="${MODEL_PROVIDER:-}"
if [ -z "$MODEL_PROVIDER" ] && [ -f .modelconf ]; then
    MODEL_PROVIDER=$(grep -E '^provider=' .modelconf 2>/dev/null | head -1 | cut -d= -f2 | tr -d ' ')
fi
MODEL_PROVIDER="${MODEL_PROVIDER:-deepseek}"
export MODEL_PROVIDER
echo "📌 Model provider: $MODEL_PROVIDER"

if [ "$MODEL_PROVIDER" = "ollama" ]; then
    # Ollama (local model) — no API key needed
    echo "✅ Using Ollama (local model) – DEEPSEEK_API_KEY not required"
    # Set OLLAMA_HOST for the container (default: reach host Ollama via Docker bridge)
    if [ -z "$OLLAMA_HOST" ]; then
        OLLAMA_HOST="http://host.docker.internal:11434"
    fi
    export OLLAMA_HOST
    echo "📌 Ollama host: $OLLAMA_HOST"
elif [ "$MODEL_PROVIDER" = "openrouter" ]; then
    # OpenRouter (cloud gateway) — API key is required
    if [ -z "$OPENROUTER_API_KEY" ] || [ "$OPENROUTER_API_KEY" = "your-openrouter-api-key-here" ]; then
        echo ""
        echo "OPENROUTER_API_KEY is required for OpenRouter provider. Enter it now (not saved to disk):"
        read -rs OPENROUTER_API_KEY
        echo ""
        if [ -z "$OPENROUTER_API_KEY" ]; then
            echo "❌ Error: OPENROUTER_API_KEY is empty"
            echo "Create .env from .env.example and set OPENROUTER_API_KEY, or run this script and enter the key when prompted."
            exit 1
        fi
        export OPENROUTER_API_KEY
        echo "✅ Using API key from this session (not stored in .env)"
    else
        export OPENROUTER_API_KEY
    fi
else
    # DeepSeek (cloud API) — API key is required
    if [ -z "$DEEPSEEK_API_KEY" ] || [ "$DEEPSEEK_API_KEY" = "your-deepseek-api-key-here" ]; then
        echo ""
        echo "DEEPSEEK_API_KEY is required for DeepSeek provider. Enter it now (not saved to disk):"
        read -rs DEEPSEEK_API_KEY
        echo ""
        if [ -z "$DEEPSEEK_API_KEY" ]; then
            echo "❌ Error: DEEPSEEK_API_KEY is empty"
            echo "Create .env from .env.example and set DEEPSEEK_API_KEY, or run this script and enter the key when prompted."
            exit 1
        fi
        export DEEPSEEK_API_KEY
        echo "✅ Using API key from this session (not stored in .env)"
    else
        export DEEPSEEK_API_KEY
    fi
fi

# CHAINLIT_AUTH_SECRET: use .env value or generate for this session (not stored)
if [ -z "$CHAINLIT_AUTH_SECRET" ] || [ "$CHAINLIT_AUTH_SECRET" = "your-secret-here-run-chainlit-create-secret" ]; then
    if command -v openssl &> /dev/null; then
        CHAINLIT_AUTH_SECRET=$(openssl rand -hex 32)
        export CHAINLIT_AUTH_SECRET
        echo "✅ Generated CHAINLIT_AUTH_SECRET for this session (not stored in .env)"
    else
        export CHAINLIT_AUTH_SECRET="default-secret-change-in-production"
        echo "⚠️  Using default CHAINLIT_AUTH_SECRET (install openssl or set in .env for production)"
    fi
else
    export CHAINLIT_AUTH_SECRET
fi

# Find first free port for Chainlit app (8000-8010 fallback)
CHAINLIT_PORT=""
for p in 8000 8001 8002 8003 8004 8005 8006 8007 8008 8009 8010; do
    if ! (echo >/dev/tcp/127.0.0.1/"$p") 2>/dev/null; then
        CHAINLIT_PORT=$p
        break
    fi
done
if [ -z "$CHAINLIT_PORT" ]; then
    echo "❌ Error: All ports 8000-8010 are in use. Cannot start Chainlit app."
    echo "Please free one of these ports or stop another process using them."
    exit 1
fi
export CHAINLIT_PORT
echo "📌 Chainlit App will use port $CHAINLIT_PORT (8000-8010 fallback)"

# Find first free port for MinIO API (9000-9010 fallback)
MINIO_PORT=""
for p in 9000 9001 9002 9003 9004 9005 9006 9007 9008 9009 9010; do
    if ! (echo >/dev/tcp/127.0.0.1/"$p") 2>/dev/null; then
        MINIO_PORT=$p
        break
    fi
done
if [ -z "$MINIO_PORT" ]; then
    echo "❌ Error: All ports 9000-9010 are in use. Cannot start MinIO."
    echo "Please free one of these ports or stop another MinIO/process using them."
    exit 1
fi
export MINIO_PORT

# Find first free port for MinIO console (9001-9010 fallback; must differ from MINIO_PORT)
MINIO_CONSOLE_PORT=""
for p in 9001 9002 9003 9004 9005 9006 9007 9008 9009 9010; do
    [ "$p" = "$MINIO_PORT" ] && continue
    if ! (echo >/dev/tcp/127.0.0.1/"$p") 2>/dev/null; then
        MINIO_CONSOLE_PORT=$p
        break
    fi
done
if [ -z "$MINIO_CONSOLE_PORT" ]; then
    echo "❌ Error: No free port for MinIO console (9001-9010, excluding $MINIO_PORT)."
    echo "Please free one of these ports or stop another MinIO/process using them."
    exit 1
fi
export MINIO_CONSOLE_PORT
echo "📌 MinIO API port $MINIO_PORT, Console port $MINIO_CONSOLE_PORT (9000-9010 fallback)"

# Find first free port for PostgreSQL (5432-5441 fallback)
POSTGRES_PORT=""
for p in 5432 5433 5434 5435 5436 5437 5438 5439 5440 5441; do
    if ! (echo >/dev/tcp/127.0.0.1/"$p") 2>/dev/null; then
        POSTGRES_PORT=$p
        break
    fi
done
if [ -z "$POSTGRES_PORT" ]; then
    echo "❌ Error: All ports 5432-5441 are in use. Cannot start PostgreSQL."
    echo "Please free one of these ports or stop another PostgreSQL/process using them."
    exit 1
fi
export POSTGRES_PORT
echo "📌 PostgreSQL will use port $POSTGRES_PORT (5432-5441 fallback)"
echo ""

# ChEMBL SQLite database (optional)
if [ -z "$CHEMBL_SQLITE_PATH" ]; then
    echo "Enter path to ChEMBL SQLite file on this machine (or press Enter to skip):"
    read -r CHEMBL_SQLITE_PATH
fi

CHEMBL_OVERRIDE=""
if [ -n "$CHEMBL_SQLITE_PATH" ]; then
    if [ ! -f "$CHEMBL_SQLITE_PATH" ]; then
        echo "❌ Error: ChEMBL SQLite file not found: $CHEMBL_SQLITE_PATH"
        exit 1
    fi
    cat > docker-compose.chembl-local.yml <<EOF
version: '3.8'
services:
  chainlit-app:
    environment:
      - CHEMBL_SQLITE_PATH=/app/chembl/chembl.db
    volumes:
      - ${CHEMBL_SQLITE_PATH}:/app/chembl/chembl.db:ro
EOF
    CHEMBL_OVERRIDE="-f docker-compose.chembl-local.yml"
    echo "📌 ChEMBL SQLite: $CHEMBL_SQLITE_PATH → /app/chembl/chembl.db (read-only)"
else
    rm -f docker-compose.chembl-local.yml
    echo "📌 ChEMBL SQLite: not configured (using REST API fallback)"
fi
echo ""

echo "✅ Configuration validated"
echo ""

# Auto-detect GPU and pick compose override
GPU_OVERRIDE=""
if docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi &>/dev/null 2>&1; then
    echo "🖥️  GPU detected — CUDA acceleration enabled"
else
    echo "⚠️  No GPU detected — falling back to CPU mode"
    GPU_OVERRIDE="-f docker-compose.cpu.yml"
fi
echo ""

# Check if running in development or production mode
echo "Select mode:"
echo "1) Production (default)"
echo "2) Development (with hot-reload)"
read -p "Enter choice [1-2]: " mode_choice

# Avoid Compose v1 "recreate" path that can raise KeyError: 'ContainerConfig' with newer Docker Engine
# (see docker/compose#11693). Down then up uses "create" instead of "recreate".
if [ "$mode_choice" = "2" ]; then
    $COMPOSE_CMD -f docker-compose.yml -f docker-compose.dev.yml $CHEMBL_OVERRIDE $GPU_OVERRIDE down --remove-orphans 2>/dev/null || true
else
    $COMPOSE_CMD -f docker-compose.yml $CHEMBL_OVERRIDE $GPU_OVERRIDE down --remove-orphans 2>/dev/null || true
fi

if [ "$mode_choice" = "2" ]; then
    echo ""
    echo "🔧 Starting in DEVELOPMENT mode..."
    echo ""
    $COMPOSE_CMD -f docker-compose.yml -f docker-compose.dev.yml $CHEMBL_OVERRIDE $GPU_OVERRIDE up -d
else
    echo ""
    echo "🚀 Starting in PRODUCTION mode..."
    echo ""
    $COMPOSE_CMD -f docker-compose.yml $CHEMBL_OVERRIDE $GPU_OVERRIDE up -d
fi

echo ""
echo "⏳ Waiting for services to be ready..."
sleep 5

# Check service health
echo ""
echo "📊 Service Status:"
$COMPOSE_CMD ps

echo ""
echo "✅ Cs_copilot is starting up!"
echo ""
echo "🌐 Access Points:"
echo "   - Chainlit App:   http://localhost:${CHAINLIT_PORT:-8000}"
echo "   - MinIO Console:  http://localhost:${MINIO_CONSOLE_PORT:-9001}"
echo "   - PostgreSQL:     localhost:${POSTGRES_PORT:-5432}"
echo ""
echo "📋 Useful Commands:"
echo "   - View logs:      $COMPOSE_CMD logs -f"
echo "   - Stop services:  $COMPOSE_CMD down"
echo "   - Restart:        $COMPOSE_CMD restart"
echo ""
echo "📖 For more information, see docs/getting-started/docker.md"
echo ""
