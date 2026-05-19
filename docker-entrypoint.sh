#!/bin/bash
# Docker entrypoint script for Cs_copilot
# Automatically generates and persists CHAINLIT_AUTH_SECRET

set -e

echo "🧪 Cs_copilot Container Starting..."

# On the arm64 (NGC PyTorch) base image, libucc.so.1 (needed by torch)
# depends on a newer libucs.so.0 that lives in /opt/hpcx/ucx/lib. A stale
# system libucs.so.0 in /lib/aarch64-linux-gnu gets loaded first on some
# import paths and causes:
#   ImportError: libucc.so.1: undefined symbol: ucs_config_doc_nop
# Prepend the HPC-X paths to LD_LIBRARY_PATH when they exist so the fresh
# libucs is picked up. On amd64 (python:3.11-slim), /opt/hpcx does not
# exist and the block is skipped.
if [ -d /opt/hpcx/ucx/lib ] && [ -d /opt/hpcx/ucc/lib ]; then
    export LD_LIBRARY_PATH="/opt/hpcx/ucx/lib:/opt/hpcx/ucc/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi

# Path to persisted secret in mounted volume
SECRET_FILE="/app/data/.chainlit_secret"

# Ensure required directories exist
mkdir -p /app/data
mkdir -p /app/.files   # Chainlit stages uploaded files here before app code processes them

# Priority order for CHAINLIT_AUTH_SECRET:
# 1. Use if already set (from .env or docker-start.sh)
# 2. Load from persisted file if exists
# 3. Generate new secret and persist it

if [ -n "$CHAINLIT_AUTH_SECRET" ] && [ "$CHAINLIT_AUTH_SECRET" != "default-secret-change-in-production" ]; then
    echo "✅ Using CHAINLIT_AUTH_SECRET from environment"
elif [ -f "$SECRET_FILE" ]; then
    echo "✅ Loading CHAINLIT_AUTH_SECRET from persisted file: $SECRET_FILE"
    export CHAINLIT_AUTH_SECRET=$(cat "$SECRET_FILE")
    if [ -z "$CHAINLIT_AUTH_SECRET" ]; then
        echo "⚠️  Warning: Secret file exists but is empty, generating new secret"
        export CHAINLIT_AUTH_SECRET=$(openssl rand -hex 32)
        echo "$CHAINLIT_AUTH_SECRET" > "$SECRET_FILE"
        chmod 600 "$SECRET_FILE"
        echo "✅ Generated and persisted new CHAINLIT_AUTH_SECRET"
    fi
else
    echo "🔐 Generating new CHAINLIT_AUTH_SECRET..."
    export CHAINLIT_AUTH_SECRET=$(openssl rand -hex 32)
    echo "$CHAINLIT_AUTH_SECRET" > "$SECRET_FILE"
    chmod 600 "$SECRET_FILE"
    echo "✅ Generated and persisted CHAINLIT_AUTH_SECRET to: $SECRET_FILE"
fi

# Verify secret is set
if [ -z "$CHAINLIT_AUTH_SECRET" ]; then
    echo "❌ Error: CHAINLIT_AUTH_SECRET is not set"
    exit 1
fi

echo "🚀 Starting Chainlit application..."
echo ""

# Execute the CMD from Dockerfile (or command passed to docker run)
exec "$@"
