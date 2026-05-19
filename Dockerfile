# syntax=docker/dockerfile:1.6

# Per-arch base:
#   - amd64: python:3.11-slim — PyPI's linux/x86_64 torch wheel already
#     ships CUDA runtime deps, so the existing path works unchanged.
#   - arm64: nvcr.io/nvidia/pytorch NGC container — ships a CUDA-enabled
#     PyTorch build that supports DGX Spark (GB10 / Blackwell). PyPI's
#     linux/aarch64 torch wheel is CPU-only, which is why torch.cuda was
#     reporting False inside the container on DGX Spark.
# TARGETARCH is declared as a build arg with an amd64 default so a plain
# `docker compose build` on an x86_64 host works without extra flags.
# docker-compose.yml forwards the TARGETARCH env var into this build arg,
# and docker-start.sh auto-detects the host arch and exports it (e.g.
# "arm64" on DGX Spark). BuildKit's auto-populated $TARGETARCH is not
# reliably substituted into FROM without --platform, so we don't rely on it.
ARG TARGETARCH=amd64
FROM python:3.11-slim AS base-amd64
FROM nvcr.io/nvidia/pytorch:25.11-py3 AS base-arm64

FROM base-${TARGETARCH} AS runtime

ARG TARGETARCH

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    UV_SYSTEM_PYTHON=1 \
    UV_PROJECT_ENVIRONMENT=/app/.venv

# System dependencies
# On amd64 (python:3.11-slim) install the full toolchain, domain dev libs,
# and Node 20 (Prisma). On arm64 (NGC PyTorch) the base image already ships
# build-essential/git/python3-dev, so we only add the domain dev libs and
# Node 20 on top.
RUN apt-get update && \
    if [ "$TARGETARCH" = "arm64" ]; then \
        apt-get install -y --no-install-recommends \
            curl \
            libpq-dev \
            libboost-all-dev \
            libcairo2-dev \
            libeigen3-dev && \
        curl -fsSL https://deb.nodesource.com/setup_20.x | bash - && \
        apt-get install -y --no-install-recommends nodejs; \
    else \
        apt-get install -y \
            build-essential \
            git \
            curl \
            libpq-dev \
            libboost-all-dev \
            libcairo2-dev \
            libeigen3-dev && \
        curl -fsSL https://deb.nodesource.com/setup_20.x | bash - && \
        apt-get install -y nodejs; \
    fi && \
    rm -rf /var/lib/apt/lists/*

# On arm64 (NGC base), PyTorch links against UCC at /opt/hpcx/ucc/lib,
# which in turn depends on UCX symbols (libucs) in /opt/hpcx/ucx/lib.
# A stale system libucs.so.0 exists in /lib/aarch64-linux-gnu (registered by
# aarch64-linux-gnu.conf which is processed earlier than "hpcx.conf" in
# lexical order), so ldconfig picks the old one first and `import torch`
# fails with "undefined symbol: ucs_config_doc_nop". We prepend HPC-X by
# using a "000-" prefix so it wins the cache ordering race, without leaking
# paths into amd64 builds.
RUN if [ "$TARGETARCH" = "arm64" ]; then \
        printf '%s\n' \
            '/opt/hpcx/ucx/lib' \
            '/opt/hpcx/ucc/lib' \
            > /etc/ld.so.conf.d/000-hpcx.conf && \
        ldconfig; \
    fi

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

# Dependency metadata (for Docker layer caching)
COPY pyproject.toml uv.lock README.md ./

# On arm64, pre-create the project venv with --system-site-packages so the
# CUDA-enabled torch/torchvision/torchaudio that NGC ships in the system
# Python are importable from inside the venv. uv sync will then reuse this
# existing venv (see UV_PROJECT_ENVIRONMENT) and install the rest of our
# dependencies on top of it. On amd64 we let uv create the venv implicitly.
RUN if [ "$TARGETARCH" = "arm64" ]; then \
        uv venv --system-site-packages /app/.venv; \
    fi

# Install third-party dependencies only (not the local project) so this
# expensive layer is cached independently of source-code changes.
# - arm64: skip torch/torchvision/torchaudio (reuse NGC's CUDA PyTorch) and
#   the SynPlanner family (no aarch64 wheels). SynPlanner is lazy-imported
#   so it is never imported unless a retrosynthesis tool is invoked.
# - amd64: install everything (PyPI torch wheel for linux/x86_64 is CUDA).
RUN if [ "$TARGETARCH" = "arm64" ]; then \
        uv sync --frozen --no-dev --no-install-project \
            --no-install-package torch \
            --no-install-package torchvision \
            --no-install-package torchaudio \
            --no-install-package synplanner \
            --no-install-package cgrtools-stable \
            --no-install-package chython-synplan \
            --no-install-package chytorch-synplan \
            --no-install-package chytorch-rxnmap-synplan; \
    else \
        uv sync --frozen --no-dev --no-install-project; \
    fi

# Application source
COPY . .

# Install the local cs_copilot package into the already-populated venv
RUN if [ "$TARGETARCH" = "arm64" ]; then \
        uv sync --frozen --no-dev \
            --no-install-package torch \
            --no-install-package torchvision \
            --no-install-package torchaudio \
            --no-install-package synplanner \
            --no-install-package cgrtools-stable \
            --no-install-package chython-synplan \
            --no-install-package chytorch-synplan \
            --no-install-package chytorch-rxnmap-synplan; \
    else \
        uv sync --frozen --no-dev; \
    fi

# Prisma / Node dependencies
COPY package.json ./
COPY prisma ./prisma

RUN npm install \
    && npx prisma generate

# Runtime prep
RUN mkdir -p /app/data

# Copy and configure entrypoint script
COPY docker-entrypoint.sh /usr/local/bin/
RUN chmod +x /usr/local/bin/docker-entrypoint.sh

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

ENTRYPOINT ["/usr/local/bin/docker-entrypoint.sh"]
CMD ["uv", "run", "--no-sync", "chainlit", "run", "chainlit_app.py", "--host", "0.0.0.0", "--port", "8000"]
