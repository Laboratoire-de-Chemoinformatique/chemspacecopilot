# syntax=docker/dockerfile:1.6

# Native CPU image on both amd64 and arm64. DGX users can opt into NGC
# with BASE_IMAGE=nvcr.io/nvidia/pytorch:25.11-py3 and USE_SYSTEM_TORCH=true.
ARG BASE_IMAGE=python:3.11-slim
FROM ${BASE_IMAGE} AS runtime
ARG USE_SYSTEM_TORCH=false
ARG INSTALL_RETROSYNTHESIS=false

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    UV_SYSTEM_PYTHON=1 \
    UV_PROJECT_ENVIRONMENT=/app/.venv

# Build tools and libraries used by the scientific stack and Prisma.
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential git curl libpq-dev libcairo2-dev libeigen3-dev && \
    curl -fsSL https://deb.nodesource.com/setup_20.x | bash - && \
    apt-get install -y --no-install-recommends nodejs && \
    rm -rf /var/lib/apt/lists/*

# On arm64 (NGC base), PyTorch links against UCC at /opt/hpcx/ucc/lib,
# which in turn depends on UCX symbols (libucs) in /opt/hpcx/ucx/lib.
# A stale system libucs.so.0 exists in /lib/aarch64-linux-gnu (registered by
# aarch64-linux-gnu.conf which is processed earlier than "hpcx.conf" in
# lexical order), so ldconfig picks the old one first and `import torch`
# fails with "undefined symbol: ucs_config_doc_nop". We prepend HPC-X by
# using a "000-" prefix so it wins the cache ordering race, without leaking
# paths into amd64 builds.
RUN if [ "$USE_SYSTEM_TORCH" = "true" ]; then \
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
RUN if [ "$USE_SYSTEM_TORCH" = "true" ]; then \
        uv venv --system-site-packages /app/.venv; \
    fi

# Install dependencies first for Docker layer caching. The optional NGC
# profile reuses its CUDA PyTorch; the default installs the locked packages.
RUN set --; \
    if [ "$USE_SYSTEM_TORCH" = "true" ]; then \
        set -- --no-install-package torch --no-install-package torchvision --no-install-package torchaudio; \
    fi; \
    if [ "$INSTALL_RETROSYNTHESIS" = "true" ]; then set -- "$@" --extra retrosynthesis; fi; \
    uv sync --frozen --no-dev --no-install-project "$@"

COPY . .

RUN set --; \
    if [ "$USE_SYSTEM_TORCH" = "true" ]; then \
        set -- --no-install-package torch --no-install-package torchvision --no-install-package torchaudio; \
    fi; \
    if [ "$INSTALL_RETROSYNTHESIS" = "true" ]; then set -- "$@" --extra retrosynthesis; fi; \
    uv sync --frozen --no-dev "$@"

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
