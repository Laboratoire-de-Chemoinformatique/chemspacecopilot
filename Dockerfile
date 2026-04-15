FROM python:3.11-slim

ARG TARGETARCH

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    UV_SYSTEM_PYTHON=1

# System dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    curl \
    libpq-dev \
    libboost-all-dev \
    libcairo2-dev \
    libeigen3-dev \
    && curl -fsSL https://deb.nodesource.com/setup_20.x | bash - \
    && apt-get install -y nodejs \
    && rm -rf /var/lib/apt/lists/*

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

# Dependency metadata (for Docker layer caching)
COPY pyproject.toml uv.lock README.md ./

# Install third-party dependencies only (not the local project) so this
# expensive layer is cached independently of source-code changes.
# The SynPlanner family has no aarch64 Linux wheels; skip them only on arm64.
# SynPlanner is lazy-loaded so it is never imported unless a retrosynthesis
# tool is explicitly invoked.
RUN if [ "$TARGETARCH" = "arm64" ]; then \
        uv sync --frozen --no-dev --no-install-project \
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
