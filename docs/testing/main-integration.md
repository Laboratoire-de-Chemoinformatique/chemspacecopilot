# Main integration validation — 2026-09-28

The integration branch combines main `cf1791256e830a28463465d616eeb156af94d07e`
and manuscript revision `351fb534cdf82ea811dd8fa3cfe99fdc358890c8` with a normal
merge. [PR #122](https://github.com/Laboratoire-de-Chemoinformatique/chemspacecopilot/pull/122)
contains the resulting source and CI history.

## Installation and runtime

The core was installed from `uv.lock` into a fresh Python 3.11 environment on
Linux ARM64, without the revision branch's supplemental dependency directory.
Deepchemography is pinned to the previously locked Git commit. Retrosynthesis
is an optional `uv sync --frozen --extra retrosynthesis` profile using
[SynPlanner 1.7.0](https://pypi.org/project/SynPlanner/1.7.0/). Its
`chytorch-synplan` dependency has no Linux ARM64 wheel; the complete profile is
validated on Linux x86_64 in CI. macOS ARM64 has published wheels but was not
executed here.

SynPlanner uses its current policy factory and chython TSV assets from the GPS
preset. The default preset revision is pinned in
`src/cs_copilot/tools/chemistry/synplanner_assets.py`. Explicit incomplete local
presets fail without silently downloading different assets. Historical
CGRtools-based studies remain attached to their original manifests and commits.

## Measured checks

- Fresh core environment: **1,196 unit tests passed, 2 skipped**, plus formatting
  and lint. The two pre-existing warnings concern an unregistered integration
  marker and a test returning a boolean.
- MCP: all **6 integration tests passed**, including stdio and HTTP sessions;
  readiness discovered 114 tools.
- Distribution build and audit: wheel catalogs, source files, both CPU/GPU Compose
  overrides, and the MinIO build recipe are included; private runtime paths are
  excluded. CI also checks an installed wheel in a fresh environment.
- Local Chainlit: authentication configuration, rejected invalid password,
  successful login, and authenticated user endpoint checked without LLM calls.
- Native ARM CPU Docker: image build, UI login, and healthy startup checked.
  Full Compose also completed PostgreSQL migrations and MinIO bucket setup, and
  passed an application-container S3 write/read round trip. Test ports were bound
  to loopback and test volumes were isolated under a separate Compose project.
- SynPlanner 1.7.0: the Linux x86 CI smoke generated three one-step aspirin routes
  (score 0.375) in about 18.5 seconds including loading. SVG and PNG files were
  verified. Rendering now preserves atom-label masks using the existing resvg
  dependency; a pixel regression check detects bonds showing through labels.
- Real offline scientific assets: molecular decoding produced all 20 requested
  raw outputs (5 valid, 2 distinct); the peptide workflow decoded 40 sequences and
  returned 10, including positional analysis, a sequence logo, and a report.
  The pinned GTM checkpoint loaded on CPU and produced finite 900-node
  responsibilities. These are software checks, not manuscript benchmark runs.

## Docker changes

The default image is native CPU Python on amd64 and arm64. The NGC CUDA build
remains an explicit profile and was not validated in this integration. MinIO's
configured registry images failed to pull, so `Dockerfile.minio` builds pinned
official server/client source releases; their licenses and source revisions are
retained. Initial builds need GitHub and Go module registry access. Both source
builds and the resulting storage service were tested on ARM64.

## Prospective paper evaluation

The earlier frozen plan belongs to the revision runtime. Regenerate it with
`scripts/plan_revision_reliability.py` against this integration and the new GPS
preset before collecting new results. The manifest now includes `pyproject.toml`
and `uv.lock` hashes and uses the normal project environment. Preflight recognizes
modern assets and respects an explicitly selected local directory.

A newly prepared plan is not evidence of execution. The full 48+12 API evaluation
still needs the previously requested provider authorization and a supported
retrosynthesis runtime. Original measured results and their immutable source
snapshots remain unchanged.
