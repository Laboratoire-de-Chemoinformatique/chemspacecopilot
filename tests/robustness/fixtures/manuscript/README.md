# Manuscript reliability fixtures

The frozen benchmark tier consumes JSON snapshots of `session_state`. The files
are intentionally not committed with fabricated or machine-specific scientific
artifacts. Build them from a successful, manually verified run, copy the referenced
artifacts to a stable location, and record both the JSON file and its SHA-256 digest.

Every reliability execution writes a redacted `session_state.json` and its digest
to the per-run metadata. A sequential live sEH run therefore captures a candidate
snapshot after each case boundary. Inspect each snapshot and its referenced files
before promoting it to a frozen fixture; stringified non-pointer Python objects are
a sign that the state must be cleaned before reuse.

Each JSON file may be either the session-state object itself or:

```json
{
  "session_state": {
    "data_file_paths": {},
    "session_objects": {}
  }
}
```

Use absolute artifact paths or stable S3 URIs. Do not store API keys, credentials,
raw model tokens, or patient/proprietary data in a fixture.

The benchmark configuration expects:

| Environment variable | Snapshot boundary |
|---|---|
| `CS_COPILOT_SEH_INPUT_FIXTURE` | Curated sEH compounds ready for Case 1 |
| `CS_COPILOT_SEH_ANALYSIS_FIXTURE` | Verified Case 1 GTM/maps/analysis state |
| `CS_COPILOT_SEH_CANDIDATES_FIXTURE` | Verified Case 2 candidate-set state |
| `CS_COPILOT_PEPTIDE_INPUT_FIXTURE` | DBAASP/WAE/GTM inputs for Case 4 |

Set the matching `*_SHA256` variables for every file. Required fixtures fail
closed: a missing path, unresolved variable, malformed JSON, or checksum mismatch
is counted as a benchmark failure and never triggers a live-data fallback.

Example:

```bash
export CS_COPILOT_SEH_INPUT_FIXTURE=/absolute/path/seh_input.json
export CS_COPILOT_SEH_INPUT_SHA256="$(sha256sum "$CS_COPILOT_SEH_INPUT_FIXTURE" | cut -d' ' -f1)"
```

For a manuscript release, archive these fixtures and their referenced artifacts,
assign a persistent DOI, and report the archive version alongside the Git commit.
