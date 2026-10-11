# Reporting evidence implementation validation

Validated on 2026-10-05 in the current worktree. No live agent benchmark was rerun and no original scores were changed.

## Automated checks

- PR preparation after rebasing onto the latest `main`: **1,568 passed, 2 skipped**; CSV output uses LF line endings and all 82 original hashes were reverified.
- Complete offline unit suite before that rebase: **1,565 passed, 2 skipped**, 3 existing warnings. Command: `PYTHONPATH=src python -m pytest tests/unit`.
- Ruff checks on all changed Python files and `git diff --check`: passed.
- All **82 original source hashes** in `reconciliation.json` were independently rechecked after implementation; none changed.
- Independent reconciliation covers all six historical sEH runs, with **32 correction ledger entries**. See [historical corrections](README.md).
- Frozen deterministic reporting replay matches every whole-set scaffold frequency and coverage total against that independent reconciliation. It uses the registered team/single-agent evidence readers and shared report renderer, without inference. Report and source hashes are recorded in [reporting_replay.json](reporting_replay.json).

Reproduce the reporting replay after running the historical reconciliation command in README.md:

```sh
PYTHONPATH=src python scripts/replay_reporting_evidence.py \
  --corrections-dir docs/manuscript/results/reporting-corrections \
  --output-dir reports/reporting_evidence_replay
```

Generated full evidence and six Markdown reports remain under that output directory. Source datasets must be available at the original recorded locations. The replay constructs the actual factory tool configurations, so the normal project environment and cached model assets are needed; it does not invoke the models.

## Context footprint

Measured UTF-8 JSON sizes across the six frozen runs (these are bytes, not token estimates):

| Payload | Range |
| --- | --- |
| Session evidence reference | at most 247 bytes |
| Coverage response | 1,137–1,162 bytes |
| First ten scaffold rows, including full-population totals | 1,531–1,594 bytes |
| Complete evidence artifact, kept outside agent context | 55,255–109,592 bytes |

Each replay makes two evidence-tool calls: coverage and the first scaffold page. There are no repeated evidence-tool reads. The renderer separately revalidates source hashes when producing the report. Detail pagination is capped at 50 rows; totals are calculated before pagination. The reader does not append results to session state.

## Behavioral checks and review

Regression checks exercise atomic ID unions, reordered combinations, missing IDs, conflicting aliases, contributing/clean disagreement, preservation of identifier aliases during cleaning, raw persistence disabled, malformed evidence containers/sections, source changes, selected populations, pagination, report fallback, and identical team/single/MCP reader behavior. Retrieval fixtures include search-matched assays with no activities and activities whose structures are excluded during standardization.

A fresh code review found three issues: alias loss could hide an upstream conflict; `save_raw=False` could fail hashing an empty path; malformed evidence could abort report export. All were reproduced in failing regression tests, fixed, and covered by the passing final suite. Additional malformed-section tests ensure invalid evidence never renders as a verified table.

A bounded instruction-adherence probe used repetition 2's evidence with the historical misleading counts. The first draft correctly replaced 167/129 with 156/76 and marked unavailable region/search evidence unresolved, but overstated the scope of missing scaffold evidence. The reporting instruction now explicitly distinguishes missing regional evidence from available whole-set evidence. A follow-up retrieved only the clean scaffold page, reported 90/1,580 exact adamantane assignments and 635 whole-set categories, and retained unresolved status for the unverified regional count. Four cumulative evidence-reader calls were used, without full-JSON, raw-CSV or identifier-list context reads. This is one instructional check, not a statistical evaluation of LLM reliability.

## Verification limits

The broad `python -m pytest` command includes unmarked live-model robustness calls. It was stopped after encountering unrelated existing failures and entering a provider-dependent ChEMBL test. Isolated diagnostics reproduce:

- `TestAutoencoderRobustness.test_basic_sampling_robustness`: fixture imports `DeepSeekChat` from the top-level `agno` package, which fails in the installed environment. The same fixture also errored for `test_gtm_guided_sampling_robustness`, `test_interpolation_robustness`, `test_latent_exploration_robustness`, and `test_molecular_validity` in the broad run.
- `test_autoencoder_prompt_variations_valid`: variation 3 of `autoencoder_sampling` fails the pre-existing semantic-similarity threshold.

These failures are outside the reporting changes; the broader suite is not claimed green. Unit-suite warnings concern an existing unregistered integration marker, installed GPU capability support, and a test returning a boolean.

Historical search-matched identifier lists and some regional memberships were not retained. Those claims remain explicitly unresolved. Artifact consistency does not establish experimental assay validity, potency, or synthetic feasibility.
