# Scoped reporting evidence implementation

Approved design: the six-step plan in this chat, including selective evidence retrieval.

- [x] Reconcile six historical reports against frozen sources; retain originals and record corrections, scope, hashes, and conclusion implications.
- [x] Compute coverage at search, retrieved, retained, standardized-contributor, and clean stages using atomic identifiers.
- [x] Attach dataset identity, selection, units, denominator and exact scaffold definition to frequencies.
- [x] Store complete evidence in artifacts; carry only references in session state; expose bounded selective retrieval shared by both reporting modes and MCP.
- [x] Require scoped supported claims; render coverage from evidence, explicitly label unavailable/stale/conflicting results.
- [x] Verify counting edge cases, source invalidation, selection/pagination, both reporting paths, frozen replay and context size.

Implementation ledger:
- Existing isolated app worktree verified; baseline 120 targeted tests passed.
- Historical correction work delegated independently; runtime implementation remains inline.
- Use existing storage/session and report machinery, no new dependencies or external model calls.
- Original audit data are outside this worktree; generated correction addenda belong in this worktree, originals remain immutable.

- Review fixes: supported ID aliases survive merging; unresolved contributing conflicts propagate to clean; save_raw=False omits unverifiable retained population; invalid JSON containers/sections render unresolved. Reproduced failures before fixes, then passing targeted tests.
- Deterministic frozen replay compares all scaffold frequencies and coverage against independent reconciliation, using each arm's registered reader. Full artifacts remain off-context; measured reference/response sizes are in reporting_replay.json.
- Historical limitations are explicit unresolved claims, not guessed repairs: absent original search ID list and incomplete regional membership.
- Final review rulings: experimental assay validity is outside reporting scope; live LLM benchmark reruns are not represented by deterministic replay. A separate bounded instruction-adherence probe checks scoped claims and selective reads; it is not a benchmark.
- The broad pytest command includes unmarked live robustness calls. Stopped that run; isolated its existing DeepSeekChat import error and autoencoder prompt-similarity failure. The complete unit suite is the offline regression gate.

Final verification: 1,565 unit tests passed, 2 skipped; Ruff and diff checks clean; six deterministic report replays matched independent counts; 82 original source hashes unchanged. Validation and measured context sizes: docs/manuscript/results/reporting-corrections/runtime-validation.md. Implementation verified; pull request requested on 2026-10-05.
