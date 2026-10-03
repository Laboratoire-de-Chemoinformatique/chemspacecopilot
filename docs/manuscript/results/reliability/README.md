# Measured reliability and architecture study (v2)

The predeclared 48+12 protocol in [`manuscript-revision.md`](../../../testing/manuscript-revision.md)
was executed on 2026-09-30. All 60 planned case/stage executions were attempted
and **all 60 are retained**, including failures. No run was repeated until it
succeeded, and no result was substituted.

The plan was hash-verified immediately before and after execution
(`plan_revision_reliability.py --verify-plan` → `verified: true, mismatches: []`),
so these runs used exactly the frozen runtime, inputs and configurations.

| Item | Value |
|---|---|
| Provider / model alias | DeepSeek API, `deepseek-flash` |
| Resolved model name | DeepSeek-V4.1-Flash (see [provider_identity.json](provider_identity.json)) |
| Inference settings | `temperature=0.0`, `max_tokens=8192` |
| Execution date (UTC) | 2026-09-30 |
| Ambient scientific seeds | 11 / 22 / 33 by repetition; peptide sampler `random_state=42` |
| Hardware | aarch64, 20 CPU, CPU-only Torch, `OMP_NUM_THREADS=1` |
| Batch wall time | 180.4 min across 9 batches |

## Reviewer 1 — reliability

| Metric | Value |
|---|---|
| Task fulfilment | **40/60 = 66.7%** (95% Wilson CI 54.1–77.3%) |
| Frozen tier | 31/48 = 64.6% |
| Live tier | 9/12 = 75.0% |
| Completed without an agent exception | 80.0% |
| Tool calls observed | 2,202, of which **499 failed (22.7%)** |
| Repeatability | 11/16 cells unanimous across 3 repetitions |
| Median wall time | 80.3 s frozen, 369.9 s live |
| Total tokens (48 runs with complete metrics) | 45,747,473, of which 87% were cache reads |

Task fulfilment, completion without tool errors, and scientific outcomes are
recorded separately. A truthful completed search that returns no route can still
fulfil an orchestration task.

**Telemetry coverage is 48/60.** The 12 runs without complete metrics are
reported as unavailable and excluded from the efficiency statistics; they are
never counted as zero.

## Reviewer 2 — architecture

Matched frozen-tier comparison, same model instance, tools, budgets and inputs,
with arm order counterbalanced across batches (n = 24 per arm):

| | team (multi-agent) | single agent (flat) |
|---|---|---|
| Task fulfilment | 15/24 = 62.5% | 16/24 = 66.7% |
| Completed without agent exception | 18/24 | 21/24 |
| Failed tool calls | 164/846 = 19.4% | 67/547 = 12.2% |
| Median wall time | 124.3 s | 58.3 s |
| Median total tokens | 743,904 | 547,617 |

Paired on identical (case, phrasing, repetition) cells: 13 both succeeded,
2 team-only, 3 single-agent-only, 6 neither. With **5 discordant pairs** an exact
test has no useful power, so the direction is descriptive only.

Per paired cell the team arm cost a median **+44.0 s** and **+107,576 tokens**.

**Conclusion:** on these four tasks the multi-agent architecture showed **no
measurable accuracy advantage** over a flat tool-calling agent using the same
model, while costing measurably more time, tokens and failed tool calls. Any
advantage claim in the manuscript should be moderated to match. This is a
small-sample exploratory tradeoff, not evidence that either architecture is
superior in general.

## A dominant, reproducible defect

`frozen_case_1_seh_analysis` failed **0/12** — in both arms — while the same
analysis succeeded **2/3** in the live tier. The frozen failures are dominated by
one reproducible crash, confirmed by an isolated diagnostic run with traceback
capture:

Agno 2.1.9 evaluates `str(result) if result else ""` for every tool result
(`agno/models/base.py`, in `run_function_call`). A pandas DataFrame raises
`ValueError: The truth value of a DataFrame is ambiguous`, aborting a run the
agent had already largely completed. It reached that line through
`delegate_task_to_member`, which is why it hit the team arm more often.

This is a defect in the system under test and its pinned dependency, **not in the
benchmark harness**, so these runs are valid measurements of the software as the
manuscript describes it. The fix (coercing Agno tool results the way the MCP
runtime already does) changes the system and therefore belongs to a separately
declared v3 measurement, not to this one.

## Limitations

- Automated artifact validators judge task fulfilment. This is **not**
  independent chemical or factual expert review, and no expert incorrect-tool-
  choice rate is available.
- Token counts are provider/Agno usage records, not a cost estimate. Team totals
  include delegation overhead by construction.
- No provider seed exists; repeated runs are not claimed to be deterministic, and
  the alias may be served by a different snapshot later.
- The three sEH stages within a live workflow share history and state, so the 12
  live executions are not 12 independent observations.
- `by_arm_pooled_across_tiers` in [reliability_summary.json](reliability_summary.json)
  pools tiers, and the live tier is team-only, so it is not a matched comparison.
  Use `by_tier_and_arm` or `architecture_paired`.

## Reproducing

```bash
python scripts/plan_revision_reliability.py --verify-plan \
  reports/reviewer_revision/reliability_study_frozen_v2/study_plan.json
python scripts/execute_revision_study.py \
  reports/reviewer_revision/reliability_study_frozen_v2/study_plan.json
python scripts/summarize_reliability_study.py \
  reports/reviewer_revision/reliability_study_frozen_v2 --output-dir <out>
```

Per-execution records are in [runs.csv](runs.csv); the attempt ledger, with each
batch's exit code and wall time, is in
[execution_ledger.jsonl](execution_ledger.jsonl).

---

# Measured effect of the fixes (v3)

The same predeclared protocol was executed again on 2026-10-02 against a fixed
runtime (commit `f69c62b`), under an independently verified plan
(`v3/study_plan.json`). Protocol, model, inputs, seeds and counterbalanced arm
order are unchanged; the v2 plan was left byte-identical and still verifies, so
the two are separately auditable rather than one overwriting the other.

| | v2 | v3 |
|---|---:|---:|
| Task fulfilment (60 runs) | 40/60 = 66.7% | **45/60 = 75.0%** |
| 95% Wilson CI | 54.1–77.3% | 62.8–84.2% |
| Completed without an agent exception | 80.0% | **100.0%** |
| Agent exceptions | 10 | **0** |
| Failed tool calls | 22.7% | **10.6%** |
| Telemetry coverage | 48/60 | **60/60** |
| Frozen tier | 31/48 | 34/48 |
| Live tier | 9/12 | 11/12 |

Per case, both arms pooled:

| Case | v2 | v3 |
|---|---:|---:|
| 1 sEH analysis | 0/12 | 3/12 |
| 2 generation | 9/12 | 7/12 |
| 3 retrosynthesis | 11/12 | **12/12** |
| 4 peptide design | 11/12 | **12/12** |

**Case 2 moved the wrong way, and it is not a regression from these changes.**
Its failures in both versions are the same check with the same cause: the task
asks for 10 analogues and a generation call returns one or two valid
structures, matching the generator's measured 29% validity and 26% uniqueness.
That mismatch was deliberately left unfixed, no provider seed exists, and the
case sits on the boundary, so movement within it is expected.

The intervals overlap, so the headline difference is suggestive rather than
established. The unambiguous results are the ones that are not rate estimates:
**no run aborted**, telemetry became complete for every run, and the failed
tool-call rate halved.

## Attribution

Measurement fixes and system fixes are separable because the retained v2 runs
were re-scored offline with the corrected validators, costing no model calls:

| | task fulfilment |
|---|---:|
| v2 as measured (51-run re-scorable subset) | 34/51 |
| v2 replayed, original validators (control) | 36/51 |
| v2 re-scored, corrected validators | 39/51 |

The control is not a no-op: two runs pass on replay because artifacts on disk
changed after they were judged. That drift belongs to replay, not to the
validator fix, which accounts for the remaining three.

## Architecture, re-measured

Matched frozen tier, n = 24 per arm, with complete telemetry for every run in
both arms this time (v2 had 18/24 and 21/24):

| | team | single agent |
|---|---:|---:|
| Task fulfilment | 18/24 = 75.0% | 16/24 = 66.7% |
| Completed without agent exception | 24/24 | 24/24 |
| Failed tool calls | 10.9% | 8.2% |

Paired on identical cells: 14 both, 4 team-only, 2 single-agent-only, 4 neither.
With 6 discordant pairs this remains underpowered. The direction reversed
between v2 and v3, which is itself a reason to treat the architecture
comparison as exploratory rather than as a result. The durable finding is the
cost: the team arm spends a median **+87 s** and **+141,000 tokens** per
matched cell.

## What the v2 numbers were measuring

Of 17 frozen-tier v2 failures, 9 were one dependency defect, 4 were a validator
demanding evidence of a stage the configuration never runs, and 4 were genuine
and unfixed. The v2 figure therefore understated the software. Both versions
are reported because the manuscript describes the v2 software; v3 measures what
the corrected software does.

## Retained but excluded

`reports/reviewer_revision/voided_v3_attempt_01_broken_pythonpath/` holds a
first v3 attempt whose generated plan omitted the vendored dependency path, so
SynPlanner could not be imported and every route search failed for a reason
unrelated to the measurement. It is kept rather than deleted and forms no part
of any denominator.

---

# Confirming the corrected check (v4)

The v3 analysis identified one check as the dominant remaining cause of failure
and predicted that correcting it would recover nine runs. That prediction was
testable, so the frozen tier was executed a third time on 2026-10-02 against
commit `f540b9f`, under an independently verified plan (`v4/study_plan.json`).
Protocol, model, inputs, seeds and counterbalanced order are unchanged, and the
v2 and v3 plans remain byte-identical and still verify.

Only the frozen tier was re-run. The corrected check cannot affect the live
tier, whose sEH analysis already passed 3/3, so re-measuring it would have cost
an hour for no new information. Results are reported per tier rather than mixed
into a single denominator across runtimes.

| Frozen tier, 48 runs each | v2 | v3 | v4 |
|---|---:|---:|---:|
| Task fulfilment | 31/48 = 64.6% | 34/48 = 70.8% | **43/48 = 89.6%** |
| 95% Wilson CI | 49.5–77.0% | 56.1–82.2% | 77.8–95.5% |
| Completed without an agent exception | 81.2% | 100% | 100% |
| Failed tool calls | 16.6% | 10.1% | 10.1% |
| Telemetry coverage | 39/48 | 48/48 | 48/48 |

Per case, both arms pooled:

| Case | v2 | v3 | v4 |
|---|---:|---:|---:|
| 1 sEH analysis | 0/12 | 3/12 | **11/12** |
| 2 generation | 9/12 | 7/12 | 8/12 |
| 3 retrosynthesis | 11/12 | 12/12 | 12/12 |
| 4 peptide design | 11/12 | 12/12 | 12/12 |

Case 1 moved as predicted. The v4 interval no longer overlaps the v2 interval.

## Architecture

| Frozen, n = 24 per arm | team | single agent |
|---|---:|---:|
| v2 | 15/24 | 16/24 |
| v3 | 18/24 | 16/24 |
| v4 | **24/24** | 19/24 |

In v4 the team arm strictly dominates: of 24 paired cells, 19 both succeeded,
5 team-only, and **no cell where the flat agent succeeded and the team did
not**. An exact two-sided sign test on five concordant discordant pairs gives
**p = 0.0625** -- not significant, and the first run in which the direction is
consistent.

The gap is almost entirely one case. On generation the team scores 6/6 against
the flat agent's 2/6, and that is the task whose tool has poor yield: a call
returns one or two valid structures against a required ten. The specialist
agent persists and accumulates across calls; the flat agent does not. Where a
multi-agent advantage appears here, it is on recovery from a weak tool rather
than on the tasks that succeed first time.

This should still be read as exploratory. The direction reversed across v2, v3
and v4, no provider seed exists, and 24 paired cells cannot separate a real
effect of this size from run-to-run variation. The durable finding is the cost:
a median **+71 s** and **+134,000 tokens** per matched cell.

## What still fails

All five remaining failures are in the flat arm. Four are the generation
yield mismatch, which was deliberately left unfixed and is the only remaining
failure mode attributable to the chemistry rather than to instrumentation. The
fifth is one sEH analysis run that failed broadly rather than on a single
check.

## Reading these three versions together

v2 measures the software as the manuscript describes it. v3 and v4 measure
corrected software, and the correction between them was to the measuring
instrument, not to the agents. The honest summary is that the v2 figure
understated the software twice over: once through a dependency defect that
aborted completed runs, and once through a check that could not pass in the
configuration being tested.
