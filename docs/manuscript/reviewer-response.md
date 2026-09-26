# Response to reviewer — working evidence ledger

This is an author-facing revision record, not a finished response for submission.
No item is closed by a code change alone when the reviewer requests experimental
evidence. Results and page/line locations will be added after the corresponding
artifacts have been inspected.

| Request | Implemented evidence | Remaining acceptance condition |
|---|---|---|
| 1. Quantitative reliability | Isolated repeated-run harness, content-based validators, timing and usage coverage, human-review packets, 48+12 prospective protocol | Verified input snapshots; completed, retained runs; expert review; numerical results table |
| 2. Architecture advantage | Same-model flat-agent baseline and paired evaluation; architecture rationale in revision text | Matched measured comparison; report both costs and success; support or moderate advantage claims |
| 3. sEH and GTM methods | Current-code search ranges/objective/representation audit; pinned model assets | Actual ChEMBL release, query/filter manifest, raw and retained counts, replicate handling, class definitions, final map parameters and criterion for the reported analysis |
| 4. Generation and synthesis | Raw generation audit; validity/uniqueness/properties/membership analysis; per-execution route summaries with error accounting | Real generation outputs; actual training-corpus linkage or explicit limitation; multiple predeclared route targets with all outcomes and expert inspection |
| 5. Reproducibility and presentation | Versioned code commits, inference/environment manifests, prompts, correction text and verified RDKit/Sattarov references | Exact measurement release and archive DOI; complete outputs; apply edits in editable paper; replace and visually inspect figures |

## 1. Reliability

We agree that selected successful demonstrations alone do not establish workflow
reliability. The revision evaluates representative tasks with explicit acceptance
criteria and retains unsuccessful executions. Task fulfillment, tool errors,
scientific outcomes, and runtime/usage are recorded separately. Repeated trials
and equivalent prompt formulations assess repeatability. Independent expert
review assesses factual grounding and inappropriate tool selection.

**Before submission:** insert success numerators/denominators, uncertainty,
repeatability, error frequency, wall time and usage, and the actual supplement
locations. Do not replace these results with unit-test pass counts.

## 2. Multi-agent architecture

The architectural rationale has been rewritten to distinguish scientific
computation from orchestration. Specialist agents restrict domain instructions
and tool schemas, while the coordinator combines workflows. A fixed pipeline
remains appropriate for a fully specified sequence, and a flat tool-calling agent
is a valid alternative. Delegation introduces measurable overhead and can fail.
The same-model comparison is designed to test these tradeoffs rather than assume
that multiple agents improve chemical computations.

**Before submission:** insert the measured paired comparison and revise any
advantage statement to match its direction, uncertainty, and scope.

## 3. Dataset construction and GTM

The revision will separate historical case-study provenance from newly measured
analyses. The supplement must specify ChEMBL release, target and assay filters,
endpoints/units/relations, exclusions, standardization, aggregation of repeat
measurements, and the exact class definitions. Distinct biological endpoints
must not silently become equivalent measurements through pooling. For GTM,
report representation, model provenance, actual parameter ranges, chosen
parameters, and selection criterion. The current optimization code maximizes
normalized node-occupancy entropy; its three effort settings are search
strategies, not three objectives.

**Before submission:** recover or regenerate a completely specified sEH analysis;
current defaults cannot retrospectively establish the original methods.

## 4. Generation and retrosynthesis

The revision measures raw-output validity and standardized uniqueness, parent
similarity, and physicochemical properties. Training-set novelty requires a
traceable training corpus and identical structure processing. Archived filtered
sets cannot establish raw validity or uniqueness. Route searches will cover a
predeclared set of distinct candidates with fixed budgets, retaining no-route
outcomes and exceptions. Route lengths and scores will be reported as
computational predictions, with experimental activity and synthetic feasibility
claims removed where unsupported.

**Before submission:** insert real counts and distributions, corpus/checkpoint
provenance, the complete target denominator, and inspected route figures.

## 5. Reproducibility and presentation

The revision package will identify the tested Git commit, exact provider model
identifier, inference settings, installed software, checkpoint/data hashes, and
prompts and outputs. The proposed manuscript edits correct the case-study count,
GTM objective description, Sattarov and RDKit citations, and the specified
typographical errors. Scientific figures must be regenerated from their source
data and inspected at their final display size.

**Before submission:** apply and verify changes in the complete editable paper,
archive the release and scientific artifacts, insert its version DOI, and provide
final page/line references. A code commit alone is not an archival data release.
