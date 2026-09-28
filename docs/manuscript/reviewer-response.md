# Response to reviewer — working evidence ledger

This is an author-facing revision record, not a finished response for submission.
No item is closed by a code change alone when the reviewer requests experimental
evidence. Results and page/line locations will be added after the corresponding
artifacts have been inspected.

| Request | Implemented evidence | Remaining acceptance condition |
|---|---|---|
| 1. Quantitative reliability | Isolated repeated-run harness, content-based validators, timing and usage coverage, human-review packets, 48+12 prospective protocol | Verified input snapshots; completed, retained runs; expert review; numerical results table |
| 2. Architecture advantage | Same-model flat-agent baseline and paired evaluation; architecture rationale in revision text | Matched measured comparison; report both costs and success; support or moderate advantage claims |
| 3. sEH and GTM methods | ChEMBL 37 extraction with exact query/filter manifest; 2,212 curated structures; raw/output hashes and identical offline replay; saved GTM parameter audit; finite projections, occupancy diagnostics and vector maps | Apply the completed prospective methods/results to the editable paper; do not attribute them to unrecovered historical runs |
| 4. Generation and synthesis | Three measured batches (300 raw outputs, 87 valid, 23 unique), exact seed repeat, full properties, MOSES reference comparison; 4/10 predicted routes with all outcomes and corrected vector figures | Preserve unverified training-linkage and chemical-feasibility limits; author/expert review; integrate results into paper |
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

We prepared a fully specified prospective replacement dataset from ChEMBL 37.
The human EPHX2 target CHEMBL2409 yielded 5,243 raw activities; confidence-9
binding/functional assays and uncensored positive IC50 measurements in nM were
retained after quality exclusions. A phosphatase-specific assay was excluded
because EPHX2 is bifunctional. After structure standardization and within-assay
duplicate collapse, 2,490 measurements represent 2,212 compounds: 1,794 active,
158 inactive and 260 intermediate under the declared median-pIC50 thresholds.
Mixed measurements are flagged and retained. Reprocessing the saved response
pages reproduces all four CSV hashes exactly. Full methods, source hashes and
exclusion evidence are in [the sEH manifest](seh-dataset.lock.json) and
[replacement methods](revision-text.md).

The molecular-generation parent has a Ki measurement in this extraction and is
therefore excluded from the IC50-only dataset. We now distinguish its verified
identity and use as a design seed from membership in the curated dataset.
For GTM, the saved checkpoint parameters have been inspected directly: the
molecular map contains 900 nodes, 225 basis functions, width 1, regularization
100 and a 256-dimensional input, despite different values in its filename.
All 2,212 compounds project successfully; 293 nodes receive a
maximum-responsibility assignment, and normalized occupancy entropy is 0.6747.
The continuous pIC50 landscape includes intermediate compounds, while binary
analyses must exclude their missing labels. The [saved map tables](results/seh/README.md)
and vector figures make the projection and display conventions reproducible.
This reused model was not selected by a new optimization. The current
optimization code maximizes normalized node-occupancy entropy;
its three effort settings are search strategies, not three objectives.

**Before submission:** integrate these methods, results and figures into the paper
as a new revision analysis. These records cannot retrospectively establish the
original case-study methods.

## 4. Generation and retrosynthesis

We retained 300 raw outputs from three seeded 100-output batches. There were
87 RDKit-valid outputs (29.0%) and 23 distinct standardized structures (26.4%
uniqueness among valid outputs), including one parent reconstruction. Median
parent Tanimoto was 0.574; the supplement contains all individual structures,
occurrence counts and physicochemical properties. A separate seed-11 repeat
reproduced every raw string exactly under the recorded environment. Raw data,
checkpoint hashes and a reproducible command accompany
[the generation results](results/generation/README.md). Vector chemical-structure
and rate figures have been regenerated and visually inspected.

All 23 unique generated structures were absent from a pinned official MOSES
train split after standardizing all 1,584,663 reference rows with the same policy.
This is explicitly a reference-set comparison. Training-set novelty remains
unavailable pending a traceable exact training corpus; the reference comparison
cannot resolve that missing provenance. Ten distinct nonparent candidates were
selected by a recorded hash-based rule before any route searches. Four returned
predicted routes (6, 9, 5 and 5 reaction steps); six exhausted the fixed
100-iteration budget without a route. There were no runtime failures or timeouts.
The median worker wall time was 23.16 s. The [complete results](results/retrosynthesis/README.md)
include all ten targets, route scores, raw plans, seeds, exact assets, actual
runtime dependency deviations and vector diagrams. Original diagrams are
preserved alongside corrected publication exports that repair clipping without
changing molecular bonds or route connectors.

These are computational predictions. We explicitly describe the planner's
small-terminal-molecule stopping rule, the absence of independent synthetic
expert assessment and the lack of experimental validation. Proposed manuscript
text removes unsupported experimental activity and synthetic-feasibility claims.

**Before submission:** perform author/chemical expert review and integrate the
completed tables and figures into the editable paper, preserving the explicit
training-provenance and experimental-validation limitations.

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
