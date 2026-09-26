# Manuscript revision protocol

This protocol addresses the five reviewer requests with a small reliability study,
an architecture comparison, and offline chemical analyses. It is a prospective
plan, not a results report. No success rates or scientific results are supplied by
the unit tests or the example data used to test the analysis scripts.

## Version and inputs

The implementation starts from `prompt-skill-source-of-truth` at
`1a25a7a0d2ec6e5d4fb74b1c3d19263c03badcac`. This includes the revised runtime,
single-agent baseline, and manuscript benchmark. Freeze the final tested commit
before measured runs; do not describe it as the original manuscript software.
The existing `v0.1.0` tag is a candidate historical reference, pending verification
against the original case-study records.

First locate the authors' Zenodo record. Its DOI and contents have not yet been
verified in this revision checkout. Inventory and hash the original raw/clean sEH
data, GTM model, generation outputs, route plans, reports, and transcripts. Reuse
these artifacts before scheduling replacement scientific computations. Retain the
archive version DOI, not only a moving latest-version link.

Create the four frozen session snapshots described in
[`fixtures/manuscript/README.md`](https://github.com/Laboratoire-de-Chemoinformatique/chemspacecopilot/blob/1a25a7a0d2ec6e5d4fb74b1c3d19263c03badcac/tests/robustness/fixtures/manuscript/README.md).
The paths must resolve to real scientific artifacts. Verify both each snapshot
and its referenced files, and perform one independent smoke run before freezing
the protocol. Record pilot runs separately; do not select successful measured
runs after seeing their outcomes.

The runner copies fixture artifacts into private per-run inputs and writes a
source-to-copy SHA-256 manifest. Local relative artifact paths are resolved from
the fixture JSON directory; remote inputs require full S3 URIs. Copies protect
the original archive, but do not replace checking the source hashes against the
published archive. Every repeat and architecture has a separate output prefix.

## Reviewer 1 and 2: one controlled study

| Component | Executions | Purpose |
|---|---:|---|
| Four frozen cases × two phrasings × three repetitions × two architectures | 48 | Repeatability and same-model architecture comparison |
| Three live sEH workflows, each containing three dependent stages | 9 stages | End-to-end retrieval, analysis, generation, planning |
| Three independent live peptide cases | 3 | Peptide workflow reliability |
| Missing-input and invalid-input recovery cases | Optional, separate | Recovery behaviour |

The 60 scientific case/stage executions are not 60 API calls or 60 independent
observations. Report per-case/per-arm denominators; the three sEH stages within a
workflow share history and state. Recovery cases must not inflate the scientific
success rate.

Hold model, tools, scientific inputs, computational budgets, and inference settings
constant between architectures. The single-agent arm is the flat Agno baseline,
not an external MCP client with a different model and harness. Predeclare arm order
and counterbalance it across batches using `--arm-order`; retain batch identifiers
when combining outputs. Do not rerun failures until success and substitute them.

With a locked environment, configured provider credentials, and verified fixtures:

```bash
USE_S3=false uv run --frozen python tests/robustness/robustness_minimal_example.py \
  --config tests/robustness/manuscript_reliability.yaml \
  --tier frozen --system both --arm-order team-first \
  --n-variations 2 --repetitions 3 \
  --test frozen_case_1_seh_analysis \
  --test frozen_case_2_seh_generation \
  --test frozen_case_3_retrosynthesis \
  --test frozen_case_4_peptide_design

USE_S3=false uv run --frozen python tests/robustness/robustness_minimal_example.py \
  --config tests/robustness/manuscript_reliability.yaml \
  --tier live --system team --repetitions 3
```

The first command is the complete 48-execution budget. If splitting it to
counterbalance order, divide that repetition budget across independently named
batches rather than running an additional unplanned full study. The study can
be reduced to one phrasing (24 frozen executions) if declared before measurement.

Use the normalized files in `reliability/` for the manuscript. The older pytest
robustness paths contain placeholder process scores and incompatible optional
reporting adapters; they are not the publication-facing evaluation. MLflow is not
required for this protocol.

Record task fulfillment separately from completion without tool errors, and from
scientific outcomes such as finding a route. A truthful completed search with no
route can fulfill an orchestration task. Missing telemetry is unavailable, not
zero; report measurement coverage and retain partial observed traces separately
from complete-run efficiency statistics. Verify incorrect tool choices and factual
grounding using the exported blinded review packets. Automated artifact checks
alone do not establish scientific correctness of the narrative.

Report success counts, uncertainty intervals, median/IQR wall time and token use,
tool failures, recovery examples, and expert review outcomes. Wall time covers the
submitted agent run, not one-time model initialization. Disclose warm-cache policy,
hardware, actual provider identifier and date, inference settings, and scientific
RNG settings. Do not infer an immutable provider model snapshot from an alias.
Describe small-sample architecture differences as exploratory tradeoffs.

Both sEH analysis prompts use the cached default map with autoencoder descriptors.
This evaluates retrieval/projection/landscape analysis, not new GTM fitting or
optimization. If the revised paper needs evidence for construction reliability,
add a separately declared small fitting task and its numerical acceptance checks.

## Reviewer 3: recover the actual methods

Prepare a methods table and machine-readable manifest with:

| Item | Evidence to recover or explicitly specify in a replacement analysis |
|---|---|
| ChEMBL | Release, backend, target IDs, organism, assay types/confidence, endpoints, units, relations/censoring, quality exclusions |
| Curation | Raw/retained counts, structure standardization, stereochemistry policy, duplicate grouping/aggregation, conflicting measurements |
| Activity | Actual regression endpoint or active/inactive thresholds, intermediate handling, any comment-based fallback |
| Representation | Actual Morgan count fingerprint settings or autoencoder checkpoint, preprocessing and dimensions |
| GTM | Trained versus pretrained map, checkpoint provenance, search ranges/trials, seeds, selected parameters, saved model and objective value |
| Environment | Git SHA, lockfile and installed versions, dataset/checkpoint hashes, device and inference configuration |

Current defaults do not establish historical methods. The inspected optimizer
maximizes normalized responsibility-density Shannon entropy; low/medium/high are
search strategies, not three objectives. Explain the actual objective and its
limits rather than adding objectives solely to repair the manuscript wording.
Do not pool unlike biological endpoints implicitly when aggregating structures.

## Reviewer 4: generation and retrosynthesis supplement

First extract physicochemical properties from full archived candidate JSON; the
compact CSV may omit properties that remain in JSON. Save raw decoder outputs
before validation/deduplication for any new generation run. For archived filtered
sets, report returned-set properties but mark raw validity/uniqueness unavailable.
Never reinterpret `count_returned` or a post-deduplication count as all attempts.

If raw attempts are absent, use three declared seeded batches of approximately
100 observed decoder outputs around the manuscript parent. Keep the engine and
sampling settings fixed. Record returned raw outputs separately from any unknown
internal attempts/rejections inside an external backend.
The new audit currently records RNG provenance as unavailable; set and record
the actual Python/NumPy/PyTorch seeds in the measurement launcher before claiming
seeded reproducibility. A parent SMILES used as a design seed is not an RNG seed.

Report validity (valid/raw), uniqueness (distinct valid/valid), parent similarity,
and MW/logP/TPSA/QED distributions. Define molecular identity and stereochemistry
handling. Count parent reconstructions explicitly. Training-set novelty requires
the actual versioned training structures; exact canonical nonmembership answers
that question. A Tanimoto threshold or uniqueness alone does not. If only a
reference dataset is available, label its membership comparison accordingly.

The offline generation summary accepts either a new raw-audit JSON or an archived
candidate JSON. With a known generator training corpus:

```bash
uv run --frozen python scripts/summarize_generation.py /path/to/generation_audit.json \
  --output-prefix reports/revision_generation/batch_01 \
  --parent-smiles 'CCC(C)C(=O)N1CCC(NC(=O)Nc2ccc(C(F)(C(F)(F)F)C(F)(F)F)cc2)CC1' \
  --training-corpus /path/to/actual_generator_training.smi
```

For existing candidate artifacts, supply the candidate JSON instead. Omit
`--training-corpus` when it is unavailable; novelty will remain unavailable. CSV
training files can specify `--training-smiles-column`. The output JSON contains
source hashes, metric definitions, counts, and distributions; the CSV includes
each observed record and its calculated properties. Molecular identity follows
the repository standardization policy, including removal of stereochemistry,
and is applied identically to candidates, parent, and training structures.
It is not a stereochemistry-sensitive novelty estimate. Preserve the raw inputs
alongside the derived table.

Select ten distinct valid molecules using a declared rule before planning routes.
Save the complete target list, each returned plan immediately, and any exceptions
that occur before a plan file is created. Use a fixed budget (for example one
120-second profile with retry profiles disabled) and retain no-route outcomes.
Run repeats in separate sessions/output roots so the same target cannot overwrite
its earlier result. Preserve the illustrated manuscript molecule separately if
it was not selected by the declared rule.

The offline command below summarizes existing plans without any API/model calls:

```bash
uv run --frozen python scripts/summarize_retrosynthesis.py \
  /path/to/target_a/plan.json /path/to/target_b/plan.json \
  --output-dir reports/revision_retrosynthesis
```

It emits per-execution `targets.csv` and `summary.json`, including distinct-target
counts, no-route/error/unavailable outcomes (including mixed no-route/error
attempts), source hashes, stored first-route
scores/lengths, and timing coverage. These are summaries of supplied files;
reconcile them with the full target list so missing failure files are not excluded
silently. Scores retain backend ordering without assuming which score direction
is best. Verify actual step counts against real route diagrams; zero-step and
inconsistent records require review. Search time is not end-to-end wall time.

## Reviewer 5: publication package

Regenerate the affected figures from original scientific data. Prefer vector
structures/routes, render raster maps at the intended publication size, and inspect
fonts, legends and multi-panel layouts at that size. Raising pixel count alone is
not sufficient. Existing image helpers and report-specific overrides differ.

Correct the case-study count, objective description, Sattarov/RDKit citations, and
listed typographical errors. Describe generated candidates and computationally
predicted routes; do not imply experimentally verified activity or synthesis.

Archive the final tested release, prompts/transcripts, fixtures and scientific
artifacts, raw generation data, all route outcomes, environment manifests, and
human-review records. Publish four result tables (reliability, architecture,
generation, retrosynthesis) plus the methods table. Report historical case studies
and revised measurements with their corresponding versions.

## Completion criteria

- Measurement and offline-analysis code passes focused tests.
- The original archive and four fixtures are verified; no synthetic test fixture
  is used as scientific evidence.
- A pilot establishes that the predeclared acceptance rules match the requested
  tasks, then the protocol and implementation are frozen.
- All planned executions, including failures, are retained and reviewed.
- Methods, figures, supplementary tables, release DOI, and response letter agree.

Locating the Zenodo record, recovering historical methods, running the scientific
study, and editing the manuscript remain separate tasks until their artifacts
and results are actually available.
