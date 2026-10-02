# Response to Reviewer 1

**Manuscript:** ChemSpace Copilot — an agentic platform for interactive chemical-space exploration
**Recommendation received:** Major revision

> **Draft status.** Every number below is drawn from executed, retained measurements in
> this repository and can be reproduced with the commands given. Items marked
> **[PENDING]** are genuinely not yet done and must be resolved before this letter is
> submitted. Page and line references cannot be inserted until the corrections are
> applied in the editable manuscript.

---

We thank the reviewer for a careful and constructive report. The central request —
evidence beyond selected successful examples — was well founded, and addressing it
changed both the manuscript's claims and the software itself.

In response we executed a predeclared quantitative evaluation of 60 task executions,
twice; added a matched single-agent ablation; rebuilt the sEH dataset and GTM
description from auditable sources; quantified molecular generation and expanded
retrosynthesis from one compound to ten; and corrected the citation, counting and
typographical errors. Two of the reviewer's points caused us to **weaken** claims in
the manuscript: the multi-agent architecture shows no measurable accuracy advantage
over a flat agent at this scale, and candidate novelty can only be established against
a public reference set, not against the generator's training data. Both are now stated
as such.

A summary of the principal changes:

| # | Request | Action |
|---|---|---|
| 1 | Quantitative evaluation | 60-execution predeclared study, run twice; new Results section and supplement |
| 2 | Multi-agent justification | Matched same-model ablation; advantage claims removed |
| 3 | Dataset and GTM detail | Fully specified replacement dataset and checkpoint audit; objective statement corrected |
| 4 | Generation and route numbers | Full generation statistics; retrosynthesis extended to ten targets |
| 5 | Reproducibility and presentation | Version, model identity and settings recorded; citations and typography corrected |

---

## Comment 1 — Quantitative evaluation

> *The case studies are currently presented mainly as qualitative workflow
> demonstrations. The authors should provide a basic quantitative evaluation of the
> system, for example the success rate across a small set of representative prompts,
> reproducibility across repeated runs, frequency of failed or incorrect tool calls,
> and approximate execution time or computational cost.*

We agree that successful demonstrations do not establish reliability, and we have
added a quantitative evaluation as a new Results subsection and supplement.

**Design.** The protocol was fixed and hash-verified *before* execution: four
representative tasks × two semantically equivalent prompt phrasings × three
repetitions × two architectures = 48 frozen-input executions, plus 12 end-to-end
executions with live ChEMBL retrieval and peptide workflows, for 60 in total. Acceptance
criteria were content-based checks on the produced artefacts and their provenance,
declared in advance. **All 60 executions are retained, including every failure.** No
run was repeated until it succeeded and no result was substituted. The plan hash was
verified immediately before and after execution.

**Results.** The study was executed on 2026-09-30 against the software as described in
the submitted manuscript (v2). The failures it exposed were diagnosed and five defects
fixed, and the identical protocol was then re-executed on 2026-10-02 against the
corrected software (v3). We report both.

| Measure | v2 (software as submitted) | v3 (corrected software) |
|---|---:|---:|
| Task fulfilment | 40/60 = 66.7% (95% Wilson CI 54.1–77.3) | **45/60 = 75.0%** (62.8–84.2) |
| — frozen tier | 31/48 | 34/48 |
| — live tier | 9/12 | 11/12 |
| Completed without an agent exception | 48/60 = 80.0% | **60/60 = 100%** |
| Tool calls observed | 2,202 | 3,281 |
| Failed tool calls | 499 = **22.7%** | 348 = **10.6%** |
| Repeatability (unanimous 3-repetition cells) | 11/16 | 12/16 = 75% |
| Median wall time | 82.4 s (IQR 38.6–233.5) | 99.0 s (IQR 55.4–229.3) |
| Median total tokens per execution | 666,819 | 752,948 |
| Telemetry coverage | 48/60 | **60/60** |

Per task, both arms pooled:

| Task | v2 | v3 |
|---|---:|---:|
| 1 — sEH map analysis | 0/12 | 3/12 |
| 2 — analogue generation | 9/12 | 7/12 |
| 3 — retrosynthetic planning | 11/12 | **12/12** |
| 4 — peptide design | 11/12 | **12/12** |

**On the v2 failures.** The complete failure of Task 1 in v2 was traced to a single
reproducible defect in the pinned agent framework, confirmed by an isolated diagnostic
run: Agno 2.1.9 evaluates `str(result) if result else ""` for every tool result, and a
pandas DataFrame raises `ValueError: The truth value of a DataFrame is ambiguous`,
aborting runs that had largely completed. Of 17 frozen-tier v2 failures, 9 were this
defect, 4 were an over-strict acceptance check demanding evidence of a stage the
configuration never runs, and 4 were genuine. Because the measurement fix and the
system fix are different things, we separated them: the retained v2 runs were re-scored
offline with corrected checks at no additional model cost, which isolates 3 of the
improvement to scoring and the remainder to the software.

**On reproducibility.** Repeatability is reported as the fraction of (task, phrasing,
arm) cells whose three repetitions agreed unanimously: 12/16 in v3. We note explicitly
that the provider exposes no seed, so repeated runs are *not* claimed to be
deterministic; the scientific tools beneath them are separately seeded and the
generation study reproduced all 100 raw outputs exactly under a fixed environment.

**On "incorrect" tool calls.** We can report *failed* tool calls objectively (10.6% in
v3). We cannot report an *inappropriate tool selection* rate, because that requires
expert judgement rather than an automated check. We state this as a limitation rather
than substituting the automated figure for it.

**On cost.** Token usage is reported per execution from provider records (median
752,948 in v3, of which a large majority are cache reads).
**[PENDING: convert the recorded token counts to a monetary estimate using the
provider's published per-token prices, and state the price date.]**

**Limitations now stated in the manuscript.** Task fulfilment is judged by automated
artefact validators, not independent chemical or factual expert review; the three sEH
stages within a live workflow share state and are not three independent observations;
and token counts are usage records rather than a cost model.

*Evidence:* `docs/manuscript/results/reliability/` (per-run records in `runs.csv`,
attempt ledger in `execution_ledger.jsonl`, aggregates in `reliability_summary.json`,
v3 under `v3/`).

---

## Comment 2 — Role and practical advantage of the multi-agent architecture

> *Since many operations are deterministic computational tasks, the authors should
> explain more explicitly why separate specialized agents are preferable to a single
> tool-calling agent or a fixed workflow. A limited comparison or ablation on several
> representative tasks would strengthen the methodological contribution.*

This was the most consequential comment, and we have acted on it in both directions:
we have stated the rationale more precisely, and we have **removed the claims the
measurement does not support.**

**Rationale, as now written.** The numerical and chemical operations are performed by
deterministic or explicitly parameterised tools. The language model selects tools,
supplies arguments, sequences dependent operations and explains the resulting
artefacts; it does not change the mathematical definition of GTM or make any individual
tool more accurate. Specialist agents partition tool schemas and domain instructions,
which bounds the context presented to each agent and makes domain-specific behaviour
easier to maintain and extend. We now state plainly that a single tool-calling agent is
a reasonable alternative where it can manage the whole tool set, and that a fixed
pipeline is preferable where inputs, sequence and acceptance conditions are known in
advance.

**Ablation.** We implemented a flat single-agent arm with the same model, tools,
budgets and frozen inputs, and ran it within the protocol above (n = 24 per arm, arm
order counterbalanced across batches):

| | team (multi-agent) | single agent (flat) |
|---|---:|---:|
| Task fulfilment, v2 | 15/24 = 62.5% | 16/24 = 66.7% |
| Task fulfilment, v3 | 18/24 = 75.0% | 16/24 = 66.7% |
| Completed without exception, v3 | 24/24 | 24/24 |
| Failed tool calls, v3 | 10.9% | 8.2% |
| Median wall time, v3 | 168.5 s | 72.4 s |
| Median total tokens, v3 | 961,217 | 541,646 |

Paired on identical (task, phrasing, repetition) cells in v3: 14 both succeeded, 4
team-only, 2 single-agent-only, 4 neither. With 6 discordant pairs an exact test has no
useful power, and the direction *reversed* between v2 and v3. We therefore report the
accuracy comparison as exploratory and descriptive only.

**Conclusion, as now written in the manuscript.** On these four tasks the multi-agent
architecture shows **no measurable accuracy advantage** over a flat tool-calling agent
using the same model, while costing a median **+87 s** and **+141,000 tokens** per
matched cell. The durable, reproducible finding is the cost, not a benefit. The
manuscript's justification for the architecture is accordingly stated in terms of
context partitioning, maintainability and extensibility — properties we can defend —
and not in terms of measured task performance. Any statement implying that
specialisation improves chemical computation has been removed.

*Evidence:* `architecture_paired` and `by_tier_and_arm` in the v3 summary. We note in
the supplement that `by_arm_pooled_across_tiers` pools a team-only live tier and must
not be read as a matched comparison.

---

## Comment 3 — sEH dataset and GTM construction

> *The authors should report the ChEMBL version, target and assay filters, activity
> types, treatment of duplicate measurements, definition of active and inactive
> compounds, molecular representation used for GTM, hyperparameter ranges, and final
> map-quality criterion. The statement that three optimization objectives are available
> should also be corrected, as only two appear to be listed.*

**[DECISION REQUIRED — see note at the end of this section.]**

**Dataset.** The dataset is now fully specified:

| Item | Value |
|---|---|
| Source | ChEMBL 37 (release 2026-05-01), extracted 2026-09-26 |
| Target | CHEMBL2409, human soluble epoxide hydrolase (EPHX2, UniProt P34913), `SINGLE PROTEIN` |
| Raw records retrieved | 5,243 activities across 289 assays, archived before filtering |
| Assay filters | Target-assignment confidence 9; assay type B or F |
| Activity filters | Standardised endpoint IC50; units nM; relation `=`; finite positive value |
| Exclusions | Non-empty ChEMBL data-validity comment; potential-duplicate flag; assay CHEMBL4415272 (phosphatase activity — EPHX2 is bifunctional) |
| Activity types | IC50 only. Ki, half-life and other endpoints were **not** converted or pooled |
| Standardisation | RDKit 2025.09.4: cleanup, largest fragment, neutralisation, canonical tautomer, stereochemistry removed |
| Duplicate treatment | Exact repeats of (structure, assay, value) counted once (2,497 → 2,490); measurements from different assays retained; per-structure activity is the **median** pIC50 |
| Final dataset | **2,212 structures** from 2,490 measurements |
| Class definition | Active: median pIC50 > 6 (IC50 < 1,000 nM) → 1,794. Inactive: median pIC50 ≤ 5 (IC50 ≥ 10,000 nM) → 158. Intermediate otherwise → 260 |
| Conflicts | 26 structures have measurements in more than one class (2 with both active and inactive); retained with explicit flags, counts and pIC50 ranges |

Intermediate compounds are retained with a *missing* binary label and do not silently
become inactive: the continuous landscape uses all 2,212 compounds, while binary
analyses use the 1,952 labelled ones. Reprocessing the archived raw response pages
reproduces all four output CSV hashes exactly.

We also note that the generation seed CHEMBL3327073 carries Ki and half-life
measurements but no qualifying IC50, so the IC50-only rule excludes it from the curated
set. We now distinguish its verified identity and use as a design seed from membership
in the dataset.

**Molecular representation.** Compounds are encoded as 256-dimensional latent vectors
from a pretrained LSTM autoencoder; these vectors are the GTM input.

**Map parameters.** The manuscript previously described the map by its checkpoint
filename, which we found to be inaccurate. We have read the parameters directly from
the saved checkpoint and report those: **900 nodes, 225 basis functions, basis width 1,
regularisation coefficient 100, 200 fitting iterations, PCA scaling, no descriptor
standardisation.** All 2,212 compounds produce finite projections; 293 of 900 nodes
receive at least one maximum-responsibility assignment; normalised occupancy entropy is
0.6747 and mean normalised per-compound responsibility entropy is 0.0373. We state
explicitly that this is a **reused pretrained map**, not one selected by a new search.

**Hyperparameter ranges.** For completeness we document the ranges the optimiser
searches when a map *is* fitted. With `k0 = round(sqrt(5·sqrt(n)) + 2)` and
`m0 = max(3, round(0.3·k0))` for `n` structures: low effort uses `k0²` nodes, `m0²`
basis functions, widths {0.5, 1, 2} and regularisation {1, 10, 100} (9 combinations);
medium uses node sides {max(8,k0−5), max(8,k0), k0+5}, basis sides {max(3,m0−5), m0,
m0+5}, widths {0.5, 1, 2, 5} and regularisation {0.1, 1, 10, 100} (up to 144); high
uses 50 Optuna TPE trials with node sides 8–40, basis sides 3–15, widths 0.1–10 and
regularisation 0.1–1000, sampler seed 42.

**Map-quality criterion, and the correction the reviewer requested.** The reviewer is
right that the statement about optimisation objectives is wrong, though the correction
differs from what the text implied. On inspecting the implementation there is **one**
objective, not three and not two: the normalised Shannon entropy of aggregate node
responsibilities. With `r_ik` the responsibility of node `k` for molecule `i`,
`p_k = Σ_i r_ik / Σ_ij r_ij` and `H_norm = −Σ_k p_k log p_k / log K` for `K` nodes,
giving a dimensionless score on [0, 1]. The three items the manuscript listed as
objectives are **search strategies** (low / medium / high effort) over that single
objective. The sentence has been rewritten accordingly. We also now state that
occupancy entropy summarises use of the latent grid and does *not* by itself establish
neighbourhood preservation, predictive performance, or the validity of an activity
landscape.

> **[DECISION REQUIRED]** The records needed to reconstruct the *original* case-study
> dataset and map could not be recovered. The specification above describes a newly
> prepared replacement dataset, and the Case 1–3 figures and numbers change
> accordingly. The letter must either (a) present Cases 1–3 as re-run on this
> fully specified dataset, or (b) retain the original figures and present this as a
> separately labelled revision analysis. Option (a) is the stronger response and is
> assumed in the text above; it requires regenerating the Case 1–3 figures and numbers
> in the manuscript. This must be settled before submission.

*Evidence:* `docs/manuscript/seh-dataset.lock.json` (query manifest, source and output
hashes, exclusion table), `docs/manuscript/results/seh/`,
`docs/manuscript/gtm-model-attributes.json`. Reproduced with
`scripts/prepare_revision_seh.py --reuse-raw`.

---

## Comment 4 — Molecular generation and retrosynthesis

> *For the generated compounds, the authors should report at least validity,
> uniqueness, similarity to the parent compound, novelty relative to the training data,
> and basic physicochemical or synthetic-accessibility properties. For retrosynthesis,
> results for more than one selected compound, together with route length or route
> score.*

**Generation.** Three independent seeded batches (seeds 11, 22, 33; latent noise 0.1;
sampling temperature 0.5; CPU) each produced 100 raw decoder outputs. **All raw
outputs, including invalid strings and duplicates, are retained**, so the rates below
describe the decoder rather than a filtered candidate set.

| Seed | Raw | RDKit-valid | Distinct standardised | Uniqueness among valid |
|---|---:|---:|---:|---:|
| 11 | 100 | 27 | 9 | 33.3% |
| 22 | 100 | 28 | 12 | 42.9% |
| 33 | 100 | 32 | 15 | 46.9% |
| **Pooled** | **300** | **87 (29.0%)** | **23 (26.4%)** | — |

The 23 distinct structures include one reconstruction of the parent. Similarity to the
parent (Morgan radius-2, 2048-bit Tanimoto) has median **0.574** (range 0.305–1.000).
Median physicochemical properties: molecular weight 467.45 Da, clogP 4.83, TPSA
70.23 Å², QED 0.585; hydrogen-bond donors/acceptors and rotatable bonds are tabulated
per compound. A separately retained seed-11 repeat reproduced all 100 raw strings
exactly in the recorded environment.

**On novelty — an important qualification.** We are unable to report novelty relative
to the training data, because the exact linkage between the distributed checkpoint and
its training corpus could not be established. Rather than substitute a weaker measure
under a stronger name, we report what we can verify: all 1,584,663 rows of a
commit- and hash-pinned official MOSES training split were standardised under the
identical identity policy, and **all 23 generated structures were absent from it
(23/23)**. This is reported as *reference-set absence*, explicitly not as confirmed
training-set novelty, and the manuscript no longer claims the latter.

**[PENDING: add a synthetic-accessibility (SA) score column for the 23 candidates. The
reviewer's request is satisfied by the physicochemical properties above, but SA pairs
naturally with the route results and is inexpensive to compute.]**

**Retrosynthesis.** The example has been extended from one compound to ten, with the
targets fixed *before* any search: all 22 distinct non-parent candidates were ranked by
SHA-256 of `42|<canonical SMILES>` and the first ten selected. The rule, declaration
time and file hashes are recorded. **No target was replaced or retried, and all ten
outcomes are reported.**

| Target | Outcome | Steps | Backend score | Iterations | Wall time (s) |
|---|---|---:|---:|---:|---:|
| generated_001 | Route found | 6 | 0.119246 | 72 | 22.75 |
| generated_002 | No route | — | — | 100 | 27.67 |
| generated_003 | No route | — | — | 100 | 26.79 |
| generated_004 | No route | — | — | 100 | 20.74 |
| generated_005 | Route found | 9 | 0.041497 | 38 | 18.19 |
| generated_006 | No route | — | — | 100 | 33.19 |
| generated_007 | Route found | 5 | 0.152778 | 5 | 12.84 |
| generated_008 | Route found | 5 | 0.152778 | 5 | 12.68 |
| generated_009 | No route | — | — | 100 | 28.36 |
| generated_010 | No route | — | — | 100 | 23.57 |

Four of ten searches returned a route; the six others exhausted the 100-iteration
budget. No worker failed or timed out. Median wall time was 23.16 s (range
12.68–33.19 s; total 226.78 s). Search parameters were identical for all targets
(≤120 s, 100 iterations, depth 9, tree size 10,000, top 50 rules, stop at first
solution, seeds 42–51, deterministic Torch, single thread). A separately labelled
aspirin positive control returned a one-step route and is excluded from the
ten-target denominator.

**Moderated claims.** As the reviewer notes, these results demonstrate successful
integration rather than candidate discovery or synthetic feasibility, and the
manuscript has been revised to say so. Specifically: backend scores are uncalibrated
search values, not probabilities of synthetic success; the planner's `min_mol_size = 6`
rule can accept small terminal molecules without exact stock membership, so a recovered
route does not establish that every precursor is purchasable; step counts describe
precursor expansions rather than the longest linear synthesis; no-route outcomes refer
only to the stated budget; and no route has been assessed by a synthetic chemist or
tested experimentally. All statements of biological activity or demonstrated synthetic
feasibility for the generated compounds have been removed.

**[PENDING: author-level chemical assessment of the 23 structures and 4 routes. The
manuscript currently promises expert review that has not been performed; either
complete it or remove the promise.]**

*Evidence:* `docs/manuscript/results/generation/` (including
`unique_candidates.csv` and `moses_reference_membership.json`) and
`docs/manuscript/results/retrosynthesis/`.

---

## Comment 5 — Reproducibility and presentation

> *The authors should provide a fixed repository version or release corresponding to
> the manuscript, together with the exact LLM model identifier, main inference settings,
> software versions, and example prompts and outputs. Several inconsistencies and
> typographical errors should also be corrected… The resolution and readability of the
> chemical structures, GTM maps, and retrosynthetic route figures should also be
> improved.*

**Execution environment, now recorded.** Provider DeepSeek API; model alias
`deepseek-flash`, resolved name **DeepSeek-V4.1-Flash**, observed at execution time with
system fingerprint `aeb56401ca74e127821c4f9126dcb669`; **temperature 0.0,
max_tokens 8192**; context window 1,048,576. We state explicitly that an alias is not an
immutable snapshot and that the provider exposes no seed. Hardware was aarch64, 20 CPU,
CPU-only Torch, `OMP_NUM_THREADS=1`. The evaluated runtime commit is recorded for each
study version, together with installed software versions, checkpoint and dataset
SHA-256 hashes, and the complete prompt set in the frozen study plan. All raw outputs
are retained.

We also disclose a measured dependency deviation rather than conceal it: on Linux
aarch64, native distributions of SynPlanner's named fork dependencies were unavailable,
so CGRtools 4.1.35 and upstream chython 3.4 supplied the imported modules. Every unmet
named requirement is listed. This is a measured compatibility environment, not a
lockfile installation, and we do not claim equivalence to the forked environment.

**[PENDING: tag a release corresponding to this manuscript and deposit it with the
scientific artefacts to obtain a version DOI; insert the DOI here and in Data
Availability. The repository currently carries no tag matching the revision, and a code
commit alone is not an archival data release.]**

**Corrections.** All of the following have been made:

| Location | Correction |
|---|---|
| Abstract, conclusion | "three tasks" → **four case studies**; Cases 1–3 described as one connected sEH workflow |
| GTM Methods | Objective statement corrected (one entropy objective; three search strategies) — see Comment 3 |
| Sattarov citation | Corrected to Sattarov *et al.*, *J. Chem. Inf. Model.* **59**, 1182–1196 (2019), DOI 10.1021/acs.jcim.8b00751. The cited reference 39 was an unrelated OpenRouter entry |
| RDKit citation | Corrected to the RDKit software citation with the version used. References 46–47 were Biopython and Clustal W |
| Spelling | "chemograpic" → "chemographic"; "retrosythetic" → "retrosynthetic"; "Case Stud y" → "Case Study" |
| Figure 3 caption | "she" → "sEH"; generated analogues distinguished from demonstrated active compounds |

Reference numbering is regenerated after these corrections.

**Figures.** All scientific figures have been regenerated from their source data as
vector graphics (SVG and PDF, with PNG previews only). This covers the chemical
structures, the generation-rate panels, the sEH density and activity landscapes, and
the retrosynthetic routes. We additionally found and fixed a rendering defect in the
route diagrams: the planner's default SVG mask bounds hid bonds lying in negative
coordinate regions. The corrected exports set explicit root-viewBox mask bounds and a
white background, leaving molecular coordinates, bond primitives, atom labels and route
connectors unchanged; all four were rendered at 2400 px and inspected against the raw
source. We recommend the routes be published as full-page supplementary figures at
readable scale rather than combined into one panel.

**[PENDING: apply all corrections in the editable manuscript, inspect every figure at
final publication size, and insert page/line references in this letter.]**

---

## Summary

The reviewer's first and second comments changed our conclusions, not only our
presentation: we now report a measured reliability figure with its failures, and we
have withdrawn the implied advantage of the multi-agent design in favour of the
properties we can actually demonstrate. The third and fourth comments led us to
specify the dataset and map precisely — which surfaced an inaccurate checkpoint
description and an incorrect statement of the optimisation objective — and to replace a
single illustrative route with a predeclared ten-target study reported in full,
including its six failures. We believe the manuscript is substantially more accurate as
a result, and we thank the reviewer for it.
