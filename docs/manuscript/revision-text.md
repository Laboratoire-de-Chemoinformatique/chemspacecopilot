# Manuscript replacement text

Editorial working copy for ChemRxiv v2. These passages are ready to insert where
indicated, but numerical results, original-run methods, figure replacements, and
the release DOI must be reconciled before submission. The original editable
manuscript has not been recovered. The inspected author-uploaded text is an
incomplete index of v2, not an authoritative editable source.

## Abstract: replace the claim and case-study sentences

ChemSpace Copilot integrates chemical-space analysis, molecular and peptide
generation, report preparation, and retrosynthetic planning through a
natural-language interface. Four case studies illustrate these capabilities:
analysis of ChEMBL-derived soluble epoxide hydrolase data, generation of
small-molecule analogues, retrosynthetic planning, and antimicrobial peptide
design. The first three form a connected workflow. The generated structures and
predicted routes are computational proposals for subsequent expert assessment;
these demonstrations do not establish biological activity, novelty relative to
all known chemistry, or experimental synthetic feasibility.

## Methods: rationale for the agent architecture

The numerical and chemical operations are implemented by deterministic or
explicitly parameterized scientific tools. Language models select tools, supply
arguments, coordinate dependent operations, and explain the resulting artifacts.
Specialist agents partition the available tool schemas and instructions into
domains: data retrieval, GTM analysis, chemoinformatics, molecular generation,
peptide generation, reporting, and retrosynthetic planning. The coordinator
connects these domains and passes task-specific context and artifact references.
This separation is intended to reduce the amount of unrelated tool and task
context presented to each specialist and to make domain-specific behavior easier
to maintain. It does not change the mathematical definition of GTM or make an
individual scientific tool more accurate.

A single tool-calling agent is a reasonable alternative when it can manage the
complete tool set. A fixed workflow is appropriate when the inputs, sequence of
operations, and acceptance conditions are known in advance; it can avoid
unnecessary model decisions and their latency. The multi-agent design is intended
for interactive requests that cross domains or change during analysis. Its
additional delegation calls may increase latency, token consumption, and the
opportunity for coordination errors. The architecture comparison therefore
measures task fulfillment and resource use under matched models, inputs, tools,
and budgets rather than assuming that specialization improves performance.

The revised implementation uses bounded, structured handoffs and explicit artifact
references. This description applies to the revision release. Historical case
studies must retain the implementation version that generated them; older shared
history behavior must not be described as if it used the revised handoff contract.

## Methods: quantitative evaluation

The prospective protocol uses four representative frozen-input tasks, two
semantically equivalent prompt formulations per task, three repetitions, and two
architectures, yielding 48 task executions. A further three connected sEH
workflows with live ChEMBL retrieval and three independent peptide workflows
provide 12 end-to-end case/stage executions. The peptide workflows begin from
a pinned public aggregate activity landscape; they do not retrieve or reconstruct
raw DBAASP measurements. The dependent sEH stages are not independent experimental units.
The exact prompts and acceptance rules are fixed before measurement. Pilot and
recovery runs are retained separately from the measured scientific tasks.

Each execution records the provider model identifier and settings, start and end
times, tool calls and reported errors, token usage where available, and resulting
artifacts. Task fulfillment requires the requested scientific outputs and their
provenance; execution without an exception alone is insufficient. A completed
retrosynthesis search that reports no route can fulfill the search task, while
route finding is reported as a separate scientific outcome. Human review checks
factual grounding and inappropriate tool selection. Missing telemetry remains
unavailable, and partial usage is excluded from complete-run usage distributions.
Success counts and uncertainty intervals are reported per task and architecture,
alongside median and interquartile range of wall time and usage. Small-sample
architecture differences are exploratory.

**Editorial gate:** change prospective wording to past tense only after the
predeclared runs and human reviews exist. Insert actual table references and
counts, including every failure. This text is a protocol, not a result.

## Methods: prospective sEH dataset for the revision

A new human soluble epoxide hydrolase dataset was extracted from ChEMBL 37
(release date 1 May 2026) on 26 September 2026. This is a prospective revision
dataset, not a reconstruction of the original case study. The target record was
CHEMBL2409, human EPHX2 (UniProt P34913), with target type `SINGLE PROTEIN`.
All 5,243 target activity records and 289 target assays were downloaded before
filtering. Original response pages, complete pagination URLs, SHA-256 hashes and
ChEMBL status responses before and after extraction were preserved.

Assays required target-assignment confidence 9 and type B or F. Activity records
required the human target assignment, standardized endpoint IC50, units nM,
relation `=`, and a finite positive value. Records with a nonempty ChEMBL data
validity comment or a potential-duplicate flag were excluded. Because EPHX2 is
bifunctional, assay CHEMBL4415272 was additionally excluded after its description
identified phosphatase activity; its five IC50 records were retained in the raw
archive and exclusion table. This prevents pooling a second enzyme activity
with epoxide-hydrolase IC50 values. The human identity criterion followed the
confidence-9 target assignment; all 151 retained assays also have explicit
`Homo sapiens` assay-organism metadata and type B. Ki, half-life and other
endpoints were not converted into IC50 or pooled with it.

RDKit 2025.09.4 performed cleanup, largest-fragment selection, neutralization,
canonical tautomer selection and removal of stereochemistry. Thus salts,
tautomers and stereoisomers that share the resulting structure were aggregated
under an explicitly defined identity rule. Exact repeats of the same standardized
structure, assay and numeric IC50 were counted once, retaining all original
activity and molecule identifiers; measurements from different assays were
retained. The resulting 2,497 eligible source records became 2,490 unique
measurements after collapsing seven repeats. Per-structure activity was the
median of `pIC50 = 9 - log10(IC50 in nM)` over these measurements. The corresponding
reported concentration is `10^(9 - median pIC50)`, not the arithmetic median
concentration.

The final dataset contains 2,212 standardized structures. A compound was labeled
active if median pIC50 was greater than 6 (IC50 below 1,000 nM), inactive if median
pIC50 was at most 5 (IC50 at least 10,000 nM), and intermediate otherwise: 1,794
active, 158 inactive and 260 intermediate structures. Intermediate compounds
remain in the dataset with a missing binary label and must not silently become
inactive in a binary activity landscape. Twenty-six structures have measurements
in more than one activity class, including two with both active and inactive
measurements; these structures were retained with explicit conflict flags,
measurement counts and pIC50 ranges. These labels are a declared visualization
policy, not new experimental measurements or a validated activity classifier.

The manuscript's molecular-generation seed CHEMBL3327073 was verified separately
against the ChEMBL molecule record. It appears in the downloaded target records
with Ki and half-life measurements, so the IC50-only rule excludes it from this
new dataset. Its use as a generation seed does not imply that it belongs to the
curated IC50 dataset. Preparation is reproduced by
`scripts/prepare_revision_seh.py --reuse-raw`; [seh-dataset.lock.json](seh-dataset.lock.json)
records source/output hashes, selection counts, domain-review evidence and the
independent parent lookup. Reprocessing the saved raw snapshot reproduced all
four output CSV hashes exactly.

## Methods: GTM objective and search

For the current implementation, GTM hyperparameters are selected by maximizing
normalized Shannon entropy of the aggregate node responsibilities. If
`r_ik` is the responsibility of node `k` for molecule `i`, then
`p_k = sum_i(r_ik) / sum_ij(r_ij)` and
`H_normalized = -sum_k(p_k log(p_k)) / log(K)`, where `K` is the number of nodes.
The score is dimensionless on the interval zero to one. It summarizes occupancy
of the latent grid; it does not independently establish neighborhood preservation,
predictive performance, or the biological validity of an activity landscape.

The current code offers three search strategies for this objective. For `n`
structures, let `k0 = round(sqrt(5 sqrt(n)) + 2)` and
`m0 = max(3, round(0.3 k0))`. Low effort uses `k0²` nodes, `m0²` basis functions,
basis widths `{0.5, 1, 2}`, and regularization coefficients `{1, 10, 100}`
(nine combinations). Medium effort uses the distinct node-side lengths
`{max(8,k0-5), max(8,k0), k0+5}`, basis-side lengths
`{max(3,m0-5), m0, m0+5}`, widths `{0.5,1,2,5}`, and regularization
`{0.1,1,10,100}` (up to 144 combinations). High effort uses 50 Optuna TPE trials
with integer node-side lengths 8–40 and basis-side lengths 3–15, and linearly
sampled widths 0.1–10 and regularization 0.1–1000. Sampler seeds are 42.
The current molecular optimizer uses 200 fitting iterations, no descriptor
standardization, and PCA scaling with the Torch PCA implementation.

These are revision-code definitions, not recovered historical hyperparameters.
For a pretrained map, report the actual saved model parameters and checkpoint
hash instead of implying that a new search was performed. The cached-map
reliability tasks evaluate projection and analysis, not GTM fitting. Replace the
claim of three optimization objectives with the objective actually used.

## Results: prospective sEH projection

The 2,212 curated compounds were encoded into 256-dimensional molecular
autoencoder representations and projected onto the pinned pretrained GTM.
The saved model contains 900 nodes, 225 basis functions, basis width 1 and
regularization coefficient 100; fitting used a maximum of 200 iterations,
PCA scaling and no descriptor standardization. These parameters were read
from the checkpoint, whose filename is not an accurate parameter record.
No new parameter search was performed for this projection study.

All 2,212 structures produced finite projections. The node responsibilities
sum to 2,212, 293 of the 900 nodes receive at least one maximum-responsibility
assignment, and normalized occupancy entropy is 0.6747. Mean normalized
per-compound responsibility entropy is 0.03731. These are descriptive
projection diagnostics and were not used to select this pretrained model.
The activity landscape is the responsibility-weighted mean of the compound
median pIC50 at each node. Nodes with responsibility mass below 0.1 are masked
in both published landscape panels. Continuous activity uses all 2,212
compounds, including the intermediate class; a binary comparison instead
contains the 1,952 explicitly active or inactive compounds. The maps and
projection statistics are provided with [their source tables](results/seh/README.md).

## Methods: chemical identity and generation metrics

For new generation runs, raw backend outputs are retained before application-level
standardization, validity filtering, and deduplication. Validity is the fraction of
observed raw outputs that RDKit parses as nonempty molecules. Uniqueness is the
number of distinct standardized valid structures divided by the number of valid
standardized outputs. Both numerator and denominator are reported. The molecular
identity policy performs cleanup, largest-fragment selection, neutralization,
canonical tautomer selection, and removal of stereochemistry; the same policy is
applied to candidates, the parent, and any comparison corpus.

Parent similarity is Morgan radius-two, 2,048-bit Tanimoto similarity. Molecular
weight, logP, topological polar surface area, hydrogen-bond donor/acceptor counts,
rotatable bonds, and QED are summarized over unique standardized molecules.
Parent reconstructions are counted separately. Exact nonmembership in a known,
versioned training corpus defines training-set novelty. A similarity threshold,
uniqueness within the generated set, or comparison with an unrelated dataset does
not establish training-set novelty. When only filtered historical candidates are
available, their properties can be analyzed but raw generation rates remain
unavailable. The count of outputs observed from a backend is not necessarily its
internal number of decoding attempts.

## Results: prospective small-molecule generation

Three independent batches produced 100 raw decoder outputs each around the
manuscript seed CHEMBL3327073, using random seeds 11, 22 and 33, latent noise
scale 0.1, sampling temperature 0.5 and CPU execution with one Torch thread.
The pinned autoencoder produced 27, 28 and 32 RDKit-valid structures in the
respective batches, giving pooled validity of 87/300 (29.0%). These batches
contained 9, 12 and 15 distinct standardized structures; uniqueness among valid
outputs was therefore 33.3%, 42.9% and 46.9%, respectively. Across batches,
87 valid outputs collapsed to 23 distinct standardized structures (26.4% pooled
uniqueness), including one reconstruction of the parent. These pooled molecules
are not 23 independent trials of biological efficacy.

Over the 23 distinct structures, parent Tanimoto similarity ranged from 0.305 to
1.000 (median 0.574). Median molecular weight, calculated logP, topological polar
surface area and QED were 467.45 Da, 4.83, 70.23 Å² and 0.585, respectively.
The full structure-level properties, occurrences and batch identities are
provided in [the candidate table](results/generation/unique_candidates.csv).
Worker wall times, including process startup and model setup, were 4.00, 4.04
and 3.68 s. These timings exclude language-model orchestration. A separately
retained seed-11 repeat reproduced all 100 raw strings exactly in the recorded
environment and is excluded from the primary denominator. Reproducibility under
this fixed environment does not establish identical sampling across hardware
or library versions.

This experiment measures local-latent analogue sampling, rather than
GTM-conditioned generation or experimental inhibitor discovery. Training-set
novelty is unavailable because the exact training-corpus/checkpoint linkage has
not been established. A separate exact-membership comparison standardized every
one of the 1,584,663 rows of a pinned official MOSES train split with the same
identity policy. All rows standardized successfully, and all 23 generated
structures were absent from that reference. This reference-set absence is not
interpreted as confirmed training-set novelty. All invalid outputs and duplicates
are retained; the rates
therefore characterize the raw decoder output, rather than only a filtered
returned candidate set. The [generation summary](results/generation/summary.json)
records model hashes, settings, software, source version and raw-output hashes.
The revised vector figures show the measured per-batch rates and the parent
alongside the ten candidates selected before retrosynthesis searches.

## Methods and results: prospective retrosynthetic planning

Before planning, the 22 distinct nonparent generated structures were ranked by
SHA-256 of `42|<canonical SMILES>`, and the first ten were selected. All ten
targets were searched once with SynPlanner 1.2.1, the pinned USPTO reaction rules
and ranking policy, and a pinned 189,144-entry building-block collection. Each
search used at most 120 s, 100 iterations, depth 9, tree size 10,000 and the top
50 reaction rules, stopping at its first solution. Retry profiles and language-model
fallback were disabled. Seeds 42–51 were applied to Python, NumPy and PyTorch;
the process hash seed was fixed and Torch used one CPU thread and deterministic
algorithms. A separate 300-s process guard applied to each worker. The backend's
120-s check occurs between iterations, so it is not a strict process deadline.

Four of ten searches returned a predicted route, containing 6, 9, 5 and 5 reaction
steps. Their backend scores were 0.119246, 0.041497, 0.152778 and 0.152778,
respectively. These are uncalibrated search scores, not probabilities of synthesis
success. The other six searches exhausted the 100-iteration budget without a
route. There were no worker errors or timeouts. Median worker wall time was
23.16 s (range 12.68–33.19 s); the ten searches took 226.78 s in total. These
timings include process startup, loading and rendering and exclude LLM
orchestration. The [complete outcome table](results/retrosynthesis/targets.csv)
retains every selected target; failures were neither replaced nor retried.

Recovered step counts describe precursor expansions and need not equal the
longest linear synthesis. The minimum-molecule-size rule (6) can accept small
terminal molecules without exact stock membership, so route finding does not
demonstrate that every precursor is purchasable. No-route outcomes refer only
to the stated search budget. Predicted routes were not experimentally tested or
independently assessed by a synthetic chemist. A separately labeled aspirin
implementation control is excluded from the ten-target denominator. Repeated
route-search reproducibility was not measured.

The recorded Linux aarch64 runtime used CGRtools 4.1.35 and upstream chython
3.4 because native distributions of SynPlanner's named fork dependencies were
unavailable. This is a measured compatibility environment, not an exact lockfile
installation or demonstrated equivalence to the forked environment. Complete
distribution versions, unmet named requirements, active module origins, pinned
assets and reproduction instructions accompany
[the retrosynthesis results](results/retrosynthesis/README.md).

## Discussion: scope of the scientific claims

Activity-enriched map regions provide a way to select structures or latent-space
locations for exploration. Proximity to known active compounds does not establish
that generated candidates retain activity. Likewise, a predicted retrosynthetic
route is evidence that a planning algorithm found a route under its reaction-rule,
building-block, and search-budget assumptions. It is not experimental evidence
of reaction success, yield, selectivity, or practical synthesis. Chemical and
synthetic expert review and experimental testing remain necessary before making
such claims.

## Presentation corrections

| Location | Required change |
|---|---|
| Abstract and conclusion | Four case studies; describe Cases 1–3 as the connected sEH workflow |
| GTM Methods | State the actual entropy objective and distinguish it from search strategies |
| Sattarov attribution | Cite Sattarov et al., *J. Chem. Inf. Model.* **59**, 1182–1196 (2019), DOI [10.1021/acs.jcim.8b00751](https://doi.org/10.1021/acs.jcim.8b00751); the inspected reference 39 is OpenRouter |
| RDKit attribution | Cite RDKit software and the version actually used; inspected references 46–47 are Biopython and Clustal W |
| Revised analysis RDKit | Version 2025.09.4 has release DOI [10.5281/zenodo.18098214](https://zenodo.org/records/18098214); do not assign this version to historical runs without evidence |
| Spelling | `chemograpic` → `chemographic`; `retrosythetic` → `retrosynthetic`; `Case Stud y` → `Case Study` |
| Figure 3 caption | Correct `she` to `sEH`; distinguish generated analogues from demonstrated active compounds |
| Figures 4–7 | Regenerate from underlying data; use vector structures/routes and readable map labels at final publication size |

Reference numbering must be regenerated in the editable paper after these
corrections. This file does not claim that the inaccessible source manuscript or
its figures have already been changed.
