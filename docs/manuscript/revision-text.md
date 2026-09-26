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
workflows and three independent peptide workflows provide 12 live case/stage
executions. The dependent sEH stages are not independent experimental units.
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
