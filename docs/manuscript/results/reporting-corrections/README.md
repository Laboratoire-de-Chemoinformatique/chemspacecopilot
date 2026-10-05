# Historical sEH reporting corrections

This additive, post-hoc correction covers the six retained sEH analysis runs. Original responses, reports, datasets, audit artifacts and task-success scores are preserved. Every run is independently recomputed and checked against the historical audit, with SHA-256 hashes in `reconciliation.json` and claim-level provenance in `correction_ledger.json`.

## Reproduce offline

```sh
PYTHONPATH=src python scripts/reconcile_reporting_evidence.py \
  --audit-dir /home/aorlov/Programs/agents/chemspacecopilot/reports/reviewer_revision/reliability_study_v4_live_single_agent_20261004/analysis \
  --source-root /home/aorlov/Programs/agents/chemspacecopilot-v4 \
  --output-dir docs/manuscript/results/reporting-corrections
```

The original artifacts must be available at the recorded paths. This command reads them and writes only this correction directory. No ChEMBL retrieval, model inference, GTM projection, or benchmark rescoring is performed. Python, pandas and RDKit are required; versions and the reconciliation script hash are recorded in the JSON.

## Counting rules

Retained raw means activity rows that survived retrieval filtering. Full retrieved means the disjoint union of retained raw and excluded rows, validated with activity IDs. Clean means standardized compound rows. Identifier counts split pipe-joined cells and count distinct nonempty atomic IDs. Distinct serialized combinations are reported only to explain the earlier error.

RDKit GetScaffoldForMol on each original clean SMILES; canonical isomeric SMILES (atom and bond types retained, no generic scaffold conversion); count every valid clean row once, including empty acyclic scaffolds. Invalid structures are counted separately and never treated as acyclic. All six historical clean datasets have zero invalid structures under this replay.

| Run | Full rows | Full assays/docs | Retained rows | Retained assays/docs | Clean compounds | Clean assays/docs | Exact Murcko scaffolds |
|---|---:|---:|---:|---:|---:|---:|---:|
| team_rep1 | 8322 | 508/169 | 5171 | 334/153 | 3028 | 334/153 | 1253 |
| team_rep2 | 8322 | 508/169 | 2185 | 156/76 | 1580 | 156/76 | 635 |
| team_rep3 | 8322 | 508/169 | 2185 | 156/76 | 1580 | 156/76 | 635 |
| single_agent_rep1 | 8322 | 508/169 | 5171 | 334/153 | 3028 | 334/153 | 1253 |
| single_agent_rep2 | 8322 | 508/169 | 2185 | 156/76 | 1580 | 156/76 | 635 |
| single_agent_rep3 | 8322 | 508/169 | 2185 | 156/76 | 1580 | 156/76 | 635 |

The original `ds_001.assay_count=525` is traced to `fetch_compounds`: it unions search-matched assay IDs before fetching activity rows. That scope is verified from the retained source code and state in each run. The original matched-ID lists were not retained, so 525 cannot be independently recounted. It is neither the 508 assays represented in retrieved activities nor retained coverage. Single-agent repetition 1's 237 documents, repetition 2's 167 assays / 129 documents, and repetition 3's 129 documents trace to stored `nunique` calls on pipe-joined clean cells. Recomputing these serialized combinations reproduces those numbers; splitting their atomic IDs gives the corrected coverage above.

## team_rep1

Original task success remains `true`. Coverage: 334 assays / 153 documents in retained activity rows; 334 / 153 atomic IDs in 3028 clean compounds. Clean-cell serialization has 311 assay combinations / 237 document combinations; these are not assay/document counts.

Whole-set exact Murcko frequencies: adamantane **99/3028**; phenyl-urea-adamantyl **61/3028**; acyclic **57/3028**; **1253** distinct scaffolds including the empty scaffold. Complete frequencies are in `murcko_scaffold_frequencies.csv`.

The original report explicitly used approximate substructure-defined chemotypes, not exact Murcko scaffolds. Its four motif categories must not be interpreted as four unique scaffolds; the exact Murcko enumeration here is a separate measurement. Substructure prevalence claims are not revalidated by exact scaffold frequencies.

Original claim excerpts and original report paths/hashes are retained in `reconciliation.json`; the ledger separates verified reconstructed values from original metadata and unresolved subset recounts.

## team_rep2

Original task success remains `true`. Coverage: 156 assays / 76 documents in retained activity rows; 156 / 76 atomic IDs in 1580 clean compounds. Clean-cell serialization has 167 assay combinations / 129 document combinations; these are not assay/document counts.

Whole-set exact Murcko frequencies: adamantane **90/1580**; phenyl-urea-adamantyl **49/1580**; acyclic **57/1580**; **635** distinct scaffolds including the empty scaffold. Complete frequencies are in `murcko_scaffold_frequencies.csv`.

The original state labels 1,563 molecules as scaffold-analyzed, while its summary frequencies match the full clean population. The explicit denominator for this correction is all 1,580 valid clean structures; the truncated original node selection cannot independently verify the 1,563 scope.
Original scaffold call 100 selected nodes truncated original node list ([1, 2, 3] …); full selection unresolved. Its stored preview reports adamantane=90, phenyl-urea-adamantyl=49 (a missing value means absent from the retained preview, not zero). Scope is a selected-node population. Independent subset recount: **unresolved**; denominator=None. Original per-molecule node membership is not available in these previews; no projection inference rerun.


Original claim excerpts and original report paths/hashes are retained in `reconciliation.json`; the ledger separates verified reconstructed values from original metadata and unresolved subset recounts.

## team_rep3

Original task success remains `true`. Coverage: 156 assays / 76 documents in retained activity rows; 156 / 76 atomic IDs in 1580 clean compounds. Clean-cell serialization has 167 assay combinations / 129 document combinations; these are not assay/document counts.

Whole-set exact Murcko frequencies: adamantane **90/1580**; phenyl-urea-adamantyl **49/1580**; acyclic **57/1580**; **635** distinct scaffolds including the empty scaffold. Complete frequencies are in `murcko_scaffold_frequencies.csv`.

Saved active subset verified independently: **1203** compounds satisfy `activity_final >= 6` in the original clean table, with **518** exact scaffolds. Adamantane=73; phenyl-urea-adamantyl=47. Missing activity is excluded. This verifies the activity-defined population without recovering original node membership.
Saved inactive subset verified independently: **225** compounds satisfy `activity_final < 6` in the original clean table, with **122** exact scaffolds. Adamantane=6; phenyl-urea-adamantyl=2. Missing activity is excluded. This verifies the activity-defined population without recovering original node membership.
Original scaffold call 105 selected nodes [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50]. Its stored preview reports adamantane=None, phenyl-urea-adamantyl=15 (a missing value means absent from the retained preview, not zero). Scope is a selected-node population. Independent subset recount: **unresolved**; denominator=None. Original per-molecule node membership is not available in these previews; no projection inference rerun.

Original scaffold call 106 selected nodes truncated original node list ([0, 1, 2] …); full selection unresolved. Its stored preview reports adamantane=90, phenyl-urea-adamantyl=49 (a missing value means absent from the retained preview, not zero). Scope is a selected-node population. Independent subset recount: **unresolved**; denominator=None. Original per-molecule node membership is not available in these previews; no projection inference rerun.

Original scaffold call 111 selected nodes truncated original node list ([0, 1, 2] …); full selection unresolved. Its stored preview reports adamantane=73, phenyl-urea-adamantyl=47 (a missing value means absent from the retained preview, not zero). Scope is a selected-node population. Independent subset recount: **unresolved**; denominator=None. Original per-molecule node membership is not available in these previews; no projection inference rerun.

Original scaffold call 113 selected nodes truncated original node list ([0, 1, 2] …); full selection unresolved. Its stored preview reports adamantane=6, phenyl-urea-adamantyl=None (a missing value means absent from the retained preview, not zero). Scope is a selected-node population. Independent subset recount: **unresolved**; denominator=None. Original per-molecule node membership is not available in these previews; no projection inference rerun.


Original claim excerpts and original report paths/hashes are retained in `reconciliation.json`; the ledger separates verified reconstructed values from original metadata and unresolved subset recounts.

## single_agent_rep1

Original task success remains `true`. Coverage: 334 assays / 153 documents in retained activity rows; 334 / 153 atomic IDs in 3028 clean compounds. Clean-cell serialization has 311 assay combinations / 237 document combinations; these are not assay/document counts.

Whole-set exact Murcko frequencies: adamantane **99/3028**; phenyl-urea-adamantyl **61/3028**; acyclic **57/3028**; **1253** distinct scaffolds including the empty scaffold. Complete frequencies are in `murcko_scaffold_frequencies.csv`.

Original scaffold call 38 selected nodes [251, 181, 250, 4, 5]. Its stored preview reports adamantane=39, phenyl-urea-adamantyl=36 (a missing value means absent from the retained preview, not zero). Scope is a selected-node population. Independent subset recount: **unresolved**; denominator=None. Original per-molecule node membership is not available in these previews; no projection inference rerun.

Original scaffold call 46 selected nodes [522, 40, 691, 57, 861]. Its stored preview reports adamantane=None, phenyl-urea-adamantyl=None (a missing value means absent from the retained preview, not zero). Scope is a selected-node population. Independent subset recount: **unresolved**; denominator=None. Original per-molecule node membership is not available in these previews; no projection inference rerun.

Original scaffold call 47 selected nodes [659, 615, 651, 660, 30]. Its stored preview reports adamantane=None, phenyl-urea-adamantyl=None (a missing value means absent from the retained preview, not zero). Scope is a selected-node population. Independent subset recount: **unresolved**; denominator=None. Original per-molecule node membership is not available in these previews; no projection inference rerun.

The scaffold numbers in the response match the first selected-node call above. They must be labeled as selected-region observations, not whole-dataset frequencies. The whole-dataset ranking is shown above.

Original claim excerpts and original report paths/hashes are retained in `reconciliation.json`; the ledger separates verified reconstructed values from original metadata and unresolved subset recounts.

## single_agent_rep2

Original task success remains `true`. Coverage: 156 assays / 76 documents in retained activity rows; 156 / 76 atomic IDs in 1580 clean compounds. Clean-cell serialization has 167 assay combinations / 129 document combinations; these are not assay/document counts.

Whole-set exact Murcko frequencies: adamantane **90/1580**; phenyl-urea-adamantyl **49/1580**; acyclic **57/1580**; **635** distinct scaffolds including the empty scaffold. Complete frequencies are in `murcko_scaffold_frequencies.csv`.

Original scaffold call 39 selected nodes [181, 5, 4]. Its stored preview reports adamantane=32, phenyl-urea-adamantyl=31 (a missing value means absent from the retained preview, not zero). Scope is a selected-node population. Independent subset recount: **unresolved**; denominator=None. Original per-molecule node membership is not available in these previews; no projection inference rerun.

Original scaffold call 40 selected nodes [3, 213, 691, 4]. Its stored preview reports adamantane=None, phenyl-urea-adamantyl=5 (a missing value means absent from the retained preview, not zero). Scope is a selected-node population. Independent subset recount: **unresolved**; denominator=None. Original per-molecule node membership is not available in these previews; no projection inference rerun.

The scaffold numbers in the response match the first selected-node call above. They must be labeled as selected-region observations, not whole-dataset frequencies. The whole-dataset ranking is shown above.

Original claim excerpts and original report paths/hashes are retained in `reconciliation.json`; the ledger separates verified reconstructed values from original metadata and unresolved subset recounts.

## single_agent_rep3

Original task success remains `true`. Coverage: 156 assays / 76 documents in retained activity rows; 156 / 76 atomic IDs in 1580 clean compounds. Clean-cell serialization has 167 assay combinations / 129 document combinations; these are not assay/document counts.

Whole-set exact Murcko frequencies: adamantane **90/1580**; phenyl-urea-adamantyl **49/1580**; acyclic **57/1580**; **635** distinct scaffolds including the empty scaffold. Complete frequencies are in `murcko_scaffold_frequencies.csv`.

Original scaffold call 40 selected nodes [181, 5, 4, 251, 211, 151, 115, 322]. Its stored preview reports adamantane=55, phenyl-urea-adamantyl=42 (a missing value means absent from the retained preview, not zero). Scope is a selected-node population. Independent subset recount: **unresolved**; denominator=None. Original per-molecule node membership is not available in these previews; no projection inference rerun.

Original scaffold call 41 selected nodes [3, 213, 691, 522, 243, 57, 861, 89]. Its stored preview reports adamantane=None, phenyl-urea-adamantyl=None (a missing value means absent from the retained preview, not zero). Scope is a selected-node population. Independent subset recount: **unresolved**; denominator=None. Original per-molecule node membership is not available in these previews; no projection inference rerun.

The scaffold numbers in the response match the first selected-node call above. They must be labeled as selected-region observations, not whole-dataset frequencies. The whole-dataset ranking is shown above.
The response also calls the 42-count scaffold most frequent although its same stored preview has adamantane at 55.

Original claim excerpts and original report paths/hashes are retained in `reconciliation.json`; the ledger separates verified reconstructed values from original metadata and unresolved subset recounts.

## Implications and remaining limits

These corrections reduce the reported breadth of retained assay/document coverage and restrict selected-node scaffold conclusions to their sampled regions. The retained sets still contain many exact scaffolds; local adamantyl-urea enrichment does not establish that one series represents the entire dataset. Corresponding team/single-agent repetitions have the same recomputed coverage and whole-set scaffold counts, so these metrics do not support a between-architecture scientific advantage.

Mixed endpoints, filtering differences between repetitions, GTM density and node activity remain exploratory context. The reporting correction does not establish improved potency, independent lead discovery, causal SAR, or experimental validation. Generation, retrosynthesis, peptide analyses and noncoverage/non-scaffold narrative claims are outside this correction's scope. Original benchmark outcomes remain unchanged; no replacement success percentage is assigned.

Selected-node scope is recoverable from the stored tool calls, but the single-agent runs retain no CSV with molecule-level node membership. Some result previews and node lists are truncated, so complete subset denominators and scaffold distributions cannot be independently verified without additional original artifacts. Reprojecting would produce reconstructed evidence and is deliberately not substituted for missing historical membership. Inventory and truncation flags are in the JSON. The original search-matched 525 assay IDs remain unresolved at the ID level.
