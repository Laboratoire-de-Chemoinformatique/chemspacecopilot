# Retained standalone-generation pilot

These are two implementation pilots run on 26 September 2026, one per arm,
using `deepseek-flash`, temperature 0 and `max_tokens=8192`. They are not the
predeclared 48-execution frozen comparison or the 12 end-to-end stages. The
task requested ten valid unique autoencoder analogues of CHEMBL3327073 with
parent provenance; it did not test activity-map-guided selection.

| Arm | Artifact task check | Wall time (s) | Observed calls | Calls reporting failure | Recorded total tokens |
|---|---|---:|---:|---:|---:|
| Coordinator plus specialists | Passed, 1/1 | 212.11 | 50 | 2 | 831,595 |
| Flat single agent | Passed, 1/1 | 132.23 | 23 | 0 | 731,144 |

The team recovered from two reported tool failures and fulfilled the artifact
checks. These counts include delegation and other captured tool calls, rather
than only chemical computations. The total tokens include repeated input
context and provider-reported usage; 667,904 and 664,576 input tokens,
respectively, were recorded as cache reads. Pricing was not configured, so no
dollar-cost estimate is supplied. Zero reported failures is not proof of correct
tool selection or an expert factuality assessment.

The team ran first; order and caching were not counterbalanced. Original
environment manifests record different source commits and dirty working trees.
There are no within-arm repetitions, and no independent human review has been
completed. These observations illustrate measured execution overhead and
recovery, but cannot establish relative architectural reliability, repeatability
or a general performance advantage. The frozen-version repeated study remains
necessary before making such claims.

After fixing the requested-count parser and transcript links, the saved outputs
were reanalyzed offline on 28 September; no additional model calls or selected
reruns were made. `summary.json` points to the reanalysis records with hashes.
The unchanged original prompts, responses, tool traces and artifacts are under
`reports/reviewer_revision/architecture_pilot` and the corresponding session
directories listed in the evidence-bundle selection. Corrected review packets
are under `reports/reviewer_revision/pilot_reanalysis_20260928`.
