# Reviewer revision workspace

Work is on `codex/reviewer-revision`, based on the existing
`prompt-skill-source-of-truth` implementation. Its runtime and benchmark machinery
were reused; the peptide aggregate-landscape sampler reuses the relevant core of
branch `114-refactor-peptide-desginer-to-new-hf-dataset-structures`. The historical
main checkout was not rewritten. These are prospective revision measurements,
not a reconstruction of the original case studies.

The shortest remaining path is to finish the small matched agent evaluation,
review the scientific outputs, and insert the prepared text/tables/figures into
the editable manuscript. No large benchmark or model retraining is needed.

| Reviewer request | Evidence now available | Work still required |
|---|---|---|
| 1. Reliability | Instrumented harness, fixed prompts, objective artifact checks, retained two-arm generation pilot, timing/tool/usage records | Run the predeclared repetitions after provider authorization; inspect outputs and finalize success/failure table |
| 2. Architecture | Same-model flat-agent comparison; explicit explanation of specialization and overhead | Complete matched repetitions; report the observed tradeoff without assuming multi-agent superiority |
| 3. sEH/GTM methods | ChEMBL 37 snapshot, exact curation, 2,212 compounds, identical offline replay, actual checkpoint parameters, finite projections, occupancy diagnostics | Apply the prospective replacement methods and maps to the paper |
| 4. Generation/routes | 300 raw outputs, 87 valid, 23 unique; complete MOSES reference comparison; 4/10 predicted routes with all failures retained | Author/chemical review; keep training-linkage and experimental-validation limitations explicit |
| 5. Reproducibility/presentation | Versioned code and data/model hashes, full raw evidence, corrected citations/text, vector structures/maps/routes | Apply changes to editable paper, archive final source/data release, insert version DOI and page/line references |

Start with [the response draft](reviewer-response.md) and
[copy-ready manuscript passages](revision-text.md). The complete prospective
protocol is in [manuscript-revision.md](../testing/manuscript-revision.md).
The compact scientific results are:

- [Generation and reference-set membership](results/generation/README.md).
- [sEH projection and source tables](results/seh/README.md).
- [All ten retrosynthesis outcomes and route figures](results/retrosynthesis/README.md).
- [Publication figure captions](figures/README.md).
- [Peptide aggregate-landscape method](peptide-landscape-methods.md).

The 48 frozen executions are four tasks × two phrasings × three repetitions ×
two architectures. Six batches counterbalance arm order. Nine additional live
ChEMBL case/stages form three connected workflows; three independent peptide
workflows start from the pinned public aggregate landscape. The latter do not
retrieve raw DBAASP measurements. Pilots, dependency controls and offline
scientific-tool runs are separate from these denominators.

External execution currently requires an explicit answer to the provider
authorization request: automatic approval review rejected the next DeepSeek
pilot because its prompts, public scientific inputs, session state and tool
outputs would be sent to `https://api.deepseek.com`. The offline analyses above
are complete independently of that request. No rejected pilot was executed.

The original editable manuscript and the author's intended Zenodo record are
still unresolved. Exact generator checkpoint/training-corpus linkage is also
unverified; absence from the official MOSES reference is reported separately.
An automated review or unit test is not independent human chemistry review.
The response draft deliberately keeps these conditions open.

## Local evidence bundle

`offline-evidence-selection.json` explicitly selects the raw data, outputs,
figures, pinned assets and original source commits for a local revision
checkpoint. It retains failed implementation controls and pilot transcripts.
The packager copies original records without rewriting absolute paths and
records their mapping and SHA-256 hashes. Source snapshots come from Git, not
from the current uncommitted working tree.

```bash
python scripts/package_revision_evidence.py \
  docs/manuscript/offline-evidence-selection.json \
  --output /absolute/path/to/new-revision-evidence.zip
```

After extraction, `python verify.py` verifies every bundled file against its
manifest. The archive is labeled an unpublished working revision; building it
does not assign a DOI or establish publication readiness. Preserve source
attributions and the measured ARM dependency deviations when preparing the
final public deposit.
