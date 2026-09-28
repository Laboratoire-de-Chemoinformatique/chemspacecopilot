# Prospective molecular generation measurements

Three independent CPU runs generated 100 raw model outputs each around CHEMBL3327073 using seeds 11, 22 and 33, latent noise scale 0.1 and sampling temperature 0.5. All outputs, including invalid strings and duplicates, were retained in the archived raw audits named and hashed in `summary.json`. The checkpoint hashes, source commit, software and hardware are also recorded there.

| Seed | Raw outputs | RDKit valid | Distinct standardized valid | Uniqueness among valid | Worker wall time (s) |
| --- | ---: | ---: | ---: | ---: | ---: |
| 11 | 100 | 27 | 9 | 33.3% | 4.00 |
| 22 | 100 | 28 | 12 | 42.9% | 4.04 |
| 33 | 100 | 32 | 15 | 46.9% | 3.68 |

Pooled validity was 87/300 (29.0%). The pooled valid outputs contained 23 distinct standardized compounds (26.4% pooled uniqueness), including one parent reconstruction. An additional seed-11 run reproduced all 100 raw strings exactly under this recorded environment; it is excluded from the primary 300-output denominator. These measurements characterize the selected local-latent generation settings and do not establish biological activity or synthetic feasibility. They do not reconstruct historical runs or evaluate GTM-conditioned selection.

`unique_candidates.csv` contains exact canonical identities, occurrences, parent similarity and physicochemical properties. Standardization includes stereochemistry removal. Parent similarity uses radius-2, 2048-bit Morgan fingerprint Tanimoto. Training-set novelty remains unavailable until the exact checkpoint training corpus is established.

As a separately labeled reference comparison, every one of the 1,584,663 rows
in the official MOSES train split at commit
`dd7ed6ab38e23afd3ef5371d67939a1760bd8599` was standardized with the identical
policy. All rows standardized successfully. None of the 23 unique generated
structures was present in that standardized reference (23/23 absent).
`moses_reference_membership.json` records the reference URL/hash, implementation
hashes, complete denominator and per-candidate membership. This is **reference
set absence**, not confirmed checkpoint training-set novelty; the precise
checkpoint/split linkage remains unavailable. The reference row count is not a
claimed count of unique standardized reference molecules.

The reference comparison is reproduced with the pinned CSV:

```bash
PYTHONPATH=src python scripts/compare_reference_corpus.py \
  docs/manuscript/results/generation/unique_candidates.csv \
  /absolute/path/to/moses_train.csv \
  --reference-name 'MOSES train at dd7ed6ab38e23afd3ef5371d67939a1760bd8599' \
  --reference-url https://media.githubusercontent.com/media/molecularsets/moses/dd7ed6ab38e23afd3ef5371d67939a1760bd8599/data/train.csv \
  --expected-reference-sha256 49fe1aae29604ec0f5023ea34edc8789415f954138f2903dabd005a4b888a961 \
  --output /absolute/path/to/new-membership.json --workers 6
```

The generation manifest recorded a dirty working tree. A later read-only
verification compared the six recorded scientific implementation files, including
the launcher and execution ledger, against their immutable recorded Git commit.
All six SHA-256 hashes match; `source_verification.json` contains this check.
This establishes the recorded files' source version without rewriting the
original run's dirty-tree flag or asserting that unlisted files were inspected.

`retrosynthesis_targets.csv` was declared before route searches: all 22 distinct nonparent candidates were ranked by SHA256 of `42|<canonical_smiles>`, then the first 10 were selected. `retrosynthesis_selection.json` records the rule, declaration time, source hashes and target-file hash. No route outcome was used for selection.

Reproduction (from a checkout of the recorded implementation, after supplying the pinned local model files):

```bash
PYTHONPATH=src python scripts/run_generation_study.py \
  --model-dir /absolute/path/to/autoencoder \
  --output-dir /absolute/path/to/new-generation-study
```

The raw audits remain under `reports/reviewer_revision/generation_3x100` for archival; they are not embedded in the compact manuscript tables. The repeat is separately stored under `reports/reviewer_revision/generation_repeat_seed11`.
