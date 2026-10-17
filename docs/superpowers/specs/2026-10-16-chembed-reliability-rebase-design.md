# Re-base the reliability study on chembed latents

Date: 2026-10-16
Branch: `claude/chemembed-maps-rebuild-5a00d4`

## Where this runs

Three worktrees are involved and the spec names each explicitly rather than
saying "the study worktree":

| Role | Path |
|---|---|
| This work; new fixtures and the rerun | `.../chemspacecopilot/.claude/worktrees/chemembed-maps-rebuild-5a00d4` |
| Curated sEH input and the LSTM study, read-only here | `.../chemspacecopilot-reviewer-revision` |
| Existing chembed generation results, read-only here | `.../chemspacecopilot-revision-metrics` |

All three sit under `/home/aorlov/Programs/agents/`. Nothing in the two
read-only worktrees is modified. `reports/` is gitignored, so study outputs stay
untracked and are archived separately for the manuscript.

## Problem

The manuscript describes "chemography-guided de novo design", but the reported
reliability study ran two latent spaces at once. `manuscript_reliability.yaml`
pins `latent_engine: "chembed"`, which fixes the *generation* engine, while
every GTM prompt in the same file pins `descriptor_type=autoencoder`, which
fixes the *map* to 256-dimensional latents from the deepchemography LSTM. The
two knobs are deliberately independent. Nothing was broken; the study simply
mapped on one latent space and generated in another.

The guard against a cross-family projection is
`validate_gtm_descriptor_backend` in `gtm_operations.py:3104`. Two things about
it matter here and are easy to get wrong:

- The docstring of `_default_gtm_source_for_descriptor` at
  `gtm_operations.py:835` names it `_assert_descriptor_backend_matches`. No
  function by that name exists anywhere in the repository. The docstring is
  stale and should be corrected while this work is in the area.
- It is called from `project_data_on_gtm` (`gtm_operations.py:1702`) and **not**
  from `data_load_and_prep`, which is the function
  `prepare_revision_fixtures.py` calls. The fixture builder's projection path is
  therefore unguarded today, which is why step 2 adds the call explicitly rather
  than relying on it.

The guard is meaningful for this work: the published chembed manifold carries
`descriptor_backend = "chembed"` as an attribute on the pickled GTM object, not
merely in its sidecar JSON. Manifolds with no such tag are let through with a
warning rather than rejected.

The Methods section compounds this by never stating which descriptors the GTM
was fitted on at all, so a reader cannot tell that every reported map runs on
LSTM latents.

The decision taken here is to tell a single-latent-space story: every reported
number comes from a chembed run, and the LSTM arm is retired rather than
reported alongside.

## Decisions

Made by the user during brainstorming on 2026-10-16:

1. **Scope** — re-base the reported numbers, not merely make chembed the
   default. Every figure and reliability statistic in the paper must come from
   a chembed run.
2. **Live tier** — hold the compound set fixed. Frozen fixtures are rebuilt by
   re-encoding the *same* curated sEH compounds with chembed, so the descriptor
   family is the only variable. The live tier is rerun separately and reported
   as a fresh prospective run with its own retrieval date, not as a matched
   replacement.
3. **LSTM arm** — retired from the paper. Not reported, not compared.
4. **No parameterisation** — hardcode chembed in the study path rather than
   adding flags, accepting that `frozen_inputs_v1` stops being rebuildable from
   HEAD.

## Non-goals

- Removing the `autoencoder` / `deepchemography` descriptor family from `src/`.
  Commit `634a877` deliberately keeps that engine separable for the package the
  peptide workflow needs, and the family remains publicly selectable. Out of
  scope here.
- Refitting any GTM manifold. The published chembed manifold is a
  hyperparameter-matched drop-in; see "Pinned assets".
- Changing `DEFAULT_DESCRIPTOR_TYPE`, currently `"morgan"`. The study pins its
  descriptor family explicitly, which is also better provenance for a paper than
  relying on a library default.
- Peptide workflows. Case 4 uses the WAE, which provides both its landscape and
  its generator in one latent space, and is unaffected.

## Pinned assets

| Asset | Identity |
|---|---|
| chembed manifold | `axelrolov/chemspacecopilot-gtm-chembed` rev `ca1d609ba58a0b2074cee8779d28af8ce435b6ef` |
| manifold file | `GTM_manifold_chembed_900_225.pkl.gz`, sha256 `45d37ae167ed4ab0c0afefcee8e76bf4c5cacbb90d660e6666d1b904a24280e0` |
| chembed checkpoint | `3BioCompBio/chembed-default/model.ckpt` |
| manifold hyperparameters | 900 nodes, 225 basis functions, basis width 1, reg 100, max_iter 200, standardize false, pca_scale true |
| hyperparameter source | `docs/manuscript/gtm-model-attributes.json`, not re-tuned |
| manifold training frameset | `gtm_frameset_training.csv`, 50000 rows, sha256 `60f43a73e072260e161714d52d8a63367ad6a8200b1d3ecd0d11603ac96d314c` |
| curated sEH set | `reports/reviewer_revision/seh/clean_compounds.csv`, 2212 compounds, sha256 `bc4b51eeb040892bb4199d4c7e09345b6e34a1a0bb78ec18fcc5a5d976024bb7` |

The manifold's hyperparameters are identical to the LSTM manifold's and were not
re-tuned, and `unknown_token_substitutions` is 0, so no SELFIES vocabulary
misses occurred during fitting. No refit is required or wanted.

## Steps

### 0. Bring the branch up to main

This worktree sits at `896b380` (PR #134) and contains no chembed code at all.
`origin/main` is `4c9a121` and already carries PR #143 (the chembed engine) and
PR #145 (per-descriptor-family map resolution, both manifolds pinned by
revision, the cross-family guard, the separate cache directory, the prefetch
entry). Merge `origin/main` into this branch before anything else; without it
none of the code the rest of this spec edits exists here.

### 1. Secure asset provenance

The published chembed manifold records its training frameset at
`/tmp/claude-1002/frameset/gtm_frameset_training.csv`. The file is still present
and its hash still matches, but it lives in `/tmp` and it is provenance for a
manifold the paper will cite. Copy it to
`reports/manuscript_reliability_chembed/assets/frameset/gtm_frameset_training.csv`
in this worktree and record both path and sha256.

Separately, the manifold is cached at
`~/.cache/cs_copilot/models/gtm/GTM_manifold_chembed_900_225.pkl.gz` — inside the
*LSTM* cache directory, predating PR #145's split. The per-family resolver now
looks in `DEFAULT_GTM_CHEMBED_MODEL_PATH`
(`~/.cache/cs_copilot/models/gtm-chembed`), misses, and re-downloads. Move the
file into the chembed directory and verify sha256 `45d37ae1…`.

### 2. Hardcode chembed in the fixture builder

`scripts/prepare_revision_fixtures.py` performs deterministic projection onto a
pretrained map — no model fitting, no provider calls. It names `autoencoder` at
six sites which mean two different things:

Descriptor family, the latent space the map is projected on — change all three
to `chembed`:

- line 62, `state["default_descriptor"]`
- lines 89–90, `ops.data_load_and_prep(..., descriptor_type="autoencoder")`
- line 175, the map `register_session_object` call's `descriptor_type`

Generation engine, describing whatever produced `--candidates-csv` rather than
any descriptor family — change both to `chembed`, but **only** because step 3
also replaces the candidate CSV with a chembed-generated one. Changing these
while leaving an LSTM-generated CSV in place would stamp the wrong engine on a
frozen artifact:

- line 196, `metadata["engine"]`
- line 206, the candidate-set `label`

Then close the unguarded path. `data_load_and_prep` does not call
`validate_gtm_descriptor_backend`, so add the call in the builder immediately
after projection, reusing the shipped function rather than writing a bespoke
assertion:

```python
ops.validate_gtm_descriptor_backend(model, "chembed")
```

That one line is what stops the builder projecting chembed vectors onto the LSTM
manifold, or the reverse. Both latents are 256 wide, so shapes agree and the
coordinates come back silently meaningless — there is no cheap trial projection
that detects it. The script's existing guards on row count and finite
responsibilities already carry over and need no change.

### 3. Build the chembed candidate set

This is the only step that is not a string swap.

The pinned LSTM candidate CSV, `unique_candidates.csv`, holds 23 distinct
compounds (22 non-parent) deduplicated from 300 raw outputs around seed
`CHEMBL3327073`, parent SMILES
`CCC(C)C(=O)N1CCC(NC(=O)Nc2ccc(C(F)(C(F)(F)F)C(F)(F)F)cc2)CC1`.

A chembed generation run for the same seed at the same raw count already exists
in the `chemspacecopilot-revision-metrics` worktree under
`docs/manuscript/results/generation_chembed/`:

```
panel            parent_id       raw  validity  distinct  nonparent  targets
generation_sweep CHEMBL3327073   300     1.0      106        105        30
```

No new generation is needed. Two consequences must be handled rather than
absorbed silently:

- **Scale.** chembed yields 105 distinct non-parent structures against the
  LSTM's 22 — roughly a fivefold increase in the frozen Case 2 and Case 3
  fixture, and Case 3 searches retrosynthetic routes against it. Apply the same
  pre-declared target rule the LSTM run used: rank candidates by
  `SHA256("42|" + canonical_smiles)` and take the first 10, with no route
  outcome informing selection. `retrosynthesis_chembed/targets_declared.csv`
  indicates this rule was already applied there; verify it against the rule
  rather than re-deriving it.
- **Validity is not comparable.** SELFIES decode to a valid molecule by
  construction, so chembed's validity is 1.0 by definition. It must never be
  compared with the LSTM decoder's pooled 29.0%. Report distinct yield,
  Tanimoto to the parent, and SA instead.

Build a chembed `unique_candidates.csv` carrying the same columns as the LSTM
file (`canonical_smiles`, occurrences, parent Tanimoto, physicochemical
properties) so the fixture builder and downstream figure scripts need no schema
changes. `all_candidates.csv` in that directory is a bare `canonical_smiles`
column of 1687 raw rows and is *not* a drop-in.

### 4. Rebuild the frozen fixtures

Run the builder into `frozen_inputs_v2_chembed/` with the same curated input and
the chembed manifold:

```bash
AGENTS=/home/aorlov/Programs/agents
PYTHONPATH=src python scripts/prepare_revision_fixtures.py \
  --curated-csv    "$AGENTS/chemspacecopilot-reviewer-revision/reports/reviewer_revision/seh/clean_compounds.csv" \
  --gtm-model      ~/.cache/cs_copilot/models/gtm-chembed/GTM_manifold_chembed_900_225.pkl.gz \
  --candidates-csv reports/manuscript_reliability_chembed/generation_chembed/unique_candidates.csv \
  --output-dir     reports/manuscript_reliability_chembed/frozen_inputs_v2_chembed
```

`--provenance` must also be supplied; it is a required argument naming the
curation provenance file recorded into `data_file_paths`. Reuse the same
provenance file the v1 build used, since the curated compound set is unchanged.

Expected: `seh_projection.csv` carries a `chembed_embedding` column and no
`autoencoder_embedding`; `manifest.json` records `model_sha256` `45d37ae1…`;
`projection_summary.json` reports `input_dimensions` 256 and `sample_count`
2212. Node occupancy and entropy will differ from v1 — that is the measurement,
not a regression.

The builder calls `destination.mkdir(parents=True, exist_ok=False)`, so it
refuses to write into an existing directory. A rebuild after a failed attempt
means removing the partial output deliberately, never silently overwriting a
promoted fixture set.

### 5. Flip the study configuration

Edit `tests/robustness/manuscript_reliability.yaml` in place. Change
`descriptor_type=autoencoder` to `descriptor_type=chembed` at lines 66, 126, 134
and 139. `latent_engine: "chembed"` is already correct and stays. Update the
file's "Latent engine" comment block, which currently explains how to reproduce
earlier LSTM results, to describe the single-latent-space configuration.

Repoint the four frozen-tier fixture environment variables at
`frozen_inputs_v2_chembed/` and set each matching `*_SHA256`:

- `CS_COPILOT_SEH_INPUT_FIXTURE`
- `CS_COPILOT_SEH_ANALYSIS_FIXTURE`
- `CS_COPILOT_SEH_CANDIDATES_FIXTURE`
- `CS_COPILOT_PEPTIDE_INPUT_FIXTURE` (unchanged; Case 4 is WAE)

Required fixtures fail closed: a missing path, unresolved variable, malformed
JSON or checksum mismatch counts as a benchmark failure and never falls back to
live data.

### 6. Human verification gate

The fixture README requires frozen fixtures to be built from a manually
verified run and inspected before promotion, with stringified non-pointer Python
objects called out as a sign the state needs cleaning. Inspect
`frozen_inputs_v2_chembed/` and its referenced artifacts, then promote. This gate
belongs to the operator, not to the implementation.

### 7. Execute and summarise

The existing chain runs unmodified:

```
scripts/plan_revision_reliability.py
scripts/execute_revision_study.py
scripts/summarize_reliability_study.py
```

60 executions: 48 frozen (6 batches of 8, both the team and single-agent arms,
alternating arm order by prompt variant) and 12 live. Provider is
`deepseek-flash` at temperature 0.0. The plan records its own
`plan_commit_at_planning`, which will differ from the LSTM study's `55863a39…`;
that is expected.

The LSTM study's retained figures — completion rate 0.80 overall and 0.8125
frozen over 60 retained executions — are recorded here as context for judging
whether the descriptor swap broke orchestration. They are not a reported
comparison.

### 8. Methods

State the single latent space explicitly, closing the gap where the Methods
never named the GTM's descriptors at all:

- Chemical-space mapping and de novo small-molecule generation both use
  256-dimensional latents from the SELFIES transformer variational autoencoder
  (chembed), whose published checkpoint is downloaded from
  `3BioCompBio/chembed-default`. No training was performed.
- All GTM maps reported were fitted and projected on these descriptors, on the
  published manifold pinned by revision.
- A manifold is valid only for the latent space it was fitted on, so each
  descriptor family resolves its own published manifold and its own cache, and
  every projection in the reported pipeline is checked against the manifold's
  recorded descriptor backend before it runs. State this as a property of the
  reported pipeline, which it is once step 2 lands, rather than as a blanket
  claim about every code path in the library.
- Peptide representation and sampling use the WAE, which provides both the
  landscape and the generator for that workflow.
- chembed validity is 1.0 by construction and is not a comparative metric.

The C57 limitation stands and is unchanged by this work: small-molecule
generation is seed-centred and never reads node coordinates, so for small
molecules the workflow is chemography-guided *seed selection* followed by
latent-space design. Node-conditioned sampling holds only for peptides, where
`sample_peptides_from_landscape` genuinely conditions on nodes. A single latent
space removes the two-space objection; it does not create node-guided sampling.

## Verification

- Unit: the fixture builder writes `descriptor_type: "chembed"` into the map
  session object and a `chembed_embedding` column into the projection, and
  records `engine: "chembed"` in the candidate metadata.
- Unit: the declared target rule is reproducible — ranking the chembed candidate
  set by `SHA256("42|" + canonical_smiles)` and taking the first 10 reproduces
  `targets_declared.csv` exactly.
- Integration, the shipped guard: projecting the chembed manifold with
  `descriptor_type="autoencoder"` must raise `ValueError` from
  `validate_gtm_descriptor_backend`, not silently succeed. This is the single
  most important check in the spec, because its failure mode is a wrong answer
  rather than an error.
- Regression: the builder calls `validate_gtm_descriptor_backend`. Without that
  call the check does not run on this path at all, so a test that only exercises
  the guard directly would still pass while the builder stayed unprotected.
- Study: the frozen tier completes at or above the configured
  `reliability_min_success_rate` of 0.80.

## Risks

| Risk | Handling |
|---|---|
| `/tmp` frameset cleared before copying | Step 1 runs first, before anything else |
| Fivefold larger candidate set inflates Case 3 runtime | Pre-declared 10-target rule caps it, as in the LSTM run |
| chembed validity misread as a quality gain | Methods states it is 1.0 by construction; report distinct yield, Tanimoto and SA |
| Live tier drift confounded with the descriptor change | Live tier reported as a fresh prospective run with its own retrieval date, never as a matched replacement |
| `frozen_inputs_v1` no longer rebuildable from HEAD | Accepted by decision 4; v1 artifacts and hashes persist on disk and remain reproducible from git history |
| Wrong-manifold projection returns meaningless coordinates silently | Both latents are 256 wide so no shape check catches it; step 2 adds the `validate_gtm_descriptor_backend` call the builder's path currently lacks |
| An untagged manifold passes the guard with only a warning | The published chembed manifold carries `descriptor_backend = "chembed"` on the pickled object; verified, and asserted as a plan step before the rebuild |
