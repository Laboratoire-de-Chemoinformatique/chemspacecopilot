# chembed Maps Re-base Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Put every reported GTM map on chembed latents, so the reliability study maps and generates in one latent space.

**Architecture:** The chembed engine, the per-descriptor-family manifold resolver and the published chembed manifold already shipped in PRs #143 and #145; no new modelling is needed. The work is to point `prepare_revision_fixtures.py` at the chembed manifold, close the unguarded projection path it uses, rebuild the frozen sEH fixtures from the *same* 2212 curated compounds, flip the four descriptor pins in the study config, and rerun the 60-execution suite. The LSTM arm is retired, not compared.

**Tech Stack:** Python 3, RDKit, PyTorch, pandas, numpy, pytest, chemographykit GTM, huggingface_hub, DeepSeek (`deepseek-flash`) as the study provider.

**Spec:** `docs/superpowers/specs/2026-10-16-chembed-reliability-rebase-design.md`

## Global Constraints

- Descriptor family for every reported map: `chembed`. Generation engine: `chembed` (`latent_engine: "chembed"`, already set).
- Manifold is pinned, never refitted: `axelrolov/chemspacecopilot-gtm-chembed` rev `ca1d609ba58a0b2074cee8779d28af8ce435b6ef`, file `GTM_manifold_chembed_900_225.pkl.gz`, sha256 `45d37ae167ed4ab0c0afefcee8e76bf4c5cacbb90d660e6666d1b904a24280e0`.
- Manifold hyperparameters, unchanged from the LSTM manifold and not re-tuned: 900 nodes, 225 basis functions, basis width 1, reg_coeff 100, max_iter 200, `standardize` false, `pca_scale` true.
- Curated compound set is held fixed: `clean_compounds.csv`, 2212 compounds, sha256 `bc4b51eeb040892bb4199d4c7e09345b6e34a1a0bb78ec18fcc5a5d976024bb7`.
- Do **not** remove the `autoencoder` / `deepchemography` descriptor family from `src/`. Out of scope.
- Do **not** change `DEFAULT_DESCRIPTOR_TYPE`, currently `"morgan"`.
- Do **not** push any branch based on the pre-scrub lineage to `origin`. This branch was rebased onto `origin/main` (`4c9a121`) and is safe.
- chembed validity is 1.0 by construction (SELFIES always decode). It is never a comparative metric against the LSTM's pooled 29.0%.
- Venv: `/home/aorlov/Programs/agents/chemspacecopilot/.venv`. Tests run with `-m "not live"` by default; subprocess/network tests carry the `live` marker.
- Three worktrees, all under `/home/aorlov/Programs/agents/`: this one (`chemspacecopilot/.claude/worktrees/chemembed-maps-rebuild-5a00d4`) is the only one written to. `chemspacecopilot-reviewer-revision` and `chemspacecopilot-revision-metrics` are read-only sources.

## Review Focus

Five failure modes the spec implies that no task's happy path exercises. Each has a test assigned to the task owning the code.

1. **A manifold with no `descriptor_backend` tag passes the guard with only a warning.** A reasonable person expects a wrong-manifold projection to be refused, not logged. Pinned in Task 1.
2. **`data_load_and_prep` does not call the guard at all.** A test that exercises `validate_gtm_descriptor_backend` directly passes while the builder stays unprotected. Pinned in Task 3 by asserting the builder calls it.
3. **A candidate CSV whose `canonical_smiles` column contains blanks or unparseable SMILES.** The builder only rejects a wholly empty list, so a partly-invalid set is frozen silently. Pinned in Task 2.
4. **A candidate set that is all parent reconstructions.** Target selection filters non-parent entries; an all-parent set yields zero targets and Case 3 has nothing to plan. Pinned in Task 2.
5. **A fixture whose recorded SHA-256 no longer matches its file.** The harness must fail closed and never fall back to live data. Pinned in Task 5.

---

### Task 1: Secure the pinned assets

The published chembed manifold's provenance points into `/tmp`, and the manifold itself sits in the LSTM cache directory where the per-family resolver will not look.

**Files:**
- Create: `reports/manuscript_reliability_chembed/assets/frameset/gtm_frameset_training.csv` (copy; `reports/` is gitignored)
- Create: `reports/manuscript_reliability_chembed/assets/frameset/provenance.json`
- Move: `~/.cache/cs_copilot/models/gtm/GTM_manifold_chembed_900_225.pkl.gz` and its `.json` sidecar into `~/.cache/cs_copilot/models/gtm-chembed/`

**Interfaces:**
- Consumes: nothing.
- Produces: the manifold at `~/.cache/cs_copilot/models/gtm-chembed/GTM_manifold_chembed_900_225.pkl.gz`, which Tasks 3 and 4 pass as `--gtm-model`.

- [ ] **Step 1: Copy the frameset out of `/tmp` before anything else touches it**

```bash
cd /home/aorlov/Programs/agents/chemspacecopilot/.claude/worktrees/chemembed-maps-rebuild-5a00d4
mkdir -p reports/manuscript_reliability_chembed/assets/frameset
cp /tmp/claude-1002/frameset/gtm_frameset_training.csv \
   reports/manuscript_reliability_chembed/assets/frameset/gtm_frameset_training.csv
```

- [ ] **Step 2: Verify the copy matches the hash the manifold records**

```bash
sha256sum reports/manuscript_reliability_chembed/assets/frameset/gtm_frameset_training.csv
```

Expected: `60f43a73e072260e161714d52d8a63367ad6a8200b1d3ecd0d11603ac96d314c`. If it differs, stop — the manifold's recorded training frameset is not this file, and the provenance chain is broken.

- [ ] **Step 3: Record the provenance**

```bash
cat > reports/manuscript_reliability_chembed/assets/frameset/provenance.json <<'EOF'
{
  "purpose": "training frameset of the published chembed GTM manifold, recovered from /tmp",
  "file": "gtm_frameset_training.csv",
  "sha256": "60f43a73e072260e161714d52d8a63367ad6a8200b1d3ecd0d11603ac96d314c",
  "rows": 50000,
  "original_path": "/tmp/claude-1002/frameset/gtm_frameset_training.csv",
  "manifold": "axelrolov/chemspacecopilot-gtm-chembed@ca1d609ba58a0b2074cee8779d28af8ce435b6ef"
}
EOF
```

- [ ] **Step 4: Move the manifold into the cache directory the resolver reads**

```bash
mkdir -p ~/.cache/cs_copilot/models/gtm-chembed
mv ~/.cache/cs_copilot/models/gtm/GTM_manifold_chembed_900_225.pkl.gz \
   ~/.cache/cs_copilot/models/gtm/GTM_manifold_chembed_900_225.json \
   ~/.cache/cs_copilot/models/gtm-chembed/
```

- [ ] **Step 5: Verify the manifold hash and its descriptor tag**

```bash
sha256sum ~/.cache/cs_copilot/models/gtm-chembed/GTM_manifold_chembed_900_225.pkl.gz
/home/aorlov/Programs/agents/chemspacecopilot/.venv/bin/python -c "
import gzip, pickle, os
p = os.path.expanduser('~/.cache/cs_copilot/models/gtm-chembed/GTM_manifold_chembed_900_225.pkl.gz')
with gzip.open(p, 'rb') as fh:
    m = pickle.load(fh)
assert getattr(m, 'descriptor_backend', None) == 'chembed', getattr(m, 'descriptor_backend', '<ABSENT>')
assert m.num_nodes == 900, m.num_nodes
print('manifold tagged chembed, 900 nodes')
"
```

Expected: hash `45d37ae167ed4ab0c0afefcee8e76bf4c5cacbb90d660e6666d1b904a24280e0`, then `manifold tagged chembed, 900 nodes`.

This assertion is Review Focus item 1. An untagged manifold passes `validate_gtm_descriptor_backend` with only a warning, so the tag is what makes every later guard check meaningful. If `descriptor_backend` is absent, stop and re-tag or refetch the manifold before continuing.

- [ ] **Step 6: Confirm the resolver points at the chembed cache for a chembed session**

```bash
cd /home/aorlov/Programs/agents/chemspacecopilot/.claude/worktrees/chemembed-maps-rebuild-5a00d4
PYTHONPATH=src /home/aorlov/Programs/agents/chemspacecopilot/.venv/bin/python -c "
from cs_copilot.tools.constants import (
    HUGGINGFACE_GTM_CHEMBED_REPO, HUGGINGFACE_GTM_CHEMBED_REVISION, DEFAULT_GTM_CHEMBED_MODEL_PATH,
)
print(HUGGINGFACE_GTM_CHEMBED_REPO)
print(HUGGINGFACE_GTM_CHEMBED_REVISION)
print(DEFAULT_GTM_CHEMBED_MODEL_PATH)
"
```

Expected: `axelrolov/chemspacecopilot-gtm-chembed`, `ca1d609ba58a0b2074cee8779d28af8ce435b6ef`, and a path ending `models/gtm-chembed` — the directory Step 4 populated.

- [ ] **Step 7: Commit the provenance record**

```bash
git add -f reports/manuscript_reliability_chembed/assets/frameset/provenance.json
git commit -m "Record the recovered chembed manifold training frameset

The published manifold recorded its training frameset at a /tmp path. The
file still hashes to the recorded value, so copy it somewhere durable and
pin path, hash and row count before /tmp is cleared.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 2: Verify the chembed candidate set

The chembed generation run for the same seed compound already exists and is already pooled. This task proves it is usable rather than rebuilding it, and pins the two ways a candidate set can be quietly wrong.

**Files:**
- Create: `reports/manuscript_reliability_chembed/generation_chembed/unique_candidates.csv` (copy)
- Create: `reports/manuscript_reliability_chembed/generation_chembed/retrosynthesis_targets.csv` (copy)
- Test: `tests/unit/test_chembed_candidate_selection.py`

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces: `unique_candidates.csv` with columns `canonical_smiles,is_parent,occurrences,seeds,parent_tanimoto,sa_score,molecular_weight,logp,tpsa,hbd,hba,rotatable_bonds,qed`, which Task 4 passes as `--candidates-csv`. Also `declared_targets(rows, count)` in the test module, used nowhere else.

- [ ] **Step 1: Copy the pooled chembed candidate set and its declared targets**

```bash
cd /home/aorlov/Programs/agents/chemspacecopilot/.claude/worktrees/chemembed-maps-rebuild-5a00d4
SRC=/home/aorlov/Programs/agents/chemspacecopilot-revision-metrics/docs/manuscript/results/generation_chembed/generation_sweep/CHEMBL3327073
mkdir -p reports/manuscript_reliability_chembed/generation_chembed
cp "$SRC/unique_candidates.csv" "$SRC/retrosynthesis_targets.csv" \
   reports/manuscript_reliability_chembed/generation_chembed/
```

- [ ] **Step 2: Confirm the shape and schema**

```bash
head -1 reports/manuscript_reliability_chembed/generation_chembed/unique_candidates.csv
wc -l < reports/manuscript_reliability_chembed/generation_chembed/unique_candidates.csv
wc -l < reports/manuscript_reliability_chembed/generation_chembed/retrosynthesis_targets.csv
```

Expected: header `canonical_smiles,is_parent,occurrences,seeds,parent_tanimoto,sa_score,molecular_weight,logp,tpsa,hbd,hba,rotatable_bonds,qed`; 107 lines (106 candidates plus header); 31 lines (30 targets plus header).

The schema differs from the LSTM file: chembed has `sa_score` and lacks `novel_to_training`, `rdkit_valid`, `standardized_valid`. This is safe — `prepare_revision_fixtures.py` reads only `canonical_smiles`, and `render_revision_figures.py` reads only `parent_tanimoto`, both of which are present.

- [ ] **Step 3: Write the failing test**

Create `tests/unit/test_chembed_candidate_selection.py`:

```python
"""The chembed candidate set reproduces its pre-declared retrosynthesis targets."""

import csv
import hashlib
from pathlib import Path

import pytest

CANDIDATES = (
    Path(__file__).parents[2]
    / "reports/manuscript_reliability_chembed/generation_chembed/unique_candidates.csv"
)
TARGETS = (
    Path(__file__).parents[2]
    / "reports/manuscript_reliability_chembed/generation_chembed/retrosynthesis_targets.csv"
)
SELECTION_SEED = 42


def declared_targets(rows, count):
    """Rank non-parent candidates by the pre-declared rule and take the first `count`."""
    eligible = sorted(
        (row for row in rows if row["is_parent"].strip().lower() not in {"true", "1"}),
        key=lambda row: hashlib.sha256(
            f"{SELECTION_SEED}|{row['canonical_smiles']}".encode()
        ).hexdigest(),
    )
    if count > len(eligible):
        raise ValueError(f"Only {len(eligible)} eligible candidates for {count} targets")
    return [row["canonical_smiles"] for row in eligible[:count]]


def _rows(path):
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def test_declared_targets_are_reproducible_from_the_rule():
    rows = _rows(CANDIDATES)
    recorded = [row["smiles"] for row in _rows(TARGETS)]
    assert declared_targets(rows, len(recorded)) == recorded


def test_candidate_smiles_are_all_present_and_parseable():
    from rdkit import Chem, rdBase

    rows = _rows(CANDIDATES)
    assert rows, "candidate set is empty"
    with rdBase.BlockLogs():
        bad = [
            row["canonical_smiles"]
            for row in rows
            if not row["canonical_smiles"].strip()
            or Chem.MolFromSmiles(row["canonical_smiles"]) is None
        ]
    assert bad == [], f"{len(bad)} unparseable candidate SMILES, first: {bad[:3]}"


def test_an_all_parent_candidate_set_cannot_yield_targets():
    rows = [{"canonical_smiles": "CCO", "is_parent": "True"}]
    with pytest.raises(ValueError, match="Only 0 eligible candidates"):
        declared_targets(rows, 10)
```

- [ ] **Step 4: Run the test to verify it fails**

```bash
cd /home/aorlov/Programs/agents/chemspacecopilot/.claude/worktrees/chemembed-maps-rebuild-5a00d4
PYTHONPATH=src /home/aorlov/Programs/agents/chemspacecopilot/.venv/bin/python -m pytest \
  tests/unit/test_chembed_candidate_selection.py -v
```

Expected if Step 1 has not run: FAIL with `FileNotFoundError`. To see the selection logic itself fail rather than the file lookup, run this before Step 1. The third test must pass regardless, since it builds its own rows.

- [ ] **Step 5: Run the test to verify it passes**

```bash
PYTHONPATH=src /home/aorlov/Programs/agents/chemspacecopilot/.venv/bin/python -m pytest \
  tests/unit/test_chembed_candidate_selection.py -v
```

Expected: 3 passed. If `test_declared_targets_are_reproducible_from_the_rule` fails, the copied `retrosynthesis_targets.csv` was not produced by this rule — stop and regenerate it with `pool_generation_study.py --target-count 30 --target-prefix chembed` rather than editing the test to match.

`test_candidate_smiles_are_all_present_and_parseable` is Review Focus item 3 — the builder rejects only a wholly empty candidate list, so without this a partly-invalid set would be frozen silently. `test_an_all_parent_candidate_set_cannot_yield_targets` is Review Focus item 4.

- [ ] **Step 6: Commit**

```bash
git add tests/unit/test_chembed_candidate_selection.py
git add -f reports/manuscript_reliability_chembed/generation_chembed/unique_candidates.csv \
          reports/manuscript_reliability_chembed/generation_chembed/retrosynthesis_targets.csv
git commit -m "Verify the chembed candidate set against its declared target rule

The pooled chembed run for CHEMBL3327073 already exists, so pin it rather
than regenerate it: reproduce the pre-declared selection from
sha256(42|<canonical_smiles>) over non-parent candidates, and reject a set
that is empty, unparseable or all parent reconstructions.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 3: Point the fixture builder at chembed and close its unguarded path

Five string swaps, plus the one call that stops a silent wrong-manifold projection.

**Files:**
- Modify: `scripts/prepare_revision_fixtures.py` lines 62, 90, 175, 196, 206
- Modify: `src/cs_copilot/tools/chemography/gtm_operations.py:835` (stale docstring reference)
- Test: `tests/unit/test_revision_fixtures.py` (append)

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces: `build(args)` emitting `state["default_descriptor"] == "chembed"`, a map session object with `descriptor_type == "chembed"`, candidate metadata with `engine == "chembed"`, and calling `ops.validate_gtm_descriptor_backend(model, "chembed")`. Task 4 runs this.

- [ ] **Step 1: Write the failing test**

Append to `tests/unit/test_revision_fixtures.py`, and add `import json` and `import types` to its imports if absent:

```python
def test_builder_projects_chembed_and_checks_the_manifold_backend(tmp_path, monkeypatch):
    """The builder must project chembed AND verify the manifold was fitted on it.

    data_load_and_prep does not call validate_gtm_descriptor_backend, so without
    an explicit call here nothing stops a chembed projection onto the LSTM
    manifold: both latents are 256 wide, so shapes agree and the coordinates come
    back silently meaningless.
    """
    import numpy as np

    calls = {}

    curated = tmp_path / "curated.csv"
    pd.DataFrame(
        {
            "canonical_smiles": ["CCO", "CCN"],
            "pIC50": [7.0, 5.0],
            "activity_class": ["active", "inactive"],
            "activity_binary": [1, 0],
            "standard_type": ["IC50"] * 2,
            "standard_units": ["nM"] * 2,
            "standard_value": [100, 10000],
        }
    ).to_csv(curated, index=False)
    provenance = tmp_path / "provenance.json"
    provenance.write_text("{}")
    model_file = tmp_path / "manifold.pkl.gz"
    model_file.write_bytes(b"not-a-real-pickle")
    candidates = tmp_path / "candidates.csv"
    pd.DataFrame({"canonical_smiles": ["CCOC", "CCNC"]}).to_csv(candidates, index=False)

    gtm = types.SimpleNamespace(num_nodes=2, descriptor_backend="chembed")
    projected = pd.DataFrame({"smi": ["CCO", "CCN"], "pIC50": [7.0, 5.0]})
    responsibilities = np.array([[0.6, 0.4], [0.3, 0.7]])
    descriptors = np.zeros((2, 256))

    def fake_data_load_and_prep(dataset, model_path, descriptor_type=None, **kwargs):
        calls["descriptor_type"] = descriptor_type
        return gtm, projected.copy(), descriptors, responsibilities

    def fake_validate(model, descriptor_type):
        calls["validated"] = (model is gtm, descriptor_type)

    import cs_copilot.tools.chemography.gtm_operations as real_ops

    monkeypatch.setattr(real_ops, "data_load_and_prep", fake_data_load_and_prep)
    monkeypatch.setattr(real_ops, "validate_gtm_descriptor_backend", fake_validate)
    monkeypatch.setattr(
        real_ops, "calculate_latent_coords", lambda r, **k: pd.DataFrame({"x": [0.0, 1.0]})
    )
    monkeypatch.setattr(real_ops, "get_density_matrix", lambda r: r)
    monkeypatch.setattr(real_ops, "density_to_table", lambda d, **k: pd.DataFrame({"node": [0, 1]}))
    monkeypatch.setattr(real_ops, "get_reg_density_matrix", lambda r, a: (r, r))
    monkeypatch.setattr(
        real_ops, "reg_density_to_table", lambda d, a, **k: pd.DataFrame({"node": [0, 1]})
    )

    args = types.SimpleNamespace(
        curated_csv=curated,
        provenance=provenance,
        gtm_model=model_file,
        candidates_csv=candidates,
        output_dir=tmp_path / "out",
    )
    manifest = module.build(args)

    assert calls["descriptor_type"] == "chembed"
    assert calls["validated"] == (True, "chembed"), "builder never checked the manifold backend"

    analysis = json.loads((tmp_path / "out" / "seh_analysis.json").read_text())["session_state"]
    assert analysis["default_descriptor"] == "chembed"
    assert '"descriptor_type": "chembed"' in json.dumps(analysis, indent=2)
    assert "autoencoder" not in json.dumps(analysis)

    candidates_state = json.loads((tmp_path / "out" / "seh_candidates.json").read_text())
    assert "autoencoder" not in json.dumps(candidates_state)
    assert "chembed" in json.dumps(candidates_state)
    assert "autoencoder" not in json.dumps(manifest)
```

The assertions deliberately check for the absence of `"autoencoder"` anywhere in the emitted JSON rather than indexing a specific nested key, because `register_session_object` owns that nesting and this test should not encode its shape.

- [ ] **Step 2: Run the test to verify it fails**

```bash
cd /home/aorlov/Programs/agents/chemspacecopilot/.claude/worktrees/chemembed-maps-rebuild-5a00d4
PYTHONPATH=src /home/aorlov/Programs/agents/chemspacecopilot/.venv/bin/python -m pytest \
  "tests/unit/test_revision_fixtures.py::test_builder_projects_chembed_and_checks_the_manifold_backend" -v
```

Expected: FAIL on `assert calls["descriptor_type"] == "chembed"` — the builder still passes `"autoencoder"`.

- [ ] **Step 3: Swap the three descriptor-family sites**

In `scripts/prepare_revision_fixtures.py`, replace the `"autoencoder"` literal at each of these three places. Line 62, inside the `state` dict:

```python
        "default_descriptor": "chembed",
```

Line 90, the projection call:

```python
        str(view_path), str(model_path), descriptor_type="chembed"
```

Line 175, inside the map `register_session_object` payload:

```python
            "descriptor_type": "chembed",
```

- [ ] **Step 4: Swap the two generation-engine sites**

These describe the engine that produced `--candidates-csv`, not a descriptor family. They change only because Task 2 supplies a chembed-generated candidate set. Line 196:

```python
        "engine": "chembed",
```

Line 206:

```python
        label="Measured seeded chembed candidates",
```

- [ ] **Step 5: Add the guard call the builder's path lacks**

Immediately after the `data_load_and_prep` call and its responsibilities check, before `calculate_latent_coords`:

```python
    ops.validate_gtm_descriptor_backend(model, "chembed")
```

Reuse the shipped function. Do not write a bespoke shape or backend assertion — `validate_gtm_descriptor_backend` already raises with the right message and is already covered by `tests/unit/test_gtm_model_loading.py`.

- [ ] **Step 6: Run the test to verify it passes**

```bash
PYTHONPATH=src /home/aorlov/Programs/agents/chemspacecopilot/.venv/bin/python -m pytest \
  tests/unit/test_revision_fixtures.py -v
```

Expected: 3 passed — the two pre-existing curated-view tests plus the new one.

- [ ] **Step 7: Prove the guard actually refuses the wrong manifold**

```bash
PYTHONPATH=src /home/aorlov/Programs/agents/chemspacecopilot/.venv/bin/python -c "
from types import SimpleNamespace
from cs_copilot.tools.chemography.gtm_operations import validate_gtm_descriptor_backend
gtm = SimpleNamespace(descriptor_backend='chembed')
try:
    validate_gtm_descriptor_backend(gtm, 'autoencoder')
except ValueError as exc:
    print('refused as expected:', str(exc)[:60])
else:
    raise SystemExit('GUARD DID NOT FIRE')
"
```

Expected: `refused as expected: GTM manifold was fitted on 'chembed' descriptors but 'autoen`

This is Review Focus item 2. Step 1's assertion proves the builder *calls* the guard; this step proves the guard *refuses*. Both are needed — either alone leaves the silent-wrong-answer path open.

- [ ] **Step 8: Correct the stale docstring**

`src/cs_copilot/tools/chemography/gtm_operations.py:835` names `_assert_descriptor_backend_matches`, which exists nowhere in the repository. Replace that name with the real one:

```python
    they project fingerprints, which ``validate_gtm_descriptor_backend``
```

- [ ] **Step 9: Run the surrounding test files**

```bash
PYTHONPATH=src /home/aorlov/Programs/agents/chemspacecopilot/.venv/bin/python -m pytest \
  tests/unit/test_revision_fixtures.py tests/unit/test_gtm_model_loading.py \
  tests/unit/test_chembed_engine.py tests/unit/test_chembed_candidate_selection.py -v
```

Expected: all pass.

- [ ] **Step 10: Commit**

```bash
git add scripts/prepare_revision_fixtures.py tests/unit/test_revision_fixtures.py \
        src/cs_copilot/tools/chemography/gtm_operations.py
git commit -m "Project the frozen sEH fixtures on chembed latents

The builder pinned descriptor_type=autoencoder, so every frozen map ran on
LSTM latents while generation ran on chembed. Point it at chembed, and
relabel the candidate-set engine because the candidate CSV is now a chembed
run rather than an LSTM one.

data_load_and_prep never calls validate_gtm_descriptor_backend, so this path
could project chembed vectors onto the LSTM manifold in silence: both
latents are 256 wide, shapes agree and the coordinates mean nothing. Call
the guard explicitly, and fix the docstring that named a function which does
not exist.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 4: Rebuild the frozen fixtures

**Files:**
- Create: `reports/manuscript_reliability_chembed/frozen_inputs_v2_chembed/` (12 files, written by the builder)

**Interfaces:**
- Consumes: the manifold from Task 1, the candidate CSV from Task 2, the builder from Task 3.
- Produces: `seh_input.json`, `seh_analysis.json`, `seh_candidates.json` and `manifest.json`, whose paths and SHA-256 digests Task 5 exports as environment variables.

- [ ] **Step 1: Resolve the provenance file the v1 build used**

The curated compound set is unchanged, so the same curation provenance file applies. Read it out of the v1 fixture rather than guessing a path:

```bash
cd /home/aorlov/Programs/agents/chemspacecopilot/.claude/worktrees/chemembed-maps-rebuild-5a00d4
V1=/home/aorlov/Programs/agents/chemspacecopilot-reviewer-revision/reports/reviewer_revision/frozen_inputs_v1/seh_input.json
PROV=$(/home/aorlov/Programs/agents/chemspacecopilot/.venv/bin/python -c "
import json, sys
state = json.load(open('$V1'))['session_state']
print(state['data_file_paths']['curation_provenance_path'])
")
echo "provenance: $PROV"
test -f "$PROV" && echo "exists" || echo "MISSING - stop and locate it before building"
```

Expected: a path that exists. Keep `$PROV` exported in the same shell for Step 2.

- [ ] **Step 2: Run the builder**

```bash
AGENTS=/home/aorlov/Programs/agents
REV=$AGENTS/chemspacecopilot-reviewer-revision/reports/reviewer_revision
OUT=reports/manuscript_reliability_chembed
PYTHONPATH=src $AGENTS/chemspacecopilot/.venv/bin/python scripts/prepare_revision_fixtures.py \
  --curated-csv    "$REV/seh/clean_compounds.csv" \
  --provenance     "$PROV" \
  --gtm-model      ~/.cache/cs_copilot/models/gtm-chembed/GTM_manifold_chembed_900_225.pkl.gz \
  --candidates-csv "$OUT/generation_chembed/unique_candidates.csv" \
  --output-dir     "$OUT/frozen_inputs_v2_chembed"
```

The builder calls `destination.mkdir(parents=True, exist_ok=False)` and refuses an existing directory. After a failed attempt, remove the partial output deliberately before retrying.

- [ ] **Step 3: Verify the projection is on chembed**

```bash
OUT=reports/manuscript_reliability_chembed/frozen_inputs_v2_chembed
head -1 $OUT/seh_projection.csv | tr ',' '\n' | grep -nE "embedding"
```

Expected: a `chembed_embedding` entry and **no** `autoencoder_embedding`.

- [ ] **Step 4: Verify the recorded identities**

```bash
/home/aorlov/Programs/agents/chemspacecopilot/.venv/bin/python -c "
import json
from pathlib import Path
out = Path('reports/manuscript_reliability_chembed/frozen_inputs_v2_chembed')
summary = json.loads((out / 'projection_summary.json').read_text())
manifest = json.loads((out / 'manifest.json').read_text())
assert summary['input_dimensions'] == 256, summary['input_dimensions']
assert summary['sample_count'] == 2212, summary['sample_count']
assert summary['node_count'] == 900, summary['node_count']
assert summary['model_sha256'] == (
    '45d37ae167ed4ab0c0afefcee8e76bf4c5cacbb90d660e6666d1b904a24280e0'
), summary['model_sha256']
assert 'autoencoder' not in json.dumps(manifest), 'autoencoder still referenced in the manifest'
print('occupancy entropy:', summary['normalized_occupancy_entropy'])
print('hard occupied nodes:', summary['hard_occupied_nodes'])
print('ok')
"
```

Expected: `ok`, preceded by the occupancy figures. Those will differ from v1's `0.6746617974445414` and 293 hard-occupied nodes — that difference is the measurement, not a regression. Record both numbers; Task 7 reports them.

- [ ] **Step 5: Commit the manifest**

```bash
git add -f reports/manuscript_reliability_chembed/frozen_inputs_v2_chembed/manifest.json \
           reports/manuscript_reliability_chembed/frozen_inputs_v2_chembed/projection_summary.json
git commit -m "Rebuild the frozen sEH fixtures on the chembed manifold

Same 2212 curated compounds, same GTM hyperparameters, chembed latents in
place of LSTM ones. Node occupancy differs from frozen_inputs_v1 because the
latent space differs; that is the measurement.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 5: Flip the study configuration and wire the fixtures

**Files:**
- Modify: `tests/robustness/manuscript_reliability.yaml` lines 66, 126, 134, 139 and the "Latent engine" comment block at lines 17–24
- Create: `reports/manuscript_reliability_chembed/fixture_env.sh`
- Test: `tests/unit/test_reliability_benchmark.py` (append)

**Interfaces:**
- Consumes: the fixture paths from Task 4.
- Produces: a config with no `descriptor_type=autoencoder`, and `fixture_env.sh` exporting eight variables, which Task 6 sources.

- [ ] **Step 1: Write the failing test**

Append to `tests/unit/test_reliability_benchmark.py`, adding `from pathlib import Path` and `import yaml` to its imports if absent:

```python
def test_manuscript_config_maps_and_generates_in_one_latent_space():
    """Every GTM prompt must pin chembed, matching general.latent_engine."""
    path = Path(__file__).parents[2] / "tests/robustness/manuscript_reliability.yaml"
    text = path.read_text()
    assert "descriptor_type=autoencoder" not in text, (
        "a GTM prompt still pins the LSTM descriptor family; maps and generation "
        "would run in different latent spaces"
    )
    assert "descriptor_type=chembed" in text
    config = yaml.safe_load(text)
    assert config["general"]["latent_engine"] == "chembed"
```

- [ ] **Step 2: Run the test to verify it fails**

```bash
cd /home/aorlov/Programs/agents/chemspacecopilot/.claude/worktrees/chemembed-maps-rebuild-5a00d4
PYTHONPATH=src /home/aorlov/Programs/agents/chemspacecopilot/.venv/bin/python -m pytest \
  "tests/unit/test_reliability_benchmark.py::test_manuscript_config_maps_and_generates_in_one_latent_space" -v
```

Expected: FAIL with "a GTM prompt still pins the LSTM descriptor family".

- [ ] **Step 3: Flip the four descriptor pins**

```bash
sed -i 's/descriptor_type=autoencoder/descriptor_type=chembed/g' \
  tests/robustness/manuscript_reliability.yaml
grep -n "descriptor_type=" tests/robustness/manuscript_reliability.yaml
```

Expected: four lines, all `descriptor_type=chembed`, at 66, 126, 134 and 139.

- [ ] **Step 4: Rewrite the "Latent engine" comment block**

Replace the block at lines 17–24, which currently explains how to reproduce the LSTM results, with:

```yaml
# Latent engine:
#   Maps and small-molecule generation run in one latent space. Generation
#   prompts name the "autoencoder design engine", which is the engine= argument
#   of the design tools and picks it over the LLM engine; it does not name a
#   latent engine, because the latent engine is fixed before the agent starts.
#   general.latent_engine below pins it and overrides AUTOENCODER_BACKEND, so a
#   run cannot inherit an engine from the operator's shell. The GTM prompts pin
#   descriptor_type=chembed to match, so every reported map is projected on the
#   same latents the generator samples. Every run's generation audit records the
#   backend that ran.
```

- [ ] **Step 5: Run the test to verify it passes**

```bash
PYTHONPATH=src /home/aorlov/Programs/agents/chemspacecopilot/.venv/bin/python -m pytest \
  tests/unit/test_reliability_benchmark.py -v
```

Expected: all pass.

- [ ] **Step 6: Write the fixture environment script**

```bash
cd /home/aorlov/Programs/agents/chemspacecopilot/.claude/worktrees/chemembed-maps-rebuild-5a00d4
OUT="$PWD/reports/manuscript_reliability_chembed/frozen_inputs_v2_chembed"
PEP=/home/aorlov/Programs/agents/chemspacecopilot-reviewer-revision/reports/reviewer_revision/reliability_study_frozen_v2/fixtures/peptide_input.json
{
  echo "# Frozen-tier fixtures for the chembed reliability study. Source before running."
  for pair in SEH_INPUT:seh_input SEH_ANALYSIS:seh_analysis SEH_CANDIDATES:seh_candidates; do
    var="${pair%%:*}"; file="${pair##*:}"
    echo "export CS_COPILOT_${var}_FIXTURE=$OUT/${file}.json"
    echo "export CS_COPILOT_${var}_SHA256=$(sha256sum "$OUT/${file}.json" | cut -d' ' -f1)"
  done
  echo "export CS_COPILOT_PEPTIDE_INPUT_FIXTURE=$PEP"
  echo "export CS_COPILOT_PEPTIDE_INPUT_SHA256=$(sha256sum "$PEP" | cut -d' ' -f1)"
} > reports/manuscript_reliability_chembed/fixture_env.sh
cat reports/manuscript_reliability_chembed/fixture_env.sh
```

Case 4's peptide fixture is unchanged — it is WAE, untouched by the descriptor swap — so it is reused from the LSTM study rather than rebuilt. If `$PEP` does not exist, locate it by reading `CS_COPILOT_PEPTIDE_INPUT_FIXTURE` from the LSTM study's recorded environment under `reliability_study_frozen_v2/`.

- [ ] **Step 7: Prove the fixtures fail closed on a checksum mismatch**

```bash
set -a; . reports/manuscript_reliability_chembed/fixture_env.sh; set +a
CS_COPILOT_SEH_ANALYSIS_SHA256=0000000000000000000000000000000000000000000000000000000000000000 \
PYTHONPATH=src /home/aorlov/Programs/agents/chemspacecopilot/.venv/bin/python \
  tests/robustness/robustness_minimal_example.py \
  --config tests/robustness/manuscript_reliability.yaml \
  --tier frozen --system team --n-variations 1 --repetitions 1 2>&1 \
  | grep -iE "checksum|sha256|mismatch|fixture" | head -5
```

Expected: a checksum-mismatch failure and **no** live-data fallback. This is Review Focus item 5. If the run proceeds past the fixture check instead, stop — a silently substituted fixture invalidates every number the study produces.

- [ ] **Step 8: Commit**

```bash
git add tests/robustness/manuscript_reliability.yaml tests/unit/test_reliability_benchmark.py
git add -f reports/manuscript_reliability_chembed/fixture_env.sh
git commit -m "Map on chembed in the reliability configuration

The config pinned latent_engine: chembed for generation while its GTM
prompts pinned descriptor_type=autoencoder, so maps and generation ran in
different latent spaces. Pin chembed in both, point the frozen tier at the
rebuilt fixtures, and say so in the comment block.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 6: Execute the study

Operator-run, against a live provider. Not a pytest target. Confirm the human verification gate on `frozen_inputs_v2_chembed/` has been cleared before starting: the fixture README requires each snapshot and its referenced artifacts to be inspected before promotion, with stringified non-pointer Python objects treated as a sign the state needs cleaning.

**Files:**
- Create: `reports/manuscript_reliability_chembed/study/` (plan, configs, per-run output, analysis)

**Interfaces:**
- Consumes: `fixture_env.sh` and the config from Task 5.
- Produces: `study/analysis/reliability_summary.json`, whose figures Task 7 quotes.

- [ ] **Step 1: Confirm readiness before spending provider budget**

```bash
cd /home/aorlov/Programs/agents/chemspacecopilot/.claude/worktrees/chemembed-maps-rebuild-5a00d4
set -a; . reports/manuscript_reliability_chembed/fixture_env.sh; set +a
PYTHONPATH=src /home/aorlov/Programs/agents/chemspacecopilot/.venv/bin/python \
  scripts/check_revision_readiness.py \
  --config tests/robustness/manuscript_reliability.yaml \
  --tier both --system both \
  --output reports/manuscript_reliability_chembed/study-readiness.json 2>&1 | tail -20
```

Expected: no missing fixture or model. Resolve anything reported before continuing. `DEEPSEEK_API_KEY` must be set.

- [ ] **Step 2: Plan the study**

`plan_revision_reliability.py` requires five paths besides `--output-dir`, and fails with `Required for planning: ...` if any is missing (`scripts/plan_revision_reliability.py:329-344`):

```bash
AGENTS=/home/aorlov/Programs/agents
REV=$AGENTS/chemspacecopilot-reviewer-revision/reports/reviewer_revision
PYTHONPATH=src $AGENTS/chemspacecopilot/.venv/bin/python \
  scripts/plan_revision_reliability.py \
  --config tests/robustness/manuscript_reliability.yaml \
  --seh-manifest reports/manuscript_reliability_chembed/frozen_inputs_v2_chembed/manifest.json \
  --peptide-bundle    <peptide landscape bundle dir: landscape.json, landscape.safetensors, nodes.parquet, sampler.json> \
  --synplanner-dir    <synplanner assets dir> \
  --autoencoder-dir   ~/.cache/cs_copilot/models/autoencoder \
  --peptide-model-dir ~/.cache/cs_copilot/models/peptide_designer \
  --output-dir        reports/manuscript_reliability_chembed/study
```

Take the three study-asset paths from the LSTM study's own plan rather than guessing: `reports/reviewer_revision/reliability_study_frozen_v2/study_plan.json` records the invocation that produced it.

Expected: 9 batches with `expected_executions: 8` each — 48 frozen and 24 live, 72 in total. `tests/unit/test_revision_study_plan.py::test_counterbalanced_matrix_is_exactly_48_frozen_plus_24_live` pins exactly that, so treat 72 as correct and any other number as drift. (The LSTM study recorded `planned_total_executions: 60`; it was planned under different settings, and that figure is not the target here.)

`--seh-manifest` points the planner at the rebuilt chembed fixture manifest. It reads `_current_gtm_model_path` **and** `default_descriptor` from that manifest's `seh_input` fixture, so the live tier inherits the chembed family rather than a hardcoded one.

- [ ] **Step 3: Execute**

The plan path is positional, not a `--plan` flag:

```bash
PYTHONPATH=src /home/aorlov/Programs/agents/chemspacecopilot/.venv/bin/python \
  scripts/execute_revision_study.py \
  reports/manuscript_reliability_chembed/study/study_plan.json \
  --stop-on-unexpected-exit
```

48 frozen executions across 6 batches plus 24 live across 3. `--stop-on-unexpected-exit` halts on any exit code other than 0, 1 or 130, so an environment failure does not silently burn the remaining provider budget. Attempt logs and the ledger land in `<plan dir>/execution` by default; `--batch <id>` reruns a single batch if one needs repeating.

The live tier retrieves from ChEMBL prospectively — record the date it runs. It is reported as a fresh prospective run, never as a matched replacement for the LSTM study's live tier.

- [ ] **Step 4: Summarise**

```bash
PYTHONPATH=src /home/aorlov/Programs/agents/chemspacecopilot/.venv/bin/python \
  scripts/summarize_reliability_study.py \
  reports/manuscript_reliability_chembed/study \
  --output-dir reports/manuscript_reliability_chembed/study/analysis
```

- [ ] **Step 5: Check the frozen tier cleared the configured floor**

```bash
/home/aorlov/Programs/agents/chemspacecopilot/.venv/bin/python -c "
import json
path = 'reports/manuscript_reliability_chembed/study/analysis/reliability_summary.json'
data = json.load(open(path))
print('planned / retained:', data['planned_total_executions'], '/', data['retained_executions'])
for tier, stats in data['by_tier'].items():
    print(tier, 'completion_rate', stats['completion_rate'], 'executions', stats['executions'])
print('overall', data['overall']['completion_rate'])
assert data['by_tier']['frozen']['completion_rate'] >= 0.80, 'frozen tier below the configured floor'
print('ok')
"
```

Expected: 72 planned, frozen completion rate at or above the configured `reliability_min_success_rate` of 0.80. The LSTM study reached 0.8125 frozen and 0.80 overall; those are context for judging whether the descriptor swap broke orchestration, not a reported comparison.

- [ ] **Step 6: Commit the summary**

```bash
git add -f reports/manuscript_reliability_chembed/study/study_plan.json \
           reports/manuscript_reliability_chembed/study/analysis/reliability_summary.json
git commit -m "Record the chembed reliability study

60 executions, 48 frozen and 12 live, with maps and generation in one latent
space. The live tier is a fresh prospective run with its own retrieval date,
not a matched replacement.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 7: State the single latent space in Methods

**Files:**
- Modify: `docs/manuscript/revision-text.md` (the GTM and generation paragraphs in Methods)

**Interfaces:**
- Consumes: the occupancy figures from Task 4 Step 4 and the completion rates from Task 6 Step 5.
- Produces: nothing consumed by later tasks.

- [ ] **Step 1: Locate the Methods paragraphs**

```bash
cd /home/aorlov/Programs/agents/chemspacecopilot/.claude/worktrees/chemembed-maps-rebuild-5a00d4
grep -nE "GTM|generative topographic|descriptor|autoencoder|MOSES" docs/manuscript/revision-text.md | head -20
```

The gap being closed: Methods describes GTM generically and never states which descriptors the manifold was fitted on, so a reader cannot tell which latent space the maps use.

- [ ] **Step 2: Write the replacement text**

```markdown
Chemical-space mapping and de novo small-molecule generation use a single
molecular representation: 256-dimensional latent vectors from the SELFIES
transformer variational autoencoder (chembed), whose published checkpoint is
downloaded from `3BioCompBio/chembed-default`. No generative model was trained
in this work. All GTM maps reported here were fitted and projected on these
descriptors, using a published manifold pinned by revision
(`axelrolov/chemspacecopilot-gtm-chembed`, 900 nodes, 225 basis functions, with
hyperparameters carried over unchanged and not re-tuned). Because a manifold is
valid only for the latent space it was fitted on, each descriptor family
resolves its own published manifold and its own cache, and every projection in
this pipeline is checked against the manifold's recorded descriptor backend
before it runs; the two latent spaces are of equal width, so this check cannot
be replaced by a shape test. Peptide representation and sampling use the WAE
described below, which provides both the landscape and the generator for that
workflow. Candidates from chembed are reported as distinct-structure yield,
Tanimoto similarity to the parent and synthetic accessibility; SELFIES decode to
a valid molecule by construction, so decoder validity is 1.0 by definition and
is not a comparative metric.
```

- [ ] **Step 3: Carry the C57 limitation through unchanged**

Confirm the limitation text still says what it said, because a single latent space does not create node-guided sampling:

```markdown
Small-molecule generation is seed-centred: the sampler perturbs the latent
vector of an encoded seed and never reads node coordinates. For small molecules
the workflow is therefore chemography-guided *seed selection* followed by
latent-space design. Node-conditioned sampling applies to the peptide workflow,
where `sample_peptides_from_landscape` conditions on landscape nodes directly.
```

- [ ] **Step 4: Check no stale two-latent-space claim survives**

```bash
grep -niE "deepchemography|LSTM autoencoder|two latent|autoencoder descriptor" \
  docs/manuscript/revision-text.md docs/manuscript/reviewer-response.md
```

Expected: no hit that still describes the reported small-molecule maps as running on LSTM latents. A hit describing the peptide WAE, or the retired LSTM arm's provenance, is fine.

- [ ] **Step 5: Commit**

```bash
git add docs/manuscript/revision-text.md
git commit -m "Say which latents the maps are built on

Methods described GTM generically and never named the descriptors the
manifold was fitted on, so a reader could not tell which latent space the
maps used. State the single chembed representation, the pinned manifold, and
that validity is 1.0 by construction and not comparative. The seed-centred
generation limitation is unchanged: one latent space does not make sampling
node-guided.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```
