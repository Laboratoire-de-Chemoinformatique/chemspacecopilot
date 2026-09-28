# Prospective retrosynthesis measurements

The ten targets were fixed before route searches using the selection recorded in `../generation/retrosynthesis_selection.json`: rank all 22 distinct nonparent generated candidates by SHA256 of `42|<canonical_smiles>`, then select the first ten. The full list is retained in `targets.csv`, including every unsuccessful search. No target was replaced or retried.

SynPlanner returned a predicted route for 4/10 targets (40%); six searches returned no route, and no worker failed or timed out. Each successful search returned one route. Recovered route lengths were 6, 9, 5 and 5 reaction steps, with backend scores 0.119246, 0.041497, 0.152778 and 0.152778 respectively. These scores are uncalibrated search values, not probabilities of synthetic success. All six unsuccessful searches exhausted the 100-iteration limit. The total worker wall time was 226.78 seconds, with a median of 23.16 seconds per target (range 12.68–33.19 seconds).

| Target | Outcome | Steps | Backend score | Iterations | Worker wall time (s) |
| --- | --- | ---: | ---: | ---: | ---: |
| generated_001 | Predicted route | 6 | 0.119246 | 72 | 22.75 |
| generated_002 | No route | — | — | 100 | 27.67 |
| generated_003 | No route | — | — | 100 | 26.79 |
| generated_004 | No route | — | — | 100 | 20.74 |
| generated_005 | Predicted route | 9 | 0.041497 | 38 | 18.19 |
| generated_006 | No route | — | — | 100 | 33.19 |
| generated_007 | Predicted route | 5 | 0.152778 | 5 | 12.84 |
| generated_008 | Predicted route | 5 | 0.152778 | 5 | 12.68 |
| generated_009 | No route | — | — | 100 | 28.36 |
| generated_010 | No route | — | — | 100 | 23.57 |

Every target used one CPU search profile: maximum 120 seconds, 100 iterations, tree depth 9, tree size 10,000, top 50 reaction rules, probability threshold 0, and `min_mol_size=6`. Search stopped at its first solution. Retry profiles and LLM fallback were disabled. The whole-worker guard was 300 seconds. Seeds 42–51 were applied to Python, NumPy, PyTorch and `PYTHONHASHSEED`; PyTorch used one thread and deterministic algorithms. Applied controls are recorded per target in `summary.json`. This study did not measure repeated retrosynthesis runs, so cross-run route reproducibility is unmeasured.

The backend reports search time separately from measured worker wall time. Its internal clock updates at iteration boundaries and can omit the last iteration. Worker wall time includes process startup, model and building-block loading, search, and route rendering. The 120-second limit is checked between iterations and is not a strict process deadline; the independent 300-second guard protects each full worker. No search reached either time limit here.

Predicted routes have not been experimentally validated or assessed by synthetic chemists. Reaction-step counts describe recovered precursor expansions, not necessarily the longest linear synthesis. In the recorded configuration, the backend can accept small terminal molecules under the `min_mol_size` rule without exact stock membership, so successful planning does not prove that every precursor is purchasable. No-route outcomes establish only that this bounded search did not find a route. No claim of biological activity, candidate discovery, or established synthetic feasibility follows from these measurements.

`predicted_routes.json` preserves the four returned step sequences and their source-plan hashes. `routes/target_*.svg` contains the original backend vector diagrams for all four successes, with copy/source hashes in `route_figures.json`. `route_structure_checks.json` confirms that every stored expansion target connects to the declared root through preceding precursor expansions, all step SMILES parse in RDKit, and reported step counts match the stored expansions. These are structural bookkeeping checks, not chemical feasibility validation.

Use `publication_routes/target_*.svg` or the corresponding standalone PDF pages for publication. The backend SVGs used default mask bounds that hid bonds in negative-coordinate regions when rendered with CairoSVG. The corrected copies set each mask to explicit root-viewBox bounds and add a white background; molecule coordinates, bond primitives, atom labels and route connections are unchanged. All four corrected figures were rendered at 2400-pixel preview width and visually inspected. Every path, line, text, polyline, circle and group attribute was checked against the raw source. The export manifest records source and corrected file hashes. Keep the routes as separate full-page supplementary figures at readable scale, rather than shrinking all four into one panel. Reproduce them with `python scripts/prepare_retrosynthesis_figures.py docs/manuscript/results/retrosynthesis/routes --output-dir /absolute/path/to/new-figures`. The raw immutable study remains under `reports/reviewer_revision/retrosynthesis_10`; `summary.json` retains source hashes, parameters, asset checksums, software versions, seed controls and all ten execution records as hashed inputs. The export is reproducible with:

```bash
PYTHONPATH=src python scripts/summarize_retrosynthesis_study.py \
  reports/reviewer_revision/retrosynthesis_10 \
  --output-dir /absolute/path/to/new-summary
```

The positive control was separately declared aspirin, with a 10-second/20-iteration pilot budget. Its first implementation run failed because MiniRacer was absent. After installing the pinned native dependency, a second separately labelled control returned one one-step predicted route (score 0.375, worker wall time 11.85 seconds). Both pilot records are retained in `implementation_controls.json` and their raw directories. Neither control is included in the ten-generated-target denominator. The first failure was an implementation/dependency check, not a failed generated-target search.

## Recorded ARM runtime deviation

These measurements used Python 3.12.13 on Linux aarch64, CPU PyTorch 2.10.0, SynPlanner 1.2.1, CGRtools 4.1.35, upstream chython 3.4, PyTorch Lightning 2.6.1, torch-geometric 2.7.0, Ray 2.53.0 and mini-racer 0.14.1. All supplemental packages were installed into an isolated target directory; the shared virtual environment was not modified. `supplemental_runtime_versions.txt` records every distribution in that isolated target. The exact active distribution/module origins and source commit where available are captured in `summary.json`.

This is a measured compatibility environment, not an exact recreation of the repository lock. SynPlanner declares `cgrtools-stable==4.2.13` and `chython-synplan>=1.91`, for which native Linux aarch64 wheels/source distributions were unavailable during setup. CGRtools 4.1.35 and upstream chython 3.4 supplied the actual imported modules. The named requirements `chytorch-rxnmap-synplan`, `chytorch-synplan`, `streamlit`, and `streamlit-ketcher` were also absent; the exercised planning and rendering paths did not use them. Every unmet named requirement is recorded rather than hidden. A standard lock installation alone does not reproduce this runtime, and these results do not establish equivalence to the forked dependency environment.

The toolkit adapter uses SynPlanner 1.2.1's actual built-in rollout evaluator (`TreeConfig(evaluation_type="rollout")`), because this version does not export the newer `RolloutEvaluationConfig`/`load_evaluation_function` API. The MiniRacer adapter supplies CGRtools' legacy module name to mini-racer 0.14.1. Both adapters were checked with unit tests and a real model/data load plus actual positive-control search. Lightning upgraded the old ranking-checkpoint metadata in memory; the pinned checkpoint file was not rewritten. Asset source pins and SHA256 values are in `../../synplanner-assets.lock.json` and the study manifest.

A new scientific run in this recorded environment uses the same immutable assets and target list, an absolute supplemental import path, and a fresh output directory:

```bash
PYTHONPATH=/absolute/checkout/src:/absolute/isolated/python-deps \
  /absolute/python scripts/run_retrosynthesis_study.py \
  --data-dir /absolute/path/to/synplanner-assets \
  --targets docs/manuscript/results/generation/retrosynthesis_targets.csv \
  --output-dir /absolute/path/to/new-retrosynthesis-study
```
