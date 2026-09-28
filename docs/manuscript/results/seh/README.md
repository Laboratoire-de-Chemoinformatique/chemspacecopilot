# Prospective sEH projection results

These compact tables describe the ChEMBL 37 revision dataset projected onto the
pinned pretrained autoencoder GTM. They are deterministic tool outputs, not
autonomous agent benchmark successes and not recovered historical manuscript
results. The complete raw extraction and curation are identified by
[`seh-dataset.lock.json`](../../seh-dataset.lock.json).

All 2,212 compounds produced finite 256-dimensional representations and finite
projections. The model has 900 nodes and 225 basis functions, width 1,
regularization 100, no input standardization and PCA scaling. The checkpoint
hash is recorded in `projection_summary.json`; parameters come from its saved
attributes, not its misleading filename. No new optimization was performed.

- `seh_density.csv`: responsibility mass over all compounds; its total is 2,212.
- `seh_activity_regression.csv`: responsibility-weighted compound median pIC50;
  `filtered_reg_density` is missing where responsibility mass is below 0.1.
- `projection_summary.json`: normalized occupancy entropy 0.6746617974,
  293 nodes with a maximum-responsibility assignment, and mean normalized
  per-compound responsibility entropy 0.0373107703.

Coordinates are one-based: node=(x−1)×30+y. The continuous landscape includes all
260 intermediate compounds. Binary active/inactive comparisons must exclude
these missing binary labels, leaving 1,794 active and 158 inactive compounds.
The occupancy and uncertainty diagnostics do not validate activity prediction
or establish neighborhood preservation.

Full descriptors, responsibilities, projections and frozen session snapshots are
retained under `reports/reviewer_revision/frozen_inputs_v1`, with hashes in its
manifest. The figure command reads only the compact tables:

```bash
python scripts/render_revision_gtm.py --input-dir docs/manuscript/results/seh
```
