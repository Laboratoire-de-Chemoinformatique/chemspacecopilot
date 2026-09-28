# Prospective aggregate-landscape peptide generation

This capability is part of the revision implementation. It reuses the selection and latent-sampling core of the author's existing branch `114-refactor-peptide-desginer-to-new-hf-dataset-structures` at `ee6156fdc9ee15b8169af8441c150e16b31ecf99`. It must not be presented as the recovered original manuscript procedure. The pinned aggregate bundle and WAE decoder are documented in [scientific-inputs.lock.json](scientific-inputs.lock.json) and [assets.lock.json](assets.lock.json).

`PeptideDesignerToolkit.sample_peptides_from_landscape` reads four explicit local files: `landscape.json`, `landscape.safetensors`, `nodes.parquet` and `sampler.json`. It does not need raw DBAASP sequences, retrain the WAE/GTM, download current HF files or deserialize a runtime pickle. The organism is resolved against the aggregate table. Default selection requires mean source activity at least 0.5, class `active_enriched` and effective node support at least 1.0. Five nodes are selected by the existing activity/support/uncertainty/density score and a greedy spatial-diversity term. Effective support is responsibility-weighted, not a count of independent experimental measurements.

For a selected one-based node ID `k`, its center in scaled WAE feature space is `(phi @ W)[k - 1]`. The sampler chooses nodes with probabilities proportional to `score - min(selected scores) + 1e-6`, adds zero-mean Gaussian noise with standard deviation 0.25 in scaled feature space, and returns to WAE coordinates using `z = (center + noise) * scaler.scale + scaler.mean`. This is the inverse of the StandardScaler transformation used to construct the published peptide GTM. The model has 100 latent dimensions. The local safe-tensor loader validates dimensions, finite parameters, positive scale values and valid one-based node identifiers.

The default bounded batch decodes four times the requested candidate count, then validates residues against the bundle alphabet, removes exact sequence duplicates and returns up to the requested number. Every observed raw decoded output, including invalid and duplicate sequences, remains in the candidate artifact. Node assignments, selection probabilities, source-file hashes, decoder hash, requested/observed/valid/unique/returned counts, decoding temperature and mode, and random seed are retained. NumPy's local generator controls node selection and latent noise; a scoped Torch seed controls conditional-prior sampling and sequence decoding without changing the caller's random state. One batch is attempted; shortfalls remain explicit.

Validity is the fraction of observed decoded sequences that pass sequence validation and the bundle alphabet. Exact-sequence uniqueness is the number of unique valid observed sequences divided by the number of valid observed sequences, before truncation to the returned set. These are generation statistics. Source-node activity is not a prediction of a generated sequence's antimicrobial activity. Training-set novelty is unavailable because the precise WAE training corpus has not been recovered. Similarity analysis, sequence logos and the report are subsequent workflow outputs and are not implied by successful decoding.

A local CPU smoke run with the pinned assets, seed 42, temperature 1.0, categorical decoding, ten requested candidates and oversampling factor four observed 40 valid, unique sequences and returned ten. It selected nodes 480, 299, 509, 479 and 269. This smoke run establishes tool integration only; it is not a reliability repetition or biological validation. Its artifacts are under ignored `reports/reviewer_revision/peptide_landscape_smoke/` and the session-local output path recorded there.

The benchmark fixture contains inputs only:

```json
{
  "peptide_landscape_bundle": {
    "bundle_path": "/absolute/path/to/pinned/peptide_landscape"
  }
}
```

Set `PEPTIDE_DESIGNER_MODEL_PATH` to the separately pinned local WAE checkpoint directory before constructing either architecture. Each benchmark repetition must receive its own staged input directory. The new sampling tool writes a current-run activity CSV and candidate artifact; downstream analysis and reports must be generated during the run rather than inherited from fixture state.
