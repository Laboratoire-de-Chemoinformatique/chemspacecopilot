# Scientific source inventory

Status: 26 September 2026. These recovered public resources support a **prospective revision study**. Their existence does not establish that a particular file or checkpoint produced a figure in ChemRxiv v2. The download URLs, byte counts and SHA-256 hashes are recorded in [scientific-inputs.lock.json](scientific-inputs.lock.json); the model checkpoints are separately recorded in [assets.lock.json](assets.lock.json).

## Manuscript and original supporting archive

The requested manuscript is [ChemRxiv v2, DOI 10.26434/chemrxiv.15000527/v2](https://chemrxiv.org/doi/full/10.26434/chemrxiv.15000527/v2). Crossref confirms this DOI and its publisher PDF URL. Direct public access to the ChemRxiv page and PDF returned HTTP 403 in this environment. An [author-uploaded ResearchGate copy](https://www.researchgate.net/publication/405278370_ChemSpace_Copilot_Agentic_AI_for_Interactive_Visualization_and_Exploration_of_Chemical_Space) is indexed with the v2 DOI and 26 May 2026 cover date. Search-index excerpts were saved; they are incomplete, not a recovered original PDF. No original manuscript figure files were recovered.

The excerpts describe four case studies; a DeepSeek v4 flash model name; an sEH analysis; molecular generation around CHEMBL3327073; and peptide generation guided by DBAASP landscapes. They point to the [author's DeepChemography repository](https://github.com/AxelRolov/deepchemography) for model training information. They do not provide enough evidence to identify the historical ChEMBL release, selected assays, exact inference settings, all generated candidates or the selected sEH map parameters.

The author reports that original supporting material is on Zenodo. The DOI remains **unresolved**, after title/author/topic searches of the public Zenodo API and review of accessible manuscript excerpts, DOI metadata, repository references and release listings. This is an unresolved link, not evidence that the archive does not exist. The unrelated dimensionality-reduction archive Zenodo 13752690 must not be substituted. Publisher/repository metadata and lookup evidence are saved under ignored `reports/reviewer_revision/sources/`.

## Molecular reference data

| Resource | Immutable source | Recovered content | Appropriate use |
| --- | --- | --- | --- |
| Author GTM frameset | [DeepChemography 9822f5c](https://github.com/AxelRolov/deepchemography/blob/9822f5c2b70a1ff5428210589fb1cdf8b1ab100f/data/gtm_frameset_training.csv) | 50,000 unique SMILES, all labeled `train`; 2,144,038 bytes | A pinned prospective GTM/reference subset |
| Official MOSES training split | [MOSES dd7ed6a](https://github.com/molecularsets/moses/blob/dd7ed6ab38e23afd3ef5371d67939a1760bd8599/data/train.csv) | 1,584,663 unique SMILES; 67,916,778 bytes | An explicit external reference for molecular overlap/novelty |
| Author sampling notebook | [Sampling_Autoencoder.ipynb](https://github.com/AxelRolov/deepchemography/blob/9822f5c2b70a1ff5428210589fb1cdf8b1ab100f/notebooks/Sampling_Autoencoder.ipynb) | Reads a local `ChemEidos/data/train.csv`; samples 50,000 rows without a fixed `random_state`; writes the frameset | Explains how the published subset was constructed, without proving the checkpoint's complete training membership |

All 50,000 author frameset SMILES match exact strings in the pinned MOSES training split. This comparison uses literal strings, with no chemical standardization or stereochemistry removal; its results are recorded in the lock file. The full MOSES file SHA-256 matches the upstream Git LFS object ID. Neither the public HF model config nor the author model config records training/validation filenames or dataset checksums. Therefore report **novelty relative to the pinned MOSES reference** until the model author confirms the exact checkpoint training split; do not relabel it confirmed training-set novelty. The 50,000-row frameset alone is insufficient for that claim.

The author repository and the current Hugging Face GTM resource have different bytes for the filename `GTM_manifold_900_100_8_100.pkl.gz` (approximately 2.8 MB versus 4.0 MB). A filename is not adequate parameter provenance. The pinned resources were inspected with the project's CPU-aware trusted loader on 28 September 2026; [actual model attributes](gtm-model-attributes.json) show that the molecular checkpoint has **900 nodes, 225 basis functions, basis width 1, regularization 100, maximum 200 iterations and 256 input dimensions**, despite its filename. The peptide runtime contains a `{model, scaler, config}` wrapper, a 100-dimensional scaler and a model with 900 nodes, 100 basis functions, width 1 and regularization 100. These are prospective checkpoint attributes; do not infer the selected historical sEH map from them.

## Peptide landscape resources

The [author's peptide landscape dataset](https://huggingface.co/datasets/axelrolov/peptide_designer_data/tree/d81356051db9e1f96e16531b78096ebb0bd7a66d), pinned to `d81356051db9e1f96e16531b78096ebb0bd7a66d`, provides a usable structured bundle at `landscapes/dbaasp_amp_v1/`:

- `landscape.json`: dataset attribution, GTM settings, organism summaries, tensor names and decoder compatibility.
- `nodes.parquet`: 14,400 aggregate records, comprising 900 nodes for each of 16 organisms, with density, mean activity, class, uncertainty and effective observation support.
- `landscape.safetensors`: GTM, scaler and aggregate landscape tensors.
- `sampler.json`: published node-selection policy.
- `runtime/gtm.pkl.gz`: compressed runtime artifact, downloaded and hashed but not deserialized during source inspection.
- `plots/`: public HTML and PNG landscapes, including E. coli; these remain available upstream and were not bulk-downloaded.

The five core bundle files were downloaded (about 2.1 MB) under ignored `reports/reviewer_revision/scientific_inputs/peptide_landscape/`. The metadata declares 900 nodes, 100 basis functions, basis width 1, regularization 100, maximum 200 iterations, latent dimension 100 and maximum sequence length 25. It identifies `axelrolov/wae_peptides` as a compatible decoder but leaves its revision as `main`; the separate model lock pins the decoder for prospective work. E. coli metadata reports 1,607 active of 5,059 labeled observations. These are source-bundle summaries, not newly measured model performance or verified manuscript counts.

The bundle attributes its source to [DBAASP](https://dbaasp.org) and [DBAASP v3, DOI 10.1093/nar/gkaa991](https://doi.org/10.1093/nar/gkaa991). Its dataset card explicitly excludes raw DBAASP records and raw peptide source datasets and declares the license as `other`; retain its attribution when using derived figures. It supports sampling from the existing landscape, not reconstructing assay filtering or retraining the WAE from raw sequences.

The [author WAE notebook](https://github.com/AxelRolov/deepchemography/blob/9822f5c2b70a1ff5428210589fb1cdf8b1ab100f/notebooks/Peptides_WAE.ipynb) refers to local `external_peptide_source_data.csv` and `external_dbaasp_activity_data.csv`, neither present in the public repository tree. It samples at most 50,000 training sequences with `random_state=42`. Its dummy-sequence fallback when input is absent must not be used as revision evidence. The [HF export notebook](https://github.com/AxelRolov/deepchemography/blob/9822f5c2b70a1ff5428210589fb1cdf8b1ab100f/notebooks/Peptides_WAE_DBAASP_HF.ipynb) requires an external `DBAASP_ACTIVITY_CSV`; it does not recover that missing raw table.

## Prospective human sEH extraction

A new ChEMBL37 extraction contains 2,212 standardized structures from 2,490 unique IC50 measurements and 151 assays, after explicit filtering and phosphatase-domain exclusion. All 5,243 original target activity records remain archived. [Dataset provenance](seh-dataset.lock.json) records complete page URLs/hashes, exact selection counts, assay-description evidence, source-code hashes, the independently verified generation parent and an offline replay with identical output hashes. The actual methods and activity thresholds are provided in [revision-text.md](revision-text.md). Local inputs are `reports/reviewer_revision/seh/clean_compounds.csv` and `clean_measurements.csv`; raw pages and excluded records are alongside them. This replaces missing historical provenance with a clearly labeled new analysis, without claiming to recover the original dataset.

## Remaining provenance limits

1. Obtain the exact Zenodo DOI and original manuscript source/figure files before declaring historical reproduction complete.
2. Confirm checkpoint-specific molecular and peptide training membership before making training-set novelty claims.
3. Keep a new ChEMBL sEH extraction, fixed prompts and newly generated candidates labeled as prospective revision experiments. Record release, complete queries, filtering counts, raw file hashes, selected map parameters, model IDs and all failed outputs with those experiments.
4. Re-export maps and structures from verified underlying outputs at publication resolution. Public aggregate landscapes may support new peptide figures, but they should not be presented as recovered original manuscript figures without a match.
