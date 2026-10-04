---
name: chemoinformatics-analysis
description: Scaffold/chemotype, clustering, SAR, and similarity/diversity analysis on prepared datasets or GTM node tables, producing structured outputs for the Report Generator.
metadata:
  title: Chemoinformatics analysis
  status: stable
  version: 2.0.0
  depends_on: []
  profiles:
    - chemoinformatics
  permissions:
    - compute:execute
    - artifact:read
    - artifact:write
  input_artifacts:
    - name: analysis_dataset
      kind: dataset
      required: true
  output_artifacts:
    - name: chemotype_analysis
      kind: analysis-result
      required: false
    - name: clustering_results
      kind: analysis-result
      required: false
    - name: sar_analysis
      kind: analysis-result
      required: false
    - name: similarity_analysis
      kind: analysis-result
      required: false
  tags:
    - chemoinformatics
    - scaffold
    - sar
    - clustering
    - similarity
  keywords:
    - scaffold
    - chemotype
    - sar
    - activity cliff
    - clustering
    - matched molecular pair
    - diversity
    - structure-activity relationship
  required_tools:
    - chem_calculate_all_similarities
    - chem_find_most_similar
  optional_tools:
    - chem_calculate_tanimoto_similarity
    - chem_calculate_similarity_matrix
    - chem_calculate_dice_similarity
    - pandas_normalize_for_analysis
    - pandas_load_dataframe_from_session
    - pandas_run_operation
    - session_list_loadable_session_data
    - session_summarize_session_memory
    - gtm_get_density_summary
  example_prompts:
    - Cluster the dataset, analyze scaffolds per cluster, and find activity cliffs.
---

# Chemoinformatics Analysis

Use this skill for chemoinformatics analysis on a prepared dataset, GTM node table, or user-provided molecular data: scaffold/chemotype profiling, clustering, structure-activity relationships (SAR), and similarity/diversity. This skill produces **structured analysis outputs** for the Report Generator — it does not make plots or write reports.

## Procedure

1. **Verify input.** Expect a SMILES column (`smiles` / `SMILES` / `canonical_smiles`), optional `cluster_id` (from GTM nodes, clustering, or user labels), and optional activity (`activity_final` / `activity`). Resolve in order: session GTM `source_mols` (use `node_index` as `cluster_id`) → `clean_dataset_path` → `dataset_path` (legacy alias) → ask the user. Normalize unfamiliar inputs with `pandas_normalize_for_analysis`; preserve `activity_mapping` (source) and `final_activity_mapping` (merged). Validate SMILES and report/drop invalid rows.
2. **Pick the analyses** the user asked for (one or several). When chained, run clustering → chemotype (using clusters) → SAR; similarity supports all three.
3. **Chemotype / scaffold analysis.** Extract Murcko scaffolds; compute frequencies overall and per cluster; identify common scaffolds; compute diversity metrics (Shannon entropy, unique-scaffold ratio); build a pairwise scaffold Tanimoto matrix (`chem_calculate_all_similarities` / `chem_find_most_similar`); identify scaffold clusters and scaffold-hopping opportunities.
4. **Clustering.** Validate/characterize existing clusters using the descriptor metric below. For fingerprints, use methods accepting precomputed Tanimoto dissimilarity (`1 - T`) and use that same matrix for silhouette scores. K-means and Davies-Bouldin assume Euclidean geometry; use them for embeddings with appropriate preprocessing. Report size distribution and intra-cluster diversity; identify representative molecules (medoid/centroid), boundary molecules, and outliers; compare scaffold distributions across clusters.
5. **SAR.** When activity is present: detect activity cliffs (similar pairs with large activity gaps — Tanimoto > 0.85 AND > 2 log-unit difference); run matched-molecular-pair (MMP) analysis (single-transformation pairs and their activity change); analyze chemical-series trends and activity distribution per cluster/scaffold.
6. **Similarity / diversity.** Choose the metric from descriptor provenance using the rules below, then compute pairwise matrices, diversity/coverage statistics, or nearest neighbors. Report the descriptor family, parameters/preprocessing, metric definition, score direction, and output paths with the results.
7. **Return** a concise bullet summary (counts, top findings, saved output paths) and indicate the data is ready for the Report Generator.

## Expected Outputs

Save structured results to session state under stable keys, export key tables to CSV, and provide paths for the Report Generator:
- `chemotype_analysis`: `scaffolds_per_cluster`, `similarity_matrix`, `summary_stats`, `output_paths`.
- `clustering_results`: `cluster_assignments`, `cluster_metrics`, `cluster_centroids`, `method`.
- `sar_analysis`: `activity_cliffs`, `mmps`, `series_analysis`, `potency_trends`.
- `similarity_analysis`: `similarity_matrix`, `diversity_metrics`, `nearest_neighbors`.

## Details

- **Fingerprint metrics**: use Tanimoto for Morgan/ECFP, RDKit, MACCS, and similar binary/count fingerprints. Preserve count values; boolean/Jaccard conversion loses multiplicity. This project's descriptor encoder produces Morgan **counts**, even when the array is stored as floats. For stored rows use `chem_calculate_similarity_matrix` with `descriptor_kind="binary"` or `"count"`; it returns similarity `T = a·b / (a·a + b·b - a·b)`. This is dot-product count Tanimoto, matching SCOPE-DEL's formula, and differs from RDKit's min/max count convention (e.g. `[2,1]` vs `[1,1]`: 0.75 vs 2/3). State the convention; do not mix their thresholds. Two all-zero fingerprints score 1; zero vs nonzero scores 0. Exclude invalid-molecule placeholder rows before analysis. SMILES-pair tools generate binary fingerprints by default; `chem_calculate_tanimoto_similarity` and Tanimoto `chem_find_most_similar` also accept `fp_type="morgan_count"` (radius 2, 2048 counts). Use stored vectors when the dataset uses another width or configuration.
- **Embedding metrics**: use `chem_calculate_similarity_matrix` with `descriptor_kind="embedding"`: default Euclidean distance, or `metric="cosine"` for cosine similarity. This covers autoencoder/WAE and other learned embeddings. Both sets must use the same model, features, and preprocessing. Inspect provenance instead of inferring representation from dtype or sign; nonnegative embeddings remain embeddings. Cosine can be negative and is undefined for a zero vector. The SMILES cosine/Euclidean tools operate on generated binary fingerprints, not stored embeddings.
- **Direction and scale**: higher Tanimoto/cosine similarity means closer; lower Euclidean distance means closer. `is_distance` in the matrix result records this distinction. Use `1 - T` for fingerprint dissimilarity; never label Euclidean distance as a bounded similarity or transfer the Tanimoto activity-cliff threshold to embeddings. GTM fitting and 2D map distances have their own geometry; map proximity is not fingerprint similarity.
- **Large matrices**: load descriptor artifacts with `pandas_*`, keep row IDs aligned, compute bounded blocks, and save outputs via the existing artifact workflow. Do not send whole descriptor tables or full pairwise matrices through chat. Unknown descriptor provenance must be resolved before metric selection.
- **Presentation boundary**: DO NOT generate plots, charts, or formatted reports — that is the Report Generator's job. Emit structured data + paths only. Depict referenced structures as `<smiles>...</smiles>`.
- **Evidence rule**: claims about potency, top actives, or SAR drivers require measured activity values from a loaded table or tool output; scaffold patterns and node density alone are not potency evidence.
- **Edge cases**: missing columns → state requirements; no activity data → skip SAR; empty clusters → report and continue; insufficient data → set minimum thresholds and warn.
- **Tool availability**: the MCP surface exposes the `chem_*` similarity tools plus `pandas_*`; Murcko-scaffold extraction, clustering metrics, and activity-cliff/MMP detection use `pandas_run_operation` and the agent's reasoning under MCP, while the Agno-team Chemoinformatician has these natively via its similarity toolkit.
