# Chemoinformatics Analysis

Use this skill for chemoinformatics analysis on a prepared dataset, GTM node table, or user-provided molecular data: scaffold/chemotype profiling, clustering, structure-activity relationships (SAR), and similarity/diversity. This skill produces **structured analysis outputs** for the Report Generator — it does not make plots or write reports.

## Procedure

1. **Verify input.** Expect a SMILES column (`smiles` / `SMILES` / `canonical_smiles`), optional `cluster_id` (from GTM nodes, clustering, or user labels), and optional activity (`activity_final` / `activity`). Resolve in order: session GTM `source_mols` (use `node_index` as `cluster_id`) → `clean_dataset_path` → `dataset_path` (legacy alias) → ask the user. Normalize unfamiliar inputs with `pandas_normalize_for_analysis`; preserve `activity_mapping` (source) and `final_activity_mapping` (merged). Validate SMILES and report/drop invalid rows.
2. **Pick the analyses** the user asked for (one or several). When chained, run clustering → chemotype (using clusters) → SAR; similarity supports all three.
3. **Chemotype / scaffold analysis.** Extract Murcko scaffolds; compute frequencies overall and per cluster; identify common scaffolds; compute diversity metrics (Shannon entropy, unique-scaffold ratio); build a pairwise scaffold Tanimoto matrix (`chem_calculate_all_similarities` / `chem_find_most_similar`); identify scaffold clusters and scaffold-hopping opportunities.
4. **Clustering.** Validate/characterize existing clusters, or offer k-means / hierarchical when none exist. Report quality metrics (silhouette, Davies-Bouldin, size distribution, intra-cluster diversity); identify representative molecules (medoid/centroid), boundary molecules, and outliers; compare scaffold distributions across clusters.
5. **SAR.** When activity is present: detect activity cliffs (similar pairs with large activity gaps — Tanimoto > 0.85 AND > 2 log-unit difference); run matched-molecular-pair (MMP) analysis (single-transformation pairs and their activity change); analyze chemical-series trends and activity distribution per cluster/scaffold.
6. **Similarity / diversity.** Pairwise Tanimoto/Dice matrices (`chem_*` tools), diversity metrics (Shannon entropy, max-dissimilarity picking, coverage), and k-nearest-neighbor search for a query.
7. **Return** a concise bullet summary (counts, top findings, saved output paths) and indicate the data is ready for the Report Generator.

## Expected Outputs

Save structured results to session state under stable keys, export key tables to CSV, and provide paths for the Report Generator:
- `chemotype_analysis`: `scaffolds_per_cluster`, `similarity_matrix`, `summary_stats`, `output_paths`.
- `clustering_results`: `cluster_assignments`, `cluster_metrics`, `cluster_centroids`, `method`.
- `sar_analysis`: `activity_cliffs`, `mmps`, `series_analysis`, `potency_trends`.
- `similarity_analysis`: `similarity_matrix`, `diversity_metrics`, `nearest_neighbors`.

## Details

- **Presentation boundary**: DO NOT generate plots, charts, or formatted reports — that is the Report Generator's job. Emit structured data + paths only. Depict referenced structures as `<smiles>...</smiles>`.
- **Evidence rule**: claims about potency, top actives, or SAR drivers require measured activity values from a loaded table or tool output; scaffold patterns and node density alone are not potency evidence.
- **Edge cases**: missing columns → state requirements; no activity data → skip SAR; empty clusters → report and continue; insufficient data → set minimum thresholds and warn.
- **Tool availability**: the MCP surface exposes the `chem_*` similarity tools plus `pandas_*`; Murcko-scaffold extraction, clustering metrics, and activity-cliff/MMP detection use `pandas_run_operation` and the agent's reasoning under MCP, while the Agno-team Chemoinformatician has these natively via its similarity toolkit.
