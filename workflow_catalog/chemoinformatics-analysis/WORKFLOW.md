# Chemoinformatics Analysis

Use this workflow for scaffold/chemotype, clustering, SAR, and similarity/diversity analysis on a prepared dataset or GTM node table. It produces structured outputs for the Report Generator and does not itself create plots or reports.

1. Resolve the active dataset (prefer `clean_dataset_path`; GTM `source_mols` `node_index` can serve as `cluster_id`) and normalize if needed.
2. Run the requested analyses; when chained, order them clustering → chemotype → SAR, with similarity supporting all three.
3. Chemotype: Murcko scaffolds, per-cluster frequencies, Shannon entropy, scaffold Tanimoto matrix → `chemotype_analysis`.
4. Clustering: k-means/hierarchical, silhouette / Davies-Bouldin, medoids and outliers → `clustering_results`.
5. SAR (requires activity): activity cliffs (Tanimoto > 0.85 and > 2 log units), MMP, series trends → `sar_analysis`.
6. Similarity/diversity: `chem_*` similarity matrices, max-dissimilarity picking, kNN → `similarity_analysis`.
7. Save structured outputs and CSV paths; hand off to the Report Generator for any plots or formatted report.
