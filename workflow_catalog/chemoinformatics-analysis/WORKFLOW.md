---
name: chemoinformatics-analysis
description: Run scaffold/chemotype, clustering, SAR, and similarity/diversity analysis on a prepared dataset or GTM node table and emit structured outputs for reporting.
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
  keywords:
    - scaffold
    - chemotype
    - sar
    - activity cliff
    - clustering
    - matched molecular pair
    - diversity
    - structure-activity relationship
  preflight_tools:
    - chemspace_plan_analysis
  required_tools:
    - chem_calculate_all_similarities
    - chem_find_most_similar
  optional_tools:
    - chem_calculate_tanimoto_similarity
    - pandas_normalize_for_analysis
    - pandas_load_dataframe_from_session
    - pandas_run_operation
    - session_summarize_session_memory
    - report_save_rich
  recommended_prompt: chemoinformatician_agent
---

# Chemoinformatics Analysis

Use this workflow for scaffold/chemotype, clustering, SAR, and similarity/diversity analysis on a prepared dataset or GTM node table. It produces structured outputs for the Report Generator and does not itself create plots or reports.

1. Resolve the active dataset (prefer `clean_dataset_path`; GTM `source_mols` `node_index` can serve as `cluster_id`) and normalize if needed.
2. Run the requested analyses; when chained, order them clustering → chemotype → SAR, with similarity supporting all three.
3. Chemotype: Murcko scaffolds, per-cluster frequencies, Shannon entropy, scaffold Tanimoto matrix → `chemotype_analysis`.
4. Clustering: k-means/hierarchical, silhouette / Davies-Bouldin, medoids and outliers → `clustering_results`.
5. SAR (requires activity): activity cliffs (Tanimoto > 0.85 and > 2 log units), MMP, series trends → `sar_analysis`.
6. Similarity/diversity: `chem_*` similarity matrices, max-dissimilarity picking, kNN → `similarity_analysis`.
7. Save structured outputs and CSV paths; hand off to the Report Generator for any plots or formatted report.
