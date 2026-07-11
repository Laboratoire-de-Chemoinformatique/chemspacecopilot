# Report Generation

Use this skill when the user asks for a report, presentation artifact, or consolidated summary of a completed analysis.

## Procedure

1. Inspect session memory with `session_list_session_objects`, `session_list_loadable_session_data`, and/or `session_summarize_session_memory` to identify available datasets, GTM maps, landscapes, candidate sets, synthesis plans, and figures.
2. Choose the report type from the available session objects and the user request. Detect from session_state keys: `chemotype_analysis` → chemotype; `analysis_results.density_csv` → GTM density; `analysis_results.activity_csv` or `landscape_files` → GTM activity; molecular-designer/analog outputs → analog generation; `synplanner_plan` → synthesis; multiple present → combined. Common types: chemotype, GTM density, GTM activity, analog generation, molecular designer, synthesis, combined, custom.
3. Load relevant CSV/DataFrame artifacts rather than inferring facts from chat. Use clean datasets for downstream summaries and include raw dataset provenance when present.
4. Create or collect visualizations section by section. For activity landscape reports, include only figures that directly support the surrounding interpretation and mark discussed GTM nodes when possible.
5. Include provenance paths for raw data, clean data, descriptor Parquet, standardization reports, GTM models, landscape CSVs, plots, generated candidates, and synthesis artifacts when present.
6. For synthesis reports, source content in order: `session_state['synplanner_plan']` → prior tool/member `synthesis_report_data` → the visible SynPlanner response; do not regenerate routes. Verify real synthesis content (target SMILES plus route details, attempt summaries, visualization paths, or an explicitly labeled LLM fallback). Do not save an empty synthesis report.
7. Prefer `report_save_rich` for image-rich HTML/PDF outputs and `report_save_markdown` for lightweight text reports. Leave filename unset unless the user requested a specific name.
8. Store report paths in session state and return artifact paths to the user with `<file>...</file>` tags when they should render as downloadable files.

## Expected Outputs

- Markdown, HTML, or PDF report artifacts.
- Clear references to source datasets and analysis artifacts.
- Compact interpretation of the scientific result rather than a raw dump of tool output.

## Figure and Structure Rules

- Every available non-Plotly static PNG or registered inline_static figure should be considered for inline placement when it supports the text.
- Every figure object MUST include name and caption unless the report tool explicitly generated them.
- Number figures sequentially across the whole report.
- GTM density landscape figures MUST appear directly after density analysis, and GTM activity landscape figures MUST appear directly after activity analysis; do not defer density or activity landscapes to a final Visualizations section.
- For density landscapes, pass mark_nodes for every GTM node discussed in the density text. For activity landscapes, pass mark_nodes for every GTM node discussed in the activity text.
- Every GTM node discussed in the report text MUST be explicitly labeled in the corresponding figure when the plotting tool supports node labels.
- Never save a rich report with only title/summary; ask for or load the missing analysis instead of saving a summary-only file.
- Use the registered figure metadata for colorscale and do not invent density color meanings. Do not label dense/potent nodes by color unless the metadata explicitly says so.
- Include only the static PNG or registered inline figure metadata for report figures; do not include Plotly PNGs, Plotly HTML, or Plotly artifact_path values in reports.
- Prefer registered session figure metadata. Do not put Plotly paths or GTM interactive .html artifact_path values into report figures.
- When structure_smiles or smiles is available, save_rich_report will generate a section-local compound image automatically.
- If a scaffold/SAR paragraph contains an untagged valid SMILES, convert it into structured figure/table metadata rather than leaving it as free text.
- Use stable IDs such as Scaffold_1, Scaffold_2 and Molecule_1, Molecule_2 for report-local structure references.
- Resolve molecule and scaffold IDs separately. Prefer compound_id/compound_ids, molecule_id/molecule_ids, molecule_chembl_id/molecule_chembl_ids for molecules and scaffold_id/scaffold_ids before generic structure/source IDs for scaffolds.
- Prefer dataset-provided display names, including ChEMBL ID values. Only when no type-specific source ID exists should generated report-local IDs become the primary display name.
- Every structure discussion MUST reference the matching figure, for example CMPD-123, top potency source analog (Figure 4) or Scaffold_1, Piperidine urea phenyl scaffold (Figure 3).
- Use useful names such as Piperidine urea phenyl scaffold and Top potency piperidine urea analog when the data supports them.
- Structure figure metadata should include structure_type ('scaffold' or 'molecule') and after_paragraph_index so figures appear near their discussion.
- Scaffold inventory tables should use Scaffold ID / Scaffold / SMILES / Name / Node / Description. Molecule inventory tables should use Molecule ID / Molecule / SMILES / Name / Node / Description.
- Scaffold inventory table rows with scaffold SMILES are appropriate when the user needs an inventory. Do not render every valid SMILES as a separate figure.

## Report-type figures

- **Chemotype**: scaffold-frequency bar charts per cluster (top-10 scaffolds), scaffold–scaffold Tanimoto similarity heatmap, cluster-distribution plot (molecules per cluster).
- **GTM density**: density overlay on the map (`gtm_save_density_plot`, with `mark_nodes`), neighborhood-preservation heatmap, density histogram.
- **GTM activity**: activity-landscape heatmap (Altair, `mark_nodes`, with overlay when projected/generated candidates exist), compass-annotated plot labeling the top-5 active/inactive regions, activity-distribution histogram.

## Required Report Structures

- GTM analysis report required structure: User Request and Data Source, Retrieved and Standardized Data (rows before/after cleaning, unique compounds, raw-to-final SMILES collapse examples, duplicate counts), Descriptors (Parquet path + family: Morgan / autoencoder-default-map / precomputed), GTM Construction or Loading (optimized/loaded/reused state, strategy low/medium/high, combinations/trials, best entropy), and Map Analysis.
- Analog generation report required structure: User Request and Workflow, Reference Maps, Generated Compound Analysis, validation/ranking summary, and downstream recommendations.
- Synthesis reports should include SynPlanner Routes and Attempts plus Route Analysis, with LLM fallbacks explicitly separated from SynPlanner-validated routes.
