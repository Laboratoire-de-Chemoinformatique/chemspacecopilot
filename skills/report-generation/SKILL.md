# Report Generation

Use this skill when the user asks for a report, presentation artifact, or consolidated summary of a completed analysis.

## Procedure

1. Inspect session memory to identify available datasets, GTM maps, landscapes, candidate sets, synthesis plans, and figures.
2. Choose the report type from the available session objects and the user request.
3. Include provenance paths for raw data, clean data, standardization reports, plots, generated candidates, and synthesis artifacts when present.
4. Prefer `report_save_rich` for image-rich HTML/PDF outputs and `report_save_markdown` for lightweight text reports.
5. Store report paths in session state and return artifact paths to the user.

## Expected Outputs

- Markdown, HTML, or PDF report artifacts.
- Clear references to source datasets and analysis artifacts.
- Compact interpretation of the scientific result rather than a raw dump of tool output.
