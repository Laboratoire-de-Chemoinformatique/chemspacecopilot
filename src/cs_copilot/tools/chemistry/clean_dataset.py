#!/usr/bin/env python
# coding: utf-8
"""Clean molecular dataset preparation and artifact export."""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

from cs_copilot.storage import S3
from cs_copilot.tools.chemistry.activity_schema import (
    ActivityMapping,
    activity_series_for_landscape,
    find_smiles_column,
    infer_activity_mapping,
)
from cs_copilot.tools.chemistry.descriptors import (
    DEFAULT_DESCRIPTOR_TYPE,
    MolecularDescriptorEncoder,
)
from cs_copilot.tools.chemistry.standardize import standardize_smiles_column

logger = logging.getLogger(__name__)

RAW_SMILES_COLUMN = "raw_smiles"
FINAL_ACTIVITY_COLUMN = "activity_final"
ACTIVITY_MERGE_POLICY = "median_p_scale"

_HELPER_COLUMNS = {
    "_activity_final_row",
    "_source_row_number",
}

_MERGE_METADATA_COLUMNS = {
    "molecule_chembl_id",
    "parent_molecule_chembl_id",
    "compound_id",
    "molecule_id",
    "id",
    "activity_id",
    "assay_chembl_id",
    "assay_id",
    "target_chembl_id",
    "target_id",
    "document_chembl_id",
    "src_id",
    "standard_type",
    "standard_units",
    "standard_relation",
    "relation",
    "query_keywords",
    "cluster_id",
    "source",
}


@dataclass
class CleanDatasetResult:
    """Artifacts and dataframes produced by clean dataset preparation."""

    raw_df: pd.DataFrame
    clean_df: pd.DataFrame
    descriptor_df: pd.DataFrame
    raw_dataset_path: str
    clean_dataset_path: str
    descriptor_parquet_path: str
    standardization_report_path: str
    standardization_summary: dict[str, Any]
    activity_mapping: ActivityMapping
    final_activity_mapping: ActivityMapping
    descriptor_type: str
    descriptor_column: str


def prepare_clean_dataset(
    df: pd.DataFrame,
    *,
    source_name: str,
    smiles_column: Optional[str] = None,
    activity_column: Optional[str] = None,
    raw_dataset_path: Optional[str] = None,
    raw_filename: Optional[str] = None,
    clean_filename: Optional[str] = None,
    descriptor_filename: Optional[str] = None,
    report_filename: Optional[str] = None,
    descriptor_type: Optional[str] = None,
    remove_stereochemistry: bool = True,
    save_raw: bool = True,
) -> CleanDatasetResult:
    """Prepare raw, clean, descriptor, and report artifacts for a molecular dataset.

    The clean CSV is intentionally descriptor-free and contains one row per final
    standardized SMILES. Descriptor vectors are written to Parquet with the same
    final activity columns for downstream ML/GTM use.
    """
    if df.empty:
        raise ValueError("Cannot prepare a clean dataset from an empty dataframe")

    raw_df = df.copy()
    source_stem = _safe_stem(source_name)

    if smiles_column is None:
        smiles_column = find_smiles_column(raw_df)
    if smiles_column not in raw_df.columns:
        raise ValueError(
            f"SMILES column '{smiles_column}' not found. Available columns: {list(raw_df.columns)}"
        )

    raw_dataset_path = _save_raw_dataset(
        raw_df,
        source_stem=source_stem,
        raw_dataset_path=raw_dataset_path,
        raw_filename=raw_filename,
        save_raw=save_raw,
    )

    standardized_df, standardization_summary = _standardize_input(
        raw_df,
        smiles_column=smiles_column,
        remove_stereochemistry=remove_stereochemistry,
    )
    if standardized_df.empty:
        raise ValueError("No valid SMILES remained after standardization")

    activity_mapping = infer_activity_mapping(
        standardized_df,
        smiles_column="smiles",
        activity_column=activity_column,
    )
    standardized_df, activity_kind = _add_row_activity(standardized_df, activity_mapping)
    clean_df = _merge_standardized_compounds(
        standardized_df,
        activity_mapping=activity_mapping,
        activity_kind=activity_kind,
    )
    final_activity_mapping = _final_activity_mapping(clean_df, activity_mapping, activity_kind)

    clean_filename = clean_filename or f"{source_stem}_clean.csv"
    clean_dataset_path = _write_csv(clean_df, clean_filename)

    descriptor_df, descriptor_type_used, descriptor_column = _build_descriptor_dataframe(
        clean_df,
        descriptor_type=descriptor_type,
    )
    descriptor_filename = descriptor_filename or f"{source_stem}_descriptors.parquet"
    descriptor_parquet_path = _write_parquet(descriptor_df, descriptor_filename)

    standardization_summary.update(
        {
            "activity_merge_policy": ACTIVITY_MERGE_POLICY,
            "activity_mapping": activity_mapping.to_dict(),
            "final_activity_mapping": final_activity_mapping.to_dict(),
            "clean_rows": int(len(clean_df)),
            "clean_unique_smiles": int(clean_df["smiles"].nunique()),
            "activity_kind": activity_kind,
            "descriptor_type": descriptor_type_used,
            "descriptor_column": descriptor_column,
            "raw_dataset_path": raw_dataset_path,
            "clean_dataset_path": clean_dataset_path,
            "descriptor_parquet_path": descriptor_parquet_path,
        }
    )

    report_filename = report_filename or f"{source_stem}_standardization_report.md"
    report_path = _write_report(standardization_summary, report_filename)
    standardization_summary["standardization_report_path"] = report_path

    logger.info(
        "Prepared clean dataset for %s: raw_rows=%d standardized_rows=%d clean_rows=%d "
        "duplicate_rows_after_standardization=%d",
        source_name,
        standardization_summary["raw_rows"],
        standardization_summary["rows_after_standardization"],
        standardization_summary["clean_rows"],
        standardization_summary["duplicate_rows_after_standardization"],
    )

    return CleanDatasetResult(
        raw_df=raw_df,
        clean_df=clean_df,
        descriptor_df=descriptor_df,
        raw_dataset_path=raw_dataset_path,
        clean_dataset_path=clean_dataset_path,
        descriptor_parquet_path=descriptor_parquet_path,
        standardization_report_path=report_path,
        standardization_summary=standardization_summary,
        activity_mapping=activity_mapping,
        final_activity_mapping=final_activity_mapping,
        descriptor_type=descriptor_type_used,
        descriptor_column=descriptor_column,
    )


def _standardize_input(
    df: pd.DataFrame,
    *,
    smiles_column: str,
    remove_stereochemistry: bool,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    working = df.copy()
    if smiles_column != "smiles":
        working = working.rename(columns={smiles_column: "smiles"})

    raw_smiles_column = _unique_column_name(working.columns, RAW_SMILES_COLUMN)
    working[raw_smiles_column] = working["smiles"]
    working["_source_row_number"] = np.arange(1, len(working) + 1)

    raw_rows = len(working)
    string_smiles_rows = int(
        working[raw_smiles_column].map(lambda value: isinstance(value, str)).sum()
    )

    working = standardize_smiles_column(
        working,
        "smiles",
        remove_stereochemistry=remove_stereochemistry,
    )
    standardized = working.dropna(subset=["smiles"]).reset_index(drop=True)
    rows_after_standardization = len(standardized)
    invalid_smiles_rows = raw_rows - rows_after_standardization

    changed_mask = standardized[raw_smiles_column].astype(str) != standardized["smiles"].astype(str)
    changed_examples = _change_examples(
        standardized.loc[changed_mask, [raw_smiles_column, "smiles"]]
    )
    collapse_examples = _collapse_examples(standardized, raw_smiles_column)
    duplicate_rows = rows_after_standardization - int(standardized["smiles"].nunique())

    summary = {
        "raw_rows": int(raw_rows),
        "string_smiles_rows": string_smiles_rows,
        "rows_after_standardization": int(rows_after_standardization),
        "invalid_smiles_rows": int(invalid_smiles_rows),
        "stereochemistry_removed": bool(remove_stereochemistry),
        "standardized_smiles_changed_rows": int(changed_mask.sum()),
        "duplicate_rows_after_standardization": int(duplicate_rows),
        "duplicate_compound_groups_after_standardization": int(
            standardized["smiles"].value_counts().gt(1).sum()
        ),
        "raw_smiles_collapse_groups": int(len(collapse_examples)),
        "standardization_change_examples": changed_examples,
        "raw_smiles_collapse_examples": collapse_examples[:20],
    }

    if raw_smiles_column != RAW_SMILES_COLUMN:
        standardized = standardized.drop(columns=[RAW_SMILES_COLUMN], errors="ignore")
        standardized = standardized.rename(columns={raw_smiles_column: RAW_SMILES_COLUMN})

    return standardized, summary


def _add_row_activity(
    df: pd.DataFrame,
    activity_mapping: ActivityMapping,
) -> tuple[pd.DataFrame, Optional[str]]:
    working = df.copy()
    if activity_mapping.activity_column is None or activity_mapping.activity_kind is None:
        return working, None

    try:
        values, activity_kind, _mapping = activity_series_for_landscape(working, activity_mapping)
    except ValueError as exc:
        logger.warning(
            "Could not compute final activity values from column '%s': %s",
            activity_mapping.activity_column,
            exc,
        )
        return working, None

    working["_activity_final_row"] = values.reindex(working.index)
    return working, activity_kind


def _merge_standardized_compounds(
    df: pd.DataFrame,
    *,
    activity_mapping: ActivityMapping,
    activity_kind: Optional[str],
) -> pd.DataFrame:
    metadata_columns = _metadata_columns(df, activity_mapping)
    records: list[dict[str, Any]] = []

    for smiles, group in df.groupby("smiles", sort=False, dropna=False):
        record: dict[str, Any] = {
            "smiles": smiles,
            RAW_SMILES_COLUMN: _joined_unique(group[RAW_SMILES_COLUMN]),
            "source_row_count": int(len(group)),
            "source_row_numbers": _joined_unique(group["_source_row_number"]),
        }

        for column in metadata_columns:
            output_column = _merged_column_name(column)
            record[output_column] = _joined_unique(group[column])

        _add_activity_record(record, group, activity_mapping, activity_kind)
        records.append(record)

    clean_df = pd.DataFrame(records)
    return _order_clean_columns(clean_df)


def _add_activity_record(
    record: dict[str, Any],
    group: pd.DataFrame,
    activity_mapping: ActivityMapping,
    activity_kind: Optional[str],
) -> None:
    if activity_kind is None or "_activity_final_row" not in group.columns:
        return

    record["activity_source_column"] = activity_mapping.activity_column
    record["activity_source_kind"] = activity_mapping.activity_kind
    record["activity_source_semantics"] = activity_mapping.activity_semantics
    record["activity_source_format"] = activity_mapping.source_format

    if activity_mapping.endpoint:
        record["activity_endpoint"] = activity_mapping.endpoint
    if activity_mapping.endpoint_column and activity_mapping.endpoint_column in group.columns:
        endpoint_values = _joined_unique(group[activity_mapping.endpoint_column])
        if endpoint_values:
            record["activity_endpoint"] = endpoint_values
    if activity_mapping.units_column and activity_mapping.units_column in group.columns:
        record["activity_units"] = _joined_unique(group[activity_mapping.units_column])
    elif activity_mapping.detected_units:
        record["activity_units"] = activity_mapping.detected_units

    if activity_kind == "classification":
        labels = group["_activity_final_row"].dropna().astype(str)
        if labels.empty:
            record["activity_n"] = 0
            return
        counts = labels.value_counts()
        active_count = int(counts.get("active", 0))
        inactive_count = int(counts.get("inactive", 0))
        record[FINAL_ACTIVITY_COLUMN] = str(counts.index[0])
        record["activity"] = record[FINAL_ACTIVITY_COLUMN]
        record["activity_n"] = int(labels.shape[0])
        record["activity_active_count"] = active_count
        record["activity_inactive_count"] = inactive_count
        record["activity_active_fraction"] = active_count / labels.shape[0]
        record["activity_label_counts"] = "|".join(
            f"{label}:{int(count)}" for label, count in counts.items()
        )
        return

    values = pd.to_numeric(group["_activity_final_row"], errors="coerce").dropna()
    if values.empty:
        record["activity_n"] = 0
        return

    median = float(values.median())
    record[FINAL_ACTIVITY_COLUMN] = median
    record["activity"] = median
    record["activity_score_median"] = median
    record["activity_score_mean"] = float(values.mean())
    record["activity_score_min"] = float(values.min())
    record["activity_score_max"] = float(values.max())
    record["activity_score_std"] = float(values.std()) if len(values) > 1 else np.nan
    record["activity_n"] = int(values.shape[0])
    record["activity_units_final"] = "p-scale"
    record["activity_semantics"] = "higher_is_better"
    if activity_mapping.activity_column and activity_mapping.activity_column in group.columns:
        record["activity_source_values"] = _joined_unique(group[activity_mapping.activity_column])


def _final_activity_mapping(
    clean_df: pd.DataFrame,
    source_mapping: ActivityMapping,
    activity_kind: Optional[str],
) -> ActivityMapping:
    if activity_kind is None or FINAL_ACTIVITY_COLUMN not in clean_df.columns:
        return ActivityMapping(
            smiles_column="smiles",
            activity_column=None,
            activity_kind=None,
            activity_semantics=None,
            source_format=source_mapping.source_format,
        )

    return ActivityMapping(
        smiles_column="smiles",
        activity_column=FINAL_ACTIVITY_COLUMN,
        activity_kind=activity_kind,  # type: ignore[arg-type]
        activity_semantics="label" if activity_kind == "classification" else "higher_is_better",
        source_format=source_mapping.source_format,
        endpoint=source_mapping.endpoint,
        endpoint_column="activity_endpoint" if "activity_endpoint" in clean_df.columns else None,
        units_column="activity_units" if "activity_units" in clean_df.columns else None,
        target_column=("target_chembl_ids" if "target_chembl_ids" in clean_df.columns else None),
        assay_column="assay_chembl_ids" if "assay_chembl_ids" in clean_df.columns else None,
        molecule_id_column=(
            "molecule_chembl_ids" if "molecule_chembl_ids" in clean_df.columns else None
        ),
        score_name=FINAL_ACTIVITY_COLUMN,
    )


def _build_descriptor_dataframe(
    clean_df: pd.DataFrame,
    *,
    descriptor_type: Optional[str],
) -> tuple[pd.DataFrame, str, str]:
    encoder = MolecularDescriptorEncoder(
        default_descriptor=descriptor_type or DEFAULT_DESCRIPTOR_TYPE
    )
    descriptor_type_used = encoder.default_descriptor
    descriptor_column = encoder.column_name(descriptor_type_used)
    descriptors = encoder.encode(clean_df["smiles"].astype(str).tolist(), descriptor_type_used)

    descriptor_df = clean_df.copy()
    descriptor_df["descriptor_type"] = descriptor_type_used
    descriptor_df["descriptor_column"] = descriptor_column
    descriptor_df[descriptor_column] = [vector.tolist() for vector in descriptors]
    return descriptor_df, descriptor_type_used, descriptor_column


def _metadata_columns(df: pd.DataFrame, activity_mapping: ActivityMapping) -> list[str]:
    candidates = set(_MERGE_METADATA_COLUMNS)
    for column in (
        activity_mapping.molecule_id_column,
        activity_mapping.assay_column,
        activity_mapping.target_column,
        activity_mapping.endpoint_column,
        activity_mapping.units_column,
        activity_mapping.relation_column,
    ):
        if column:
            candidates.add(column)

    excluded = {"smiles", RAW_SMILES_COLUMN, activity_mapping.activity_column} | _HELPER_COLUMNS
    return [column for column in df.columns if column in candidates and column not in excluded]


def _order_clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    preferred = [
        "smiles",
        RAW_SMILES_COLUMN,
        "source_row_count",
        "source_row_numbers",
        "molecule_chembl_ids",
        "compound_ids",
        "molecule_ids",
        "ids",
        "activity_ids",
        "assay_chembl_ids",
        "target_chembl_ids",
        "standard_types",
        "standard_units",
        "cluster_id",
        FINAL_ACTIVITY_COLUMN,
        "activity",
        "activity_score_median",
        "activity_score_mean",
        "activity_score_min",
        "activity_score_max",
        "activity_score_std",
        "activity_n",
    ]
    ordered = [column for column in preferred if column in df.columns]
    ordered.extend(column for column in df.columns if column not in ordered)
    return df[ordered]


def _save_raw_dataset(
    df: pd.DataFrame,
    *,
    source_stem: str,
    raw_dataset_path: Optional[str],
    raw_filename: Optional[str],
    save_raw: bool,
) -> str:
    if raw_dataset_path:
        return raw_dataset_path
    if not save_raw:
        return ""
    return _write_csv(df, raw_filename or f"{source_stem}_raw.csv")


def _write_csv(df: pd.DataFrame, filename: str) -> str:
    with S3.open(filename, "w") as handle:
        df.to_csv(handle, index=False)
    return S3.path(filename)


def _write_parquet(df: pd.DataFrame, filename: str) -> str:
    with S3.open(filename, "wb") as handle:
        df.to_parquet(handle, index=False)
    return S3.path(filename)


def _write_report(summary: dict[str, Any], filename: str) -> str:
    report = _format_report(summary)
    with S3.open(filename, "w") as handle:
        handle.write(report)
    return S3.path(filename)


def _format_report(summary: dict[str, Any]) -> str:
    artifacts = [
        ("Raw dataset", summary.get("raw_dataset_path")),
        ("Clean dataset", summary.get("clean_dataset_path")),
        ("Descriptor Parquet", summary.get("descriptor_parquet_path")),
    ]
    artifact_lines = "\n".join(f"- {label}: `{path}`" for label, path in artifacts if path)

    counts = {
        key: summary.get(key)
        for key in (
            "raw_rows",
            "string_smiles_rows",
            "rows_after_standardization",
            "invalid_smiles_rows",
            "standardized_smiles_changed_rows",
            "duplicate_rows_after_standardization",
            "duplicate_compound_groups_after_standardization",
            "raw_smiles_collapse_groups",
            "clean_rows",
            "clean_unique_smiles",
        )
    }

    examples = summary.get("raw_smiles_collapse_examples") or []
    examples_table = pd.DataFrame(examples).to_markdown(index=False) if examples else "None"

    change_examples = summary.get("standardization_change_examples") or []
    changes_table = (
        pd.DataFrame(change_examples).to_markdown(index=False) if change_examples else "None"
    )

    return (
        "# Dataset Standardization Report\n\n"
        "## Artifacts\n"
        f"{artifact_lines}\n\n"
        "## Procedure\n"
        "- Preserved the raw dataset for provenance.\n"
        "- Standardized molecules with RDKit cleanup, largest-fragment selection, uncharging, "
        "tautomer canonicalization, canonical SMILES generation, and default stereochemistry "
        "removal.\n"
        "- Dropped rows with invalid or unstandardizable SMILES.\n"
        "- Merged rows by final standardized achiral SMILES.\n"
        "- Converted numeric potency activity to final higher-is-better p-scale values where "
        "possible and merged replicates with the median p-scale policy.\n"
        "- Wrote descriptors to Parquet, not into the clean CSV, while preserving final "
        "activity values in the Parquet file.\n\n"
        "## Counts\n"
        f"```json\n{json.dumps(counts, indent=2, sort_keys=True, default=str)}\n```\n\n"
        "## Activity\n"
        f"```json\n{json.dumps(summary.get('activity_mapping', {}), indent=2, sort_keys=True, default=str)}\n```\n\n"
        "## Final Activity\n"
        f"```json\n{json.dumps(summary.get('final_activity_mapping', {}), indent=2, sort_keys=True, default=str)}\n```\n\n"
        "## Standardization Changes\n"
        f"{changes_table}\n\n"
        "## Raw SMILES Collapse Examples\n"
        f"{examples_table}\n"
    )


def _safe_stem(value: str) -> str:
    stem = Path(str(value).rstrip("/")).stem or "dataset"
    stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", stem)
    stem = stem.strip("._-")
    return stem or "dataset"


def _unique_column_name(columns: Any, base_name: str) -> str:
    existing = set(columns)
    if base_name not in existing:
        return base_name
    suffix = 1
    while True:
        candidate = f"{base_name}_{suffix}"
        if candidate not in existing:
            return candidate
        suffix += 1


def _change_examples(df: pd.DataFrame, *, limit: int = 20) -> list[dict[str, Any]]:
    examples = []
    for _idx, row in df.head(limit).iterrows():
        examples.append(
            {
                "raw_smiles": _format_scalar(row.iloc[0]),
                "standardized_smiles": _format_scalar(row.iloc[1]),
            }
        )
    return examples


def _collapse_examples(
    df: pd.DataFrame,
    raw_smiles_column: str,
    *,
    limit: int = 50,
) -> list[dict[str, Any]]:
    examples = []
    for smiles, group in df.groupby("smiles", sort=False):
        raw_values = _unique_values(group[raw_smiles_column])
        if len(raw_values) <= 1:
            continue
        examples.append(
            {
                "standardized_smiles": smiles,
                "raw_smiles": "|".join(raw_values),
                "row_count": int(len(group)),
            }
        )
        if len(examples) >= limit:
            break
    return examples


def _merged_column_name(column: str) -> str:
    if column == "cluster_id":
        return "cluster_id"
    if column == "query_keywords":
        return "query_keywords"
    if column == "source":
        return "source"
    if column == "id":
        return "ids"
    if column.endswith("_id"):
        return f"{column}s"
    if column.endswith("y"):
        return f"{column[:-1]}ies"
    return f"{column}s"


def _joined_unique(series: pd.Series) -> Optional[str]:
    values = _unique_values(series)
    return "|".join(values) if values else None


def _unique_values(series: pd.Series) -> list[str]:
    values: list[str] = []
    seen = set()
    for value in series.tolist():
        if _is_missing(value):
            continue
        for token in str(_format_scalar(value)).split("|"):
            token = token.strip()
            if not token or token in seen:
                continue
            seen.add(token)
            values.append(token)
    return values


def _format_scalar(value: Any) -> Any:
    if hasattr(value, "item"):
        try:
            value = value.item()
        except Exception:
            pass
    return value


def _is_missing(value: Any) -> bool:
    try:
        missing = pd.isna(value)
    except Exception:
        return False
    if isinstance(missing, (bool, np.bool_)):
        return bool(missing)
    return False
