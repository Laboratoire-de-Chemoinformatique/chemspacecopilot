#!/usr/bin/env python
# coding: utf-8
"""Activity and compound column inference for tabular molecular datasets."""

from __future__ import annotations

import math
import re
from dataclasses import asdict, dataclass
from typing import Any, Dict, Iterable, Literal, Optional

import numpy as np
import pandas as pd

from cs_copilot.tools.chemistry.smiles_columns import find_smiles_column_name

ActivityKind = Literal["regression", "classification"]
ActivitySemantics = Literal["higher_is_better", "lower_is_better", "label"]

_SMILES_EXACT_NAMES = (
    "smiles",
    "smi",
    "canonical_smiles",
    "canonical_smiles_standardized",
    "SMILES",
    "Smiles",
)

_PVALUE_COLUMNS = {
    "pchemblvalue": ("pchembl_value", "pChEMBL"),
    "pic50": ("pIC50", "IC50"),
    "pec50": ("pEC50", "EC50"),
    "pki": ("pKi", "Ki"),
    "pkd": ("pKd", "Kd"),
    "pmic": ("pMIC", "MIC"),
}

_POTENCY_COLUMNS = {
    "ic50": "IC50",
    "ec50": "EC50",
    "ki": "Ki",
    "kd": "Kd",
    "mic": "MIC",
    "standardvalue": None,
    "value": None,
    "potency": None,
}

_CLASSIFICATION_COLUMNS = {
    "activity",
    "activitycomment",
    "label",
    "class",
}

_UNIT_FACTORS_TO_MOLAR = {
    "m": 1.0,
    "mol/l": 1.0,
    "molar": 1.0,
    "mm": 1e-3,
    "millimolar": 1e-3,
    "um": 1e-6,
    "µm": 1e-6,
    "μm": 1e-6,
    "micromolar": 1e-6,
    "nm": 1e-9,
    "nanomolar": 1e-9,
    "pm": 1e-12,
    "picomolar": 1e-12,
}

_ACTIVE_LABELS = {"active", "act", "positive", "pos", "hit", "yes", "true", "1"}
_INACTIVE_LABELS = {
    "inactive",
    "inact",
    "negative",
    "neg",
    "no",
    "false",
    "0",
    "non-active",
    "not active",
    "inconclusive",
}


@dataclass(frozen=True)
class ActivityMapping:
    """Compact description of inferred dataset activity columns."""

    smiles_column: Optional[str]
    activity_column: Optional[str]
    activity_kind: Optional[ActivityKind]
    activity_semantics: Optional[ActivitySemantics]
    source_format: str
    endpoint: Optional[str] = None
    endpoint_column: Optional[str] = None
    units_column: Optional[str] = None
    detected_units: Optional[str] = None
    relation_column: Optional[str] = None
    target_column: Optional[str] = None
    assay_column: Optional[str] = None
    molecule_id_column: Optional[str] = None
    score_name: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-friendly mapping, dropping empty values."""
        return {key: value for key, value in asdict(self).items() if value is not None}


def _canonical_name(column_name: object) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(column_name).lower())


def _column_lookup(columns: Iterable[object]) -> Dict[str, str]:
    lookup: Dict[str, str] = {}
    for column in columns:
        lookup.setdefault(_canonical_name(column), str(column))
    return lookup


def _find_first_column(df: pd.DataFrame, names: Iterable[str]) -> Optional[str]:
    lookup = _column_lookup(df.columns)
    for name in names:
        match = lookup.get(_canonical_name(name))
        if match is not None:
            return match
    return None


def _find_units_column(df: pd.DataFrame, activity_column: str) -> Optional[str]:
    candidates = (
        f"{activity_column}_units",
        f"{activity_column}_unit",
        "standard_units",
        "units",
        "unit",
        "activity_units",
        "value_units",
    )
    return _find_first_column(df, candidates)


def _find_units_in_name(column_name: str) -> Optional[str]:
    for match in re.finditer(r"(?:^|[^a-zA-Z])(pM|nM|uM|µM|μM|mM|M)(?:$|[^a-zA-Z])", column_name):
        return _normalize_units(match.group(1))
    return None


def _infer_endpoint_from_column(column_name: str) -> Optional[str]:
    canonical = _canonical_name(column_name)
    if canonical in _PVALUE_COLUMNS:
        return _PVALUE_COLUMNS[canonical][1]
    for key, endpoint in _POTENCY_COLUMNS.items():
        if endpoint and key in canonical:
            return endpoint
    return None


def _normalize_units(units: Any) -> Optional[str]:
    if units is None or (isinstance(units, float) and math.isnan(units)):
        return None
    normalized = str(units).strip()
    if not normalized:
        return None
    normalized = normalized.replace("μ", "µ")
    lower = normalized.lower().replace(" ", "")
    if lower == "um":
        lower = "µm"
    return lower


def _detect_constant_units(df: pd.DataFrame, units_column: Optional[str]) -> Optional[str]:
    if units_column is None or units_column not in df.columns:
        return None
    values = {_normalize_units(value) for value in df[units_column].dropna().unique()}
    values.discard(None)
    if len(values) == 1:
        return next(iter(values))
    if len(values) > 1:
        return "mixed"
    return None


def _looks_like_numeric(series: pd.Series) -> bool:
    numeric = pd.to_numeric(series, errors="coerce")
    return bool(numeric.notna().any())


def _parse_activity_label(value: Any) -> Optional[str]:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    text = str(value).strip().lower()
    if not text or text == "nan":
        return None
    if text in _INACTIVE_LABELS:
        return "inactive"
    if text in _ACTIVE_LABELS:
        return "active"
    if "inactive" in text or "not active" in text or "inconclusive" in text:
        return "inactive"
    if "active" in text or "hit" in text:
        return "active"
    return None


def normalize_activity_labels(series: pd.Series) -> pd.Series:
    """Normalize common activity labels to active/inactive, preserving nulls."""
    return series.apply(_parse_activity_label).replace([None], pd.NA)


def find_smiles_column(df: pd.DataFrame) -> Optional[str]:
    """Find a likely SMILES column in a dataset."""
    return find_smiles_column_name(df.columns, exact_names=_SMILES_EXACT_NAMES)


def infer_activity_mapping(
    df: pd.DataFrame,
    *,
    smiles_column: Optional[str] = None,
    activity_column: Optional[str] = None,
) -> ActivityMapping:
    """Infer SMILES and activity columns from a molecular dataset."""
    smiles_found = smiles_column if smiles_column in df.columns else find_smiles_column(df)
    source_format = _infer_source_format(df)
    target_column = _find_first_column(df, ("target_chembl_id", "target", "target_id", "protein"))
    assay_column = _find_first_column(df, ("assay_chembl_id", "assay_id", "assay", "source"))
    molecule_id_column = _find_first_column(
        df, ("molecule_chembl_id", "compound_id", "molecule_id", "id")
    )
    relation_column = _find_first_column(
        df, ("standard_relation", "relation", "operator", "activity_relation")
    )

    activity_found = None
    if activity_column and activity_column in df.columns:
        activity_found = activity_column
    if activity_found:
        explicit = _mapping_for_explicit_activity(
            df,
            activity_found,
            smiles_found,
            source_format,
            target_column=target_column,
            assay_column=assay_column,
            molecule_id_column=molecule_id_column,
            relation_column=relation_column,
        )
        if explicit.activity_kind is not None:
            return explicit

    pvalue = _find_pvalue_column(df)
    if pvalue is not None:
        canonical = _canonical_name(pvalue)
        score_name, endpoint = _PVALUE_COLUMNS[canonical]
        return ActivityMapping(
            smiles_column=smiles_found,
            activity_column=pvalue,
            activity_kind="regression",
            activity_semantics="higher_is_better",
            endpoint=endpoint,
            target_column=target_column,
            assay_column=assay_column,
            molecule_id_column=molecule_id_column,
            relation_column=relation_column,
            source_format=source_format,
            score_name=score_name,
        )

    raw = _find_raw_potency_column(df)
    if raw is not None:
        units_column = _find_units_column(df, raw)
        endpoint_column = "standard_type" if "standard_type" in df.columns else None
        endpoint = _infer_endpoint_from_column(raw)
        detected_units = _find_units_in_name(raw) or _detect_constant_units(df, units_column)
        return ActivityMapping(
            smiles_column=smiles_found,
            activity_column=raw,
            activity_kind="regression",
            activity_semantics="lower_is_better",
            endpoint=endpoint,
            endpoint_column=endpoint_column,
            units_column=units_column,
            detected_units=detected_units,
            target_column=target_column,
            assay_column=assay_column,
            molecule_id_column=molecule_id_column,
            relation_column=relation_column,
            source_format=source_format,
            score_name=f"p{endpoint}" if endpoint else "activity_score",
        )

    label = _find_label_column(df)
    if label is not None:
        return ActivityMapping(
            smiles_column=smiles_found,
            activity_column=label,
            activity_kind="classification",
            activity_semantics="label",
            target_column=target_column,
            assay_column=assay_column,
            molecule_id_column=molecule_id_column,
            relation_column=relation_column,
            source_format=source_format,
            score_name="activity_label",
        )

    return ActivityMapping(
        smiles_column=smiles_found,
        activity_column=None,
        activity_kind=None,
        activity_semantics=None,
        target_column=target_column,
        assay_column=assay_column,
        molecule_id_column=molecule_id_column,
        relation_column=relation_column,
        source_format=source_format,
    )


def _mapping_for_explicit_activity(
    df: pd.DataFrame,
    activity_column: str,
    smiles_column: Optional[str],
    source_format: str,
    *,
    target_column: Optional[str],
    assay_column: Optional[str],
    molecule_id_column: Optional[str],
    relation_column: Optional[str],
) -> ActivityMapping:
    labels = normalize_activity_labels(df[activity_column])
    canonical = _canonical_name(activity_column)
    if labels.notna().any() and (
        canonical in _CLASSIFICATION_COLUMNS or not _looks_like_numeric(df[activity_column])
    ):
        return ActivityMapping(
            smiles_column=smiles_column,
            activity_column=activity_column,
            activity_kind="classification",
            activity_semantics="label",
            target_column=target_column,
            assay_column=assay_column,
            molecule_id_column=molecule_id_column,
            relation_column=relation_column,
            source_format=source_format,
            score_name="activity_label",
        )

    if _looks_like_numeric(df[activity_column]):
        if canonical in _PVALUE_COLUMNS:
            score_name, endpoint = _PVALUE_COLUMNS[canonical]
            semantics: ActivitySemantics = "higher_is_better"
        elif canonical in _POTENCY_COLUMNS or _infer_endpoint_from_column(activity_column):
            score_name = "activity_score"
            endpoint = _infer_endpoint_from_column(activity_column)
            semantics = "lower_is_better"
        else:
            score_name = "activity"
            endpoint = None
            semantics = "higher_is_better"
        units_column = _find_units_column(df, activity_column)
        return ActivityMapping(
            smiles_column=smiles_column,
            activity_column=activity_column,
            activity_kind="regression",
            activity_semantics=semantics,
            endpoint=endpoint,
            units_column=units_column,
            detected_units=_find_units_in_name(activity_column)
            or _detect_constant_units(df, units_column),
            target_column=target_column,
            assay_column=assay_column,
            molecule_id_column=molecule_id_column,
            relation_column=relation_column,
            source_format=source_format,
            score_name=score_name,
        )

    return ActivityMapping(
        smiles_column=smiles_column,
        activity_column=activity_column,
        activity_kind=None,
        activity_semantics=None,
        target_column=target_column,
        assay_column=assay_column,
        molecule_id_column=molecule_id_column,
        relation_column=relation_column,
        source_format=source_format,
    )


def _find_pvalue_column(df: pd.DataFrame) -> Optional[str]:
    lookup = _column_lookup(df.columns)
    for canonical in _PVALUE_COLUMNS:
        column = lookup.get(canonical)
        if column is not None and _looks_like_numeric(df[column]):
            return column
    return None


def _find_raw_potency_column(df: pd.DataFrame) -> Optional[str]:
    lookup = _column_lookup(df.columns)
    for canonical in _POTENCY_COLUMNS:
        column = lookup.get(canonical)
        if column is not None and _looks_like_numeric(df[column]):
            return column

    for column in df.columns:
        endpoint = _infer_endpoint_from_column(str(column))
        if endpoint and _looks_like_numeric(df[column]):
            return str(column)
    return None


def _find_label_column(df: pd.DataFrame) -> Optional[str]:
    lookup = _column_lookup(df.columns)
    for canonical in _CLASSIFICATION_COLUMNS:
        column = lookup.get(canonical)
        if column is None:
            continue
        labels = normalize_activity_labels(df[column])
        if labels.notna().any():
            return column
    return None


def _infer_source_format(df: pd.DataFrame) -> str:
    chembl_columns = {
        "molecule_chembl_id",
        "assay_chembl_id",
        "target_chembl_id",
        "pchembl_value",
        "standard_type",
        "standard_value",
        "standard_units",
        "activity_comment",
    }
    return "chembl" if chembl_columns.intersection(set(df.columns)) else "user_dataset"


def activity_series_for_landscape(
    df: pd.DataFrame,
    mapping: Optional[ActivityMapping] = None,
) -> tuple[pd.Series, ActivityKind, ActivityMapping]:
    """Return a normalized activity series suitable for GTM landscapes."""
    mapping = mapping or infer_activity_mapping(df)
    if mapping.activity_column is None or mapping.activity_kind is None:
        raise ValueError(
            "No valid activity column found. Expected p-scale potency columns "
            "such as pIC50/pKi/pChEMBL, raw potency columns with detectable units, "
            "or active/inactive label columns. "
            f"Available columns: {list(df.columns)}"
        )

    if mapping.activity_kind == "classification":
        labels = normalize_activity_labels(df[mapping.activity_column])
        if not labels.notna().any():
            raise ValueError(
                f"Activity column '{mapping.activity_column}' does not contain usable "
                "active/inactive labels."
            )
        return labels, "classification", mapping

    values = pd.to_numeric(df[mapping.activity_column], errors="coerce")
    if mapping.activity_semantics == "higher_is_better":
        return values, "regression", mapping

    score = potency_to_pvalue_series(df, mapping)
    if score.notna().any():
        return score, "regression", mapping
    raise ValueError(
        f"Activity column '{mapping.activity_column}' looks like a lower-is-better potency "
        "field, but units could not be inferred. Add a units column such as 'standard_units' "
        "or encode units in the column name, e.g. 'IC50_nM'."
    )


def potency_to_pvalue_series(df: pd.DataFrame, mapping: ActivityMapping) -> pd.Series:
    """Convert lower-is-better concentration values to p-scale scores."""
    if mapping.activity_column is None:
        return pd.Series([np.nan] * len(df), index=df.index, dtype=float)
    values = pd.to_numeric(df[mapping.activity_column], errors="coerce")
    units = _units_for_rows(df, mapping)

    def convert(value: Any, unit: Any) -> float:
        if pd.isna(value):
            return np.nan
        normalized_unit = _normalize_units(unit)
        if normalized_unit not in _UNIT_FACTORS_TO_MOLAR:
            return np.nan
        molar = float(value) * _UNIT_FACTORS_TO_MOLAR[normalized_unit]
        if molar <= 0:
            return np.nan
        return -math.log10(molar)

    return pd.Series(
        [convert(value, unit) for value, unit in zip(values, units, strict=True)],
        index=df.index,
        dtype=float,
    )


def _units_for_rows(df: pd.DataFrame, mapping: ActivityMapping) -> pd.Series:
    if mapping.units_column and mapping.units_column in df.columns:
        return df[mapping.units_column]
    if mapping.detected_units and mapping.detected_units != "mixed":
        return pd.Series([mapping.detected_units] * len(df), index=df.index)
    return pd.Series([None] * len(df), index=df.index)


def add_normalized_activity_column(
    df: pd.DataFrame,
    mapping: ActivityMapping,
    *,
    output_column: str = "activity",
) -> pd.DataFrame:
    """Return a copy with a normalized higher-is-better or label activity column."""
    if mapping.activity_column is None or mapping.activity_kind is None:
        return df.copy()
    normalized = df.copy()
    series, _kind, _mapping = activity_series_for_landscape(normalized, mapping)
    if output_column in normalized.columns and output_column != mapping.activity_column:
        normalized = normalized.drop(columns=[output_column])
    normalized[output_column] = series
    return normalized


def build_compound_memory_preview(
    df: pd.DataFrame,
    mapping: Optional[ActivityMapping] = None,
    *,
    limit: int = 50,
) -> list[Dict[str, Any]]:
    """Build compact compound records with sparse activity metadata."""
    mapping = mapping or infer_activity_mapping(df)
    smiles_col = mapping.smiles_column or find_smiles_column(df)
    if smiles_col is None or smiles_col not in df.columns:
        return []

    preview = df.copy()
    activity_values = None
    activity_kind = None
    if mapping.activity_column and mapping.activity_kind:
        try:
            activity_values, activity_kind, _mapping = activity_series_for_landscape(
                preview, mapping
            )
            sort_values = _activity_sort_values(activity_values, activity_kind)
            preview = preview.assign(_session_activity_sort=sort_values)
            preview = preview.sort_values(
                "_session_activity_sort", ascending=False, na_position="last"
            )
        except ValueError:
            activity_values = None

    preview = preview.drop_duplicates(subset=[smiles_col]).head(limit)
    records: list[Dict[str, Any]] = []
    for idx, row in preview.iterrows():
        smiles = row.get(smiles_col)
        if pd.isna(smiles):
            continue
        item: Dict[str, Any] = {"smiles": smiles}
        for column in (mapping.molecule_id_column, "canonical_smiles", "smi"):
            if column and column in row.index and pd.notna(row.get(column)):
                item[column] = row.get(column)

        activity = _activity_payload_for_row(row, mapping, activity_values, activity_kind, idx)
        if activity:
            item["activity"] = activity
        records.append(item)
    return records


def _activity_sort_values(series: pd.Series, kind: Optional[ActivityKind]) -> pd.Series:
    if kind == "classification":
        return series.map({"active": 1, "inactive": 0})
    return pd.to_numeric(series, errors="coerce")


def _activity_payload_for_row(
    row: pd.Series,
    mapping: ActivityMapping,
    activity_values: Optional[pd.Series],
    activity_kind: Optional[ActivityKind],
    row_index: Any,
) -> Dict[str, Any]:
    if mapping.activity_column is None or mapping.activity_column not in row.index:
        return {}

    activity: Dict[str, Any] = {
        "source_column": mapping.activity_column,
        "kind": mapping.activity_kind,
        "semantics": mapping.activity_semantics,
        "source_format": mapping.source_format,
    }
    if mapping.endpoint:
        activity["endpoint"] = mapping.endpoint
    if (
        mapping.endpoint_column
        and mapping.endpoint_column in row.index
        and pd.notna(row.get(mapping.endpoint_column))
    ):
        activity["endpoint"] = row.get(mapping.endpoint_column)
    raw_value = row.get(mapping.activity_column)
    if pd.notna(raw_value):
        activity["value"] = raw_value
    if (
        mapping.units_column
        and mapping.units_column in row.index
        and pd.notna(row.get(mapping.units_column))
    ):
        activity["units"] = row.get(mapping.units_column)
    elif mapping.detected_units:
        activity["units"] = mapping.detected_units
    for key, column in (
        ("relation", mapping.relation_column),
        ("target", mapping.target_column),
        ("assay_id", mapping.assay_column),
    ):
        if column and column in row.index and pd.notna(row.get(column)):
            activity[key] = row.get(column)
    if activity_values is not None and row_index in activity_values.index:
        normalized_value = activity_values.loc[row_index]
        if pd.notna(normalized_value):
            if activity_kind == "classification":
                activity["label"] = normalized_value
            else:
                activity["score"] = normalized_value
                if mapping.score_name:
                    activity["score_name"] = mapping.score_name
    return activity
