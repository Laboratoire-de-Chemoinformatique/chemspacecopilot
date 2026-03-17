#!/usr/bin/env python
"""Unit tests for PointerPandasTools column parsing fixes."""

import pandas as pd
import pytest

from cs_copilot.tools.io.pointer_pandas_tools import PointerPandasTools, _coerce_columns


class TestCoerceColumns:
    """Tests for the _coerce_columns helper function."""

    def test_single_string(self):
        """Test single column name as string."""
        result = _coerce_columns("col1", "test")
        assert result == ["col1"]

    def test_comma_separated_string(self):
        """Test comma-separated column names."""
        result = _coerce_columns("col1,col2,col3", "test")
        assert result == ["col1", "col2", "col3"]

    def test_comma_separated_with_spaces(self):
        """Test comma-separated names with spaces."""
        result = _coerce_columns("col1, col2 , col3", "test")
        assert result == ["col1", "col2", "col3"]

    def test_list_of_strings(self):
        """Test list of column names."""
        result = _coerce_columns(["col1", "col2"], "test")
        assert result == ["col1", "col2"]

    def test_tuple_of_strings(self):
        """Test tuple of column names."""
        result = _coerce_columns(("col1", "col2"), "test")
        assert result == ["col1", "col2"]

    def test_string_representation_of_list(self):
        """Test string that looks like a list."""
        result = _coerce_columns("['col1', 'col2', 'col3']", "test")
        assert result == ["col1", "col2", "col3"]

    def test_string_representation_with_spaces(self):
        """Test string list with spaces."""
        result = _coerce_columns("['col1', 'col2 name', 'col3']", "test")
        assert result == ["col1", "col2 name", "col3"]

    def test_string_representation_with_newlines(self):
        """Test string list with newlines and indentation."""
        messy_string = "['canonical_smiles', 'standard_value',\n         'standard_type', 'molecule_chembl_id']"
        result = _coerce_columns(messy_string, "test")
        assert result == [
            "canonical_smiles",
            "standard_value",
            "standard_type",
            "molecule_chembl_id",
        ]

    def test_string_representation_with_leading_whitespace(self):
        """Test string list with leading/trailing whitespace."""
        result = _coerce_columns("  ['col1', 'col2']  ", "test")
        assert result == ["col1", "col2"]

    def test_none_raises_error(self):
        """Test that None raises ValueError."""
        with pytest.raises(ValueError, match="test parameter must be provided"):
            _coerce_columns(None, "test")

    def test_invalid_type_raises_error(self):
        """Test that invalid type raises ValueError."""
        with pytest.raises(ValueError, match="must be a string or list"):
            _coerce_columns(123, "test")


class TestPointerPandasTools:
    """Tests for PointerPandasTools operations with various column input formats."""

    @pytest.fixture
    def tools(self):
        """Create toolkit instance."""
        return PointerPandasTools()

    @pytest.fixture
    def sample_df(self):
        """Create sample DataFrame."""
        return pd.DataFrame(
            {
                "molecule_chembl_id": ["CHEMBL1", "CHEMBL2", "CHEMBL3"],
                "canonical_smiles": ["C", "CC", "CCC"],
                "standard_value": [100, 200, 300],
                "standard_type": ["IC50", "IC50", "Ki"],
                "standard_units": ["nM", "nM", "nM"],
                "assay_chembl_id": ["ASSAY1", "ASSAY2", "ASSAY3"],
            }
        )

    def test_filter_with_comma_separated_string(self, tools, sample_df):
        """Test filter() with comma-separated column string."""
        tools.dataframes["test_df"] = sample_df

        result = tools.run_dataframe_operation(
            dataframe_name="test_df",
            operation="filter",
            operation_parameters={"items": "molecule_chembl_id,canonical_smiles,standard_value"},
        )

        assert "dataframe_name" in result
        result_df = tools.dataframes[result["dataframe_name"]]
        assert result_df.shape == (3, 3)
        assert list(result_df.columns) == [
            "molecule_chembl_id",
            "canonical_smiles",
            "standard_value",
        ]

    def test_filter_with_list(self, tools, sample_df):
        """Test filter() with proper list."""
        tools.dataframes["test_df"] = sample_df

        result = tools.run_dataframe_operation(
            dataframe_name="test_df",
            operation="filter",
            operation_parameters={"items": ["molecule_chembl_id", "standard_value"]},
        )

        result_df = tools.dataframes[result["dataframe_name"]]
        assert result_df.shape == (3, 2)
        assert list(result_df.columns) == ["molecule_chembl_id", "standard_value"]

    def test_getitem_with_comma_separated_string(self, tools, sample_df):
        """Test __getitem__ with comma-separated string."""
        tools.dataframes["test_df"] = sample_df

        result = tools.run_dataframe_operation(
            dataframe_name="test_df",
            operation="__getitem__",
            operation_parameters={"columns": "molecule_chembl_id,canonical_smiles"},
        )

        result_df = tools.dataframes[result["dataframe_name"]]
        assert result_df.shape == (3, 2)
        assert list(result_df.columns) == ["molecule_chembl_id", "canonical_smiles"]

    def test_getitem_with_string_list(self, tools, sample_df):
        """Test __getitem__ with string representation of list."""
        tools.dataframes["test_df"] = sample_df

        result = tools.run_dataframe_operation(
            dataframe_name="test_df",
            operation="__getitem__",
            operation_parameters={
                "columns": "['molecule_chembl_id', 'canonical_smiles', 'standard_value']"
            },
        )

        result_df = tools.dataframes[result["dataframe_name"]]
        assert result_df.shape == (3, 3)
        assert list(result_df.columns) == [
            "molecule_chembl_id",
            "canonical_smiles",
            "standard_value",
        ]

    def test_loc_with_columns(self, tools, sample_df):
        """Test loc with columns parameter."""
        tools.dataframes["test_df"] = sample_df

        result = tools.run_dataframe_operation(
            dataframe_name="test_df",
            operation="loc",
            operation_parameters={"columns": "molecule_chembl_id,canonical_smiles"},
        )

        result_df = tools.dataframes[result["dataframe_name"]]
        assert result_df.shape == (3, 2)
        assert list(result_df.columns) == ["molecule_chembl_id", "canonical_smiles"]

    def test_select_with_comma_separated_string(self, tools, sample_df):
        """Test select with comma-separated string."""
        tools.dataframes["test_df"] = sample_df

        result = tools.run_dataframe_operation(
            dataframe_name="test_df",
            operation="select",
            operation_parameters={"columns": "standard_type,standard_units"},
        )

        result_df = tools.dataframes[result["dataframe_name"]]
        assert result_df.shape == (3, 2)
        assert list(result_df.columns) == ["standard_type", "standard_units"]

    def test_invalid_column_names(self, tools, sample_df):
        """Test that invalid column names raise appropriate errors."""
        tools.dataframes["test_df"] = sample_df

        with pytest.raises(ValueError, match="not found in DataFrame"):
            tools.run_dataframe_operation(
                dataframe_name="test_df",
                operation="filter",
                operation_parameters={"items": "invalid_column,another_invalid"},
            )

    def test_single_column_select(self, tools, sample_df):
        """Test selecting a single column."""
        tools.dataframes["test_df"] = sample_df

        result = tools.run_dataframe_operation(
            dataframe_name="test_df",
            operation="__getitem__",
            operation_parameters={"columns": "canonical_smiles"},
        )

        # Single column returns a Series, which gets serialized to dict
        assert isinstance(result, dict)
        assert "sample" in result or "dataframe_name" in result
        # If it's a Series, it should have a sample
        if "sample" in result:
            assert result["length"] == 3
            assert result["name"] == "canonical_smiles"

    def test_describe_with_comma_separated_columns(self, tools, sample_df):
        """Test describe operation with comma-separated columns."""
        tools.dataframes["test_df"] = sample_df

        result = tools.run_dataframe_operation(
            dataframe_name="test_df",
            operation="describe",
            operation_parameters={"column": "standard_value"},
        )

        # describe returns a DataFrame summary
        assert isinstance(result, dict)

    def test_describe_with_include_all(self, tools, sample_df):
        """Test describe operation with include='all' parameter."""
        tools.dataframes["test_df"] = sample_df

        # This should not treat 'all' as a column name
        result = tools.run_dataframe_operation(
            dataframe_name="test_df", operation="describe", operation_parameters={"include": "all"}
        )

        # describe returns a DataFrame summary
        assert isinstance(result, dict)
        assert "dataframe_name" in result

    def test_unique_operation(self, tools, sample_df):
        """Test unique operation on a column."""
        tools.dataframes["test_df"] = sample_df

        result = tools.run_dataframe_operation(
            dataframe_name="test_df",
            operation="unique",
            operation_parameters={"column": "standard_type"},
        )

        assert sorted(result) == ["IC50", "Ki"]

    def test_value_counts_operation(self, tools, sample_df):
        """Test value_counts operation on a column."""
        tools.dataframes["test_df"] = sample_df

        result = tools.run_dataframe_operation(
            dataframe_name="test_df",
            operation="value_counts",
            operation_parameters={"column": "standard_type"},
        )

        assert isinstance(result, dict)
        assert result["IC50"] == 2
        assert result["Ki"] == 1
