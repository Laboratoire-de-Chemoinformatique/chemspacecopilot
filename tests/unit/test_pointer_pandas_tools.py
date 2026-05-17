#!/usr/bin/env python
"""Unit tests for PointerPandasTools column parsing fixes."""

import json

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

    def test_create_pandas_dataframe_accepts_json_string_parameters(
        self, tools, sample_df, tmp_path
    ):
        """Tool runtimes may pass object parameters as JSON strings."""
        csv_path = tmp_path / "activity.csv"
        sample_df.to_csv(csv_path, index=False)

        result = tools.create_pandas_dataframe(
            dataframe_name="loaded_activity",
            create_using_function="read_csv",
            function_parameters=json.dumps({"path_or_buf": str(csv_path)}),
        )

        assert result["dataframe_name"] == "loaded_activity"
        assert tools.dataframes["loaded_activity"].shape == sample_df.shape

    def test_run_dataframe_operation_accepts_json_string_parameters(self, tools, sample_df):
        """Operation parameters are parsed when the tool runtime stringifies them."""
        tools.dataframes["test_df"] = sample_df

        result = tools.run_dataframe_operation(
            dataframe_name="test_df",
            operation="select",
            operation_parameters=json.dumps({"columns": ["molecule_chembl_id", "standard_value"]}),
        )

        result_df = tools.dataframes[result["dataframe_name"]]
        assert list(result_df.columns) == ["molecule_chembl_id", "standard_value"]

    def test_run_dataframe_operation_accepts_function_parameters_alias(self, tools, sample_df):
        """Some model gateways use function_parameters instead of operation_parameters."""
        tools.dataframes["test_df"] = sample_df

        result = tools.run_dataframe_operation(
            dataframe_name="test_df",
            operation="sort_values",
            function_parameters=json.dumps({"by": "standard_value", "ascending": False}),
        )

        result_df = tools.dataframes[result["dataframe_name"]]
        assert result_df["standard_value"].tolist() == [300, 200, 100]

    def test_run_dataframe_operation_infers_sort_for_boolean_operation(self, tools, sample_df):
        """Recover from OpenRouter tool calls that put True in operation for sort args."""
        tools.dataframes["test_df"] = sample_df

        result = tools.run_dataframe_operation(
            dataframe_name="test_df",
            operation=True,
            function_parameters={"by": "standard_value", "ascending": False},
        )

        result_df = tools.dataframes[result["dataframe_name"]]
        assert result_df["standard_value"].tolist() == [300, 200, 100]

    def test_run_dataframe_operation_prefers_operation_parameters(self, tools, sample_df):
        """The canonical parameter name wins if both parameter aliases are present."""
        tools.dataframes["test_df"] = sample_df

        result = tools.run_dataframe_operation(
            dataframe_name="test_df",
            operation="select",
            operation_parameters={"columns": ["molecule_chembl_id"]},
            function_parameters={"columns": ["standard_value"]},
        )

        result_df = tools.dataframes[result["dataframe_name"]]
        assert list(result_df.columns) == ["molecule_chembl_id"]

    def test_load_dataframe_from_session_resolves_top_level_dataframe(self, tools):
        session_df = pd.DataFrame({"smi": ["CCO"], "activity_final": [7.0]})

        result = tools.load_dataframe_from_session(
            "loaded",
            "analysis_input",
            session_state={"analysis_input": session_df},
        )

        assert result["session_key"] == "analysis_input"
        assert tools.dataframes["loaded"].equals(session_df)

    def test_load_dataframe_from_session_resolves_dotted_csv_path(self, tools, tmp_path):
        csv_path = tmp_path / "landscape.csv"
        pd.DataFrame({"nodes": [1], "filtered_reg_density": [8.0]}).to_csv(csv_path, index=False)

        result = tools.load_dataframe_from_session(
            "landscape",
            "landscape_files.landscape_data_csv",
            session_state={"landscape_files": {"landscape_data_csv": str(csv_path)}},
        )

        assert result["session_key"] == "landscape_files.landscape_data_csv"
        assert tools.dataframes["landscape"]["filtered_reg_density"].tolist() == [8.0]

    def test_load_dataframe_from_session_resolves_container_primary_csv(self, tools, tmp_path):
        primary_path = tmp_path / "primary.csv"
        supplementary_path = tmp_path / "supplementary.csv"
        pd.DataFrame({"kind": ["primary"]}).to_csv(primary_path, index=False)
        pd.DataFrame({"kind": ["supplementary"]}).to_csv(supplementary_path, index=False)

        result = tools.load_dataframe_from_session(
            "resolved",
            "analysis_outputs",
            session_state={
                "analysis_outputs": {
                    "supplementary_data": [str(supplementary_path)],
                    "primary_data_csv": str(primary_path),
                }
            },
        )

        assert result["session_key"] == "analysis_outputs.primary_data_csv"
        assert tools.dataframes["resolved"]["kind"].tolist() == ["primary"]

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

    def test_normalize_for_analysis_detects_derived_smiles_column(self, tools):
        """Test normalize_for_analysis detects SMILES columns by containment."""
        tools.dataframes["derived_smiles_df"] = pd.DataFrame(
            {
                "standardized_smiles": ["CCO", "CCN"],
                "node_index": [1, 2],
                "standard_value": [10.0, 20.0],
                "standard_units": ["nM", "nM"],
            }
        )

        result = tools.normalize_for_analysis("derived_smiles_df")
        normalized = tools.dataframes[result["dataframe_name"]]

        assert result["columns_mapped"]["smiles"] == "standardized_smiles"
        assert result["columns_mapped"]["cluster_id"] == "node_index"
        assert result["columns_mapped"]["activity"] == "standard_value"
        assert result["activity_mapping"]["activity_column"] == "standard_value"
        assert result["final_activity_mapping"]["activity_column"] == "activity_final"
        assert normalized["smiles"].tolist() == ["CCO", "CCN"]
        assert "morgan_fingerprint" not in normalized.columns
        assert (
            result["clean_dataset_path"] == result["standardization_summary"]["clean_dataset_path"]
        )
        assert result["descriptor_parquet_path"].endswith(".parquet")

    def test_normalize_for_analysis_registers_user_dataset_activity_memory(self, tools):
        """User datasets get sparse activity memory without ChEMBL-specific fields."""
        tools.dataframes["user_activity_df"] = pd.DataFrame(
            {
                "canonical_smiles": ["CCO", "CCN"],
                "IC50_nM": [10.0, 100.0],
            }
        )
        session_state = {}

        result = tools.normalize_for_analysis("user_activity_df", session_state=session_state)

        memory = session_state["session_objects"]
        dataset = next(iter(memory["datasets"].values()))
        compounds = list(memory["compounds"].values())

        assert result["activity_mapping"]["activity_column"] == "IC50_nM"
        assert result["activity_mapping"]["activity_semantics"] == "lower_is_better"
        assert result["final_activity_mapping"]["activity_column"] == "activity_final"
        assert session_state["data_file_paths"]["dataset_path"] == result["clean_dataset_path"]
        assert session_state["data_file_paths"]["raw_dataset_path"] == result["raw_dataset_path"]
        assert (
            session_state["data_file_paths"]["descriptor_parquet_path"]
            == result["descriptor_parquet_path"]
        )
        assert dataset["activity_mapping"]["activity_column"] == "IC50_nM"
        assert dataset["clean_dataset_path"] == result["clean_dataset_path"]
        assert dataset["descriptor_parquet_path"] == result["descriptor_parquet_path"]
        assert compounds[0]["activity"]["endpoint"] == "IC50"
        assert compounds[0]["activity"]["score"] == 8.0
        assert "assay_chembl_id" not in compounds[0]

    def test_normalize_for_analysis_prefers_exact_smiles_column(self, tools):
        """Test normalize_for_analysis does not remap when exact smiles exists."""
        tools.dataframes["exact_smiles_df"] = pd.DataFrame(
            {
                "canonical_smiles": ["CCN"],
                "smiles": ["CCO"],
            }
        )

        result = tools.normalize_for_analysis("exact_smiles_df")
        normalized = tools.dataframes[result["dataframe_name"]]

        assert "smiles" not in result["columns_mapped"]
        assert normalized["smiles"].tolist() == ["CCO"]
