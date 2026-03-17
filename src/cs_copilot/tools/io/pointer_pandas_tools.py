#!/usr/bin/env python
# coding: utf-8
"""
Enhanced PandasTools with pointer-based dataframe management and S3 support.
"""

import logging
from typing import Dict, Optional, Union
from uuid import uuid4

import pandas as pd
from agno.tools.pandas import PandasTools

from cs_copilot.storage import S3
from cs_copilot.tools.chemistry.standardize import standardize_smiles_column
from cs_copilot.tools.constants import MAX_COL_WIDTH, SAMPLE_COLS, SAMPLE_ROWS

logger = logging.getLogger(__name__)

_OPERATION_ALIASES = {
    "summary": "describe",
    "stats": "describe",
    "stat": "describe",
    "describe_dataframe": "describe",
    "describe_data": "describe",
    "head_rows": "head",
    "tail_rows": "tail",
    "preview": "head",
    "sort": "sort_values",
    "sort_by": "sort_values",
    "order_by": "sort_values",
    "save_csv": "to_csv",
    "write_csv": "to_csv",
    "export_csv": "to_csv",
    "concatenate": "concat",
    "select": "select",
    "select_columns": "select",
    "subset": "select",
    "drop_columns": "drop",
    "remove_columns": "drop",
    "drop_rows": "drop",
    "len": "_len",
    "length": "_len",
    "n_rows": "_len",
    "num_rows": "_len",
    "row_count": "_len",
    "count_rows": "_len",
    "t": "transpose",
    "unique_values": "unique",  # LLM confusion
    "get_unique": "unique",
}

_NUMERIC_ONLY_OPS = {"sum", "mean", "median", "std", "var"}
_SERIES_OPS = {
    "mean",
    "median",
    "min",
    "max",
    "sum",
    "std",
    "var",
    "count",
    "nunique",
    "mode",
    "unique",
    "value_counts",  # Value operations (work on single column)
}

_NULL_CHECK_OPS = {
    "isnull",
    "isna",
    "notna",
    "notnull",  # Null checking operations (can work on DataFrame or subset)
}


def _preview(df: pd.DataFrame) -> str:
    """Generate a preview string for a DataFrame."""
    head = df.iloc[:SAMPLE_ROWS, :SAMPLE_COLS]
    return (
        f"(preview rows={SAMPLE_ROWS}, cols={SAMPLE_COLS}, shape={df.shape})\n"
        + head.to_markdown(maxcolwidths=[MAX_COL_WIDTH])
    )


def _normalize_csv(params: dict) -> dict:
    """Map legacy aliases -> path_or_buf."""
    for old in ("path", "filepath_or_buffer", "file_path", "filename"):
        if old in params:
            params["path_or_buf"] = params.pop(old)
    return params


def _normalize_operation_name(operation: str) -> str:
    op = operation.strip()
    if op.endswith("()"):
        op = op[:-2]
    if op.startswith("df."):
        op = op[3:]
    if op.startswith("DataFrame."):
        op = op.split(".", 1)[1]
    return op


def _normalize_param_aliases(params: dict, canonical: str, aliases: tuple[str, ...]) -> None:
    if canonical in params:
        return
    for alias in aliases:
        if alias in params:
            params[canonical] = params.pop(alias)
            return


def _coerce_columns(value, param_name: str) -> list[str]:
    if value is None:
        raise ValueError(f"{param_name} parameter must be provided")
    if isinstance(value, str):
        # Strip surrounding whitespace first
        value = value.strip()

        # Handle string representation of lists like "['col1', 'col2']" FIRST
        # (before checking for commas, since these strings contain commas)
        if value.startswith("[") and value.endswith("]"):
            import ast
            import re

            try:
                # Try to parse as-is first
                parsed = ast.literal_eval(value)
                if isinstance(parsed, (list, tuple)):
                    return [str(item).strip() for item in parsed]
            except (ValueError, SyntaxError):
                # If parsing fails, try to clean up whitespace/newlines and retry
                try:
                    # Remove extra whitespace and newlines
                    cleaned = re.sub(r"\s+", " ", value)
                    parsed = ast.literal_eval(cleaned)
                    if isinstance(parsed, (list, tuple)):
                        return [str(item).strip() for item in parsed]
                except (ValueError, SyntaxError):
                    # Last resort: extract comma-separated values from within brackets
                    # Remove brackets and parse as comma-separated
                    inner = value[1:-1]
                    # Try to extract quoted strings
                    matches = re.findall(r"['\"]([^'\"]+)['\"]", inner)
                    if matches:
                        return [m.strip() for m in matches]
                    # Otherwise fall through to comma-separated parsing
        # Handle comma-separated strings (common LLM mistake)
        if "," in value:
            return [col.strip() for col in value.split(",") if col.strip()]
        return [value]
    if isinstance(value, (list, tuple)):
        return list(value)
    raise ValueError(f"{param_name} parameter must be a string or list of strings")


def _validate_columns(df: pd.DataFrame, columns: list[str], param_name: str) -> None:
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise ValueError(f"{param_name} {missing} not found in DataFrame")


def _serialize_series(series: pd.Series) -> Dict[str, Union[int, dict, str]]:
    if series.empty:
        return {"note": "empty series", "sample": {}, "length": 0}
    payload: Dict[str, Union[int, dict, str]] = {
        "name": str(series.name) if series.name is not None else "",
        "dtype": str(series.dtype),
    }
    if len(series) > SAMPLE_ROWS:
        payload["note"] = "sample only – full series omitted to save tokens"
        payload["sample"] = series.head(SAMPLE_ROWS).to_dict()
        payload["length"] = int(series.shape[0])
        return payload
    payload["sample"] = series.to_dict()
    payload["length"] = int(series.shape[0])
    return payload


def _resolve(obj, registry: dict):
    """Turn dataframe-name strings *or* {'dataframe_name': …} dicts into real DataFrames."""
    if isinstance(obj, str) and obj in registry:
        return registry[obj]
    if isinstance(obj, dict) and "dataframe_name" in obj:
        return registry[obj["dataframe_name"]]
    return obj


class PointerPandasTools(PandasTools):
    """
    Enhanced PandasTools with pointer-based dataframe management.

    Features:
    - S3/local file system abstraction
    - DataFrame pointer system to avoid context overflow
    - Support for concat operations with dataframe references
    - Automatic CSV path normalization
    """

    def create_pandas_dataframe(
        self,
        dataframe_name: str,
        create_using_function: str,
        function_parameters: Optional[Dict] = None,
    ) -> Dict[str, Union[str, pd.DataFrame]]:
        """Create a pandas DataFrame using various methods.

        Args:
            dataframe_name: Name to store the DataFrame under
            create_using_function: Method to use for creation. Options:
                - "read_csv": Load from CSV file (requires 'path_or_buf' in function_parameters)
                - "from_dict": Create from dictionary (requires 'data' in function_parameters)
                - "from_records": Create from records (requires 'data' in function_parameters)
                - "concat": Concatenate DataFrames (requires 'objs' list in function_parameters)
            function_parameters: Parameters for the creation method. Examples:
                - For read_csv: {"path_or_buf": "s3://bucket/file.csv"} or {"path_or_buf": "local.csv"}
                - For from_dict: {"data": {"col1": [1, 2], "col2": [3, 4]}}
                - For concat: {"objs": ["df1", "df2"]}

        Returns:
            Dictionary with dataframe name and preview

        Raises:
            ValueError: If required parameters are missing
            FileNotFoundError: If specified file doesn't exist

        Examples:
            # Load CSV file
            create_pandas_dataframe(
                dataframe_name="my_data",
                create_using_function="read_csv",
                function_parameters={"path_or_buf": "s3://bucket/data.csv"}
            )

            # Create from dict
            create_pandas_dataframe(
                dataframe_name="my_data",
                create_using_function="from_dict",
                function_parameters={"data": {"A": [1, 2], "B": [3, 4]}}
            )
        """
        if not dataframe_name:
            raise ValueError("dataframe_name cannot be empty")

        if not create_using_function:
            raise ValueError("create_using_function cannot be empty")

        function_parameters = function_parameters or {}

        # Normalize CSV params early for common cases
        if create_using_function in {"read_csv", "to_csv"}:
            function_parameters = _normalize_csv(function_parameters)

        if create_using_function == "concat":
            objs = function_parameters.get("objs", [])
            if not objs:
                raise ValueError("concat requires 'objs' parameter with list of DataFrames")
            try:
                objs = [_resolve(o, self.dataframes) for o in objs]
                function_parameters["objs"] = objs
            except Exception as e:
                logger.error(f"Error resolving dataframes for concat: {e}")
                raise
        elif create_using_function == "read_csv":
            # Ensure S3/local-safe CSV loading
            params = function_parameters.copy()
            path = params.pop("path_or_buf", None)

            # Also check for common alternative parameter names
            if path is None:
                path = params.pop("filepath_or_buffer", None)
            if path is None:
                path = params.pop("filepath", None)
            if path is None:
                path = params.pop("file_path", None)
            if path is None:
                path = params.pop("path", None)

            if path is None:
                raise ValueError(
                    "read_csv requires 'path_or_buf' parameter. "
                    "Example: function_parameters={'path_or_buf': 's3://bucket/file.csv'} "
                    "or function_parameters={'path_or_buf': 'local_file.csv'}. "
                    f"Received function_parameters: {function_parameters}"
                )
            try:
                with S3.open(path, "r") as fh:
                    df = pd.read_csv(fh, **params)
                self.dataframes[dataframe_name] = df
                logger.info(f"Successfully loaded CSV from {path} with shape {df.shape}")
                return {"dataframe_name": dataframe_name, "preview": _preview(df)}
            except Exception as e:
                logger.error(f"Error reading CSV from {path}: {e}")
                raise
        elif create_using_function == "from_s3":
            # Handle loading from S3 path
            s3_path = function_parameters.get("s3_path")
            if not s3_path:
                raise ValueError("s3_path parameter is required for from_s3 function")
            try:
                with S3.open(s3_path, "r") as f:
                    df = pd.read_csv(f)
                self.dataframes[dataframe_name] = df
                logger.info(f"Successfully loaded file from {s3_path} with shape {df.shape}")
                return {"dataframe_name": dataframe_name, "preview": _preview(df)}
            except Exception as e:
                logger.error(f"Error loading file from {s3_path}: {e}")
                raise
        elif create_using_function == "from_file" or dataframe_name.endswith(".csv"):
            # Handle direct file loading (S3 or local)
            file_path = function_parameters.get("file_path", dataframe_name)
            try:
                # Use S3.open which now supports s3://, local absolute, and relative
                with S3.open(file_path, "r") as f:
                    df = pd.read_csv(f)
                self.dataframes[dataframe_name] = df
                logger.info(f"Successfully loaded file from {file_path} with shape {df.shape}")
                return {"dataframe_name": dataframe_name, "preview": _preview(df)}
            except Exception as e:
                logger.error(f"Error loading file from {file_path}: {e}")
                raise

        # Map bare DataFrame class method names to full form
        # LLMs often use "from_dict" instead of "DataFrame.from_dict"
        DATAFRAME_CLASS_METHODS = {"from_dict", "from_records"}
        if create_using_function in DATAFRAME_CLASS_METHODS:
            create_using_function = f"DataFrame.{create_using_function}"

        # Handle DataFrame class methods like "DataFrame.from_dict", "DataFrame.from_records", etc.
        if "." in create_using_function:
            parts = create_using_function.split(".")
            if len(parts) == 2 and parts[0] == "DataFrame":
                class_method = parts[1]
                if hasattr(pd.DataFrame, class_method):
                    try:
                        logger.info(f"Creating DataFrame using DataFrame.{class_method}")
                        method = getattr(pd.DataFrame, class_method)
                        df = method(**function_parameters)
                        if not isinstance(df, pd.DataFrame):
                            raise ValueError(f"DataFrame.{class_method} did not return a DataFrame")
                        self.dataframes[dataframe_name] = df
                        logger.info(
                            f"Successfully created DataFrame using DataFrame.{class_method} with shape {df.shape}"
                        )
                        return {"dataframe_name": dataframe_name, "preview": _preview(df)}
                    except Exception as e:
                        logger.error(
                            f"Error creating DataFrame using DataFrame.{class_method}: {e}"
                        )
                        raise
                else:
                    error_msg = f"DataFrame has no method '{class_method}'. Valid methods include: from_dict, from_records, etc."
                    logger.error(error_msg)
                    raise AttributeError(error_msg)

        # Check if user is trying to use a DataFrame operation instead of creation function
        DATAFRAME_OPERATIONS = {
            "query",
            "filter",
            "head",
            "tail",
            "sample",
            "describe",
            "groupby",
            "sort_values",
            "drop_duplicates",
            "fillna",
            "dropna",
            "merge",
            "join",
            "select",
            "agg",
            "aggregate",
            "apply",
            "transform",
        }
        if create_using_function in DATAFRAME_OPERATIONS:
            raise AttributeError(
                f"'{create_using_function}' is a DataFrame operation, not a creation function. "
                f"Use run_dataframe_operation() instead of create_pandas_dataframe(). "
                f"Example: run_dataframe_operation(dataframe_name='existing_df', "
                f"operation='{create_using_function}', operation_parameters={{...}})"
            )

        # Validate that the function exists on pandas before calling parent
        if not hasattr(pd, create_using_function):
            error_msg = (
                f"pandas has no function '{create_using_function}'. "
                f"Valid pandas functions include: read_csv, read_json, read_excel, DataFrame, "
                f"from_dict, from_records, etc. "
                f"If you want to filter/query an existing DataFrame, use run_dataframe_operation() instead."
            )
            logger.error(error_msg)
            raise AttributeError(error_msg)

        return super().create_pandas_dataframe(
            dataframe_name=dataframe_name,
            create_using_function=create_using_function,
            function_parameters=function_parameters,
        )

    def run_dataframe_operation(
        self,
        dataframe_name: str,
        operation: str,
        operation_parameters: Optional[Dict] = None,
    ) -> Union[pd.DataFrame, pd.Series, Dict, str, float, int]:
        """Run operations on existing DataFrames.

        Args:
            dataframe_name: Name of the DataFrame to operate on
            operation: Operation to perform
            operation_parameters: Parameters for the operation

        Returns:
            Operation result (DataFrame, Series, scalar, etc.)

        Raises:
            KeyError: If dataframe not found
            ValueError: If operation parameters are invalid
        """
        if not dataframe_name:
            raise ValueError("dataframe_name cannot be empty")

        if not operation:
            raise ValueError("operation cannot be empty")

        operation = _normalize_operation_name(operation)
        operation = _OPERATION_ALIASES.get(operation.lower(), operation)
        params = (operation_parameters or {}).copy()

        # Fix legacy to_csv aliases
        if operation == "to_csv":
            params = _normalize_csv(params)
            # Intercept to route writes through S3/local abstraction
            df = self._get_or_load_dataframe(dataframe_name)
            path = params.pop("path_or_buf", None)
            if path is None:
                # no path provided → behave like pandas and return CSV string
                return df.to_csv(**params)
            try:
                # pandas decides newline/encoding; we just provide a text handle
                with S3.open(path, "w") as fh:
                    result = df.to_csv(fh, **params)
                logger.info(f"Successfully wrote CSV to {path}")
                return result
            except Exception as e:
                logger.error(f"Error writing CSV to {path}: {e}")
                raise

        # Resolve concat objs that arrive via run_dataframe_operation
        if operation == "concat":
            try:
                params["objs"] = [_resolve(o, self.dataframes) for o in params.get("objs", [])]
                result = pd.concat(**params)
            except Exception as e:
                logger.error(f"Error in concat operation: {e}")
                raise
        else:
            # Get or load the DataFrame
            df = self._get_or_load_dataframe(dataframe_name)
            result = None

            if operation == "_len":
                return int(len(df))

            # Intercept to_dict to avoid context blow-up
            if operation == "to_dict":
                orient = params.get("orient", "records")
                sample = df.head(SAMPLE_ROWS).to_dict(orient=orient)
                return {
                    "dataframe_name": dataframe_name,
                    "note": "sample only – full dict omitted to save tokens",
                    "sample": sample,
                }

            # Normalize common parameter names for rows/columns
            if operation in {"head", "tail", "sample"}:
                _normalize_param_aliases(
                    params, "n", ("rows", "n_rows", "num_rows", "count", "size")
                )
                # Coerce string numbers to int (LLM often passes "5" instead of 5)
                if "n" in params and isinstance(params["n"], str):
                    try:
                        params["n"] = int(params["n"])
                    except ValueError as e:
                        raise ValueError(
                            f"Parameter 'n' must be an integer, got string '{params['n']}' "
                            f"that cannot be converted to int"
                        ) from e
                if operation == "sample":
                    _normalize_param_aliases(params, "frac", ("fraction",))
                    # Coerce string floats to float
                    if "frac" in params and isinstance(params["frac"], str):
                        try:
                            params["frac"] = float(params["frac"])
                        except ValueError as e:
                            raise ValueError(
                                f"Parameter 'frac' must be a float, got string '{params['frac']}' "
                                f"that cannot be converted to float"
                            ) from e

            # Allow describe(column=...) to target a subset without passing column to pandas
            # Note: 'include' is NOT aliased - it's a valid pandas parameter for data type selection
            if operation == "describe":
                _normalize_param_aliases(params, "column", ("columns", "cols"))
            if operation == "describe" and "column" in params:
                columns = _coerce_columns(params.pop("column"), param_name="column")
                if not columns:
                    raise ValueError(
                        "describe() 'column' parameter must include at least one column"
                    )
                _validate_columns(df, columns, param_name="column")
                df = df[columns]

            # Handle unique() - it's a Series method, not DataFrame
            if operation == "unique":
                _normalize_param_aliases(params, "column", ("columns", "col"))
                column = params.get("column")
                if column is None:
                    raise ValueError("unique() operation requires a 'column' parameter")
                columns = _coerce_columns(column, param_name="column")
                if len(columns) != 1:
                    raise ValueError("unique() operation requires a single column")
                _validate_columns(df, columns, param_name="column")
                result = df[columns[0]].unique()
                # Convert numpy array to list for better serialization
                return result.tolist() if hasattr(result, "tolist") else list(result)

            # Handle value_counts() - it's a Series method, not DataFrame
            if operation == "value_counts":
                _normalize_param_aliases(params, "column", ("columns", "col", "subset"))
                column = params.get("column")
                if column is None:
                    raise ValueError(
                        "value_counts() operation requires a 'column' parameter. "
                        "Example: operation_parameters={'column': 'target_chembl_id'}"
                    )
                columns = _coerce_columns(column, param_name="column")
                if len(columns) != 1:
                    raise ValueError("value_counts() operation requires a single column")
                _validate_columns(df, columns, param_name="column")
                # Remove 'column' from params before passing to value_counts()
                params_without_column = {k: v for k, v in params.items() if k != "column"}
                result = df[columns[0]].value_counts(**params_without_column)
                # Convert Series to dict for better serialization
                return result.to_dict()

            # Handle groupby() - convert column_name to 'by' parameter
            if operation == "groupby":
                _normalize_param_aliases(
                    params, "by", ("column_name", "column", "columns", "group_by", "groupby")
                )
                by_value = params.get("by")
                if by_value is None:
                    raise ValueError("groupby() operation requires a 'by' parameter")
                by_columns = _coerce_columns(by_value, param_name="by")
                _validate_columns(df, by_columns, param_name="by")
                params["by"] = by_columns if len(by_columns) > 1 else by_columns[0]
                agg = None
                for key in ("agg", "aggregation", "agg_func"):
                    if key in params:
                        agg = params.pop(key)
                        break
                grouped = df.groupby(**params)
                if agg is not None:
                    result = grouped.agg(agg)
                else:
                    result = grouped.size()
                if isinstance(result, pd.Series):
                    return _serialize_series(result)

            # Normalize query() parameter names
            if operation == "query":
                _normalize_param_aliases(params, "expr", ("expression", "query", "where"))
                if "expr" not in params or params["expr"] in (None, ""):
                    raise ValueError("query() operation requires an 'expr' parameter")

            if operation == "sort_values":
                _normalize_param_aliases(params, "by", ("column", "columns", "sort_by", "order_by"))

            if operation == "drop":
                _normalize_param_aliases(params, "columns", ("column", "cols"))
                _normalize_param_aliases(params, "index", ("rows", "row", "indices", "row_index"))

            if operation in {"dropna", "drop_duplicates"}:
                _normalize_param_aliases(params, "subset", ("column", "columns", "cols"))

            if operation == "rename":
                _normalize_param_aliases(params, "columns", ("column_map", "mapping", "rename_map"))

            if operation == "fillna":
                _normalize_param_aliases(params, "value", ("fill_value", "default", "replace_with"))
                if "column" in params and "value" in params:
                    columns = _coerce_columns(params.pop("column"), param_name="column")
                    value = params.get("value")
                    if not isinstance(value, dict):
                        params["value"] = {c: value for c in columns}

            if operation == "select":
                columns = params.get("columns", params.get("column"))
                columns = _coerce_columns(columns, param_name="columns")
                _validate_columns(df, columns, param_name="columns")
                result = df[columns]

            # Handle filter() with items parameter (for filtering columns)
            if operation == "filter" and "items" in params:
                items = params.get("items")
                if items is not None:
                    items_list = _coerce_columns(items, param_name="items")
                    _validate_columns(df, items_list, param_name="items")
                    params["items"] = items_list

            # Handle __getitem__ for column selection (df[columns])
            if operation == "__getitem__":
                # __getitem__ expects a single positional argument, not keyword args
                if params:
                    # Try to extract column specification from various parameter names
                    key = params.get("key") or params.get("columns") or params.get("column")
                    if key is not None:
                        columns = _coerce_columns(key, param_name="key")
                        _validate_columns(df, columns, param_name="key")
                        # For single column, pass as string; for multiple, pass as list
                        result = df[columns[0] if len(columns) == 1 else columns]
                    else:
                        raise ValueError(
                            "__getitem__ operation requires a 'key', 'column', or 'columns' parameter. "
                            "Example: operation_parameters={'columns': ['col1', 'col2']} or "
                            "operation_parameters={'columns': 'col1,col2,col3'}"
                        )

            # Handle loc indexer (not a regular method)
            if operation == "loc":
                rows = params.get("rows") or params.get("row") or params.get("index", slice(None))
                cols = params.get("columns") or params.get("column") or params.get("cols")

                if cols is not None:
                    cols_list = _coerce_columns(cols, param_name="columns")
                    _validate_columns(df, cols_list, param_name="columns")
                    result = df.loc[rows, cols_list]
                else:
                    result = df.loc[rows]

            # Handle iloc indexer (not a regular method)
            if operation == "iloc":
                rows = params.get("rows") or params.get("row") or params.get("index", slice(None))
                cols = params.get("columns") or params.get("column") or params.get("cols")

                if cols is not None:
                    # For iloc, columns should be integer indices, not names
                    if isinstance(cols, str) and not cols.isdigit():
                        raise ValueError(
                            "iloc requires integer indices, not column names. "
                            "Use loc for column names or provide integer positions."
                        )
                    result = df.iloc[rows, cols]
                else:
                    result = df.iloc[rows]

            # Handle null-checking operations (can work on DataFrame subsets)
            if operation in _NULL_CHECK_OPS:
                _normalize_param_aliases(params, "column", ("columns", "cols", "subset"))
                if "column" in params:
                    columns = _coerce_columns(params.pop("column"), param_name="column")
                    _validate_columns(df, columns, param_name="column")
                    # Select subset of columns and apply operation
                    df_subset = df[columns]
                    result = getattr(df_subset, operation)(**params)
                else:
                    # No subset specified, apply to whole DataFrame
                    result = getattr(df, operation)(**params)

            if operation in _SERIES_OPS:
                _normalize_param_aliases(params, "column", ("columns", "col", "subset"))
            if operation in _SERIES_OPS and "column" in params:
                columns = _coerce_columns(params.pop("column"), param_name="column")
                if len(columns) != 1:
                    raise ValueError(f"{operation}() operation requires a single column")
                _validate_columns(df, columns, param_name="column")
                series = df[columns[0]]
                result = getattr(series, operation)(**params)
                return _serialize_series(result) if isinstance(result, pd.Series) else result

            # Special handling for aggregate/agg operations
            if operation in ("agg", "aggregate"):
                func = params.get("func")
                if func is None:
                    # LLM might pass it as a list of functions
                    if isinstance(params.get("operation_parameters"), list):
                        func = params.pop("operation_parameters")
                    elif "functions" in params:
                        func = params.pop("functions")
                    else:
                        raise ValueError(
                            f"'{operation}' requires 'func' parameter with aggregation functions. "
                            f"Examples: func='mean', func=['mean', 'std'], "
                            f"func={{'column': 'mean'}}, func={{'column': ['min', 'max']}}. "
                            f"Received params: {params}"
                        )

                # Ensure func is properly formatted
                if isinstance(func, str):
                    params["func"] = func
                elif isinstance(func, list):
                    # Convert list to dict format: all columns get all functions
                    params["func"] = func
                elif isinstance(func, dict):
                    params["func"] = func
                else:
                    raise ValueError(
                        f"'func' must be a string, list, or dict. Got: {type(func).__name__}"
                    )

            # Perform the operation
            if result is None:
                try:
                    attr = getattr(df, operation)
                except AttributeError as e:
                    raise AttributeError(f"DataFrame has no operation '{operation}'") from e

                # If it's callable (method), call it; otherwise just use the value.
                if callable(attr):
                    # Handle filter() with query parameter - should use query() instead
                    if operation == "filter" and any(
                        k in params for k in ("query", "expression", "expr", "condition", "where")
                    ):
                        _normalize_param_aliases(
                            params, "expr", ("query", "expression", "where", "condition")
                        )
                        query_expr = params.pop("expr")
                        if not query_expr:
                            raise ValueError(
                                "filter() with condition requires a query expression. "
                                "Example: operation_parameters={'condition': \"standard_type == 'IC50'\"}"
                            )
                        result = df.query(query_expr)
                    else:
                        if (
                            operation in _NUMERIC_ONLY_OPS
                            and "numeric_only" not in params
                            and "column" not in params
                        ):
                            params["numeric_only"] = True
                        try:
                            result = attr(**params)
                        except Exception as e:
                            logger.error(f"Error executing operation '{operation}': {e}")
                            raise
                else:
                    if params:
                        raise TypeError(
                            f"Operation '{operation}' is not callable; you supplied parameters {params}"
                        )
                    result = attr

                    # Add this check to handle Index objects properly
                    if isinstance(result, pd.Index):
                        # Convert Index to a more string-friendly format
                        result = result.tolist() if len(result) > 0 else []

        if result is None and params.get("inplace"):
            return {"dataframe_name": dataframe_name, "preview": _preview(df)}

        # Store new DataFrames; return pointer + preview
        if isinstance(result, pd.DataFrame):
            key = f"{dataframe_name}_{uuid4().hex[:6]}"
            self.dataframes[key] = result
            logger.debug(f"Stored new DataFrame '{key}' with shape {result.shape}")
            return {"dataframe_name": key, "preview": _preview(result)}

        if isinstance(result, pd.Series):
            return _serialize_series(result)

        return result  # scalars / Series / etc.

    def normalize_for_analysis(
        self,
        df_path: str,
        cluster_col: Optional[str] = None,
        smiles_col: Optional[str] = None,
        activity_col: Optional[str] = None,
    ) -> Dict[str, Union[str, int]]:
        """Normalize a DataFrame to standard analysis format.

        Standardizes column names for downstream chemoinformatics analysis:
        - SMILES column → 'smiles'
        - Cluster column → 'cluster_id' (from node_index, cluster, group, etc.)
        - Activity column → 'activity' (optional)

        Args:
            df_path: Path to the CSV file (S3 or local) or name of existing DataFrame
            cluster_col: Name of column to use as cluster_id. If None, auto-detects from:
                        ['node_index', 'cluster_id', 'cluster', 'group', 'class', 'label']
            smiles_col: Name of SMILES column. If None, auto-detects from:
                       ['smiles', 'SMILES', 'canonical_smiles', 'Smiles']
            activity_col: Name of activity column. If None, auto-detects from:
                         ['activity', 'pIC50', 'pKi', 'standard_value', 'value']

        Returns:
            Dictionary with:
            - dataframe_name: Name of the normalized DataFrame in registry
            - n_rows: Number of rows
            - n_clusters: Number of unique clusters (if cluster column found)
            - has_activity: Boolean indicating if activity column was found
            - columns_mapped: Dict showing original → normalized column mappings

        Examples:
            # Normalize GTM output
            normalize_for_analysis(
                df_path="s3://bucket/gtm_source_mols.csv",
                cluster_col="node_index"
            )

            # Auto-detect columns
            normalize_for_analysis(df_path="molecules.csv")
        """
        # Auto-detect column name patterns
        SMILES_PATTERNS = ["smiles", "SMILES", "canonical_smiles", "Smiles", "smi"]
        CLUSTER_PATTERNS = [
            "node_index",
            "cluster_id",
            "cluster",
            "group",
            "class",
            "label",
            "node",
        ]
        ACTIVITY_PATTERNS = [
            "activity",
            "pIC50",
            "pKi",
            "pEC50",
            "standard_value",
            "value",
            "potency",
        ]

        # Load the DataFrame
        df = self._get_or_load_dataframe(df_path)
        df = df.copy()  # Don't modify original
        columns_mapped = {}

        # Normalize SMILES column
        smiles_found = None
        if smiles_col and smiles_col in df.columns:
            smiles_found = smiles_col
        else:
            for pattern in SMILES_PATTERNS:
                if pattern in df.columns:
                    smiles_found = pattern
                    break

        if not smiles_found:
            raise ValueError(
                f"No SMILES column found. Expected one of {SMILES_PATTERNS}. "
                f"Available columns: {list(df.columns)}"
            )

        if smiles_found != "smiles":
            df = df.rename(columns={smiles_found: "smiles"})
            columns_mapped["smiles"] = smiles_found
            logger.info(f"Mapped '{smiles_found}' → 'smiles'")

        pre_std = len(df)
        df = standardize_smiles_column(df, "smiles")
        df = df.dropna(subset=["smiles"]).reset_index(drop=True)
        dropped = pre_std - len(df)
        if dropped:
            logger.info(f"Dropped {dropped} rows with unstandardizable SMILES")

        # Normalize cluster column (optional)
        cluster_found = None
        n_clusters = None
        if cluster_col and cluster_col in df.columns:
            cluster_found = cluster_col
        else:
            for pattern in CLUSTER_PATTERNS:
                if pattern in df.columns:
                    cluster_found = pattern
                    break

        if cluster_found:
            if cluster_found != "cluster_id":
                df = df.rename(columns={cluster_found: "cluster_id"})
                columns_mapped["cluster_id"] = cluster_found
                logger.info(f"Mapped '{cluster_found}' → 'cluster_id'")
            n_clusters = int(df["cluster_id"].nunique())

        # Normalize activity column (optional)
        activity_found = None
        has_activity = False
        if activity_col and activity_col in df.columns:
            activity_found = activity_col
        else:
            for pattern in ACTIVITY_PATTERNS:
                if pattern in df.columns:
                    activity_found = pattern
                    break

        if activity_found:
            if activity_found != "activity":
                df = df.rename(columns={activity_found: "activity"})
                columns_mapped["activity"] = activity_found
                logger.info(f"Mapped '{activity_found}' → 'activity'")
            has_activity = True

        # Store in registry with standardized name
        normalized_name = f"analysis_input_{uuid4().hex[:6]}"
        self.dataframes[normalized_name] = df
        logger.info(f"Normalized DataFrame stored as '{normalized_name}' with shape {df.shape}")

        result = {
            "dataframe_name": normalized_name,
            "n_rows": int(len(df)),
            "has_activity": has_activity,
            "columns_mapped": columns_mapped,
            "preview": _preview(df),
        }

        if n_clusters is not None:
            result["n_clusters"] = n_clusters

        return result

    def _get_or_load_dataframe(self, dataframe_name: str) -> pd.DataFrame:
        """Get DataFrame from registry or load from file if it's a CSV path.

        Args:
            dataframe_name: Name or path of DataFrame

        Returns:
            The requested DataFrame

        Raises:
            KeyError: If DataFrame not found and not a loadable path
        """
        # Check if it's already in the registry
        if dataframe_name in self.dataframes:
            return self.dataframes[dataframe_name]

        # If it looks like a CSV file, try to load it
        if dataframe_name.endswith((".csv", ".csv.gz", ".tsv", ".tab", ".txt")):
            try:
                sep = "\t" if dataframe_name.endswith((".tsv", ".tab")) else ","
                with S3.open(dataframe_name, "r") as f:
                    df = pd.read_csv(f, sep=sep)
                # Store for future use
                self.dataframes[dataframe_name] = df
                logger.info(f"Auto-loaded DataFrame from {dataframe_name} with shape {df.shape}")
                return df
            except Exception as e:
                raise KeyError(f"Could not load DataFrame from path '{dataframe_name}': {e}") from e

        # Not found and not a loadable path
        available = list(self.dataframes.keys())

        # Suggest similar names using Levenshtein distance
        import difflib

        close_matches = difflib.get_close_matches(dataframe_name, available, n=3, cutoff=0.6)

        error_msg = f"DataFrame '{dataframe_name}' not found. Available: {available}"
        if close_matches:
            error_msg += f"\n\nDid you mean one of these? {close_matches}"

        raise KeyError(error_msg)
