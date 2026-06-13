"""PointerPandas facade for MCP tool registration."""

from __future__ import annotations

import functools
from typing import Any


class PointerPandasFacade:
    """MCP-safe wrapper around PointerPandasTools with JSON-friendly schemas."""

    def __init__(self) -> None:
        from cs_copilot.tools.io.pointer_pandas_tools import PointerPandasTools

        self._toolkit = PointerPandasTools()

    def load_dataframe_from_session(
        self,
        dataframe_name: str,
        session_key: str,
        session_state: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Load a session DataFrame or CSV path into the pandas registry."""
        return dict(
            self._toolkit.load_dataframe_from_session(
                dataframe_name=dataframe_name,
                session_key=session_key,
                session_state=session_state,
            )
        )

    def create_dataframe(
        self,
        dataframe_name: str,
        create_using_function: str,
        function_parameters: Any | None = None,
    ) -> dict[str, Any]:
        """Create a DataFrame and store it in the pandas registry."""
        return dict(
            self._toolkit.create_pandas_dataframe(
                dataframe_name=dataframe_name,
                create_using_function=create_using_function,
                function_parameters=function_parameters,
            )
        )

    def run_operation(
        self,
        dataframe_name: str,
        operation: str,
        operation_parameters: Any | None = None,
        function_parameters: Any | None = None,
    ) -> Any:
        """Run a pandas operation against a registered DataFrame."""
        return self._toolkit.run_dataframe_operation(
            dataframe_name=dataframe_name,
            operation=operation,
            operation_parameters=operation_parameters,
            function_parameters=function_parameters,
        )

    def normalize_for_analysis(
        self,
        df_path: str,
        cluster_col: str | None = None,
        smiles_col: str | None = None,
        activity_col: str | None = None,
        agent: Any | None = None,
        session_state: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Normalize a DataFrame to the standard analysis format."""
        return dict(
            self._toolkit.normalize_for_analysis(
                df_path=df_path,
                cluster_col=cluster_col,
                smiles_col=smiles_col,
                activity_col=activity_col,
                agent=agent,
                session_state=session_state,
            )
        )


@functools.lru_cache(maxsize=1)
def pointer_pandas_facade() -> PointerPandasFacade:
    return PointerPandasFacade()
