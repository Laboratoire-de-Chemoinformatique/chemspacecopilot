"""Buffer for streaming events in MLflow tracking."""

from typing import Any, Dict, List, Optional


class StreamingBuffer:
    """Buffer for accumulating streaming events before logging to MLflow.

    This class buffers streaming chunks (text, tool calls, etc.) and provides
    aggregated metrics when streaming completes.
    """

    def __init__(self):
        """Initialize streaming buffer."""
        self.text_chunks: List[str] = []
        self.tool_calls: List[Dict[str, Any]] = []
        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None
        self.metadata: Dict[str, Any] = {}

    def add_text_chunk(self, chunk: str):
        """Add a text chunk to the buffer.

        Args:
            chunk: Text chunk from streaming response
        """
        self.text_chunks.append(chunk)

    def add_tool_call(self, tool_name: str, args: Dict[str, Any], result: Any = None):
        """Add a tool call to the buffer.

        Args:
            tool_name: Name of the tool
            args: Tool arguments
            result: Tool result (optional)
        """
        self.tool_calls.append({"tool": tool_name, "args": args, "result": result})

    def set_metadata(self, key: str, value: Any):
        """Set metadata value.

        Args:
            key: Metadata key
            value: Metadata value
        """
        self.metadata[key] = value

    def get_full_text(self) -> str:
        """Get concatenated full text from all chunks.

        Returns:
            Complete text
        """
        return "".join(self.text_chunks)

    def get_metrics(self) -> Dict[str, float]:
        """Get aggregated metrics from buffered data.

        Returns:
            Dictionary of metrics
        """
        metrics = {
            "text_chunks_count": float(len(self.text_chunks)),
            "tool_calls_count": float(len(self.tool_calls)),
            "total_text_length": float(len(self.get_full_text())),
        }

        if self.start_time and self.end_time:
            metrics["streaming_duration_seconds"] = self.end_time - self.start_time

        return metrics

    def get_params(self) -> Dict[str, Any]:
        """Get parameters from buffered data.

        Returns:
            Dictionary of parameters
        """
        params = {}
        if self.tool_calls:
            params["tools_used"] = ",".join({tc["tool"] for tc in self.tool_calls})
        return params

    def clear(self):
        """Clear all buffered data."""
        self.text_chunks.clear()
        self.tool_calls.clear()
        self.metadata.clear()
        self.start_time = None
        self.end_time = None
