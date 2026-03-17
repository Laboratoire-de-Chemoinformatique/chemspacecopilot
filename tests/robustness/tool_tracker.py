#!/usr/bin/env python
# coding: utf-8
"""
Tool sequence tracking and comparison for robustness testing.

This module provides utilities to track and compare sequences of tool calls
across different prompt variations, helping assess whether agents follow
similar execution paths despite prompt variation.
"""

from __future__ import annotations

import logging
from difflib import SequenceMatcher
from typing import Any, Dict, List

import numpy as np

logger = logging.getLogger(__name__)


class ToolSequenceComparator:
    """Compare sequences of tool calls across agent runs."""

    @staticmethod
    def extract_tool_sequence(agent_response: Any) -> List[str]:
        """
        Extract tool call sequence from agent response.

        Supports multiple response formats:
        - Agno agent responses with messages attribute
        - Response objects with tool_calls
        - Session state with tool call history

        Args:
            agent_response: Agent response object or session state

        Returns:
            List of tool names in order of execution
        """
        sequence = []

        # Try different response formats
        try:
            # Format 1: Agno agent response with messages
            if hasattr(agent_response, "messages"):
                for msg in agent_response.messages:
                    if hasattr(msg, "tool_calls") and msg.tool_calls:
                        for tc in msg.tool_calls:
                            # Extract tool name from different structures
                            if hasattr(tc, "function") and hasattr(tc.function, "name"):
                                sequence.append(tc.function.name)
                            elif hasattr(tc, "name"):
                                sequence.append(tc.name)
                            elif isinstance(tc, dict) and "function" in tc:
                                sequence.append(tc["function"].get("name", "unknown"))

            # Format 2: Direct tool_calls attribute
            elif hasattr(agent_response, "tool_calls"):
                for tc in agent_response.tool_calls:
                    if hasattr(tc, "function") and hasattr(tc.function, "name"):
                        sequence.append(tc.function.name)
                    elif hasattr(tc, "name"):
                        sequence.append(tc.name)

            # Format 3: Session state with tool history
            elif isinstance(agent_response, dict):
                if "tool_calls" in agent_response:
                    tool_calls = agent_response["tool_calls"]
                    if isinstance(tool_calls, list):
                        for tc in tool_calls:
                            if isinstance(tc, dict):
                                sequence.append(tc.get("name", "unknown"))
                            elif isinstance(tc, str):
                                sequence.append(tc)

                # Check for agent_messages in session state
                if "agent_messages" in agent_response:
                    for msg in agent_response["agent_messages"]:
                        if isinstance(msg, dict) and "tool_calls" in msg:
                            for tc in msg["tool_calls"]:
                                if isinstance(tc, dict):
                                    sequence.append(tc.get("name", "unknown"))

        except Exception as e:
            logger.debug(f"Failed to extract tool sequence: {e}")

        logger.debug(f"Extracted tool sequence: {sequence}")
        return sequence

    @classmethod
    def compare_sequences(cls, sequences: List[List[str]]) -> float:
        """
        Compare multiple tool call sequences using edit distance.

        Computes pairwise similarity between all sequences and returns
        the average. Similarity is based on the longest common subsequence
        ratio (similar to difflib.SequenceMatcher).

        Args:
            sequences: List of tool call sequences to compare

        Returns:
            Average pairwise similarity (0.0 to 1.0)
        """
        if len(sequences) < 2:
            return 1.0  # Perfect similarity if only one sequence

        # Remove empty sequences
        non_empty = [seq for seq in sequences if seq]
        if len(non_empty) < 2:
            # If all sequences are empty, they're perfectly similar
            # If only one is non-empty, similarity is low
            return 1.0 if len(non_empty) == 0 else 0.5

        similarities = []
        for i in range(len(non_empty)):
            for j in range(i + 1, len(non_empty)):
                matcher = SequenceMatcher(None, non_empty[i], non_empty[j])
                similarity = matcher.ratio()
                similarities.append(similarity)

        avg_similarity = np.mean(similarities) if similarities else 0.0
        logger.info(f"Tool sequence similarity: {avg_similarity:.3f}")
        return avg_similarity

    @classmethod
    def analyze_sequence_patterns(cls, sequences: List[List[str]]) -> Dict[str, Any]:
        """
        Analyze patterns in tool call sequences.

        Provides detailed metrics about tool usage:
        - Most common tools
        - Sequence length statistics
        - Tool transition patterns (which tools follow which)

        Args:
            sequences: List of tool call sequences

        Returns:
            Dictionary with analysis results
        """
        if not sequences:
            return {"error": "No sequences provided"}

        # Remove empty sequences for analysis
        non_empty = [seq for seq in sequences if seq]
        if not non_empty:
            return {"error": "All sequences are empty"}

        # 1. Tool frequency analysis
        tool_counts = {}
        for seq in non_empty:
            for tool in seq:
                tool_counts[tool] = tool_counts.get(tool, 0) + 1

        most_common_tools = sorted(tool_counts.items(), key=lambda x: x[1], reverse=True)

        # 2. Sequence length statistics
        lengths = [len(seq) for seq in non_empty]
        length_stats = {
            "mean": np.mean(lengths),
            "std": np.std(lengths),
            "min": min(lengths),
            "max": max(lengths),
        }

        # 3. Tool transition patterns (bigrams)
        transitions = {}
        for seq in non_empty:
            for i in range(len(seq) - 1):
                current_tool = seq[i]
                next_tool = seq[i + 1]
                transition_key = f"{current_tool} → {next_tool}"
                transitions[transition_key] = transitions.get(transition_key, 0) + 1

        most_common_transitions = sorted(transitions.items(), key=lambda x: x[1], reverse=True)[:10]

        # 4. Unique sequence count
        unique_sequences = len({tuple(seq) for seq in non_empty})

        return {
            "total_sequences": len(sequences),
            "non_empty_sequences": len(non_empty),
            "unique_sequences": unique_sequences,
            "tool_usage": dict(most_common_tools),
            "most_common_tools": most_common_tools[:10],
            "sequence_length_stats": length_stats,
            "tool_transitions": dict(transitions),
            "most_common_transitions": most_common_transitions,
        }

    @classmethod
    def find_common_subsequence(cls, sequences: List[List[str]]) -> List[str]:
        """
        Find longest common subsequence across all tool sequences.

        Identifies the longest sequence of tool calls that appears in all runs.
        Useful for understanding the "core workflow" that's robust to prompt variation.

        Args:
            sequences: List of tool call sequences

        Returns:
            Longest common subsequence as list of tool names
        """
        if len(sequences) < 2:
            return sequences[0] if sequences else []

        # Remove empty sequences
        non_empty = [seq for seq in sequences if seq]
        if len(non_empty) < 2:
            return non_empty[0] if non_empty else []

        # Start with first sequence and iteratively find common with others
        common = non_empty[0]
        for seq in non_empty[1:]:
            matcher = SequenceMatcher(None, common, seq)
            # Get longest matching block
            match = matcher.find_longest_match(0, len(common), 0, len(seq))
            if match.size > 0:
                common = common[match.a : match.a + match.size]
            else:
                common = []
                break

        logger.info(f"Common tool subsequence: {common}")
        return common


def compare_tool_usage_across_runs(outputs: List[Dict]) -> Dict[str, Any]:
    """
    High-level function to compare tool usage across multiple runs.

    Args:
        outputs: List of output dictionaries from robustness test runs

    Returns:
        Dictionary with tool sequence comparison metrics
    """
    # Extract sequences from each output
    sequences = []
    for output in outputs:
        # Try to get sequence from response object
        response = output.get("response_object") or output.get("agent_response")
        if response:
            seq = ToolSequenceComparator.extract_tool_sequence(response)
            sequences.append(seq)
        else:
            # Fallback: empty sequence if no response object
            sequences.append([])

    # Calculate similarity
    similarity = ToolSequenceComparator.compare_sequences(sequences)

    # Analyze patterns
    patterns = ToolSequenceComparator.analyze_sequence_patterns(sequences)

    # Find common workflow
    common_workflow = ToolSequenceComparator.find_common_subsequence(sequences)

    return {
        "tool_sequence_similarity": similarity,
        "sequences": sequences,
        "patterns": patterns,
        "common_workflow": common_workflow,
    }
