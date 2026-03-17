#!/usr/bin/env python
# coding: utf-8
"""
Session toolkit for managing session state and uploaded files.

Provides tools for accessing and managing files uploaded by users during a chat session.
"""

from typing import Optional

from agno.agent import Agent
from agno.tools.toolkit import Toolkit


class SessionToolkit(Toolkit):
    """
    Toolkit for session state operations, particularly managing uploaded files.

    This toolkit provides access to files that users have uploaded via the Chainlit UI,
    which are automatically stored in the agent's session state.
    """

    def __init__(self):
        """Initialize the SessionToolkit."""
        super().__init__("session_operations")
        self.register(self.list_uploaded_files)
        self.register(self.get_uploaded_file_path)
        self.register(self.check_file_uploaded)

    def list_uploaded_files(self, agent: Optional[Agent] = None) -> str:
        """
        List all files that have been uploaded by the user in this session.

        Returns a formatted list of uploaded files with their S3 paths.
        Use this tool when you need to know what files are available to work with.

        Args:
            agent: Agent instance (automatically provided)

        Returns:
            String containing list of uploaded files or message if no files

        Example:
            >>> list_uploaded_files()
            "Uploaded files:
            - data.csv: s3://bucket/sessions/abc123/uploads/data.csv
            - molecules.sdf: s3://bucket/sessions/abc123/uploads/molecules.sdf"
        """
        if agent is None:
            return "Error: Agent not available"

        uploaded_files = agent.session_state.get("uploaded_files", {})

        if not uploaded_files:
            return "No files have been uploaded yet. The user can upload files using the file attachment button (📎) in the chat interface."

        file_info = []
        for filename, s3_path in uploaded_files.items():
            file_info.append(f"- {filename}: {s3_path}")

        count = len(uploaded_files)
        return f"Found {count} uploaded file{'s' if count != 1 else ''}:\n" + "\n".join(file_info)

    def get_uploaded_file_path(self, filename: str, agent: Optional[Agent] = None) -> str:
        """
        Get the S3 path for a specific uploaded file.

        Use this tool when you need to access a file that the user uploaded.
        Once you have the path, you can read the file using S3.open() or pandas tools.

        Args:
            filename: Name of the file (e.g., 'data.csv', 'molecules.sdf')
            agent: Agent instance (automatically provided)

        Returns:
            S3 path to the file or error message if not found

        Example:
            >>> get_uploaded_file_path("data.csv")
            "s3://bucket/sessions/abc123/uploads/data.csv"
        """
        if agent is None:
            return "Error: Agent not available"

        uploaded_files = agent.session_state.get("uploaded_files", {})

        if not uploaded_files:
            return "No files have been uploaded yet. Please ask the user to upload a file first."

        if filename not in uploaded_files:
            available = ", ".join(uploaded_files.keys())
            return f"File '{filename}' not found in uploaded files. Available files: {available}"

        return uploaded_files[filename]

    def check_file_uploaded(self, filename: str, agent: Optional[Agent] = None) -> str:
        """
        Check if a specific file has been uploaded.

        Use this tool to verify if a file exists before trying to process it.

        Args:
            filename: Name of the file to check
            agent: Agent instance (automatically provided)

        Returns:
            Message indicating whether the file exists and its path if it does

        Example:
            >>> check_file_uploaded("data.csv")
            "Yes, 'data.csv' is available at: s3://bucket/sessions/abc123/uploads/data.csv"
        """
        if agent is None:
            return "Error: Agent not available"

        uploaded_files = agent.session_state.get("uploaded_files", {})

        if not uploaded_files:
            return f"No files have been uploaded yet, so '{filename}' is not available."

        if filename in uploaded_files:
            s3_path = uploaded_files[filename]
            return f"Yes, '{filename}' is available at: {s3_path}"
        else:
            available = ", ".join(uploaded_files.keys())
            return f"No, '{filename}' has not been uploaded. Available files: {available}"
