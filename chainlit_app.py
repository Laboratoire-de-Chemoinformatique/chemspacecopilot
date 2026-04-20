#!/usr/bin/env python
# coding: utf-8
"""
Chainlit application for Cs_copilot - AI-powered chemical data analysis.
"""

import asyncio
import base64
import hashlib
import json
import logging
import mimetypes
import os
import re
from pathlib import Path

import chainlit as cl
from chainlit.input_widget import Switch, Select
from chainlit.types import ThreadDict
from dotenv import load_dotenv

from cs_copilot.agents.teams import get_cs_copilot_agent_team
from cs_copilot.model_config import _is_retriable, arun_with_retry, load_model_from_config
from cs_copilot.storage import S3
from cs_copilot.tools.io.formatting import smiles_to_png_bytes

load_dotenv()

# Ensure the data/ directory exists (used by Agno for its SQLite session DB).
# The Dockerfile creates /app/data; this handles local runs outside Docker.
Path("data").mkdir(exist_ok=True)

# Set up logger
logger = logging.getLogger(__name__)

# ---------- User Management System ----------------------------------------- #
# Simple in-memory user storage (in production, use a proper database)
USERS = {
    "admin": {"password_hash": hashlib.sha256("admin123".encode()).hexdigest(), "role": "admin"},
}


def verify_password(username: str, password: str) -> bool:
    """Verify user credentials against stored users."""
    if username not in USERS:
        return False

    password_hash = hashlib.sha256(password.encode()).hexdigest()
    return USERS[username]["password_hash"] == password_hash


def get_user_role(username: str) -> str:
    """Get user role for authorization."""
    return USERS.get(username, {}).get("role", "guest")


# ---------- Session map settings helper ------------------------------------ #
def _apply_map_settings(session_agent, map_choice: str) -> None:
    """Propagate the selected map to the team's session_state.

    When the user picks the Universal Map, molecular descriptors default to
    autoencoder embeddings (compatible with the HuggingFace GTM model).
    Otherwise, the project keeps its historical Morgan-fingerprint default.
    """
    if session_agent is None:
        return

    if getattr(session_agent, "session_state", None) is None:
        session_agent.session_state = {}

    map_type = map_choice if map_choice in ("new_map", "universal_map") else "new_map"
    session_agent.session_state["map_type"] = map_type
    session_agent.session_state["default_descriptor"] = (
        "autoencoder" if map_type == "universal_map" else "morgan"
    )


# ---------- Authentication Callback ---------------------------------------- #
@cl.password_auth_callback
async def auth_callback(username: str, password: str) -> cl.User | None:
    """Authenticate users based on username and password."""
    if verify_password(username, password):
        return cl.User(
            identifier=username,
            display_name=username.title(),
            metadata={"role": get_user_role(username), "username": username},
        )
    else:
        return None


# ❶ Instantiate the LLM (configured via .modelconf or MODEL_PROVIDER env var)
model = load_model_from_config()

# ❷ Define the agent factory (per chat thread)


# ---------- Chat lifecycle --------------------------------------------------- #
@cl.on_chat_start
async def on_chat_start():
    """Create a fresh agent for this chat thread and stash in session."""
    # Synchronize S3 session with Chainlit thread ID
    thread_id = cl.context.session.thread_id
    if thread_id:
        # Update S3 prefix to match Chainlit session
        S3.prefix = f"sessions/{thread_id}"
        logger.info(f"Set S3 session prefix to: {S3.prefix}")

    # Initialize session state for this chat thread
    session_agent = get_cs_copilot_agent_team(
        model,
        show_members_responses=False,
    )
    cl.user_session.set("agent", session_agent)
    cl.user_session.set("title_set", False)
    cl.user_session.set("session_initialized", True)

    # Initialize ChatSettings with tool call toggle
    settings = await cl.ChatSettings(
        [
            Switch(
                id="show_tool_calls",
                label="Show Tool Calls",
                initial=True,
            ),
            Select(
                id="map",
                label="Map for Chemography",
                # values=["New map in this session", "Universal Map"],
                items={
                    "New map in this session": "new_map",
                    "Universal Map": "universal_map",
                    },
                initial_value="new_map",
            )
        ]
    ).send()
    cl.user_session.set("show_tool_calls", settings["show_tool_calls"])
    cl.user_session.set("map", settings["map"])
    _apply_map_settings(session_agent, settings["map"])

@cl.on_chat_resume
async def on_chat_resume(thread: ThreadDict):
    """Resume existing chat session or create new agent if needed."""
    # Synchronize S3 session with Chainlit thread ID
    thread_id = cl.context.session.thread_id
    if thread_id:
        # Update S3 prefix to match Chainlit session
        S3.prefix = f"sessions/{thread_id}"
        logger.info(f"Resumed S3 session prefix: {S3.prefix}")

    # Only create a new agent if none exists and session wasn't properly initialized
    if not cl.user_session.get("session_initialized") or cl.user_session.get("agent") is None:
        session_agent = get_cs_copilot_agent_team(
            model,
            show_members_responses=False,
        )
        cl.user_session.set("agent", session_agent)
        cl.user_session.set("session_initialized", True)

    session_agent = cl.user_session.get("agent")

    # Restore session state from Agno's persisted DB so that map settings and
    # uploaded_files are recovered without the agent having to run first.
    restored_map = "new_map"
    if thread_id and session_agent is not None:
        try:
            team_session = session_agent.get_session(session_id=thread_id)
            if team_session and team_session.session_data:
                saved_state = team_session.session_data.get("session_state", {})
                if saved_state:
                    restored_map = saved_state.get("map_type", "new_map")
                    # Seed the in-memory session_state from the DB so the agent
                    # doesn't lose uploaded_files or other state before the first arun.
                    session_agent.session_state = saved_state.copy()
                    logger.info(
                        f"Restored Agno session state for thread {thread_id}: "
                        f"map={restored_map}, "
                        f"files={list(saved_state.get('uploaded_files', {}).keys())}"
                    )
        except Exception as e:
            logger.warning(f"Could not restore Agno session state for thread {thread_id}: {e}")

    # Restore ChatSettings on resume using the persisted map selection.
    settings = await cl.ChatSettings(
        [
            Switch(
                id="show_tool_calls",
                label="Show Tool Calls",
                initial=True,
            ),
            Select(
                id="map",
                label="Map for Chemography",
                items={
                    "New map in this session": "new_map",
                    "Universal Map": "universal_map",
                },
                initial_value=restored_map,
            )
        ]
    ).send()
    cl.user_session.set("show_tool_calls", settings["show_tool_calls"])
    cl.user_session.set("map", settings["map"])
    _apply_map_settings(session_agent, settings["map"])


@cl.on_settings_update
async def on_settings_update(settings):
    """Handle settings updates from the UI."""
    cl.user_session.set("show_tool_calls", settings["show_tool_calls"])
    cl.user_session.set("map", settings["map"])
    _apply_map_settings(cl.user_session.get("agent"), settings["map"])

# Note: on_chat_end can cause issues with some Chainlit versions
# Session cleanup is handled automatically by Chainlit
@cl.on_chat_end
async def on_chat_end():
    """Save the messages history"""
    session_agent = cl.user_session.get("agent")
    if session_agent is not None:
        try:
            # Get thread_id for session identification
            thread_id = cl.context.session.thread_id if hasattr(cl.context, "session") else None

            # Try to get messages from the Team's database if available
            # Note: Team doesn't have get_session_messages(), messages are stored in the database
            if hasattr(session_agent, "db") and session_agent.db is not None and thread_id:
                # Messages are stored in the database, but accessing them requires
                # the Agno database API which may not be available at chat end
                # Chainlit already handles message persistence, so we skip manual saving
                logger.debug(f"Chat ended for thread {thread_id}, messages persisted by Chainlit")
            else:
                logger.debug("No database available or thread_id missing, skipping message save")
        except Exception as e:
            # Gracefully handle any errors since this is optional functionality
            logger.debug(f"Could not save messages on chat end: {e}")


# ---------- regex helpers --------------------------------------------------- #
PATH_RX = re.compile(
    r"^\s*(.*?)\s*[:\-]\s*(/[^ \t]+?\.(?:png|jpe?g|gif|svg))\s*$", re.I
)  # Caption: /path/file.png
SMI_RX = re.compile(r"`?<smiles>([^<]+)</smiles>`?")  # explicit SMILES tags
INLINE_ELEMENT_RX = re.compile(
    r"!\[([^\]]*)\]\(([^)]+)\)|<file>(.*?)</file>",
    re.I,
)

FILE_DOWNLOAD_MODE = os.getenv("CHAINLIT_FILE_DOWNLOAD_MODE", "local").strip().lower()
MAX_INLINE_FILE_BYTES = int(os.getenv("CHAINLIT_FILE_INLINE_MAX_BYTES", str(10 * 1024 * 1024)))


# ---------- utilities ------------------------------------------------------- #
def _pretty(x):
    try:
        return json.dumps(x, ensure_ascii=False)
    except Exception:
        return str(x)


def _process_smiles_in_text(text: str, callback):
    """
    Process SMILES patterns in text and call callback for each part.

    Args:
        text: Text to process
        callback: Function called with (text_part, is_smiles, smiles_string)
                 where is_smiles is True for SMILES tokens and False for regular text
    """
    pos = 0
    for m in SMI_RX.finditer(text):
        smi = m.group(1)

        # Add text before SMILES
        if m.start() > pos:
            callback(text[pos : m.start()], False, None)

        # Add SMILES token
        callback(f"`{smi}`", True, smi)

        pos = m.end()

    # Add remaining text
    if pos < len(text):
        callback(text[pos:], False, None)


async def _image_bubble(caption: str, src: str) -> cl.Message:
    """Generic local/remote image → cl.Image bubble → new assistant msg."""
    logger.debug(f"_image_bubble called with caption='{caption}', src='{src}'")
    p = Path(src).expanduser()
    name = caption or p.name

    def _to_data_url(data: bytes, filename: str) -> str:
        mt = mimetypes.guess_type(filename)[0] or "image/png"
        b64 = base64.b64encode(data).decode()
        data_size = len(data)
        logger.debug(f"Converted {data_size} bytes to data URL (mime: {mt})")
        return f"data:{mt};base64,{b64}"

    # Prefer local file → data URL; HTTP(S)/data: → pass-through; otherwise try S3 → data URL
    if p.is_file():
        logger.info(f"Loading image from local file: {p}")
        file_data = p.read_bytes()
        logger.debug(f"Read {len(file_data)} bytes from local file")
        data_url = _to_data_url(file_data, p.name)
        img_el = cl.Image(url=data_url, name=name, display="inline")
        logger.debug(f"Created cl.Image element from local file: {name}")
    elif isinstance(src, str) and (
        src.startswith("http://") or src.startswith("https://") or src.startswith("data:")
    ):
        logger.info(f"Using image from URL/data URL: {src[:100]}...")
        img_el = cl.Image(url=src, name=name, display="inline")
        logger.debug(f"Created cl.Image element from URL: {name}")
    else:
        try:
            logger.info(f"Attempting to load image from S3: {src}")
            with S3.open(src, "rb") as fh:
                data = fh.read()
            logger.debug(f"Read {len(data)} bytes from S3")
            data_url = _to_data_url(data, name)
            img_el = cl.Image(url=data_url, name=name, display="inline")
            logger.debug(f"Created cl.Image element from S3: {name}")
        except Exception as e:
            logger.warning(f"Error loading from S3, falling back to URL: {type(e).__name__}: {e}")
            # Fallback: let client try to fetch as URL (e.g. if it's a presigned S3 HTTP URL)
            img_el = cl.Image(url=src, name=name, display="inline")
            logger.debug(f"Created cl.Image element with fallback URL: {name}")

    logger.info(f"Sending image message with caption='{caption or img_el.name}'")
    await cl.Message(content=f"{caption or img_el.name}", elements=[img_el]).send()
    logger.debug("Image message sent successfully")


async def _create_streaming_message() -> cl.Message:
    """Create a message for streaming - only send when we have content"""
    msg = cl.Message(content="", author="assistant")
    return msg


async def _finalize_message(msg: cl.Message | None) -> None:
    """Persist the accumulated streaming content to the database.

    ``stream_token()`` accumulates content in memory and pushes it to the
    client via websocket, but never writes it back to the DB.  Calling
    ``update()`` after streaming ends ensures the final text is persisted so
    that resumed sessions display the assistant's messages.
    """
    if msg is None:
        return
    # Only update messages that were actually sent and have content.
    if getattr(msg, "id", None) and msg.content:
        await msg.update()


async def _stream_text_to_message(text: str, msg: cl.Message):
    """Stream text to message, handling SMILES with cl.Image when interrupted"""
    if not text:
        return

    # Send message if not sent yet
    if not hasattr(msg, "_sent") or not msg._sent:
        await msg.send()

    # Process text with SMILES
    pos = 0
    for m in SMI_RX.finditer(text):
        smi = m.group(1)
        logger.debug(f"Detected SMILES in stream: '{smi}'")

        # Stream text up to SMILES
        if m.start() > pos:
            await msg.stream_token(text[pos : m.start()])

        # Stream SMILES token
        await msg.stream_token(f"`{smi}`")
        logger.debug(f"Streamed SMILES token: `{smi}`")

        # Try to create molecule image and send as cl.Image
        try:
            logger.info(f"Attempting to convert SMILES to PNG: '{smi}'")
            png = smiles_to_png_bytes(smi)
            if png is not None:
                png_size = len(png)
                logger.info(f"Successfully generated PNG from SMILES '{smi}' ({png_size} bytes)")
                b64 = base64.b64encode(png).decode()
                data_url = f"data:image/png;base64,{b64}"
                logger.debug(f"Created data URL from PNG (size: {len(b64)} chars)")
                img_el = cl.Image(url=data_url, name=smi, display="inline")
                logger.debug(f"Created cl.Image element for SMILES: '{smi}'")
                await _finalize_message(msg)
                await cl.Message(content=f"`{smi}`", elements=[img_el]).send()
                logger.info(f"Sent SMILES image message for: '{smi}'")
                # Return new streaming message for continuation
                return await _create_streaming_message()
            else:
                logger.info(f"smiles_to_png_bytes returned None for SMILES: '{smi}'")
        except ValueError as ve:
            logger.info(f"ValueError converting SMILES '{smi}' to image: {ve}")
            # Invalid SMILES, just continue without image
        except Exception as e:
            logger.info(f"Exception converting SMILES '{smi}' to image: {type(e).__name__}: {e}")
            # Invalid SMILES, just continue without image

        pos = m.end()

    # Stream remaining text
    if pos < len(text):
        await msg.stream_token(text[pos:])


async def _image_bubble_streaming(caption: str, src: str) -> cl.Message:
    """Send image bubble and return new streaming message"""
    logger.debug(f"_image_bubble_streaming called with caption='{caption}', src='{src}'")
    p = Path(src).expanduser()
    name = caption or p.name

    def _to_data_url(data: bytes, filename: str) -> str:
        mt = mimetypes.guess_type(filename)[0] or "image/png"
        b64 = base64.b64encode(data).decode()
        data_size = len(data)
        logger.debug(f"Converted {data_size} bytes to data URL (mime: {mt})")
        return f"data:{mt};base64,{b64}"

    # Prefer local file → data URL; HTTP(S)/data: → pass-through; otherwise try S3 → data URL
    if p.is_file():
        logger.info(f"Loading image from local file (streaming): {p}")
        file_data = p.read_bytes()
        logger.debug(f"Read {len(file_data)} bytes from local file")
        data_url = _to_data_url(file_data, p.name)
        img_el = cl.Image(url=data_url, name=name, display="inline")
        logger.debug(f"Created cl.Image element from local file (streaming): {name}")
    elif isinstance(src, str) and (
        src.startswith("http://") or src.startswith("https://") or src.startswith("data:")
    ):
        logger.info(f"Using image from URL/data URL (streaming): {src[:100]}...")
        img_el = cl.Image(url=src, name=name, display="inline")
        logger.debug(f"Created cl.Image element from URL (streaming): {name}")
    else:
        try:
            logger.info(f"Attempting to load image from S3 (streaming): {src}")
            with S3.open(src, "rb") as fh:
                data = fh.read()
            logger.debug(f"Read {len(data)} bytes from S3")
            data_url = _to_data_url(data, name)
            img_el = cl.Image(url=data_url, name=name, display="inline")
            logger.debug(f"Created cl.Image element from S3 (streaming): {name}")
        except Exception as e:
            logger.warning(f"Error loading from S3 (streaming), falling back to URL: {type(e).__name__}: {e}")
            # Fallback: let client try to fetch as URL (e.g. if it's a presigned S3 HTTP URL)
            img_el = cl.Image(url=src, name=name, display="inline")
            logger.debug(f"Created cl.Image element with fallback URL (streaming): {name}")

    logger.info(f"Sending image message (streaming) with caption='{caption or img_el.name}'")
    await cl.Message(content=f"{caption or img_el.name}", elements=[img_el]).send()
    logger.debug("Image message sent successfully (streaming)")
    return await _create_streaming_message()


def _is_web_url(path: str) -> bool:
    return path.startswith("http://") or path.startswith("https://")


def _guess_file_name(path: str) -> str:
    cleaned = path.strip().strip("`").strip("\"").strip("'")
    without_query = cleaned.split("?", 1)[0]
    name = Path(without_query).name
    if not name:
        digest = hashlib.sha256(cleaned.encode("utf-8")).hexdigest()[:10]
        return f"download_{digest}"
    return name


def _safe_file_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]", "_", name)


def _read_file_bytes_from_storage(file_ref: str) -> bytes:
    with S3.open(file_ref, "rb") as fh:
        return fh.read()


async def _materialize_file_to_local(file_ref: str, file_name: str) -> Path:
    cache = cl.user_session.get("downloadable_files_cache") or {}
    cached = cache.get(file_ref)
    if cached:
        cached_path = Path(cached)
        if cached_path.exists():
            return cached_path

    downloads_dir = Path(".files") / "downloads"
    downloads_dir.mkdir(parents=True, exist_ok=True)

    digest = hashlib.sha256(file_ref.encode("utf-8")).hexdigest()[:12]
    local_path = downloads_dir / f"{digest}_{_safe_file_name(file_name)}"

    if not local_path.exists():
        data = await asyncio.to_thread(_read_file_bytes_from_storage, file_ref)
        await asyncio.to_thread(local_path.write_bytes, data)
        logger.info("Downloaded file from storage to local cache: %s", local_path)

    cache[file_ref] = str(local_path)
    cl.user_session.set("downloadable_files_cache", cache)
    return local_path


async def _build_download_file_element(file_ref: str) -> cl.File:
    file_name = _guess_file_name(file_ref)

    if _is_web_url(file_ref):
        return cl.File(name=file_name, url=file_ref, display="inline")

    if FILE_DOWNLOAD_MODE == "content":
        try:
            data = await asyncio.to_thread(_read_file_bytes_from_storage, file_ref)
            if len(data) <= MAX_INLINE_FILE_BYTES:
                return cl.File(name=file_name, content=data, display="inline")
            logger.info(
                "File %s is too large for inline bytes (%d > %d), falling back to local path mode",
                file_ref,
                len(data),
                MAX_INLINE_FILE_BYTES,
            )
        except Exception as e:
            logger.warning(
                "Could not stream file bytes directly for %s (%s: %s). Falling back to local path mode.",
                file_ref,
                type(e).__name__,
                e,
            )

    local_path = await _materialize_file_to_local(file_ref, file_name)
    return cl.File(name=file_name, path=str(local_path), display="inline")


async def _file_bubble_streaming(file_ref: str) -> cl.Message:
    normalized_ref = file_ref.strip()
    if not normalized_ref:
        return await _create_streaming_message()

    try:
        file_el = await _build_download_file_element(normalized_ref)
        await cl.Message(content=f"Download `{file_el.name}`", elements=[file_el]).send()
    except Exception as e:
        logger.error(
            "Failed to create downloadable file for %s: %s: %s",
            normalized_ref,
            type(e).__name__,
            e,
            exc_info=True,
        )
        await cl.Message(
            content=f"Could not prepare downloadable file: `{normalized_ref}`",
            author="assistant",
        ).send()

    return await _create_streaming_message()


async def _stream_line_with_elements(
    line: str,
    assistant: cl.Message | None,
    append_newline: bool = True,
) -> cl.Message:
    # 1) stand-alone Caption: /path/img.png
    if m := PATH_RX.fullmatch(line.strip()):
        caption, src = m.groups()
        logger.info(f"Relay detected image path pattern: caption='{caption}', src='{src}'")
        if assistant is None:
            assistant = await _create_streaming_message()
        await _stream_text_to_message(line, assistant)
        await _finalize_message(assistant)
        return await _image_bubble_streaming(caption.strip(), src)

    # 2) inline markdown images and <file> tags, preserving order
    pos = 0
    for m in INLINE_ELEMENT_RX.finditer(line):
        if m.start() > pos:
            if assistant is None:
                assistant = await _create_streaming_message()
            await _stream_text_to_message(line[pos : m.start()], assistant)

        image_alt, image_src, file_src = m.groups()
        if image_src is not None:
            logger.info(f"Relay detected markdown image: alt='{image_alt}', src='{image_src}'")
            await _finalize_message(assistant)
            assistant = await _image_bubble_streaming(image_alt, image_src)
        elif file_src is not None:
            normalized_file_src = file_src.strip()
            logger.info("Relay detected file tag: '%s'", normalized_file_src)
            await _finalize_message(assistant)
            assistant = await _file_bubble_streaming(normalized_file_src)

        pos = m.end()

    # 3) tail text
    if assistant is None:
        assistant = await _create_streaming_message()
    tail = line[pos:] + ("\n" if append_newline else "")
    new_assistant = await _stream_text_to_message(tail, assistant)
    if new_assistant is not None:
        assistant = new_assistant

    return assistant


async def _send_text_with_smiles(text: str):
    """Send text message with SMILES processing using cl.Image"""
    logger.debug(f"_send_text_with_smiles called with text length: {len(text)}")
    # Check if there are any SMILES patterns
    if not SMI_RX.search(text):
        # No SMILES, send as regular message
        logger.debug("No SMILES patterns found in text, sending as regular message")
        await cl.Message(content=text, author="assistant").send()
        return

    logger.info("SMILES patterns detected, processing...")
    # Process text with SMILES
    pos = 0
    for m in SMI_RX.finditer(text):
        smi = m.group(1)
        logger.debug(f"Detected SMILES: '{smi}'")

        # Send text up to SMILES
        if m.start() > pos:
            await cl.Message(content=text[pos : m.start()], author="assistant").send()

        # Send SMILES token
        await cl.Message(content=f"`{smi}`", author="assistant").send()
        logger.debug(f"Sent SMILES token message: `{smi}`")

        # Try to create molecule image and send as cl.Image
        try:
            logger.info(f"Attempting to convert SMILES to PNG: '{smi}'")
            png = smiles_to_png_bytes(smi)
            if png is not None:
                png_size = len(png)
                logger.info(f"Successfully generated PNG from SMILES '{smi}' ({png_size} bytes)")
                b64 = base64.b64encode(png).decode()
                data_url = f"data:image/png;base64,{b64}"
                logger.debug(f"Created data URL from PNG (size: {len(b64)} chars)")
                img_el = cl.Image(url=data_url, name=smi, display="inline")
                logger.debug(f"Created cl.Image element for SMILES: '{smi}'")
                await cl.Message(content=f"`{smi}`", elements=[img_el], author="assistant").send()
                logger.info(f"Sent SMILES image message for: '{smi}'")
            else:
                logger.warning(f"smiles_to_png_bytes returned None for SMILES: '{smi}'")
        except ValueError as ve:
            logger.warning(f"ValueError converting SMILES '{smi}' to image: {ve}")
            # Invalid SMILES, just continue without image
        except Exception as e:
            logger.error(f"Exception converting SMILES '{smi}' to image: {type(e).__name__}: {e}")
            # Invalid SMILES, just continue without image

        pos = m.end()

    # Send remaining text
    if pos < len(text):
        await cl.Message(content=text[pos:], author="assistant").send()


async def _handle_file_uploads(files: list, session_id: str) -> list[str]:
    """
    Upload files to S3 in the session-specific folder.

    Args:
        files: List of cl.File objects from the message
        session_id: Current Chainlit thread/session ID

    Returns:
        List of S3 paths where files were uploaded
    """
    uploaded_paths = []
    logger.debug(f"_handle_file_uploads called with {len(files)} files")

    for file in files:
        try:
            logger.debug(f"Processing file: {file.name if hasattr(file, 'name') else 'unknown'}")
            logger.debug(f"File attributes: {dir(file)}")

            # Read file content
            file_content = None

            if hasattr(file, 'content') and file.content:
                file_content = file.content
                logger.debug(f"Got content from file.content ({len(file_content)} bytes)")
            elif hasattr(file, 'path') and file.path:
                # Read from file path
                logger.debug(f"Reading from file.path: {file.path}")
                with open(file.path, 'rb') as f:
                    file_content = f.read()
                logger.debug(f"Read {len(file_content)} bytes from file")

            if file_content is None:
                logger.warning(f"Could not read content from file {file.name}")
                continue

            # Upload to S3 using relative path
            # S3.prefix is already set to sessions/{session_id} by on_chat_start/resume
            relative_path = f"{file.name}"
            logger.debug(f"Relative S3 path: {relative_path}")
            logger.debug(f"Current S3.prefix: {S3.prefix}")

            # Write file to S3 using S3.open with relative path
            logger.debug("Opening S3 file for writing...")
            with S3.open(relative_path, 'wb') as s3_file:
                s3_file.write(file_content)

            # Get the full S3 URL for display
            full_s3_url = S3.path(relative_path)
            uploaded_paths.append(full_s3_url)
            logger.info(f"Uploaded file {file.name} to {full_s3_url}")

        except Exception as e:
            logger.error(f"Error uploading file {getattr(file, 'name', 'unknown')}: {type(e).__name__}: {e}", exc_info=True)
            # Continue with other files even if one fails
            continue

    logger.debug(f"Upload complete. {len(uploaded_paths)} files uploaded successfully")
    return uploaded_paths


# ---------- main relay ------------------------------------------------------ #
async def relay(stream):
    assistant = None  # Will be created when we have content
    current_step = None  # active tool Step
    buf = ""  # accumulate until newline
    # Check if tool calls should be displayed
    show_tool_calls = cl.user_session.get("show_tool_calls", True)

    async for chunk in stream:
        # ── tool events → COT sidebar as Steps ───────────────────────────────
        ev = getattr(chunk, "event", None)
        if ev == "ToolCallStarted":
            if show_tool_calls:
                t = chunk.tool
                current_step = cl.Step(name=t.tool_name or t.name or "tool", type="tool")
                current_step.input = t.tool_args or getattr(t, "arguments", {})
                await current_step.send()
            continue

        if ev and ev.endswith("Completed"):
            if show_tool_calls and current_step:
                current_step.output = chunk.content or "✅ done"
                await current_step.update()
                current_step = None
            continue

        # ── plain text from the LLM / agent ─────────────────────────────────
        text = (
            chunk
            if isinstance(chunk, str)
            else getattr(chunk, "content", "") or getattr(chunk, "text", "")
        )
        if not text:
            continue

        buf += text
        while "\n" in buf:  # process complete lines
            line, buf = buf.split("\n", 1)
            assistant = await _stream_line_with_elements(line, assistant, append_newline=True)

    # ── flush tail (no final newline) ────────────────────────────────────────
    if buf:
        assistant = await _stream_line_with_elements(buf, assistant, append_newline=False)

    # Persist the final streaming message so resumed sessions show the text.
    await _finalize_message(assistant)


# ---------- Chainlit entry-point ------------------------------------------- #
@cl.on_message
async def main(user_msg: cl.Message):
    try:
        # Ensure session is properly initialized
        if not cl.user_session.get("session_initialized"):
            await on_chat_start()

        # Lazily set chat title from the first user message
        if not cl.user_session.get("title_set"):
            try:
                await cl.set_chat_title(user_msg.content[:60] or "New chat")
            except Exception:
                pass
            cl.user_session.set("title_set", True)

        # Get or create session agent first (needed for session_state)
        session_agent = cl.user_session.get("agent")
        if session_agent is None:
            # Ensure S3 session is synchronized
            thread_id = cl.context.session.thread_id
            if thread_id:
                S3.prefix = f"sessions/{thread_id}"
                logger.info(f"Set S3 session prefix in main(): {S3.prefix}")

            session_agent = get_cs_copilot_agent_team(
                model,
                show_members_responses=False,
            )
            cl.user_session.set("agent", session_agent)

        # Re-apply map settings so the agent's session_state is up to date
        # even when a fresh agent was just created above.
        _apply_map_settings(session_agent, cl.user_session.get("map") or "new_map")

        # Handle file uploads if present
        # Debug: Check multiple possible locations for files
        files = None

        # Try different ways files might be attached
        if hasattr(user_msg, 'files') and user_msg.files:
            files = user_msg.files
            logger.debug(f"Found files in user_msg.files: {[f.name for f in files]}")
        elif hasattr(user_msg, 'elements') and user_msg.elements:
            # Filter for File elements
            files = [el for el in user_msg.elements if isinstance(el, cl.File)]
            if files:
                logger.debug(f"Found files in user_msg.elements: {[f.name for f in files]}")

        if files:
            logger.debug(f"Processing {len(files)} file(s)")
            # Get thread ID for session-specific folder
            thread_id = cl.context.session.thread_id
            logger.debug(f"Thread ID: {thread_id}")

            if thread_id:
                uploaded_paths = await _handle_file_uploads(files, thread_id)
                logger.debug(f"Uploaded paths: {uploaded_paths}")

                if uploaded_paths:
                    # Store uploaded files in agent's session state
                    # Ensure session_state exists
                    if session_agent.session_state is None:
                        session_agent.session_state = {}
                        logger.info("Initialized session_state")

                    # Initialize uploaded_files dict if it doesn't exist
                    if "uploaded_files" not in session_agent.session_state:
                        session_agent.session_state["uploaded_files"] = {}
                        logger.info("Initialized uploaded_files in agent session state")

                    # Add new files (basename: s3_path) without overwriting existing ones
                    for s3_path in uploaded_paths:
                        filename = s3_path.split('/')[-1]
                        session_agent.session_state["uploaded_files"][filename] = s3_path
                        logger.info(f"Added to session state: {filename} → {s3_path}")

                    logger.info(f"Total files in session state: {len(session_agent.session_state['uploaded_files'])}")

                    # Display confirmation message
                    file_list = "\n".join([f"- `{path.split('/')[-1]}` → {path}" for path in uploaded_paths])
                    await cl.Message(
                        content=f"📁 **Files uploaded to S3:**\n{file_list}",
                        author="assistant",
                    ).send()
                    logger.debug("Confirmation message sent to UI")
                else:
                    logger.debug("No files were successfully uploaded")
            else:
                logger.debug("No thread_id available, skipping upload")
        else:
            logger.debug("No files found in message")

        # Process the message with session-scoped memory.
        # Two layers of retry protect against transient Ollama errors
        # (e.g. malformed tool-call JSON):
        #  - Inner: arun_with_retry wraps the async stream with retry
        #    logic that is transparent to relay().
        #  - Outer: this loop catches errors that surface after relay()
        #    has already emitted partial UI content.  On retry a fresh
        #    stream + relay is started and the user is notified.
        thread_id = cl.context.session.thread_id
        max_retries = 3
        base_delay = 2.0
        for attempt in range(max_retries + 1):
            try:
                stream = await arun_with_retry(
                    session_agent,
                    user_msg.content,
                    stream=True,
                    session_id=thread_id,  # Isolate memory per chat thread
                    max_retries=1,  # Light inner retry; outer loop is primary
                )
                await relay(stream)
                break  # Success – exit retry loop
            except Exception as e:
                if _is_retriable(e) and attempt < max_retries:
                    delay = base_delay * (2 ** attempt)
                    logger.warning(
                        "Retriable error in main() on attempt %d/%d: %s "
                        "– retrying in %.1fs …",
                        attempt + 1,
                        max_retries + 1,
                        e,
                        delay,
                    )
                    await cl.Message(
                        content=(
                            f"The model encountered a transient error. "
                            f"Retrying (attempt {attempt + 2}/{max_retries + 1})..."
                        ),
                        author="assistant",
                    ).send()
                    await asyncio.sleep(delay)
                    continue
                # Non-retriable or final attempt – fall through to error handler
                raise

    except Exception as e:
        # Log error and send user-friendly message
        logger.error(f"Error processing message: {e}", exc_info=True)
        await cl.Message(
            content="Sorry, I encountered an error processing your message. Please try again.",
            author="assistant",
        ).send()
