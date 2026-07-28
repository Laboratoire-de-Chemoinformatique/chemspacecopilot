#!/usr/bin/env python
# coding: utf-8
"""
S3 storage client for unified file operations.

Provides transparent access to files stored in S3/MinIO or locally,
with automatic session-based path management.
"""

import builtins
import contextvars
import datetime
import errno
import hashlib
import io
import logging
import os
import stat
import tempfile
import uuid
from contextlib import contextmanager
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Iterator, Mapping
from urllib.parse import unquote, urlsplit

import fsspec

from .config import get_s3_config, is_s3_enabled

logger = logging.getLogger(__name__)
LOCAL_STORAGE_ROOT = Path("data")


def _normalize_storage_prefix(value: str) -> str:
    """Return a safe backend-relative storage prefix."""

    raw = str(value).strip()
    if raw.startswith("/") or "\\" in raw or "://" in raw:
        raise ValueError("storage prefix must be a safe relative path")
    text = raw.rstrip("/")
    candidate = PurePosixPath(text)
    if (
        not text
        or not candidate.parts
        or candidate.is_absolute()
        or any(part in {"", ".", ".."} for part in candidate.parts)
    ):
        raise ValueError("storage prefix must be a safe relative path")
    return candidate.as_posix()


# Generate a per-run session ID when SESSION_ID is unset or blank
_ENV_SESSION_ID = os.getenv("SESSION_ID")
if _ENV_SESSION_ID is None or _ENV_SESSION_ID.strip() == "":
    logger.info("SESSION_ID is not set, generating new session ID")
    # Timestamp + short uuid for readability and uniqueness
    _ts = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%d-%H%M%S")
    SESSION_ID = f"{_ts}-{uuid.uuid4().hex[:6]}"
else:
    logger.info("SESSION_ID is set from environment")
    SESSION_ID = _ENV_SESSION_ID

_DEFAULT_PREFIX = _normalize_storage_prefix(f"sessions/{SESSION_ID}")
_SESSION_PREFIX: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "cs_copilot_s3_prefix",
    default=None,
)
_WRITE_BOUNDARY: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "cs_copilot_storage_write_boundary",
    default=None,
)
_WRITE_PROTECTED_PATHS: contextvars.ContextVar[frozenset[str]] = contextvars.ContextVar(
    "cs_copilot_storage_write_protected_paths",
    default=frozenset(),
)
_PENDING_ARTIFACT_WRITES: contextvars.ContextVar[list[Any] | None] = contextvars.ContextVar(
    "cs_copilot_storage_pending_artifact_writes",
    default=None,
)
_VERIFIED_ARTIFACT_READS: contextvars.ContextVar[Mapping[str, tuple[str, int]] | None] = (
    contextvars.ContextVar(
        "cs_copilot_storage_verified_artifact_reads",
        default=None,
    )
)
_STAGED_ARTIFACT_WRITES: contextvars.ContextVar[dict[str, Any] | None] = contextvars.ContextVar(
    "cs_copilot_storage_staged_artifact_writes",
    default=None,
)
_EXTERNAL_STAGING_ID: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "cs_copilot_storage_external_staging_id",
    default=None,
)
_EXTERNAL_STAGED_PUBLICATIONS: contextvars.ContextVar[dict[str, str] | None] = (
    contextvars.ContextVar(
        "cs_copilot_storage_external_staged_publications",
        default=None,
    )
)
_OPEN_SUPPORTS_DIR_FD = os.open in os.supports_dir_fd
_O_NOFOLLOW = getattr(os, "O_NOFOLLOW", 0)
_RESERVED_RUN_FILES = frozenset({"manifest.json", "artifacts/index.json"})


def _directory_open_flags() -> int:
    return os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_DIRECTORY", 0) | _O_NOFOLLOW


def _open_existing_directory_tree(path: Path) -> int:
    """Open every component of an absolute directory without following links."""

    if not _OPEN_SUPPORTS_DIR_FD or not _O_NOFOLLOW:
        raise RuntimeError("secure local storage requires directory-relative no-follow support")
    absolute = path.absolute()
    if not absolute.is_absolute():
        raise ValueError("secure directory traversal requires an absolute path")
    flags = _directory_open_flags()
    current_fd = os.open(absolute.anchor, flags)
    try:
        for component in absolute.parts[1:]:
            next_fd = os.open(component, flags, dir_fd=current_fd)
            os.close(current_fd)
            current_fd = next_fd
        result = current_fd
        current_fd = -1
        return result
    except OSError as exc:
        if exc.errno in {errno.ELOOP, errno.ENOTDIR}:
            raise PermissionError(
                "local storage path contains a symlink or non-directory component"
            ) from exc
        raise
    finally:
        if current_fd >= 0:
            os.close(current_fd)


def _stream_digest(
    source: Any,
    *,
    maximum_size: int | None = None,
) -> tuple[str, int]:
    """Digest a stream, optionally failing before consuming oversized content."""

    digest = hashlib.sha256()
    size = 0
    while True:
        chunk = source.read(1024 * 1024)
        if not chunk:
            break
        if isinstance(chunk, str):
            chunk = chunk.encode("utf-8")
        size += len(chunk)
        if maximum_size is not None and size > maximum_size:
            raise PermissionError("artifact exceeded its expected size during verification")
        digest.update(chunk)
    return digest.hexdigest(), size


def _secure_local_unlink_verified(
    path: Path,
    *,
    expected_sha256: str,
    expected_size: int,
    expected_identity: tuple[int, int] | None = None,
) -> bool:
    """Unlink one exact regular file without following any path component."""

    absolute = path.absolute()
    try:
        parent_fd = _open_existing_directory_tree(absolute.parent)
    except FileNotFoundError:
        return False
    file_fd: int | None = None
    try:
        flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | _O_NOFOLLOW
        try:
            file_fd = os.open(absolute.name, flags, dir_fd=parent_fd)
        except FileNotFoundError:
            return False
        metadata = os.fstat(file_fd)
        identity = (metadata.st_dev, metadata.st_ino)
        if (
            not stat.S_ISREG(metadata.st_mode)
            or metadata.st_nlink != 1
            or (expected_identity is not None and identity != expected_identity)
        ):
            raise PermissionError("refusing to remove a replaced or linked artifact")
        with os.fdopen(os.dup(file_fd), "rb") as source:
            digest, size = _stream_digest(source, maximum_size=expected_size)
        if size != expected_size or digest != expected_sha256:
            raise PermissionError("refusing to remove an artifact whose content changed")
        current = os.stat(absolute.name, dir_fd=parent_fd, follow_symlinks=False)
        if (current.st_dev, current.st_ino) != identity:
            raise PermissionError("refusing to remove a replaced artifact")
        os.unlink(absolute.name, dir_fd=parent_fd)
        os.fsync(parent_fd)
        return True
    except OSError as exc:
        if exc.errno in {errno.ELOOP, errno.ENOTDIR}:
            raise PermissionError("refusing to remove a symlinked artifact") from exc
        raise
    finally:
        if file_fd is not None:
            os.close(file_fd)
        os.close(parent_fd)


def _secure_local_unlink(path: Path) -> bool:
    """Unlink a final directory entry beneath a no-follow absolute parent."""

    absolute = path.absolute()
    try:
        parent_fd = _open_existing_directory_tree(absolute.parent)
    except FileNotFoundError:
        return False
    try:
        try:
            os.unlink(absolute.name, dir_fd=parent_fd)
        except FileNotFoundError:
            return False
        os.fsync(parent_fd)
        return True
    finally:
        os.close(parent_fd)


def _secure_rmtree_contents(directory_fd: int) -> None:
    """Remove directory contents relative to an already verified descriptor."""

    for name in os.listdir(directory_fd):
        metadata = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
        if stat.S_ISDIR(metadata.st_mode):
            child_fd = os.open(name, _directory_open_flags(), dir_fd=directory_fd)
            try:
                _secure_rmtree_contents(child_fd)
                current = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
                if (current.st_dev, current.st_ino) != (metadata.st_dev, metadata.st_ino):
                    raise PermissionError("refusing to remove a replaced staging directory")
            finally:
                os.close(child_fd)
            os.rmdir(name, dir_fd=directory_fd)
        else:
            os.unlink(name, dir_fd=directory_fd)


def _secure_local_rmtree(path: Path) -> bool:
    """Remove one exact directory tree without following any path component."""

    absolute = path.absolute()
    try:
        parent_fd = _open_existing_directory_tree(absolute.parent)
    except FileNotFoundError:
        return False
    root_fd: int | None = None
    try:
        try:
            root_fd = os.open(absolute.name, _directory_open_flags(), dir_fd=parent_fd)
        except FileNotFoundError:
            return False
        metadata = os.fstat(root_fd)
        _secure_rmtree_contents(root_fd)
        current = os.stat(absolute.name, dir_fd=parent_fd, follow_symlinks=False)
        if (current.st_dev, current.st_ino) != (metadata.st_dev, metadata.st_ino):
            raise PermissionError("refusing to remove a replaced staging directory")
        os.rmdir(absolute.name, dir_fd=parent_fd)
        os.fsync(parent_fd)
        return True
    except OSError as exc:
        if exc.errno in {errno.ELOOP, errno.ENOTDIR}:
            raise PermissionError("refusing to traverse a symlinked staging directory") from exc
        raise
    finally:
        if root_fd is not None:
            os.close(root_fd)
        os.close(parent_fd)


def _remove_fsspec_path_verified(
    filesystem: Any,
    path: str,
    *,
    expected_sha256: str,
    expected_size: int,
) -> bool:
    """Remove one backend object only while its bytes match the publication."""

    try:
        source = filesystem.open(path, mode="rb")
    except FileNotFoundError:
        return False
    with source:
        digest, size = _stream_digest(source, maximum_size=expected_size)
    if size != expected_size or digest != expected_sha256:
        raise PermissionError("refusing to remove an artifact whose content changed")
    filesystem.rm(path)
    return True


class _AtomicLocalCreateFile:
    """File-like wrapper that publishes a complete local artifact exactly once."""

    def __init__(
        self,
        handle: Any,
        *,
        directory_fd: int,
        temporary_name: str,
        destination_name: str,
        destination_path: Path,
        staged_key: str | None,
        deferred: bool,
    ) -> None:
        self._handle = handle
        self._directory_fd = directory_fd
        self._temporary_name = temporary_name
        self._destination_name = destination_name
        self._destination_path = destination_path
        self._staged_key = staged_key
        self._deferred = deferred
        self._closed = False
        self._prepared = False
        self._finalized = False
        self._published = False
        self._expected_sha256: str | None = None
        self._expected_size: int | None = None
        self._expected_identity: tuple[int, int] | None = None

    def __getattr__(self, name: str) -> Any:
        return getattr(self._handle, name)

    def __enter__(self) -> "_AtomicLocalCreateFile":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> bool:
        if exc_type is not None:
            self._abort()
            return False
        self.close()
        return False

    def __iter__(self):
        return iter(self._handle)

    def __next__(self):
        return next(self._handle)

    @property
    def closed(self) -> bool:
        return self._closed

    def write(self, value):
        return self._handle.write(value)

    def writelines(self, lines) -> None:
        self._handle.writelines(lines)

    def flush(self) -> None:
        self._handle.flush()

    def fileno(self) -> int:
        return self._handle.fileno()

    def seek(self, *args):
        return self._handle.seek(*args)

    def tell(self) -> int:
        return self._handle.tell()

    def truncate(self, *args):
        return self._handle.truncate(*args)

    def readable(self) -> bool:
        return self._handle.readable()

    def writable(self) -> bool:
        return self._handle.writable()

    def seekable(self) -> bool:
        return self._handle.seekable()

    def close(self) -> None:
        if self._closed:
            return
        self._prepare()
        pending = _PENDING_ARTIFACT_WRITES.get()
        if self._deferred and pending is not None:
            pending.append(self)
            staged = _STAGED_ARTIFACT_WRITES.get()
            if staged is not None and self._staged_key is not None:
                staged[self._staged_key] = self
        else:
            self._commit()
        self._closed = True

    def _prepare(self) -> None:
        if self._prepared:
            return
        try:
            self._handle.flush()
            os.fsync(self._handle.fileno())
            self._handle.close()
            self._prepared = True
        except BaseException:
            self._abort()
            raise

    def _commit(self) -> None:
        if self._finalized:
            return
        self._prepare()
        try:
            self._capture_integrity()
            os.link(
                self._temporary_name,
                self._destination_name,
                src_dir_fd=self._directory_fd,
                dst_dir_fd=self._directory_fd,
                follow_symlinks=False,
            )
            self._published = True
            os.unlink(self._temporary_name, dir_fd=self._directory_fd)
            os.fsync(self._directory_fd)
        except OSError as exc:
            if exc.errno == errno.EEXIST:
                raise PermissionError(
                    "scoped artifact writes are create-only; destination already exists"
                ) from exc
            raise
        finally:
            if not self._published:
                self._unlink_temporary()
            os.close(self._directory_fd)
            self._finalized = True

    def _abort(self) -> None:
        if self._finalized:
            return
        try:
            if not self._handle.closed:
                self._handle.close()
        finally:
            self._unlink_temporary()
            os.close(self._directory_fd)
            self._closed = True
            self._finalized = True

    def _rollback_published(self) -> None:
        if not self._published or self._expected_sha256 is None or self._expected_size is None:
            return
        _secure_local_unlink_verified(
            self._destination_path,
            expected_sha256=self._expected_sha256,
            expected_size=self._expected_size,
            expected_identity=self._expected_identity,
        )

    def _open_staged_read(self, mode: str):
        if mode not in {"r", "rt", "rb"}:
            raise PermissionError("staged artifacts are read-only")
        flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | _O_NOFOLLOW
        fd = os.open(self._temporary_name, flags, dir_fd=self._directory_fd)
        metadata = os.fstat(fd)
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
            os.close(fd)
            raise PermissionError("staged artifact must be a regular, non-linked file")
        return os.fdopen(fd, "rb" if "b" in mode else "r")

    def _unlink_temporary(self) -> None:
        try:
            os.unlink(self._temporary_name, dir_fd=self._directory_fd)
        except FileNotFoundError:
            pass

    def _capture_integrity(self) -> None:
        if self._expected_sha256 is not None:
            return
        with self._open_staged_read("rb") as source:
            metadata = os.fstat(source.fileno())
            digest, size = _stream_digest(source)
        self._expected_sha256 = digest
        self._expected_size = size
        self._expected_identity = (metadata.st_dev, metadata.st_ino)


class _AtomicS3CreateFile:
    """Publish a non-empty S3 artifact through one conditional create."""

    def __init__(
        self,
        handle: Any,
        *,
        deferred: bool,
        mode: str,
        staged_key: str | None,
    ) -> None:
        self._handle = handle
        self._deferred = deferred
        self._binary = "b" in mode
        self._staged_key = staged_key
        self._mirror = tempfile.SpooledTemporaryFile(
            max_size=16 * 1024 * 1024,
            mode="w+b" if self._binary else "w+",
            encoding=None if self._binary else "utf-8",
        )
        self._closed = False
        self._finalized = False
        self._published = False
        self._expected_sha256: str | None = None
        self._expected_size: int | None = None

    def __getattr__(self, name: str) -> Any:
        return getattr(self._handle, name)

    def __enter__(self) -> "_AtomicS3CreateFile":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> bool:
        if exc_type is not None:
            self._abort()
            return False
        self.close()
        return False

    def __iter__(self):
        return iter(self._handle)

    def __next__(self):
        return next(self._handle)

    @property
    def closed(self) -> bool:
        return self._closed

    def write(self, value):
        written = self._handle.write(value)
        self._mirror.write(value)
        return written

    def writelines(self, lines) -> None:
        for line in lines:
            self.write(line)

    def flush(self) -> None:
        self._handle.flush()
        self._mirror.flush()

    def seek(self, *args):
        result = self._handle.seek(*args)
        self._mirror.seek(*args)
        return result

    def tell(self) -> int:
        return self._handle.tell()

    def truncate(self, *args):
        result = self._handle.truncate(*args)
        self._mirror.truncate(*args)
        return result

    def close(self) -> None:
        if self._closed:
            return
        try:
            if int(self._mirror.tell()) <= 0:
                self._abort()
                raise ValueError("scoped workflow artifacts must not be empty")
            self._handle.close()
            if self._deferred:
                pending = _PENDING_ARTIFACT_WRITES.get()
                if pending is None:
                    raise RuntimeError("deferred S3 write lost its invocation transaction")
                pending.append(self)
                staged = _STAGED_ARTIFACT_WRITES.get()
                if staged is not None and self._staged_key is not None:
                    staged[self._staged_key] = self
            else:
                self._published = True
                self._finalized = True
                self._mirror.close()
        except (FileExistsError, IsADirectoryError) as exc:
            self._abort()
            raise PermissionError(
                "scoped artifact writes are create-only; destination already exists"
            ) from exc
        except OSError as exc:
            self._abort()
            if exc.errno == errno.EEXIST:
                raise PermissionError(
                    "scoped artifact writes are create-only; destination already exists"
                ) from exc
            raise
        finally:
            self._closed = True

    def _commit(self) -> None:
        if self._finalized:
            return
        try:
            self._capture_integrity()
            self._handle.commit()
            self._published = True
            self._finalized = True
            self._mirror.close()
        except (FileExistsError, IsADirectoryError) as exc:
            self._abort()
            raise PermissionError(
                "scoped artifact writes are create-only; destination already exists"
            ) from exc
        except OSError as exc:
            self._abort()
            if exc.errno == errno.EEXIST:
                raise PermissionError(
                    "scoped artifact writes are create-only; destination already exists"
                ) from exc
            raise

    def _abort(self) -> None:
        if self._finalized:
            return
        discard = getattr(self._handle, "discard", None)
        if callable(discard):
            discard()
        try:
            self._handle.closed = True
        except (AttributeError, TypeError):
            pass
        self._closed = True
        self._finalized = True
        self._mirror.close()

    def _rollback_published(self) -> None:
        if (
            self._published
            and self._expected_sha256 is not None
            and self._expected_size is not None
        ):
            _remove_fsspec_path_verified(
                self._handle.fs,
                self._handle.path,
                expected_sha256=self._expected_sha256,
                expected_size=self._expected_size,
            )

    def _open_staged_read(self, mode: str):
        if mode not in {"r", "rt", "rb"}:
            raise PermissionError("staged artifacts are read-only")
        self._mirror.flush()
        position = self._mirror.tell()
        self._mirror.seek(0)
        target_binary = "b" in mode
        snapshot = tempfile.SpooledTemporaryFile(
            max_size=16 * 1024 * 1024,
            mode="w+b" if target_binary else "w+",
            encoding=None if target_binary else "utf-8",
        )
        while True:
            chunk = self._mirror.read(1024 * 1024)
            if not chunk:
                break
            if target_binary and isinstance(chunk, str):
                chunk = chunk.encode("utf-8")
            elif not target_binary and isinstance(chunk, bytes):
                chunk = chunk.decode("utf-8")
            snapshot.write(chunk)
        self._mirror.seek(position)
        snapshot.seek(0)
        return snapshot

    def _capture_integrity(self) -> None:
        if self._expected_sha256 is not None:
            return
        with self._open_staged_read("rb") as source:
            digest, size = _stream_digest(source)
        self._expected_sha256 = digest
        self._expected_size = size


class _S3Meta(type):
    @property
    def prefix(cls) -> str:
        return cls.current_prefix()

    @prefix.setter
    def prefix(cls, value: str) -> None:
        cls.set_session_prefix(value)


class S3(metaclass=_S3Meta):
    """
    Unified storage client for S3/MinIO and local file operations.

    All file operations are scoped to the current session by default,
    unless an absolute S3 URL or local path is provided.

    Class Attributes:
    -----------------
    prefix : str
        Session-scoped prefix for all relative paths

    Methods:
    --------
    path(rel: str) -> str
        Convert a relative path to an S3 URL

    open(rel: str, mode: str = "rb")
        Open a file for reading or writing

    Examples:
    ---------
    >>> # Write to session-scoped S3 path
    >>> with S3.open("results.csv", "w") as f:
    ...     f.write("data")

    >>> # Read from absolute S3 URL
    >>> with S3.open("s3://bucket/key.csv", "r") as f:
    ...     data = f.read()

    >>> # Read from local file
    >>> with S3.open("/tmp/local.csv", "r") as f:
    ...     data = f.read()
    """

    _fallback_prefix = _DEFAULT_PREFIX
    logger.info(f"Initialized S3 client with SESSION_ID: {SESSION_ID}")

    @classmethod
    def set_session_prefix(cls, prefix: str) -> None:
        """Set the storage prefix for the current execution context.

        Chainlit serves multiple chat sessions in one Python process.  A plain
        class-level prefix is shared across concurrent chats, so relative
        storage paths can leak into another user's session.  The context-local
        prefix isolates async tasks while keeping ``S3.prefix`` as a fallback
        for CLI/tests/backwards compatibility.
        """
        normalized = _normalize_storage_prefix(prefix)
        cls._fallback_prefix = normalized
        _SESSION_PREFIX.set(normalized)

    @classmethod
    def current_prefix(cls) -> str:
        """Return the active context-local prefix, falling back to ``S3.prefix``."""
        return _SESSION_PREFIX.get() or cls._fallback_prefix

    @classmethod
    @contextmanager
    def confine_writes(
        cls,
        boundary: str,
        *,
        protected_paths: Iterable[str] = (),
        publication_receipt: dict[str, dict[str, Any]] | None = None,
    ) -> Iterator[None]:
        """Confine relative writes to one session subtree for this execution context.

        The MCP adapter activates this guard while a session-scoped tool runs.
        It is context-local, propagates through ``asyncio.to_thread``, and is
        enforced again by the worker process. Existing registered artifacts
        can be supplied as protected paths so a tool cannot mutate immutable
        scientific evidence. Reads remain unaffected.
        """

        normalized = cls._normalize_relative_path(boundary, allow_empty=True)
        boundary_path = PurePosixPath(normalized) if normalized else PurePosixPath()
        protected: set[str] = set()
        for path in protected_paths:
            protected_path = cls._normalize_relative_path(path)
            try:
                PurePosixPath(protected_path).relative_to(boundary_path)
            except ValueError as exc:
                raise ValueError(
                    "protected write path must remain inside the active boundary"
                ) from exc
            protected.add(protected_path)
        normalized_protected = frozenset(protected)
        outer_pending = _PENDING_ARTIFACT_WRITES.get()
        pending: list[Any] = outer_pending if outer_pending is not None else []
        owns_transaction = outer_pending is None
        outer_staged = _STAGED_ARTIFACT_WRITES.get()
        staged: dict[str, Any] = outer_staged if outer_staged is not None else {}
        boundary_token = _WRITE_BOUNDARY.set(normalized)
        protected_token = _WRITE_PROTECTED_PATHS.set(normalized_protected)
        pending_token = _PENDING_ARTIFACT_WRITES.set(pending)
        staged_token = _STAGED_ARTIFACT_WRITES.set(staged)
        try:
            yield
        except BaseException:
            if owns_transaction:
                for writer in reversed(pending):
                    writer._abort()
            raise
        else:
            if owns_transaction:
                committed: list[Any] = []
                prepared_publications: dict[str, dict[str, Any]] = {}
                try:
                    if publication_receipt is not None:
                        for writer in pending:
                            staged_key = getattr(writer, "_staged_key", None)
                            if staged_key is None:
                                continue
                            prepared_publications[staged_key] = cls._pending_writer_metadata(
                                writer, staged_key
                            )
                    for writer in pending:
                        writer._commit()
                        committed.append(writer)
                except BaseException:
                    cleanup_errors: list[BaseException] = []
                    for writer in reversed(pending[len(committed) :]):
                        try:
                            writer._abort()
                        except BaseException as exc:
                            cleanup_errors.append(exc)
                            logger.warning(
                                "Could not abort a pending artifact publication",
                                exc_info=True,
                            )
                    for writer in reversed(committed):
                        try:
                            writer._rollback_published()
                        except BaseException as exc:
                            cleanup_errors.append(exc)
                            logger.warning(
                                "Refused or failed to roll back a published artifact",
                                exc_info=True,
                            )
                    if cleanup_errors:
                        raise RuntimeError(
                            "artifact transaction failed and could not be fully rolled back"
                        ) from cleanup_errors[0]
                    raise
                else:
                    if publication_receipt is not None:
                        publication_receipt.update(prepared_publications)
        finally:
            _STAGED_ARTIFACT_WRITES.reset(staged_token)
            _PENDING_ARTIFACT_WRITES.reset(pending_token)
            _WRITE_PROTECTED_PATHS.reset(protected_token)
            _WRITE_BOUNDARY.reset(boundary_token)

    @staticmethod
    def _pending_writer_metadata(writer: Any, final_key: str) -> dict[str, Any]:
        """Checksum a private transaction object before it is published."""

        digest = hashlib.sha256()
        size = 0
        with writer._open_staged_read("rb") as source:
            while True:
                chunk = source.read(1024 * 1024)
                if not chunk:
                    break
                if isinstance(chunk, str):
                    chunk = chunk.encode("utf-8")
                digest.update(chunk)
                size += len(chunk)
        return {
            "staged_path": final_key,
            "sha256": digest.hexdigest(),
            "size_bytes": size,
        }

    @staticmethod
    def current_write_boundary() -> str | None:
        """Return the active write boundary, or ``None`` outside guarded execution."""

        return _WRITE_BOUNDARY.get()

    @staticmethod
    def current_write_protected_paths() -> frozenset[str]:
        """Return immutable session-relative paths for the active tool call."""

        return _WRITE_PROTECTED_PATHS.get()

    @classmethod
    @contextmanager
    def stage_publications(cls, staging_id: str) -> Iterator[None]:
        """Redirect scoped writes to a worker-private publication namespace."""

        normalized_id = cls._normalize_relative_path(staging_id)
        if len(PurePosixPath(normalized_id).parts) != 1:
            raise ValueError("staging_id must be one safe path component")
        publications: dict[str, str] = {}
        id_token = _EXTERNAL_STAGING_ID.set(normalized_id)
        publications_token = _EXTERNAL_STAGED_PUBLICATIONS.set(publications)
        try:
            yield
        finally:
            _EXTERNAL_STAGED_PUBLICATIONS.reset(publications_token)
            _EXTERNAL_STAGING_ID.reset(id_token)

    @staticmethod
    def current_staged_publications() -> dict[str, str]:
        """Return final-to-staged keys created by the active worker invocation."""

        return dict(_EXTERNAL_STAGED_PUBLICATIONS.get() or {})

    @classmethod
    def staged_publication_metadata(cls) -> dict[str, dict[str, Any]]:
        """Checksum committed worker staging objects through a no-follow read."""

        metadata: dict[str, dict[str, Any]] = {}
        for final_key, staged_key in cls.current_staged_publications().items():
            digest = hashlib.sha256()
            size = 0
            if is_s3_enabled():
                config = get_s3_config()
                source = fsspec.open(
                    cls.path(staged_key),
                    mode="rb",
                    **config.to_storage_options(),
                ).open()
            else:
                source = cls._open_local_verified_source(staged_key)
            with source:
                while True:
                    chunk = source.read(1024 * 1024)
                    if not chunk:
                        break
                    if isinstance(chunk, str):
                        chunk = chunk.encode("utf-8")
                    digest.update(chunk)
                    size += len(chunk)
            metadata[final_key] = {
                "staged_path": staged_key,
                "sha256": digest.hexdigest(),
                "size_bytes": size,
            }
        return metadata

    @classmethod
    def promote_staged_publications(
        cls,
        publications: Mapping[str, Mapping[str, Any]],
    ) -> None:
        """Promote complete worker outputs to final create-only artifact keys."""

        promoted: dict[str, Mapping[str, Any]] = {}
        try:
            for final_key, metadata in publications.items():
                final = cls._normalize_relative_path(final_key)
                staged = cls._normalize_relative_path(str(metadata["staged_path"]))
                source = cls._open_verified_snapshot(
                    staged,
                    "rb",
                    expected_sha256=str(metadata["sha256"]),
                    expected_size=int(metadata["size_bytes"]),
                )
                with source:
                    with cls.open_atomic(final, "xb") as destination:
                        while True:
                            chunk = source.read(1024 * 1024)
                            if not chunk:
                                break
                            destination.write(chunk)
                promoted[final] = metadata
            for metadata in publications.values():
                cls._remove_storage_key(cls._normalize_relative_path(str(metadata["staged_path"])))
        except BaseException:
            cls.rollback_promoted_publications(promoted)
            raise

    @classmethod
    def rollback_promoted_publications(
        cls,
        publications: Mapping[str, Mapping[str, Any]],
    ) -> None:
        """Remove only finals that still match the worker's promoted bytes."""

        for final_key, metadata in reversed(tuple(publications.items())):
            final = cls._normalize_relative_path(final_key)
            try:
                cls._remove_storage_key_verified(
                    final,
                    expected_sha256=str(metadata["sha256"]),
                    expected_size=int(metadata["size_bytes"]),
                )
            except FileNotFoundError:
                continue
            except Exception:
                logger.warning(
                    "Refusing to roll back changed worker publication %s",
                    final,
                    exc_info=True,
                )
                continue

    @classmethod
    def discard_staging_prefix(cls, boundary: str, staging_id: str) -> None:
        """Remove one exact worker staging subtree after failure or cancellation."""

        normalized_boundary = cls._normalize_relative_path(boundary)
        normalized_id = cls._normalize_relative_path(staging_id)
        if len(PurePosixPath(normalized_id).parts) != 1:
            raise ValueError("staging_id must be one safe path component")
        prefix = PurePosixPath(
            normalized_boundary,
            ".staging",
            normalized_id,
        ).as_posix()
        if is_s3_enabled():
            config = get_s3_config()
            fs, fs_path = fsspec.core.url_to_fs(
                cls.path(prefix),
                **config.to_storage_options(),
            )
            if fs.exists(fs_path):
                fs.rm(fs_path, recursive=True)
            return
        _secure_local_rmtree(cls._local_session_path(prefix).absolute())

    @classmethod
    def _remove_storage_key(cls, key: str) -> None:
        if is_s3_enabled():
            config = get_s3_config()
            fs, fs_path = fsspec.core.url_to_fs(
                cls.path(key),
                **config.to_storage_options(),
            )
            if fs.exists(fs_path):
                fs.rm(fs_path)
            return
        _secure_local_unlink(cls._local_session_path(key).absolute())

    @classmethod
    def _remove_storage_key_verified(
        cls,
        key: str,
        *,
        expected_sha256: str,
        expected_size: int,
    ) -> bool:
        """Remove one exact publication only while its registered bytes match."""

        if is_s3_enabled():
            config = get_s3_config()
            fs, fs_path = fsspec.core.url_to_fs(
                cls.path(key),
                **config.to_storage_options(),
            )
            return _remove_fsspec_path_verified(
                fs,
                fs_path,
                expected_sha256=expected_sha256,
                expected_size=expected_size,
            )
        return _secure_local_unlink_verified(
            cls._local_session_path(key).absolute(),
            expected_sha256=expected_sha256,
            expected_size=expected_size,
        )

    @classmethod
    @contextmanager
    def confine_artifact_reads(
        cls,
        artifacts: Mapping[str, tuple[str, int]],
    ) -> Iterator[None]:
        """Pin registered artifact reads to checksum-verified byte snapshots."""

        inherited = dict(_VERIFIED_ARTIFACT_READS.get() or {})
        for path, expected in artifacts.items():
            normalized = cls._normalize_relative_path(path)
            if (
                not isinstance(expected, tuple)
                or len(expected) != 2
                or not isinstance(expected[0], str)
                or not isinstance(expected[1], int)
                or expected[1] < 0
            ):
                raise ValueError("verified artifact metadata must be (sha256, size_bytes)")
            inherited[normalized] = (expected[0], expected[1])
        token = _VERIFIED_ARTIFACT_READS.set(inherited)
        try:
            yield
        finally:
            _VERIFIED_ARTIFACT_READS.reset(token)

    @staticmethod
    def current_verified_artifact_reads() -> dict[str, tuple[str, int]]:
        """Return a detached copy of invocation-pinned artifact read metadata."""

        return dict(_VERIFIED_ARTIFACT_READS.get() or {})

    @classmethod
    def path(cls, rel: str) -> str:
        """
        Convert a relative path to an S3 URL or local session path.

        Explicit S3 and local paths are returned unchanged. Relative paths are
        resolved against the active backend.

        Args:
            rel: Relative path or absolute S3 URL

        Returns:
            str: Full S3 URL or local path

        Examples:
        ---------
        >>> S3.path("data.csv")
        's3://chatbot-assets/sessions/20250121-123456-abc123/data.csv'

        >>> S3.path("s3://mybucket/data.csv")
        's3://mybucket/data.csv'
        """
        if cls._is_s3_url(rel) or cls._is_explicit_local_path(rel):
            return rel

        config = get_s3_config()
        if not is_s3_enabled():
            return os.fspath(cls._local_session_path(rel))

        key = PurePosixPath(cls.current_prefix().strip("/")) / str(rel).lstrip("/")
        return f"s3://{config.bucket_name}/{key.as_posix()}"

    @classmethod
    def open(cls, rel: str, mode: str = "rb"):
        """
        Open a file for reading or writing.

        Supports three types of paths:
        1. Absolute S3 URLs (s3://...)
        2. Local absolute paths (/ or file://)
        3. Relative paths (scoped to current session)

        Args:
            rel: File path (relative, absolute, or S3 URL)
            mode: File mode ('r', 'w', 'rb', 'wb', etc.)

        Returns:
            File-like object opened with the specified mode

        Examples:
        ---------
        >>> # Read from session-scoped S3 path
        >>> with S3.open("data.csv", "r") as f:
        ...     df = pd.read_csv(f)

        >>> # Write to session-scoped S3 path
        >>> with S3.open("output.csv", "w") as f:
        ...     df.to_csv(f)

        >>> # Read from absolute S3 URL
        >>> with S3.open("s3://bucket/data.csv", "r") as f:
        ...     df = pd.read_csv(f)

        >>> # Read from local file
        >>> with S3.open("/tmp/data.csv", "r") as f:
        ...     df = pd.read_csv(f)
        """
        config = get_s3_config()
        write_mode = cls._is_write_mode(mode)
        if not write_mode:
            read_key = cls._verified_read_key(rel)
            staged = (
                _STAGED_ARTIFACT_WRITES.get().get(read_key)
                if read_key is not None and _STAGED_ARTIFACT_WRITES.get() is not None
                else None
            )
            if staged is not None:
                return staged._open_staged_read(mode)
            expected_reads = _VERIFIED_ARTIFACT_READS.get() or {}
            expected = expected_reads.get(read_key) if read_key else None
            if read_key is not None and expected is not None:
                return cls._open_verified_snapshot(
                    read_key,
                    mode,
                    expected_sha256=expected[0],
                    expected_size=expected[1],
                )

        write_boundary = cls.current_write_boundary() if write_mode else None
        scoped_write_key: str | None = None
        if write_boundary is not None:
            scoped_write_key = cls._validate_scoped_write(rel, boundary=write_boundary)
        storage_target = rel
        staging_id = _EXTERNAL_STAGING_ID.get()
        publications = _EXTERNAL_STAGED_PUBLICATIONS.get()
        if scoped_write_key is not None and staging_id is not None and publications is not None:
            relative_to_boundary = PurePosixPath(scoped_write_key).relative_to(
                PurePosixPath(write_boundary or "")
            )
            staged_key = PurePosixPath(
                write_boundary or "",
                ".staging",
                staging_id,
                relative_to_boundary,
            ).as_posix()
            publications[scoped_write_key] = staged_key
            storage_target = staged_key

        if cls._is_s3_url(storage_target):
            if scoped_write_key is None:
                return fsspec.open(
                    storage_target,
                    mode=mode,
                    **config.to_storage_options(),
                )
            return cls._open_confined_s3_path(
                storage_target,
                mode,
                storage_options=config.to_storage_options(),
                public_write_key=scoped_write_key,
            )

        if cls._is_explicit_local_path(storage_target):
            if storage_target.startswith("file://"):
                return fsspec.open(storage_target, mode=mode)
            return cls._open_local_path(Path(storage_target), mode)

        if not is_s3_enabled():
            return cls._open_local_path(
                cls._local_session_path(storage_target),
                mode,
                scoped_write_key=scoped_write_key,
            )

        if scoped_write_key is None:
            return fsspec.open(
                cls.path(storage_target),
                mode=mode,
                **config.to_storage_options(),
            )
        return cls._open_confined_s3_path(
            cls.path(storage_target),
            mode,
            storage_options=config.to_storage_options(),
            public_write_key=scoped_write_key,
        )

    @classmethod
    def _verified_read_key(cls, path: str) -> str | None:
        """Resolve a public storage spelling to a session-relative artifact key."""

        if not isinstance(path, str) or not path.strip():
            return None
        candidate = path.strip()
        try:
            if cls._is_s3_url(candidate):
                parsed = urlsplit(candidate)
                config = get_s3_config()
                prefix = cls.current_prefix().strip("/")
                object_path = unquote(parsed.path).lstrip("/")
                expected_prefix = f"{prefix}/"
                if (
                    parsed.query
                    or parsed.fragment
                    or parsed.netloc != config.bucket_name
                    or not object_path.startswith(expected_prefix)
                ):
                    return None
                return cls._normalize_relative_path(object_path[len(expected_prefix) :])
            if cls._is_explicit_local_path(candidate):
                local_value = (
                    unquote(urlsplit(candidate).path)
                    if candidate.startswith("file://")
                    else candidate
                )
                local_path = Path(local_value).absolute()
                session_root = cls._local_session_path("").absolute()
                return cls._normalize_relative_path(local_path.relative_to(session_root).as_posix())
            expanded_prefix = PurePosixPath(
                LOCAL_STORAGE_ROOT.as_posix(),
                cls.current_prefix(),
            )
            candidate_path = PurePosixPath(candidate)
            try:
                expanded_relative = candidate_path.relative_to(expanded_prefix)
            except ValueError:
                pass
            else:
                return cls._normalize_relative_path(expanded_relative.as_posix())
            return cls._normalize_relative_path(candidate)
        except (ValueError, OSError):
            return None

    @classmethod
    def _open_verified_snapshot(
        cls,
        key: str,
        mode: str,
        *,
        expected_sha256: str,
        expected_size: int,
    ):
        """Copy one artifact once, verify it, and return the pinned byte snapshot."""

        if mode not in {"r", "rt", "rb"}:
            raise PermissionError("verified artifacts are available only in read mode")
        if is_s3_enabled():
            config = get_s3_config()
            source = fsspec.open(
                cls.path(key),
                mode="rb",
                **config.to_storage_options(),
            ).open()
        else:
            source = cls._open_local_verified_source(key)

        snapshot = tempfile.SpooledTemporaryFile(max_size=16 * 1024 * 1024, mode="w+b")
        digest = hashlib.sha256()
        size = 0
        try:
            with source:
                while True:
                    chunk = source.read(1024 * 1024)
                    if not chunk:
                        break
                    if isinstance(chunk, str):
                        chunk = chunk.encode("utf-8")
                    size += len(chunk)
                    if size > expected_size:
                        raise PermissionError(
                            f"registered artifact {key!r} exceeded its expected size"
                        )
                    digest.update(chunk)
                    snapshot.write(chunk)
            if size != expected_size or digest.hexdigest() != expected_sha256:
                raise PermissionError(f"registered artifact {key!r} failed checksum verification")
            snapshot.seek(0)
            if "b" in mode:
                return snapshot
            return io.TextIOWrapper(snapshot, encoding="utf-8")
        except BaseException:
            snapshot.close()
            raise

    @classmethod
    def _open_local_verified_source(cls, key: str):
        """Descriptor-open a registered local artifact without following links."""

        if not _OPEN_SUPPORTS_DIR_FD or not _O_NOFOLLOW:
            raise RuntimeError("secure local reads require directory-relative no-follow support")
        relative = PurePosixPath(cls._normalize_relative_path(key))
        common_flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | _O_NOFOLLOW
        directory_flags = common_flags | getattr(os, "O_DIRECTORY", 0)
        current_fd: int | None = None
        file_fd: int | None = None
        try:
            current_fd = _open_existing_directory_tree(cls._local_session_path("").absolute())
            for component in relative.parts[:-1]:
                next_fd = os.open(component, directory_flags, dir_fd=current_fd)
                os.close(current_fd)
                current_fd = next_fd
            file_fd = os.open(relative.parts[-1], common_flags, dir_fd=current_fd)
            metadata = os.fstat(file_fd)
            if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
                raise PermissionError("registered artifact must be a regular, non-linked file")
            handle = os.fdopen(file_fd, "rb")
            file_fd = None
            return handle
        except OSError as exc:
            if exc.errno in {errno.ELOOP, errno.ENOTDIR}:
                raise PermissionError(
                    "registered artifact contains a symlink or non-directory component"
                ) from exc
            raise
        finally:
            if file_fd is not None:
                os.close(file_fd)
            if current_fd is not None:
                os.close(current_fd)

    @classmethod
    def open_atomic(cls, rel: str, mode: str = "xb"):
        """Create one session-relative object that is invisible until complete."""

        if not cls._is_write_mode(mode):
            raise ValueError("open_atomic requires a write-capable mode")
        if cls._is_s3_url(rel) or cls._is_explicit_local_path(rel):
            raise ValueError("open_atomic requires a session-relative storage path")
        normalized = cls._normalize_relative_path(rel)
        config = get_s3_config()
        if not is_s3_enabled():
            return cls._open_confined_local_path(
                cls._local_session_path(normalized),
                mode,
                boundary="",
                scoped_write_key=normalized,
                defer_transaction=False,
            )
        return cls._open_confined_s3_path(
            cls.path(normalized),
            mode,
            storage_options=config.to_storage_options(),
            defer_transaction=False,
            public_write_key=None,
        )

    @staticmethod
    def _is_s3_url(path: str) -> bool:
        return isinstance(path, str) and path.startswith("s3://")

    @staticmethod
    def _is_explicit_local_path(path: str) -> bool:
        return isinstance(path, str) and (path.startswith("/") or path.startswith("file://"))

    @classmethod
    def _local_session_path(cls, rel: str) -> Path:
        return LOCAL_STORAGE_ROOT / Path(cls.current_prefix().strip("/")) / Path(rel)

    @staticmethod
    def _is_write_mode(mode: str) -> bool:
        return any(flag in mode for flag in ("w", "a", "x", "+"))

    @classmethod
    def _open_local_path(
        cls,
        path: Path,
        mode: str,
        *,
        scoped_write_key: str | None = None,
    ):
        if cls._is_write_mode(mode):
            boundary = cls.current_write_boundary()
            if boundary is not None:
                return cls._open_confined_local_path(
                    path,
                    mode,
                    boundary=boundary,
                    scoped_write_key=scoped_write_key,
                )
            path.parent.mkdir(parents=True, exist_ok=True)
        return builtins.open(path, mode)

    @classmethod
    def _validate_scoped_write(cls, rel: str, *, boundary: str) -> str:
        """Validate a write destination against the active session/run prefix."""

        if not isinstance(rel, str) or not rel.strip():
            raise PermissionError("write destination must be a non-empty storage path")
        candidate = rel.strip()
        if "\x00" in candidate or "\\" in candidate:
            raise PermissionError("write destination contains an unsafe path component")

        if cls._is_s3_url(candidate):
            parsed = urlsplit(candidate)
            expected = urlsplit(cls.path(boundary))
            decoded_path = unquote(parsed.path)
            expected_path = unquote(expected.path).rstrip("/")
            if (
                parsed.query
                or parsed.fragment
                or parsed.netloc != expected.netloc
                or parsed.scheme.lower() != "s3"
                or expected.scheme.lower() != "s3"
                or not decoded_path.startswith(f"{expected_path}/")
            ):
                raise PermissionError("S3 write destination is outside the active boundary")
            relative = decoded_path[len(expected_path) + 1 :]
            cls._normalize_relative_path(relative)
            cls._reject_reserved_run_path(relative)
            session_relative = (
                PurePosixPath(boundary, relative).as_posix() if boundary else relative
            )
            cls._reject_protected_write(session_relative)
            return session_relative

        if cls._is_explicit_local_path(candidate):
            raise PermissionError(
                "absolute and file:// write destinations are forbidden during scoped execution"
            )

        normalized = cls._normalize_relative_path(candidate)
        normalized_path = PurePosixPath(normalized)
        boundary_path = PurePosixPath(boundary) if boundary else PurePosixPath()
        try:
            relative = normalized_path.relative_to(boundary_path)
        except ValueError as exc:
            raise PermissionError("write destination is outside the active boundary") from exc
        cls._reject_reserved_run_path(relative.as_posix())
        cls._reject_protected_write(normalized)
        return normalized

    @staticmethod
    def _normalize_relative_path(value: str, *, allow_empty: bool = False) -> str:
        raw = str(value).strip()
        if allow_empty and not raw:
            return ""
        if raw.startswith("/"):
            raise ValueError("storage boundary must be a safe relative path")
        text = raw.rstrip("/") if allow_empty else raw
        candidate = PurePosixPath(text)
        if (
            not text
            or not candidate.parts
            or candidate.is_absolute()
            or "\\" in text
            or any(part in {"", ".", ".."} for part in candidate.parts)
        ):
            raise ValueError("storage boundary must be a safe relative path")
        return candidate.as_posix()

    @staticmethod
    def _reject_reserved_run_path(relative: str) -> None:
        path = PurePosixPath(relative)
        parts = path.parts
        if len(parts) >= 3 and parts[0] == "workflows":
            path = PurePosixPath(*parts[2:])
            parts = path.parts
        normalized = path.as_posix()
        if normalized in _RESERVED_RUN_FILES or (parts and parts[0] in {".staging", "events"}):
            raise PermissionError("write destination is reserved for workflow runtime metadata")

    @staticmethod
    def _reject_protected_write(session_relative: str) -> None:
        if session_relative in _WRITE_PROTECTED_PATHS.get():
            raise PermissionError("write destination is an immutable registered workflow artifact")

    @classmethod
    def _open_confined_s3_path(
        cls,
        path: str,
        mode: str,
        *,
        storage_options: dict,
        defer_transaction: bool = True,
        public_write_key: str | None = None,
    ):
        """Open one scoped S3 artifact with create-once semantics."""

        selected_mode = cls._exclusive_creation_mode(mode)
        deferred = defer_transaction and _PENDING_ARTIFACT_WRITES.get() is not None
        try:
            filesystem, filesystem_path = fsspec.core.url_to_fs(
                path,
                **storage_options,
            )
            handle = filesystem.open(
                filesystem_path,
                mode=selected_mode,
                autocommit=not deferred,
            )
        except (FileExistsError, IsADirectoryError) as exc:
            raise PermissionError(
                "scoped artifact writes are create-only; destination already exists"
            ) from exc
        except OSError as exc:
            if exc.errno == errno.EEXIST:
                raise PermissionError(
                    "scoped artifact writes are create-only; destination already exists"
                ) from exc
            raise
        return _AtomicS3CreateFile(
            handle,
            deferred=deferred,
            mode=selected_mode,
            staged_key=public_write_key or cls._verified_read_key(path),
        )

    @staticmethod
    def _exclusive_creation_mode(mode: str) -> str:
        """Translate the first scoped S3 write into an exclusive create."""

        if not isinstance(mode, str):
            raise ValueError("file mode must be a string")
        bases = [character for character in "rwax" if character in mode]
        if len(bases) != 1:
            raise ValueError(f"invalid file mode: {mode!r}")
        base = bases[0]
        if base == "x":
            return mode
        if base in {"w", "a"}:
            return mode.replace(base, "x", 1)
        raise PermissionError("scoped update modes cannot modify a pre-existing artifact")

    @classmethod
    def _open_confined_local_path(
        cls,
        path: Path,
        mode: str,
        *,
        boundary: str,
        scoped_write_key: str | None,
        defer_transaction: bool = True,
    ):
        """Open a local write by descriptor without following any symlink."""

        if not _OPEN_SUPPORTS_DIR_FD or not _O_NOFOLLOW:
            raise RuntimeError("secure local writes require directory-relative no-follow support")

        session_root = cls._local_session_path("").absolute()
        boundary_root = session_root.joinpath(*PurePosixPath(boundary).parts).absolute()
        destination = path.absolute()
        try:
            relative = destination.relative_to(boundary_root)
        except ValueError as exc:
            raise PermissionError("write destination is outside the active boundary") from exc
        if not relative.parts:
            raise PermissionError("write destination must identify a file")

        directory_flags = (
            os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_DIRECTORY", 0) | _O_NOFOLLOW
        )
        current_fd: int | None = None
        file_fd: int | None = None
        try:
            current_fd = cls._open_or_create_directory_tree(
                boundary_root,
                flags=directory_flags,
            )
            for component in relative.parts[:-1]:
                try:
                    os.mkdir(component, mode=0o700, dir_fd=current_fd)
                except FileExistsError:
                    pass
                next_fd = os.open(component, directory_flags, dir_fd=current_fd)
                os.close(current_fd)
                current_fd = next_fd

            if scoped_write_key is not None:
                destination_name = relative.parts[-1]
                temporary_name = f".{destination_name}.{uuid.uuid4().hex}.pending"
                flags, truncate = cls._write_open_flags(mode, exclusive_create=True)
                file_fd = os.open(temporary_name, flags, 0o600, dir_fd=current_fd)
            else:
                destination_name = relative.parts[-1]
                temporary_name = ""
                flags, truncate = cls._write_open_flags(mode)
                file_fd = os.open(destination_name, flags, 0o600, dir_fd=current_fd)
            metadata = os.fstat(file_fd)
            if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
                raise PermissionError(
                    "write destination must be a regular, non-linked artifact file"
                )
            if truncate:
                os.ftruncate(file_fd, 0)
            fdopen_mode = mode.replace("x", "w", 1) if "x" in mode else mode
            handle = os.fdopen(file_fd, fdopen_mode)
            file_fd = None
            if scoped_write_key is not None:
                owned_directory_fd = current_fd
                current_fd = None
                return _AtomicLocalCreateFile(
                    handle,
                    directory_fd=owned_directory_fd,
                    temporary_name=temporary_name,
                    destination_name=destination_name,
                    destination_path=destination,
                    staged_key=scoped_write_key,
                    deferred=(defer_transaction and _PENDING_ARTIFACT_WRITES.get() is not None),
                )
            return handle
        except OSError as exc:
            if exc.errno == errno.EEXIST:
                raise PermissionError(
                    "scoped artifact writes are create-only; destination already exists"
                ) from exc
            if exc.errno in {errno.ELOOP, errno.ENOTDIR}:
                raise PermissionError(
                    "write destination contains a symlink or non-directory component"
                ) from exc
            raise
        finally:
            if file_fd is not None:
                os.close(file_fd)
            if current_fd is not None:
                os.close(current_fd)

    @staticmethod
    def _open_or_create_directory_tree(path: Path, *, flags: int) -> int:
        """Return a descriptor for an absolute directory tree, creating it safely."""

        if not path.is_absolute():
            raise ValueError("secure directory traversal requires an absolute path")
        current_fd = os.open(path.anchor, flags)
        try:
            for component in path.parts[1:]:
                try:
                    os.mkdir(component, mode=0o700, dir_fd=current_fd)
                except FileExistsError:
                    pass
                next_fd = os.open(component, flags, dir_fd=current_fd)
                os.close(current_fd)
                current_fd = next_fd
            result = current_fd
            current_fd = -1
            return result
        finally:
            if current_fd >= 0:
                os.close(current_fd)

    @staticmethod
    def _write_open_flags(
        mode: str,
        *,
        exclusive_create: bool = False,
    ) -> tuple[int, bool]:
        """Translate a Python write mode into no-follow ``os.open`` flags."""

        if not isinstance(mode, str):
            raise ValueError("file mode must be a string")
        if any(character not in "rwaxbt+" for character in mode):
            raise ValueError(f"unsupported file mode: {mode!r}")
        bases = [character for character in "rwax" if character in mode]
        if len(bases) != 1 or ("b" in mode and "t" in mode):
            raise ValueError(f"invalid file mode: {mode!r}")

        base = bases[0]
        if base == "r" and "+" not in mode:
            raise ValueError("confined writer requires a write-capable file mode")
        access = os.O_RDWR if "+" in mode else os.O_WRONLY
        flags = access | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NONBLOCK", 0) | _O_NOFOLLOW
        truncate = False
        if base == "w":
            flags |= os.O_CREAT
            if exclusive_create:
                flags |= os.O_EXCL
            truncate = True
        elif base == "a":
            flags |= os.O_CREAT | os.O_APPEND
            if exclusive_create:
                flags |= os.O_EXCL
        elif base == "x":
            flags |= os.O_CREAT | os.O_EXCL
        elif base == "r" and exclusive_create:
            raise PermissionError("scoped update modes cannot modify a pre-existing artifact")
        return flags, truncate
