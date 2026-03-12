"""Centralized logging infrastructure with file-based sinks.

Provides:
    - ``setup_logging()`` — configures structlog + stdlib logging to write to
      both the console **and** timestamped files under ``logs/``.
    - ``LogAccessLayer`` — convenience wrapper for querying, tailing, and listing
      log files from the ``logs/`` directory.

Every log file is named ``{prefix}_{YYYYMMDD_HHMMSS}.log`` so they sort
chronologically and are easy to correlate to sessions.
"""

from __future__ import annotations

import logging
import re
import sys

from datetime import UTC, datetime
from pathlib import Path

import structlog

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

_DEFAULT_LOG_DIR = Path("logs")
_LOG_FORMAT = "%(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


# ---------------------------------------------------------------------------
# Setup helper
# ---------------------------------------------------------------------------


def setup_logging(
    *,
    log_dir: Path | str = _DEFAULT_LOG_DIR,
    level: int | str = logging.INFO,
    prefix: str = "session",
    capture_stdout: bool = True,
    capture_stderr: bool = True,
) -> Path:
    """Configure structured logging with file + console sinks.

    Args:
        log_dir: Directory to write log files into (created if needed).
        level: Logging level (int or string like ``"DEBUG"``).
        prefix: Filename prefix for the log file.
        capture_stdout: If True, tee stdout to the log file.
        capture_stderr: If True, tee stderr to the log file.

    Returns:
        Absolute path to the log file created for this session.

    """
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    ts = datetime.now(tz=UTC).strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"{prefix}_{ts}.log"

    # -- stdlib root logger ------------------------------------------------
    numeric_level = logging.getLevelName(level) if isinstance(level, str) else level
    root = logging.getLogger()
    root.setLevel(numeric_level)
    # Remove any pre-existing handlers to avoid duplicates
    root.handlers.clear()

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(numeric_level)
    console_handler.setFormatter(logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT))
    root.addHandler(console_handler)

    # File handler
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(numeric_level)
    file_handler.setFormatter(logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT))
    root.addHandler(file_handler)

    # -- structlog ---------------------------------------------------------
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.StackInfoRenderer(),
            structlog.dev.set_exc_info,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.dev.ConsoleRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(numeric_level),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )

    # -- stdout / stderr tee -----------------------------------------------
    if capture_stdout:
        sys.stdout = _TeeStream(sys.stdout, log_file)  # type: ignore[assignment]
    if capture_stderr:
        sys.stderr = _TeeStream(sys.stderr, log_file)  # type: ignore[assignment]

    return log_file.resolve()


# ---------------------------------------------------------------------------
# Tee helper — mirrors writes to both the original stream and the log file
# ---------------------------------------------------------------------------


class _TeeStream:
    """Write to both an original stream and a file simultaneously."""

    def __init__(self, original: object, log_path: Path) -> None:
        self._original = original
        self._log_path = log_path

    def write(self, data: str) -> int:
        """Write to original stream and append to log file."""
        if data:
            self._original.write(data)  # type: ignore[union-attr]
            with open(self._log_path, "a", encoding="utf-8") as f:
                f.write(data)
        return len(data) if data else 0

    def flush(self) -> None:
        """Flush the original stream."""
        self._original.flush()  # type: ignore[union-attr]

    # Forwarding so the object behaves enough like a real stream
    def fileno(self) -> int:
        """Return the original file descriptor."""
        return self._original.fileno()  # type: ignore[union-attr]

    @property
    def encoding(self) -> str:
        """Return the original stream encoding."""
        return getattr(self._original, "encoding", "utf-8")


# ---------------------------------------------------------------------------
# Log access layer — convenient querying of log files
# ---------------------------------------------------------------------------


class LogAccessLayer:
    """Convenient read-only access to log files in a directory.

    Usage::

        logs = LogAccessLayer()          # defaults to ./logs
        logs.list_sessions()             # sorted list of log files
        logs.latest()                    # path to newest log
        logs.tail("session_20260305_120000.log", n=20)  # last 20 lines
        logs.read("session_20260305_120000.log")         # full contents
        logs.search("ERROR")             # lines matching a pattern

    """

    def __init__(self, log_dir: Path | str = _DEFAULT_LOG_DIR) -> None:
        self._dir = Path(log_dir)

    # -- listing -----------------------------------------------------------

    def list_sessions(self, *, prefix: str | None = None) -> list[Path]:
        """Return log files sorted newest-first.

        Args:
            prefix: Optional filter — only files whose name starts with this.

        """
        if not self._dir.exists():
            return []
        files = sorted(
            self._dir.glob("*.log"),
            key=self._sort_key,
            reverse=True,
        )
        if prefix:
            files = [f for f in files if f.name.startswith(prefix)]
        return files

    def latest(self, *, prefix: str | None = None) -> Path | None:
        """Return the most recent log file, or None if none exist."""
        sessions = self.list_sessions(prefix=prefix)
        return sessions[0] if sessions else None

    # -- reading -----------------------------------------------------------

    def read(self, filename: str | Path) -> str:
        """Read the full contents of a log file.

        Args:
            filename: Name (or path) of the log file.

        """
        path = self._resolve(filename)
        return path.read_text(encoding="utf-8")

    def tail(self, filename: str | Path, n: int = 30) -> list[str]:
        """Return the last *n* lines of a log file.

        Args:
            filename: Name (or path) of the log file.
            n: Number of trailing lines.

        """
        path = self._resolve(filename)
        lines = path.read_text(encoding="utf-8").splitlines()
        return lines[-n:]

    def head(self, filename: str | Path, n: int = 30) -> list[str]:
        """Return the first *n* lines of a log file.

        Args:
            filename: Name (or path) of the log file.
            n: Number of leading lines.

        """
        path = self._resolve(filename)
        lines = path.read_text(encoding="utf-8").splitlines()
        return lines[:n]

    def search(self, pattern: str, *, filename: str | Path | None = None) -> list[str]:
        """Return lines containing *pattern* (case-insensitive).

        Args:
            pattern: Substring to search for.
            filename: If given, search only this file; otherwise search all.

        """
        target_files = [self._resolve(filename)] if filename else self.list_sessions()
        pat = pattern.lower()
        hits: list[str] = []
        for path in target_files:
            hits.extend(
                f"[{path.name}] {line}"
                for line in path.read_text(encoding="utf-8").splitlines()
                if pat in line.lower()
            )
        return hits

    # -- internal ----------------------------------------------------------

    @staticmethod
    def _sort_key(path: Path) -> str:
        """Extract a sort key from the log filename.

        Filenames are ``{prefix}_{YYYYMMDD_HHMMSS}.log``.  We extract the
        timestamp portion so sorting is chronological regardless of prefix.
        Falls back to the full stem for non-matching filenames.
        """
        stem = path.stem  # e.g. "session_20250305_120000"
        match = re.search(r"(\d{8}_\d{6})$", stem)
        return match.group(1) if match else stem

    def _resolve(self, filename: str | Path) -> Path:
        """Resolve a filename to a full path inside the log directory."""
        p = Path(filename)
        if p.is_absolute():
            return p
        # If the path already lives under _dir (e.g. returned by list_sessions),
        # don't prepend _dir again.
        try:
            p.resolve().relative_to(self._dir.resolve())
        except ValueError:
            return self._dir / p
        else:
            return p
