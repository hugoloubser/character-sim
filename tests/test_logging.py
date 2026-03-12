"""Tests for the logging infrastructure."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from pathlib import Path

from character_creator.utils.logging import LogAccessLayer, setup_logging

# ---------------------------------------------------------------------------
# setup_logging tests
# ---------------------------------------------------------------------------


class TestSetupLogging:
    """Tests for the setup_logging function."""

    def test_creates_log_directory(self, tmp_path: Path) -> None:
        """setup_logging creates the log directory if missing."""
        log_dir = tmp_path / "newlogs"
        assert not log_dir.exists()
        setup_logging(log_dir=log_dir, capture_stdout=False, capture_stderr=False)
        assert log_dir.exists()

    def test_returns_log_file_path(self, tmp_path: Path) -> None:
        """Returned path points to a .log file inside the directory."""
        log_file = setup_logging(
            log_dir=tmp_path, prefix="test", capture_stdout=False, capture_stderr=False
        )
        assert log_file.suffix == ".log"
        assert log_file.parent == tmp_path
        assert "test_" in log_file.name

    def test_log_file_created_on_disk(self, tmp_path: Path) -> None:
        """The file is created (by the FileHandler) after setup."""
        log_file = setup_logging(
            log_dir=tmp_path, capture_stdout=False, capture_stderr=False
        )
        assert log_file.exists()

    def test_custom_prefix(self, tmp_path: Path) -> None:
        """Log file name starts with the given prefix."""
        log_file = setup_logging(
            log_dir=tmp_path, prefix="demo", capture_stdout=False, capture_stderr=False
        )
        assert log_file.name.startswith("demo_")


# ---------------------------------------------------------------------------
# LogAccessLayer tests
# ---------------------------------------------------------------------------


class TestLogAccessLayer:
    """Tests for the log access layer."""

    @pytest.fixture
    def populated_dir(self, tmp_path: Path) -> Path:
        """Create a log dir with two sample files."""
        (tmp_path / "session_20250101_100000.log").write_text(
            "line1\nline2\nline3\n", encoding="utf-8"
        )
        (tmp_path / "demo_20250102_120000.log").write_text(
            "alpha\nbeta\ngamma\ndelta\n", encoding="utf-8"
        )
        return tmp_path

    def test_list_sessions_sorted(self, populated_dir: Path) -> None:
        """list_sessions returns newest first."""
        layer = LogAccessLayer(populated_dir)
        sessions = layer.list_sessions()
        assert len(sessions) == 2
        assert sessions[0].name == "demo_20250102_120000.log"

    def test_list_sessions_with_prefix(self, populated_dir: Path) -> None:
        """Prefix filter narrows results."""
        layer = LogAccessLayer(populated_dir)
        assert len(layer.list_sessions(prefix="session")) == 1
        assert len(layer.list_sessions(prefix="demo")) == 1
        assert len(layer.list_sessions(prefix="unknown")) == 0

    def test_list_sessions_empty_dir(self, tmp_path: Path) -> None:
        """Empty directory returns empty list."""
        layer = LogAccessLayer(tmp_path)
        assert layer.list_sessions() == []

    def test_list_sessions_missing_dir(self, tmp_path: Path) -> None:
        """Non-existent directory returns empty list."""
        layer = LogAccessLayer(tmp_path / "nope")
        assert layer.list_sessions() == []

    def test_latest(self, populated_dir: Path) -> None:
        """latest returns the most recent log file."""
        layer = LogAccessLayer(populated_dir)
        latest = layer.latest()
        assert latest is not None
        assert latest.name == "demo_20250102_120000.log"

    def test_latest_none(self, tmp_path: Path) -> None:
        """latest returns None when no logs exist."""
        layer = LogAccessLayer(tmp_path)
        assert layer.latest() is None

    def test_read(self, populated_dir: Path) -> None:
        """read returns the full file content."""
        layer = LogAccessLayer(populated_dir)
        content = layer.read("session_20250101_100000.log")
        assert "line1" in content
        assert "line3" in content

    def test_tail(self, populated_dir: Path) -> None:
        """tail returns last n lines."""
        layer = LogAccessLayer(populated_dir)
        lines = layer.tail("demo_20250102_120000.log", n=2)
        assert lines == ["gamma", "delta"]

    def test_head(self, populated_dir: Path) -> None:
        """head returns first n lines."""
        layer = LogAccessLayer(populated_dir)
        lines = layer.head("demo_20250102_120000.log", n=2)
        assert lines == ["alpha", "beta"]

    def test_search_single_file(self, populated_dir: Path) -> None:
        """search finds matching lines in a specific file."""
        layer = LogAccessLayer(populated_dir)
        hits = layer.search("line", filename="session_20250101_100000.log")
        assert len(hits) == 3

    def test_search_all_files(self, populated_dir: Path) -> None:
        """search across all files finds matches."""
        layer = LogAccessLayer(populated_dir)
        hits = layer.search("alpha")
        assert len(hits) == 1
        assert "demo_20250102_120000.log" in hits[0]

    def test_search_case_insensitive(self, populated_dir: Path) -> None:
        """search is case-insensitive."""
        layer = LogAccessLayer(populated_dir)
        hits = layer.search("ALPHA")
        assert len(hits) == 1
