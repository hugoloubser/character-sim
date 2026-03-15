"""Interaction repository for persisting dialogue sessions.

Mirrors the CharacterRepository pattern — abstract base with in-memory
(dev/test) and SQLite (persistence) backends.  Each interaction captures
the full dialogue context including scene, participants, exchanges, and
metadata (timestamps, exchange count, provider used).
"""

from __future__ import annotations

import json
import sqlite3
import threading
import uuid

from abc import ABC, abstractmethod
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


class InteractionRecord(BaseModel):
    """Immutable snapshot of a dialogue session.

    Fields:
        interaction_id: Unique identifier (UUID).
        scene_description: The scene that was set for the interaction.
        topic: The conversational topic.
        characters: Names of characters involved.
        exchanges: List of exchange dicts (speaker, text, emotional_context, internal_thought).
        provider: LLM provider name used (e.g. "anthropic").
        model: Model identifier (e.g. "claude-3-5-sonnet-20241022").
        max_exchanges: Configured exchange cap.
        started_at: When the interaction began.
        finished_at: When the interaction ended (None while in progress).
        metadata: Arbitrary extra data (temperature, tokens, etc.).
    """

    interaction_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    scene_description: str = ""
    topic: str = ""
    characters: list[str] = Field(default_factory=list)
    exchanges: list[dict[str, str]] = Field(default_factory=list)
    provider: str = ""
    model: str = ""
    max_exchanges: int = 0
    started_at: datetime = Field(default_factory=lambda: datetime.now(tz=UTC))
    finished_at: datetime | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    # -- helpers -----------------------------------------------------------

    @property
    def exchange_count(self) -> int:
        """Number of exchanges recorded."""
        return len(self.exchanges)

    @property
    def duration_seconds(self) -> float | None:
        """Wall-clock duration in seconds, or None if still running."""
        if self.finished_at is None:
            return None
        return (self.finished_at - self.started_at).total_seconds()

    def add_exchange(
        self,
        speaker: str,
        text: str,
        emotional_context: str = "",
        internal_thought: str = "",
    ) -> None:
        """Append an exchange to the record."""
        self.exchanges.append(
            {
                "speaker": speaker,
                "text": text,
                "emotional_context": emotional_context,
                "internal_thought": internal_thought,
            }
        )

    def finish(self) -> None:
        """Mark the interaction as finished."""
        self.finished_at = datetime.now(tz=UTC)

    # -- serialization -----------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a plain dict (JSON-safe)."""
        d = self.model_dump()
        d["started_at"] = self.started_at.isoformat()
        d["finished_at"] = self.finished_at.isoformat() if self.finished_at else None
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> InteractionRecord:
        """Deserialize from a plain dict."""
        started = data.get("started_at")
        finished = data.get("finished_at")
        return cls.model_validate({
            **data,
            "started_at": (
                datetime.fromisoformat(started) if isinstance(started, str) else (started or datetime.now(tz=UTC))
            ),
            "finished_at": (
                datetime.fromisoformat(finished) if isinstance(finished, str) else finished
            ),
        })


# ---------------------------------------------------------------------------
# Abstract repository
# ---------------------------------------------------------------------------


class InteractionRepository(ABC):
    """Abstract base class for interaction storage backends."""

    @abstractmethod
    def save(self, record: InteractionRecord) -> None:
        """Save (insert or update) an interaction record."""

    @abstractmethod
    def get(self, interaction_id: str) -> InteractionRecord | None:
        """Retrieve a single interaction by ID."""

    @abstractmethod
    def list_all(self) -> list[InteractionRecord]:
        """List all interaction records, newest first."""

    @abstractmethod
    def list_by_character(self, character_name: str) -> list[InteractionRecord]:
        """List interactions involving a specific character."""

    @abstractmethod
    def delete(self, interaction_id: str) -> bool:
        """Delete an interaction, returning True if found."""

    @abstractmethod
    def count(self) -> int:
        """Return total number of stored interactions."""


# ---------------------------------------------------------------------------
# In-memory implementation (dev / test)
# ---------------------------------------------------------------------------


class InMemoryInteractionRepository(InteractionRepository):
    """Thread-safe in-memory interaction store."""

    def __init__(self) -> None:
        """Initialize empty store."""
        self._store: dict[str, InteractionRecord] = {}
        self._lock = threading.RLock()

    def save(self, record: InteractionRecord) -> None:
        """Save (insert or update) an interaction record."""
        with self._lock:
            self._store[record.interaction_id] = record

    def get(self, interaction_id: str) -> InteractionRecord | None:
        """Retrieve a single interaction by ID."""
        with self._lock:
            return self._store.get(interaction_id)

    def list_all(self) -> list[InteractionRecord]:
        """List all interaction records, newest first."""
        with self._lock:
            return sorted(
                self._store.values(),
                key=lambda r: r.started_at,
                reverse=True,
            )

    def list_by_character(self, character_name: str) -> list[InteractionRecord]:
        """List interactions involving a specific character."""
        with self._lock:
            return sorted(
                (r for r in self._store.values() if character_name in r.characters),
                key=lambda r: r.started_at,
                reverse=True,
            )

    def delete(self, interaction_id: str) -> bool:
        """Delete an interaction, returning True if found."""
        with self._lock:
            if interaction_id in self._store:
                del self._store[interaction_id]
                return True
            return False

    def count(self) -> int:
        """Return total number of stored interactions."""
        with self._lock:
            return len(self._store)


# ---------------------------------------------------------------------------
# SQLite implementation (persistence)
# ---------------------------------------------------------------------------


class SQLiteInteractionRepository(InteractionRepository):
    """SQLite-backed interaction store with the same interface."""

    def __init__(self, db_path: str | Path = "interactions.db") -> None:
        """Initialize and ensure schema exists.

        Args:
            db_path: Path to the SQLite database file.

        """
        self.db_path = Path(db_path)
        self._lock = threading.RLock()
        self._init_db()

    def _init_db(self) -> None:
        """Create the interactions table if it does not exist."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS interactions (
                    interaction_id TEXT PRIMARY KEY,
                    scene_description TEXT NOT NULL,
                    topic TEXT NOT NULL,
                    characters TEXT NOT NULL,
                    exchanges TEXT NOT NULL,
                    provider TEXT,
                    model TEXT,
                    max_exchanges INTEGER,
                    started_at TEXT NOT NULL,
                    finished_at TEXT,
                    metadata TEXT
                )
                """
            )
            conn.commit()

    # -- CRUD --------------------------------------------------------------

    def save(self, record: InteractionRecord) -> None:
        """Save (insert or update) an interaction record."""
        d = record.to_dict()
        with self._lock, sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO interactions
                    (interaction_id, scene_description, topic, characters,
                     exchanges, provider, model, max_exchanges,
                     started_at, finished_at, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    d["interaction_id"],
                    d["scene_description"],
                    d["topic"],
                    json.dumps(d["characters"]),
                    json.dumps(d["exchanges"]),
                    d["provider"],
                    d["model"],
                    d["max_exchanges"],
                    d["started_at"],
                    d["finished_at"],
                    json.dumps(d["metadata"]),
                ),
            )
            conn.commit()

    def get(self, interaction_id: str) -> InteractionRecord | None:
        """Retrieve a single interaction by ID."""
        with self._lock, sqlite3.connect(self.db_path) as conn:
            row = conn.execute(
                "SELECT * FROM interactions WHERE interaction_id = ?",
                (interaction_id,),
            ).fetchone()
            if row is None:
                return None
            return self._row_to_record(row)

    def list_all(self) -> list[InteractionRecord]:
        """List all interaction records, newest first."""
        with self._lock, sqlite3.connect(self.db_path) as conn:
            rows = conn.execute(
                "SELECT * FROM interactions ORDER BY started_at DESC"
            ).fetchall()
            return [self._row_to_record(r) for r in rows]

    def list_by_character(self, character_name: str) -> list[InteractionRecord]:
        """List interactions involving a specific character."""
        with self._lock, sqlite3.connect(self.db_path) as conn:
            # JSON array stored as text — use LIKE for portability
            rows = conn.execute(
                "SELECT * FROM interactions WHERE characters LIKE ? ORDER BY started_at DESC",
                (f'%"{character_name}"%',),
            ).fetchall()
            return [self._row_to_record(r) for r in rows]

    def delete(self, interaction_id: str) -> bool:
        """Delete an interaction, returning True if found."""
        with self._lock, sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                "DELETE FROM interactions WHERE interaction_id = ?",
                (interaction_id,),
            )
            conn.commit()
            return cursor.rowcount > 0

    def count(self) -> int:
        """Return total number of stored interactions."""
        with self._lock, sqlite3.connect(self.db_path) as conn:
            row = conn.execute("SELECT COUNT(*) FROM interactions").fetchone()
            return row[0] if row else 0

    # -- internal ----------------------------------------------------------

    @staticmethod
    def _row_to_record(row: tuple) -> InteractionRecord:
        """Convert a database row to an InteractionRecord."""
        return InteractionRecord.from_dict(
            {
                "interaction_id": row[0],
                "scene_description": row[1],
                "topic": row[2],
                "characters": json.loads(row[3]),
                "exchanges": json.loads(row[4]),
                "provider": row[5],
                "model": row[6],
                "max_exchanges": row[7],
                "started_at": row[8],
                "finished_at": row[9],
                "metadata": json.loads(row[10]) if row[10] else {},
            }
        )
