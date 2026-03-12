"""Emotional-state model and repository.

Provides a validated ``EmotionalState`` Pydantic model backed by a dedicated
database table.  The ``emotional_states`` table acts as the canonical registry
of allowed emotions; the ``characters`` table references it via a foreign key.
"""

from __future__ import annotations

import sqlite3
import threading

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, field_validator

from character_creator.core.constants import (
    DEFAULT_EMOTIONAL_STATE,
    EMOTIONAL_MODIFIERS,
    EMOTIONAL_STATES,
    TRAIT_DEFAULT,
    TRAIT_MAX,
    TRAIT_MIN,
)

# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


class EmotionalState(BaseModel):
    """A validated emotional state with optional intensity and modifier.

    Attributes:
        base: Primary emotion label (must be a recognised state).
        intensity: Strength of the emotion on a 0-1 scale.
        modifier: Optional qualifier (e.g. "frustrated", "hopeful").

    """

    base: str = DEFAULT_EMOTIONAL_STATE
    intensity: float = Field(TRAIT_DEFAULT, ge=TRAIT_MIN, le=TRAIT_MAX)
    modifier: str | None = None

    @field_validator("base")
    @classmethod
    def _valid_base(cls, v: str) -> str:
        v = v.strip().lower()
        if v not in EMOTIONAL_STATES:
            msg = f"Unknown emotional state: {v!r}. Must be one of {EMOTIONAL_STATES}"
            raise ValueError(msg)
        return v

    # -- derived properties ------------------------------------------------

    @property
    def self_perception(self) -> str | None:
        """First-person self-perception blurb, or *None* for neutral."""
        return EMOTIONAL_MODIFIERS.get(self.base)

    @property
    def label(self) -> str:
        """Human-readable label (e.g. ``'angry:frustrated'``)."""
        return str(self)

    # -- serialisation helpers ---------------------------------------------

    def __str__(self) -> str:
        """Compact string form understood by ``from_string``."""
        s = self.base
        if self.modifier:
            s += f":{self.modifier}"
        return s

    @classmethod
    def from_string(cls, raw: str) -> EmotionalState:
        """Parse a string like ``'angry'`` or ``'angry:frustrated'``.

        This preserves backward compatibility with the previous bare-string
        representation stored in the database and used throughout the UI.
        """
        parts = raw.strip().lower().split(":", maxsplit=1)
        base = parts[0]
        modifier = parts[1] if len(parts) > 1 else None
        return cls(base=base, modifier=modifier)

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a JSON-safe dict."""
        return self.model_dump()

    @classmethod
    def neutral(cls) -> EmotionalState:
        """Convenience factory for the default neutral state."""
        return cls()


# ---------------------------------------------------------------------------
# Abstract repository
# ---------------------------------------------------------------------------


class EmotionalStateRepository(ABC):
    """Abstract base for emotional-state storage."""

    @abstractmethod
    def list_states(self) -> list[str]:
        """Return all registered base-emotion labels."""

    @abstractmethod
    def add_state(self, label: str, self_perception: str | None = None) -> None:
        """Register a new base emotion."""

    @abstractmethod
    def remove_state(self, label: str) -> bool:
        """Remove a registered emotion; returns *True* if found."""

    @abstractmethod
    def exists(self, label: str) -> bool:
        """Check whether a base emotion is registered."""

    @abstractmethod
    def get_perception(self, label: str) -> str | None:
        """Return the self-perception blurb for *label*, or *None*."""


# ---------------------------------------------------------------------------
# In-memory implementation (dev / test)
# ---------------------------------------------------------------------------


class InMemoryEmotionalStateRepository(EmotionalStateRepository):
    """Thread-safe in-memory emotional-state store."""

    def __init__(self, *, seed: bool = True) -> None:
        """Initialise the store, optionally seeding canonical emotions.

        Args:
            seed: If *True* (default), pre-populate with ``EMOTIONAL_STATES``.

        """
        self._store: dict[str, str | None] = {}
        self._lock = threading.RLock()
        if seed:
            self._seed()

    def _seed(self) -> None:
        for label in EMOTIONAL_STATES:
            self._store[label] = EMOTIONAL_MODIFIERS.get(label)

    def list_states(self) -> list[str]:
        """Return all registered base-emotion labels."""
        with self._lock:
            return sorted(self._store)

    def add_state(self, label: str, self_perception: str | None = None) -> None:
        """Register a new base emotion."""
        with self._lock:
            if label in self._store:
                msg = f"Emotional state '{label}' already registered"
                raise ValueError(msg)
            self._store[label] = self_perception

    def remove_state(self, label: str) -> bool:
        """Remove a registered emotion; returns *True* if found."""
        with self._lock:
            if label in self._store:
                del self._store[label]
                return True
            return False

    def exists(self, label: str) -> bool:
        """Check whether a base emotion is registered."""
        with self._lock:
            return label in self._store

    def get_perception(self, label: str) -> str | None:
        """Return the self-perception blurb for *label*, or *None*."""
        with self._lock:
            return self._store.get(label)


# ---------------------------------------------------------------------------
# SQLite implementation (persistence)
# ---------------------------------------------------------------------------


class SQLiteEmotionalStateRepository(EmotionalStateRepository):
    """SQLite-backed emotional-state store.

    The table schema is::

        CREATE TABLE emotional_states (
            label TEXT PRIMARY KEY,
            self_perception TEXT
        );

    This table is designed to be referenced by a foreign key on the
    ``characters`` table (``current_emotional_state → emotional_states.label``).
    """

    def __init__(self, db_path: str | Path = "character_creator.db") -> None:
        """Initialise and ensure the schema exists.

        Args:
            db_path: Path to the SQLite database file.

        """
        self.db_path = Path(db_path)
        self._lock = threading.RLock()
        self._init_db()

    def _init_db(self) -> None:
        """Create the ``emotional_states`` table and seed canonical rows."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS emotional_states (
                    label TEXT PRIMARY KEY,
                    self_perception TEXT
                )
                """
            )
            # Seed canonical states (ignore duplicates)
            for label in EMOTIONAL_STATES:
                conn.execute(
                    "INSERT OR IGNORE INTO emotional_states (label, self_perception) VALUES (?, ?)",
                    (label, EMOTIONAL_MODIFIERS.get(label)),
                )
            conn.commit()

    # -- CRUD --------------------------------------------------------------

    def list_states(self) -> list[str]:
        """Return all registered base-emotion labels."""
        with self._lock, sqlite3.connect(self.db_path) as conn:
            rows = conn.execute(
                "SELECT label FROM emotional_states ORDER BY label"
            ).fetchall()
            return [r[0] for r in rows]

    def add_state(self, label: str, self_perception: str | None = None) -> None:
        """Register a new base emotion."""
        with self._lock, sqlite3.connect(self.db_path) as conn:
            try:
                conn.execute(
                    "INSERT INTO emotional_states (label, self_perception) VALUES (?, ?)",
                    (label, self_perception),
                )
                conn.commit()
            except sqlite3.IntegrityError as exc:
                msg = f"Emotional state '{label}' already registered"
                raise ValueError(msg) from exc

    def remove_state(self, label: str) -> bool:
        """Remove a registered emotion; returns *True* if found."""
        with self._lock, sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                "DELETE FROM emotional_states WHERE label = ?", (label,)
            )
            conn.commit()
            return cursor.rowcount > 0

    def exists(self, label: str) -> bool:
        """Check whether a base emotion is registered."""
        with self._lock, sqlite3.connect(self.db_path) as conn:
            row = conn.execute(
                "SELECT 1 FROM emotional_states WHERE label = ? LIMIT 1", (label,)
            ).fetchone()
            return row is not None

    def get_perception(self, label: str) -> str | None:
        """Return the self-perception blurb for *label*, or *None*."""
        with self._lock, sqlite3.connect(self.db_path) as conn:
            row = conn.execute(
                "SELECT self_perception FROM emotional_states WHERE label = ?", (label,)
            ).fetchone()
            return row[0] if row else None
