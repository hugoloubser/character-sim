"""Tests for the interaction repository module."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from pathlib import Path

from character_creator.core.interaction import (
    InMemoryInteractionRepository,
    InteractionRecord,
    SQLiteInteractionRepository,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_record() -> InteractionRecord:
    """Return a minimal populated InteractionRecord."""
    rec = InteractionRecord(
        interaction_id="test-001",
        scene_description="A quiet park bench.",
        topic="philosophy",
        characters=["Alice", "Kai"],
        provider="anthropic",
        model="claude-3-5-sonnet-20241022",
        max_exchanges=5,
    )
    rec.add_exchange("Alice", "What is freedom?", "curious", "I wonder…")
    rec.add_exchange("Kai", "Freedom is chaos!", "enthusiastic", "This is fun.")
    return rec


@pytest.fixture
def memory_repo() -> InMemoryInteractionRepository:
    """Return a fresh in-memory repo."""
    return InMemoryInteractionRepository()


@pytest.fixture
def sqlite_repo(tmp_path: Path) -> SQLiteInteractionRepository:
    """Return a SQLite repo backed by a temp file."""
    return SQLiteInteractionRepository(tmp_path / "test_interactions.db")


# ---------------------------------------------------------------------------
# InteractionRecord unit tests
# ---------------------------------------------------------------------------


class TestInteractionRecord:
    """Tests for the InteractionRecord dataclass."""

    def test_default_id_generated(self) -> None:
        """A UUID is assigned when no ID is provided."""
        rec = InteractionRecord()
        assert rec.interaction_id  # non-empty string

    def test_exchange_count(self, sample_record: InteractionRecord) -> None:
        """exchange_count returns number of exchanges added."""
        assert sample_record.exchange_count == 2

    def test_add_exchange(self) -> None:
        """add_exchange appends to exchanges list."""
        rec = InteractionRecord()
        rec.add_exchange("Bob", "Hello!", "neutral", "")
        assert len(rec.exchanges) == 1
        assert rec.exchanges[0]["speaker"] == "Bob"

    def test_finish_sets_timestamp(self) -> None:
        """finish() fills finished_at."""
        rec = InteractionRecord()
        assert rec.finished_at is None
        rec.finish()
        assert rec.finished_at is not None
        assert isinstance(rec.finished_at, datetime)

    def test_duration_none_before_finish(self) -> None:
        """duration_seconds is None while running."""
        rec = InteractionRecord()
        assert rec.duration_seconds is None

    def test_duration_after_finish(self) -> None:
        """duration_seconds returns a non-negative float."""
        rec = InteractionRecord()
        rec.finish()
        assert rec.duration_seconds is not None
        assert rec.duration_seconds >= 0

    def test_round_trip_serialization(self, sample_record: InteractionRecord) -> None:
        """to_dict/from_dict produce an equivalent record."""
        sample_record.finish()
        d = sample_record.to_dict()
        restored = InteractionRecord.from_dict(d)
        assert restored.interaction_id == sample_record.interaction_id
        assert restored.exchange_count == sample_record.exchange_count
        assert restored.characters == sample_record.characters
        assert restored.topic == sample_record.topic

    def test_from_dict_missing_fields(self) -> None:
        """from_dict handles missing optional fields gracefully."""
        rec = InteractionRecord.from_dict({})
        assert rec.interaction_id  # generated
        assert rec.exchange_count == 0


# ---------------------------------------------------------------------------
# InMemoryInteractionRepository tests
# ---------------------------------------------------------------------------


class TestInMemoryInteractionRepository:
    """Tests for the in-memory backend."""

    def test_save_and_get(
        self,
        memory_repo: InMemoryInteractionRepository,
        sample_record: InteractionRecord,
    ) -> None:
        """save then get returns the same record."""
        memory_repo.save(sample_record)
        fetched = memory_repo.get(sample_record.interaction_id)
        assert fetched is not None
        assert fetched.interaction_id == sample_record.interaction_id

    def test_get_missing(self, memory_repo: InMemoryInteractionRepository) -> None:
        """get returns None for unknown ID."""
        assert memory_repo.get("nonexistent") is None

    def test_list_all_empty(self, memory_repo: InMemoryInteractionRepository) -> None:
        """list_all on empty repo returns empty list."""
        assert memory_repo.list_all() == []

    def test_list_all_ordering(
        self, memory_repo: InMemoryInteractionRepository
    ) -> None:
        """list_all returns records newest first."""
        now = datetime.now(tz=UTC)
        r1 = InteractionRecord(
            interaction_id="r1", topic="first",
            started_at=now - timedelta(seconds=10),
        )
        r2 = InteractionRecord(
            interaction_id="r2", topic="second",
            started_at=now,
        )
        memory_repo.save(r1)
        memory_repo.save(r2)
        items = memory_repo.list_all()
        assert items[0].interaction_id == "r2"

    def test_list_by_character(
        self,
        memory_repo: InMemoryInteractionRepository,
        sample_record: InteractionRecord,
    ) -> None:
        """list_by_character filters correctly."""
        memory_repo.save(sample_record)
        assert len(memory_repo.list_by_character("Alice")) == 1
        assert len(memory_repo.list_by_character("Zoey")) == 0

    def test_delete(
        self,
        memory_repo: InMemoryInteractionRepository,
        sample_record: InteractionRecord,
    ) -> None:
        """delete removes the record and returns True."""
        memory_repo.save(sample_record)
        assert memory_repo.delete(sample_record.interaction_id) is True
        assert memory_repo.get(sample_record.interaction_id) is None

    def test_delete_missing(self, memory_repo: InMemoryInteractionRepository) -> None:
        """delete returns False for unknown ID."""
        assert memory_repo.delete("nonexistent") is False

    def test_count(
        self,
        memory_repo: InMemoryInteractionRepository,
        sample_record: InteractionRecord,
    ) -> None:
        """count returns the number of stored records."""
        assert memory_repo.count() == 0
        memory_repo.save(sample_record)
        assert memory_repo.count() == 1


# ---------------------------------------------------------------------------
# SQLiteInteractionRepository tests
# ---------------------------------------------------------------------------


class TestSQLiteInteractionRepository:
    """Tests for the SQLite backend."""

    def test_save_and_get(
        self,
        sqlite_repo: SQLiteInteractionRepository,
        sample_record: InteractionRecord,
    ) -> None:
        """save then get round-trips through SQLite."""
        sample_record.finish()
        sqlite_repo.save(sample_record)
        fetched = sqlite_repo.get(sample_record.interaction_id)
        assert fetched is not None
        assert fetched.interaction_id == sample_record.interaction_id
        assert fetched.exchange_count == 2
        assert fetched.characters == ["Alice", "Kai"]

    def test_get_missing(self, sqlite_repo: SQLiteInteractionRepository) -> None:
        """get returns None for unknown ID."""
        assert sqlite_repo.get("nonexistent") is None

    def test_list_all(
        self, sqlite_repo: SQLiteInteractionRepository
    ) -> None:
        """list_all returns all records."""
        r1 = InteractionRecord(interaction_id="s1", characters=["A"])
        r2 = InteractionRecord(interaction_id="s2", characters=["B"])
        r1.finish()
        r2.finish()
        sqlite_repo.save(r1)
        sqlite_repo.save(r2)
        assert sqlite_repo.count() == 2
        items = sqlite_repo.list_all()
        assert len(items) == 2

    def test_list_by_character(
        self,
        sqlite_repo: SQLiteInteractionRepository,
        sample_record: InteractionRecord,
    ) -> None:
        """list_by_character filters via LIKE on JSON array."""
        sample_record.finish()
        sqlite_repo.save(sample_record)
        assert len(sqlite_repo.list_by_character("Alice")) == 1
        assert len(sqlite_repo.list_by_character("Zoey")) == 0

    def test_delete(
        self,
        sqlite_repo: SQLiteInteractionRepository,
        sample_record: InteractionRecord,
    ) -> None:
        """delete removes the record."""
        sample_record.finish()
        sqlite_repo.save(sample_record)
        assert sqlite_repo.delete(sample_record.interaction_id) is True
        assert sqlite_repo.get(sample_record.interaction_id) is None

    def test_delete_missing(self, sqlite_repo: SQLiteInteractionRepository) -> None:
        """delete returns False for unknown ID."""
        assert sqlite_repo.delete("nonexistent") is False

    def test_upsert(
        self,
        sqlite_repo: SQLiteInteractionRepository,
        sample_record: InteractionRecord,
    ) -> None:
        """Saving twice with the same ID updates rather than duplicating."""
        sample_record.finish()
        sqlite_repo.save(sample_record)
        sample_record.add_exchange("Alice", "Interesting…", "thoughtful", "")
        sqlite_repo.save(sample_record)
        assert sqlite_repo.count() == 1
        fetched = sqlite_repo.get(sample_record.interaction_id)
        assert fetched is not None
        assert fetched.exchange_count == 3
