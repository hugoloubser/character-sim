"""Tests for the EmotionalState model and repository."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from pathlib import Path

from character_creator.core.emotional_state import (
    EmotionalState,
    InMemoryEmotionalStateRepository,
    SQLiteEmotionalStateRepository,
)

# ---------------------------------------------------------------------------
# EmotionalState model tests
# ---------------------------------------------------------------------------


class TestEmotionalState:
    """Tests for the EmotionalState Pydantic model."""

    def test_default_is_neutral(self) -> None:
        state = EmotionalState()
        assert state.base == "neutral"
        assert state.intensity == 0.5
        assert state.modifier is None

    def test_neutral_factory(self) -> None:
        state = EmotionalState.neutral()
        assert state.base == "neutral"

    def test_valid_base_states(self) -> None:
        for label in ("happy", "angry", "sad", "anxious", "confident", "vulnerable"):
            state = EmotionalState(base=label)
            assert state.base == label

    def test_invalid_base_rejected(self) -> None:
        with pytest.raises(ValueError, match="Unknown emotional state"):
            EmotionalState(base="ecstatic")

    def test_base_is_lowercased(self) -> None:
        state = EmotionalState(base="HAPPY")
        assert state.base == "happy"

    def test_base_is_stripped(self) -> None:
        state = EmotionalState(base="  sad  ")
        assert state.base == "sad"

    def test_intensity_bounds(self) -> None:
        EmotionalState(base="happy", intensity=0.0)
        EmotionalState(base="happy", intensity=1.0)
        with pytest.raises(ValueError):
            EmotionalState(base="happy", intensity=-0.1)
        with pytest.raises(ValueError):
            EmotionalState(base="happy", intensity=1.1)

    def test_modifier(self) -> None:
        state = EmotionalState(base="angry", modifier="frustrated")
        assert state.modifier == "frustrated"

    def test_str_without_modifier(self) -> None:
        assert str(EmotionalState(base="happy")) == "happy"

    def test_str_with_modifier(self) -> None:
        state = EmotionalState(base="angry", modifier="frustrated")
        assert str(state) == "angry:frustrated"

    def test_label_property(self) -> None:
        state = EmotionalState(base="sad", modifier="lonely")
        assert state.label == "sad:lonely"

    def test_self_perception_non_neutral(self) -> None:
        state = EmotionalState(base="happy")
        assert state.self_perception is not None
        assert "uplifted" in state.self_perception

    def test_self_perception_neutral_is_none(self) -> None:
        state = EmotionalState(base="neutral")
        assert state.self_perception is None

    def test_from_string_simple(self) -> None:
        state = EmotionalState.from_string("angry")
        assert state.base == "angry"
        assert state.modifier is None

    def test_from_string_compound(self) -> None:
        state = EmotionalState.from_string("angry:frustrated")
        assert state.base == "angry"
        assert state.modifier == "frustrated"

    def test_from_string_whitespace(self) -> None:
        state = EmotionalState.from_string("  happy  ")
        assert state.base == "happy"

    def test_to_dict(self) -> None:
        state = EmotionalState(base="sad", intensity=0.8, modifier="lonely")
        d = state.to_dict()
        assert d["base"] == "sad"
        assert d["intensity"] == 0.8
        assert d["modifier"] == "lonely"

    def test_roundtrip_model_validate(self) -> None:
        state = EmotionalState(base="confident", intensity=0.9, modifier="bold")
        d = state.to_dict()
        restored = EmotionalState.model_validate(d)
        assert restored == state


# ---------------------------------------------------------------------------
# In-memory repository tests
# ---------------------------------------------------------------------------


class TestInMemoryEmotionalStateRepository:
    """Tests for the in-memory emotional-state repository."""

    def test_seeded_by_default(self) -> None:
        repo = InMemoryEmotionalStateRepository()
        states = repo.list_states()
        assert "neutral" in states
        assert "happy" in states
        assert len(states) == 7

    def test_unseeded(self) -> None:
        repo = InMemoryEmotionalStateRepository(seed=False)
        assert repo.list_states() == []

    def test_add_and_exists(self) -> None:
        repo = InMemoryEmotionalStateRepository(seed=False)
        repo.add_state("excited", "I'm buzzing with energy!")
        assert repo.exists("excited")
        assert repo.get_perception("excited") == "I'm buzzing with energy!"

    def test_add_duplicate_raises(self) -> None:
        repo = InMemoryEmotionalStateRepository()
        with pytest.raises(ValueError, match="already registered"):
            repo.add_state("neutral")

    def test_remove(self) -> None:
        repo = InMemoryEmotionalStateRepository()
        assert repo.remove_state("happy")
        assert not repo.exists("happy")

    def test_remove_missing_returns_false(self) -> None:
        repo = InMemoryEmotionalStateRepository()
        assert not repo.remove_state("nonexistent")

    def test_get_perception_missing(self) -> None:
        repo = InMemoryEmotionalStateRepository()
        assert repo.get_perception("nonexistent") is None


# ---------------------------------------------------------------------------
# SQLite repository tests
# ---------------------------------------------------------------------------


class TestSQLiteEmotionalStateRepository:
    """Tests for the SQLite emotional-state repository."""

    @pytest.fixture
    def repo(self, tmp_path: Path) -> SQLiteEmotionalStateRepository:
        return SQLiteEmotionalStateRepository(db_path=tmp_path / "emotions.db")

    def test_seeded_on_init(self, repo: SQLiteEmotionalStateRepository) -> None:
        states = repo.list_states()
        assert "neutral" in states
        assert len(states) == 7

    def test_add_and_exists(self, repo: SQLiteEmotionalStateRepository) -> None:
        repo.add_state("excited", "I'm buzzing!")
        assert repo.exists("excited")
        assert repo.get_perception("excited") == "I'm buzzing!"

    def test_add_duplicate_raises(self, repo: SQLiteEmotionalStateRepository) -> None:
        with pytest.raises(ValueError, match="already registered"):
            repo.add_state("neutral")

    def test_remove(self, repo: SQLiteEmotionalStateRepository) -> None:
        assert repo.remove_state("happy")
        assert not repo.exists("happy")

    def test_remove_missing(self, repo: SQLiteEmotionalStateRepository) -> None:
        assert not repo.remove_state("nonexistent")

    def test_perception_for_missing(self, repo: SQLiteEmotionalStateRepository) -> None:
        assert repo.get_perception("nonexistent") is None

    def test_perception_for_neutral(self, repo: SQLiteEmotionalStateRepository) -> None:
        assert repo.get_perception("neutral") is None

    def test_perception_for_happy(self, repo: SQLiteEmotionalStateRepository) -> None:
        perception = repo.get_perception("happy")
        assert perception is not None
        assert "uplifted" in perception
