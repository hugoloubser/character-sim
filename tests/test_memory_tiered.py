"""Tests for the tiered memory system: models, decay, condensation, and integration."""

from __future__ import annotations

import json

from unittest.mock import AsyncMock, MagicMock

import pytest

from character_creator.core.character import Character
from character_creator.core.constants import (
    CORE_MEMORY_FLOOR,
    CORE_MEMORY_SALIENCE,
    DECAY_EMOTIONAL_ANCHOR,
    DEFAULT_DECAY_RATE,
    LONG_TERM_CAPACITY,
    MAX_DISTORTION_BUDGET,
    MIN_SALIENCE_THRESHOLD,
    SHORT_TERM_CAPACITY,
    WORKING_MEMORY_CAPACITY,
)
from character_creator.core.memory_tiered import (
    CondensedMemory,
    MemoryCondenser,
    MemoryStore,
    MemoryTier,
)
from character_creator.core.personality import Personality, Values

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_character(name: str = "Alice", **kwargs) -> Character:
    """Create a minimal test character."""
    return Character(
        name=name,
        description="Test character",
        personality=Personality(
            values=Values(priority_keywords=["honesty", "courage"]),
        ),
        **kwargs,
    )


def _make_condensed(
    tier: MemoryTier = MemoryTier.SHORT_TERM,
    salience: float = 0.5,
    current_salience: float | None = None,
    topics: list[str] | None = None,
    **kwargs,
) -> CondensedMemory:
    """Create a condensed memory with sensible defaults."""
    return CondensedMemory(
        tier=tier,
        summary="Test memory",
        salience=salience,
        current_salience=current_salience if current_salience is not None else salience,
        topics=topics or [],
        **kwargs,
    )


def _mock_llm_provider(response_data: dict | None = None) -> MagicMock:
    """Create a mock LLM provider that returns a JSON string."""
    provider = MagicMock()
    data = response_data or {
        "summary": "Summary from LLM",
        "emotional_tone": "curious",
        "salience": 0.6,
        "topics": ["topic_a", "topic_b"],
        "perspective_bias": "slightly optimistic",
    }
    provider.generate = AsyncMock(return_value=json.dumps(data))
    return provider


# ===================================================================
# MemoryTier enum
# ===================================================================


class TestMemoryTier:
    def test_values(self) -> None:
        assert MemoryTier.WORKING == "working"
        assert MemoryTier.SHORT_TERM == "short_term"
        assert MemoryTier.LONG_TERM == "long_term"

    def test_membership(self) -> None:
        assert "working" in MemoryTier.__members__.values()


# ===================================================================
# CondensedMemory model
# ===================================================================


class TestCondensedMemory:
    def test_defaults(self) -> None:
        mem = CondensedMemory(tier=MemoryTier.SHORT_TERM, summary="hello")
        assert mem.tier == MemoryTier.SHORT_TERM
        assert mem.summary == "hello"
        assert mem.emotional_tone == "neutral"
        assert mem.salience == 0.5
        assert mem.current_salience == 0.5
        assert mem.perspective_bias is None
        assert mem.original_exchange_count == 0
        assert mem.source_speakers == []
        assert mem.topics == []
        assert mem.decay_rate == DEFAULT_DECAY_RATE
        assert mem.distortion_budget == 0.0
        assert mem.protected is False

    def test_uuid_is_unique(self) -> None:
        a = _make_condensed()
        b = _make_condensed()
        assert a.memory_id != b.memory_id

    def test_salience_bounds(self) -> None:
        with pytest.raises(ValueError):
            _make_condensed(salience=1.5)
        with pytest.raises(ValueError):
            _make_condensed(salience=-0.1)

    def test_to_dict(self) -> None:
        mem = _make_condensed(topics=["war", "peace"])
        d = mem.to_dict()
        assert isinstance(d, dict)
        assert d["summary"] == "Test memory"
        assert d["topics"] == ["war", "peace"]
        # datetimes should be ISO strings
        assert isinstance(d["created_at"], str)
        assert isinstance(d["last_accessed"], str)

    def test_protected_flag(self) -> None:
        mem = _make_condensed(salience=CORE_MEMORY_SALIENCE)
        assert mem.protected is False  # not set automatically
        mem.protected = True
        assert mem.protected is True


# ===================================================================
# MemoryStore model
# ===================================================================


class TestMemoryStore:
    def test_empty_defaults(self) -> None:
        store = MemoryStore()
        assert store.working == []
        assert store.short_term == []
        assert store.long_term == []
        assert store.ground_truth_log == []

    def test_add_exchange(self) -> None:
        store = MemoryStore()
        store.add_exchange("Alice", "Hello!")
        assert len(store.working) == 1
        assert store.working[0] == {"speaker": "Alice", "text": "Hello!"}
        # Ground truth is an independent copy
        assert len(store.ground_truth_log) == 1
        assert store.ground_truth_log[0] == {"speaker": "Alice", "text": "Hello!"}
        assert store.working[0] is not store.ground_truth_log[0]

    def test_working_is_full(self) -> None:
        store = MemoryStore()
        for i in range(WORKING_MEMORY_CAPACITY - 1):
            store.add_exchange("A", f"msg {i}")
        assert not store.working_is_full
        store.add_exchange("A", "last")
        assert store.working_is_full

    def test_short_term_is_full(self) -> None:
        store = MemoryStore()
        for _ in range(SHORT_TERM_CAPACITY):
            store.short_term.append(_make_condensed())
        assert store.short_term_is_full

    def test_get_working_text(self) -> None:
        store = MemoryStore()
        store.add_exchange("Alice", "Hi")
        store.add_exchange("Bob", "Hey")
        assert store.get_working_text() == "Alice: Hi\nBob: Hey"

    def test_get_short_term_summaries(self) -> None:
        store = MemoryStore()
        store.short_term.append(
            _make_condensed(emotional_tone="happy")
        )
        result = store.get_short_term_summaries()
        assert "Test memory" in result
        assert "happy" in result

    def test_get_short_term_summaries_empty(self) -> None:
        store = MemoryStore()
        assert store.get_short_term_summaries() == ""

    def test_get_long_term_summaries(self) -> None:
        store = MemoryStore()
        store.long_term.append(
            _make_condensed(tier=MemoryTier.LONG_TERM, emotional_tone="sad")
        )
        result = store.get_long_term_summaries()
        assert "sad" in result

    def test_ground_truth_immutability(self) -> None:
        """Ground truth should not be affected by clearing working memory."""
        store = MemoryStore()
        store.add_exchange("A", "msg1")
        store.add_exchange("B", "msg2")
        store.working.clear()
        assert len(store.ground_truth_log) == 2


# ===================================================================
# Character integration
# ===================================================================


class TestCharacterMemoryStoreIntegration:
    def test_character_has_memory_store(self) -> None:
        c = _make_character()
        assert isinstance(c.memory_store, MemoryStore)
        assert c.memory_store.working == []

    def test_add_memory_to_history_populates_both(self) -> None:
        c = _make_character()
        c.add_memory_to_history("Alice", "Hello")
        # Legacy conversation_history
        assert len(c.conversation_history) == 1
        # Tiered memory store
        assert len(c.memory_store.working) == 1
        assert len(c.memory_store.ground_truth_log) == 1

    def test_profile_includes_tiered_memory(self) -> None:
        c = _make_character()
        c.memory_store.short_term.append(
            _make_condensed(emotional_tone="reflective")
        )
        c.memory_store.long_term.append(
            _make_condensed(tier=MemoryTier.LONG_TERM, emotional_tone="wistful")
        )
        profile = c.get_character_profile()
        assert "Recent Memories" in profile
        assert "reflective" in profile
        assert "Long-Term Memories" in profile
        assert "wistful" in profile

    def test_profile_falls_back_to_conversation_history(self) -> None:
        """When memory_store.working is empty but conversation_history has data."""
        c = _make_character()
        # Directly manipulate legacy history (simulating old data)
        c.conversation_history.append({"speaker": "Bob", "text": "Legacy msg"})
        profile = c.get_character_profile()
        assert "Legacy msg" in profile

    def test_profile_prefers_working_memory(self) -> None:
        c = _make_character()
        c.memory_store.add_exchange("Alice", "Tiered msg")
        c.conversation_history.append({"speaker": "Bob", "text": "Legacy msg"})
        profile = c.get_character_profile()
        assert "Tiered msg" in profile


# ===================================================================
# Decay mechanics
# ===================================================================


class TestDecayMechanics:
    def test_basic_decay(self) -> None:
        character = _make_character()
        mem = _make_condensed(salience=0.5, current_salience=0.5)
        result = MemoryCondenser.apply_decay([mem], character)
        assert len(result) == 1
        assert result[0].current_salience < 0.5

    def test_decay_removes_below_threshold(self) -> None:
        character = _make_character()
        mem = _make_condensed(
            salience=0.05,
            current_salience=MIN_SALIENCE_THRESHOLD + 0.001,
        )
        # One decay pass should push it below MIN_SALIENCE_THRESHOLD
        mem.decay_rate = 0.1
        result = MemoryCondenser.apply_decay([mem], character)
        assert len(result) == 0

    def test_emotional_anchor_slows_decay(self) -> None:
        character = _make_character()
        high = _make_condensed(salience=DECAY_EMOTIONAL_ANCHOR + 0.1, current_salience=0.8)
        low = _make_condensed(salience=0.2, current_salience=0.8)

        [high_result] = MemoryCondenser.apply_decay([high], character)
        [low_result] = MemoryCondenser.apply_decay([low], character)

        # High-emotional memory should retain more salience
        assert high_result.current_salience > low_result.current_salience

    def test_value_anchor_slows_decay(self) -> None:
        character = _make_character()  # values: honesty, courage
        anchored = _make_condensed(current_salience=0.5, topics=["honesty"])
        unanchored = _make_condensed(current_salience=0.5, topics=["weather"])

        [a] = MemoryCondenser.apply_decay([anchored], character)
        [u] = MemoryCondenser.apply_decay([unanchored], character)

        assert a.current_salience > u.current_salience

    def test_core_memory_protection(self) -> None:
        character = _make_character()
        mem = _make_condensed(salience=CORE_MEMORY_SALIENCE, current_salience=CORE_MEMORY_SALIENCE)
        mem.decay_rate = 1.0  # Aggressive decay
        result = MemoryCondenser.apply_decay([mem], character)
        assert len(result) == 1
        assert result[0].current_salience >= CORE_MEMORY_FLOOR
        assert result[0].protected is True

    def test_protected_flag_persists(self) -> None:
        character = _make_character()
        mem = _make_condensed(salience=0.5, current_salience=0.6)
        mem.protected = True
        mem.decay_rate = 1.0
        result = MemoryCondenser.apply_decay([mem], character)
        assert len(result) == 1
        assert result[0].current_salience >= CORE_MEMORY_FLOOR

    def test_ground_truth_untouched(self) -> None:
        """Decay should have no effect on the ground truth log."""
        character = _make_character()
        store = MemoryStore()
        store.add_exchange("A", "msg1")
        store.short_term.append(_make_condensed(current_salience=0.05, decay_rate=1.0))
        # Apply decay directly
        store.short_term = MemoryCondenser.apply_decay(store.short_term, character)
        assert len(store.ground_truth_log) == 1


class TestCondensationLLM:
    @pytest.mark.asyncio
    async def test_condense_working_to_short_term(self) -> None:
        provider = _mock_llm_provider()
        condenser = MemoryCondenser(provider)
        character = _make_character()
        exchanges = [
            {"speaker": "Alice", "text": "Hello!"},
            {"speaker": "Bob", "text": "Hi there!"},
        ]

        result = await condenser.condense_working_to_short_term(character, exchanges)

        assert isinstance(result, CondensedMemory)
        assert result.tier == MemoryTier.SHORT_TERM
        assert result.summary == "Summary from LLM"
        assert result.emotional_tone == "curious"
        assert result.original_exchange_count == 2
        assert set(result.source_speakers) == {"Alice", "Bob"}
        provider.generate.assert_called_once()

    @pytest.mark.asyncio
    async def test_condense_working_heuristic_fallback(self) -> None:
        provider = MagicMock()
        provider.generate = AsyncMock(side_effect=OSError("LLM down"))
        condenser = MemoryCondenser(provider)
        character = _make_character()
        exchanges = [{"speaker": "Alice", "text": "Hello world!"}]

        result = await condenser.condense_working_to_short_term(character, exchanges)

        assert isinstance(result, CondensedMemory)
        assert "Alice" in result.summary
        assert result.emotional_tone == "neutral"  # from character's state

    @pytest.mark.asyncio
    async def test_condense_short_to_long_term(self) -> None:
        provider = _mock_llm_provider({
            "summary": "Long-term theme",
            "emotional_tone": "reflective",
            "salience": 0.7,
            "topics": ["friendship"],
            "perspective_bias": None,
        })
        condenser = MemoryCondenser(provider)
        character = _make_character()
        memories = [
            _make_condensed(original_exchange_count=5, source_speakers=["Alice"]),
            _make_condensed(original_exchange_count=3, source_speakers=["Bob"]),
        ]

        result = await condenser.condense_short_to_long_term(character, memories)

        assert result.tier == MemoryTier.LONG_TERM
        assert result.summary == "Long-term theme"
        assert result.original_exchange_count == 8
        assert set(result.source_speakers) == {"Alice", "Bob"}

    @pytest.mark.asyncio
    async def test_long_term_condense_tracks_distortion(self) -> None:
        provider = _mock_llm_provider()
        condenser = MemoryCondenser(provider)
        character = _make_character()
        memories = [
            _make_condensed(distortion_budget=1.0),
            _make_condensed(distortion_budget=1.0),
        ]

        result = await condenser.condense_short_to_long_term(character, memories)

        # avg existing (1.0) + 1.0 = 2.0
        assert result.distortion_budget == pytest.approx(2.0)

    @pytest.mark.asyncio
    async def test_distortion_budget_capped(self) -> None:
        provider = _mock_llm_provider()
        condenser = MemoryCondenser(provider)
        character = _make_character()
        memories = [
            _make_condensed(distortion_budget=MAX_DISTORTION_BUDGET),
        ]

        result = await condenser.condense_short_to_long_term(character, memories)

        assert result.distortion_budget <= MAX_DISTORTION_BUDGET


# ===================================================================
# Trigger condensation orchestrator
# ===================================================================


class TestTriggerCondensation:
    @pytest.mark.asyncio
    async def test_no_action_when_under_capacity(self) -> None:
        provider = _mock_llm_provider()
        condenser = MemoryCondenser(provider)
        character = _make_character()
        store = MemoryStore()
        store.add_exchange("A", "hi")

        await condenser.trigger_condensation(store, character)

        # No LLM calls — under capacity
        provider.generate.assert_not_called()
        assert len(store.working) == 1

    @pytest.mark.asyncio
    async def test_condenses_when_working_full(self) -> None:
        provider = _mock_llm_provider()
        condenser = MemoryCondenser(provider)
        character = _make_character()
        store = MemoryStore()
        for i in range(WORKING_MEMORY_CAPACITY):
            store.add_exchange("A", f"msg {i}")

        assert store.working_is_full
        await condenser.trigger_condensation(store, character)

        # Working memory cleared, one short-term memory created
        assert len(store.working) == 0
        assert len(store.short_term) == 1
        assert store.short_term[0].tier == MemoryTier.SHORT_TERM

    @pytest.mark.asyncio
    async def test_condenses_short_term_overflow(self) -> None:
        provider = _mock_llm_provider()
        condenser = MemoryCondenser(provider)
        character = _make_character()
        store = MemoryStore()
        for _ in range(SHORT_TERM_CAPACITY):
            store.short_term.append(_make_condensed())

        assert store.short_term_is_full
        await condenser.trigger_condensation(store, character)

        # Half should have been merged into one long-term memory
        assert len(store.short_term) < SHORT_TERM_CAPACITY
        assert len(store.long_term) == 1
        assert store.long_term[0].tier == MemoryTier.LONG_TERM

    @pytest.mark.asyncio
    async def test_ground_truth_survives_condensation(self) -> None:
        provider = _mock_llm_provider()
        condenser = MemoryCondenser(provider)
        character = _make_character()
        store = MemoryStore()
        for i in range(WORKING_MEMORY_CAPACITY):
            store.add_exchange("A", f"msg {i}")

        await condenser.trigger_condensation(store, character)

        # Ground truth should still have all entries
        assert len(store.ground_truth_log) == WORKING_MEMORY_CAPACITY
        # Working should be cleared
        assert len(store.working) == 0

    @pytest.mark.asyncio
    async def test_long_term_capped(self) -> None:
        provider = _mock_llm_provider()
        condenser = MemoryCondenser(provider)
        character = _make_character()
        store = MemoryStore()
        for _ in range(LONG_TERM_CAPACITY + 5):
            store.long_term.append(_make_condensed(tier=MemoryTier.LONG_TERM))

        await condenser.trigger_condensation(store, character)

        assert len(store.long_term) <= LONG_TERM_CAPACITY


# ===================================================================
# Prompt template coverage
# ===================================================================


class TestMemoryPromptTemplates:
    def test_working_to_short_term_substitution(self) -> None:
        from character_creator.llm.prompts import MemoryPrompts, substitute_prompt

        result = substitute_prompt(
            MemoryPrompts.WORKING_TO_SHORT_TERM,
            character_name="Alice",
            personality_summary="Warm and open",
            emotional_state="happy",
            core_values="honesty, courage",
            exchanges="Alice: Hello\nBob: Hi",
        )
        assert "Alice" in result
        assert "honesty" in result
        assert "Hello" in result

    def test_short_to_long_term_substitution(self) -> None:
        from character_creator.llm.prompts import MemoryPrompts, substitute_prompt

        result = substitute_prompt(
            MemoryPrompts.SHORT_TO_LONG_TERM,
            character_name="Alice",
            personality_summary="Warm and open",
            emotional_state="happy",
            core_values="honesty, courage",
            memories="- A discussion about trust (tone: curious)",
        )
        assert "Alice" in result
        assert "trust" in result
