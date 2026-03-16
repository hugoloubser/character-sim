"""Tests for experience-triggered personality micro-shifts."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from character_creator.core.character import Character
from character_creator.core.constants import (
    MAX_TRAIT_DELTA_PER_EXCHANGE,
    TRAIT_MAX,
    TRAIT_MIN,
)
from character_creator.core.personality import Personality, PersonalityTraits, Values
from character_creator.core.trait_evolution import (
    EXPERIENCE_INFLUENCE_VECTORS,
    ExperienceClassifier,
    ExperienceType,
    TraitDelta,
    TraitHistory,
    TraitShiftEngine,
    apply_trait_delta,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_character(name: str = "Alice", **traits_overrides) -> Character:
    return Character(
        name=name,
        description="Test character",
        personality=Personality(
            traits=PersonalityTraits(**traits_overrides),
            values=Values(priority_keywords=["honesty", "courage"]),
        ),
    )


def _mock_llm(response: str = "neutral") -> MagicMock:
    provider = MagicMock()
    provider.generate = AsyncMock(return_value=response)
    return provider


class TestExperienceType:
    def test_all_enum_values(self) -> None:
        expected = {
            "conflict", "vulnerability", "success", "rejection",
            "connection", "betrayal", "discovery", "loss", "humor",
            "humiliation", "triumph", "forgiveness", "loneliness",
            "inspiration", "gratitude", "neutral",
        }
        assert {e.value for e in ExperienceType} == expected

    def test_all_types_have_influence_vectors(self) -> None:
        for exp in ExperienceType:
            assert exp in EXPERIENCE_INFLUENCE_VECTORS


class TestInfluenceVectors:
    def test_conflict_increases_assertiveness(self) -> None:
        vec = EXPERIENCE_INFLUENCE_VECTORS[ExperienceType.CONFLICT]
        assert vec["assertiveness"] > 0

    def test_conflict_decreases_agreeableness(self) -> None:
        vec = EXPERIENCE_INFLUENCE_VECTORS[ExperienceType.CONFLICT]
        assert vec["agreeableness"] < 0

    def test_connection_increases_warmth(self) -> None:
        vec = EXPERIENCE_INFLUENCE_VECTORS[ExperienceType.CONNECTION]
        assert vec["warmth"] > 0

    def test_neutral_has_no_deltas(self) -> None:
        vec = EXPERIENCE_INFLUENCE_VECTORS[ExperienceType.NEUTRAL]
        assert vec == {}

    def test_all_deltas_within_cap(self) -> None:
        for vec in EXPERIENCE_INFLUENCE_VECTORS.values():
            for delta in vec.values():
                assert abs(delta) <= MAX_TRAIT_DELTA_PER_EXCHANGE


class TestTraitDelta:
    def test_construction(self) -> None:
        td = TraitDelta(
            trait_name="warmth",
            delta=0.02,
            source=ExperienceType.CONNECTION,
            exchange_index=3,
        )
        assert td.trait_name == "warmth"
        assert td.delta == 0.02
        assert td.source == ExperienceType.CONNECTION
        assert td.exchange_index == 3
        assert td.timestamp is not None


class TestTraitHistory:
    def test_record_delta(self) -> None:
        history = TraitHistory(character_name="Alice")
        td = TraitDelta(
            trait_name="warmth", delta=0.01,
            source=ExperienceType.CONNECTION, exchange_index=0,
        )
        history.record_delta(td)
        assert len(history.deltas) == 1

    def test_record_snapshot(self) -> None:
        history = TraitHistory(character_name="Alice")
        history.record_snapshot({"warmth": 0.7, "openness": 0.5})
        assert len(history.snapshots) == 1
        assert history.snapshots[0]["warmth"] == 0.7

    def test_net_deltas(self) -> None:
        history = TraitHistory(character_name="Alice")
        history.record_delta(TraitDelta(
            trait_name="warmth", delta=0.02,
            source=ExperienceType.CONNECTION, exchange_index=0,
        ))
        history.record_delta(TraitDelta(
            trait_name="warmth", delta=-0.01,
            source=ExperienceType.CONFLICT, exchange_index=1,
        ))
        history.record_delta(TraitDelta(
            trait_name="openness", delta=0.03,
            source=ExperienceType.DISCOVERY, exchange_index=2,
        ))
        net = history.net_deltas()
        assert net["warmth"] == pytest.approx(0.01)
        assert net["openness"] == pytest.approx(0.03)


class TestApplyTraitDelta:
    def test_basic_application(self) -> None:
        traits = {"warmth": 0.5, "openness": 0.5}
        actual = apply_trait_delta(traits, {"warmth": 0.02})
        assert traits["warmth"] == pytest.approx(0.52)
        assert actual["warmth"] == pytest.approx(0.02)

    def test_clamped_to_max(self) -> None:
        traits = {"warmth": 0.99}
        actual = apply_trait_delta(traits, {"warmth": 0.03})
        assert traits["warmth"] == TRAIT_MAX
        assert actual["warmth"] == pytest.approx(0.01)

    def test_clamped_to_min(self) -> None:
        traits = {"warmth": 0.01}
        actual = apply_trait_delta(traits, {"warmth": -0.03})
        assert traits["warmth"] == TRAIT_MIN
        assert actual["warmth"] == pytest.approx(-0.01)

    def test_delta_capped_per_exchange(self) -> None:
        traits = {"warmth": 0.5}
        actual = apply_trait_delta(traits, {"warmth": 0.1})
        # Should be capped at MAX_TRAIT_DELTA_PER_EXCHANGE
        assert actual["warmth"] == pytest.approx(MAX_TRAIT_DELTA_PER_EXCHANGE)

    def test_unknown_trait_ignored(self) -> None:
        traits = {"warmth": 0.5}
        actual = apply_trait_delta(traits, {"nonexistent": 0.02})
        assert actual == {}
        assert traits == {"warmth": 0.5}

    def test_no_change_returns_empty(self) -> None:
        traits = {"warmth": 0.0}
        actual = apply_trait_delta(traits, {"warmth": -0.02})
        # Already at min, no change
        assert actual == {}


class TestExperienceClassifier:
    @pytest.mark.asyncio
    async def test_llm_classification(self) -> None:
        provider = _mock_llm("conflict")
        classifier = ExperienceClassifier(provider)
        character = _make_character()

        result = await classifier.classify(character, "I disagree!", "frustrated")
        assert result == ExperienceType.CONFLICT

    @pytest.mark.asyncio
    async def test_fallback_to_heuristic(self) -> None:
        provider = MagicMock()
        provider.generate = AsyncMock(side_effect=OSError("down"))
        classifier = ExperienceClassifier(provider)
        character = _make_character()

        result = await classifier.classify(character, "I disagree and argue", "angry")
        # Heuristic should pick up conflict keywords
        assert result == ExperienceType.CONFLICT

    def test_heuristic_conflict(self) -> None:
        result = ExperienceClassifier.classify_heuristic("We argue constantly", "angry")
        assert result == ExperienceType.CONFLICT

    def test_heuristic_connection(self) -> None:
        result = ExperienceClassifier.classify_heuristic("I feel connected to you", "warm")
        assert result == ExperienceType.CONNECTION

    def test_heuristic_neutral_fallback(self) -> None:
        result = ExperienceClassifier.classify_heuristic("The weather is nice", "calm")
        assert result == ExperienceType.NEUTRAL

    def test_heuristic_humor(self) -> None:
        result = ExperienceClassifier.classify_heuristic("That was hilarious, I laughed so hard", "happy")
        assert result == ExperienceType.HUMOR

    @pytest.mark.asyncio
    async def test_invalid_llm_response_falls_back(self) -> None:
        provider = _mock_llm("not_a_valid_type")
        classifier = ExperienceClassifier(provider)
        character = _make_character()

        result = await classifier.classify(character, "hello", "neutral")
        # Invalid enum value → fallback to heuristic → NEUTRAL
        assert isinstance(result, ExperienceType)


class TestTraitShiftEngine:
    @pytest.mark.asyncio
    async def test_process_exchange_conflict(self) -> None:
        provider = _mock_llm("conflict")
        engine = TraitShiftEngine(provider)
        character = _make_character(assertiveness=0.5, agreeableness=0.5)

        deltas = await engine.process_exchange(character, "I disagree!", "frustrated", 0)

        assert len(deltas) > 0
        # Assertiveness should have gone up
        assert character.personality.traits.assertiveness > 0.5
        # Agreeableness should have gone down
        assert character.personality.traits.agreeableness < 0.5

    @pytest.mark.asyncio
    async def test_process_exchange_neutral_no_deltas(self) -> None:
        provider = _mock_llm("neutral")
        engine = TraitShiftEngine(provider)
        character = _make_character()

        deltas = await engine.process_exchange(character, "Ok", "calm", 0)
        assert deltas == []

    @pytest.mark.asyncio
    async def test_history_recorded(self) -> None:
        provider = _mock_llm("connection")
        engine = TraitShiftEngine(provider)
        character = _make_character()

        await engine.process_exchange(character, "I feel close to you", "warm", 0)
        history = engine.get_history("Alice")

        assert len(history.deltas) > 0
        assert all(d.source == ExperienceType.CONNECTION for d in history.deltas)

    @pytest.mark.asyncio
    async def test_trait_bounds_respected(self) -> None:
        provider = _mock_llm("conflict")
        engine = TraitShiftEngine(provider)
        # Start with traits near bounds
        character = _make_character(assertiveness=0.99, agreeableness=0.01)

        await engine.process_exchange(character, "Fight!", "angry", 0)

        assert character.personality.traits.assertiveness <= TRAIT_MAX
        assert character.personality.traits.agreeableness >= TRAIT_MIN

    @pytest.mark.asyncio
    async def test_multiple_exchanges_accumulate(self) -> None:
        provider = _mock_llm("conflict")
        engine = TraitShiftEngine(provider)
        character = _make_character(assertiveness=0.5)
        original = character.personality.traits.assertiveness

        for i in range(5):
            await engine.process_exchange(character, "Argue!", "angry", i)

        assert character.personality.traits.assertiveness > original
        history = engine.get_history("Alice")
        assert len(history.deltas) >= 5  # At least one delta per exchange


class TestPromptTemplate:
    def test_experience_classification_substitution(self) -> None:
        from character_creator.llm.prompts import TraitEvolutionPrompts, substitute_prompt

        result = substitute_prompt(
            TraitEvolutionPrompts.EXPERIENCE_CLASSIFICATION,
            character_name="Alice",
            emotional_context="frustrated",
            exchange_text="I disagree!",
            experience_types="conflict, neutral",
        )
        assert "Alice" in result
        assert "frustrated" in result
        assert "I disagree!" in result
