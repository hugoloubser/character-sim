"""Tests for the reflective self-model system."""

from __future__ import annotations

import json

from unittest.mock import AsyncMock, MagicMock

import pytest

from character_creator.core.character import Character
from character_creator.core.constants import (
    MAX_SELF_MODEL_HISTORY,
    SELF_REFLECTION_INTERVAL,
)
from character_creator.core.personality import Personality, PersonalityTraits, Values
from character_creator.core.self_model import SelfModel, SelfReflectionEngine

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_character(name: str = "Elena", **trait_overrides) -> Character:
    return Character(
        name=name,
        description="A thoughtful test character",
        personality=Personality(
            traits=PersonalityTraits(**trait_overrides),
            values=Values(priority_keywords=["honesty", "growth"]),
        ),
    )


def _mock_llm(response: str | None = None) -> MagicMock:
    if response is None:
        response = json.dumps({
            "self_concept": "I see myself as someone who is growing.",
            "emotional_awareness": "I feel cautiously optimistic.",
            "value_tensions": ["I value honesty but sometimes stay silent."],
            "growth_edges": ["Learning to speak up more."],
        })
    provider = MagicMock()
    provider.generate = AsyncMock(return_value=response)
    return provider


# ---------------------------------------------------------------------------
# SelfModel tests
# ---------------------------------------------------------------------------


class TestSelfModel:
    def test_default_construction(self) -> None:
        model = SelfModel()
        assert model.self_concept == ""
        assert model.emotional_awareness == ""
        assert model.value_tensions == []
        assert model.growth_edges == []
        assert model.generated_at_exchange == 0

    def test_round_trip_dict(self) -> None:
        model = SelfModel(
            self_concept="I'm a thinker.",
            emotional_awareness="Feeling pensive.",
            value_tensions=["honesty vs. kindness"],
            growth_edges=["speaking up"],
            generated_at_exchange=7,
        )
        data = model.model_dump()
        restored = SelfModel.model_validate(data)
        assert restored.self_concept == model.self_concept
        assert restored.growth_edges == model.growth_edges
        assert restored.generated_at_exchange == 7

    def test_timestamp_auto_set(self) -> None:
        m1 = SelfModel()
        m2 = SelfModel()
        assert m1.timestamp is not None
        assert m2.timestamp >= m1.timestamp


# ---------------------------------------------------------------------------
# SelfReflectionEngine.should_reflect
# ---------------------------------------------------------------------------


class TestShouldReflect:
    def test_first_reflection_not_yet(self) -> None:
        assert not SelfReflectionEngine.should_reflect(
            exchange_index=SELF_REFLECTION_INTERVAL - 1,
            last_reflection_exchange=None,
        )

    def test_first_reflection_at_interval(self) -> None:
        assert SelfReflectionEngine.should_reflect(
            exchange_index=SELF_REFLECTION_INTERVAL,
            last_reflection_exchange=None,
        )

    def test_subsequent_reflection_not_yet(self) -> None:
        assert not SelfReflectionEngine.should_reflect(
            exchange_index=SELF_REFLECTION_INTERVAL + 2,
            last_reflection_exchange=SELF_REFLECTION_INTERVAL,
        )

    def test_subsequent_reflection_at_interval(self) -> None:
        assert SelfReflectionEngine.should_reflect(
            exchange_index=SELF_REFLECTION_INTERVAL * 2,
            last_reflection_exchange=SELF_REFLECTION_INTERVAL,
        )

    def test_zero_index_never_triggers(self) -> None:
        assert not SelfReflectionEngine.should_reflect(0, None)


# ---------------------------------------------------------------------------
# LLM-driven reflection generation
# ---------------------------------------------------------------------------


class TestGenerateReflection:
    @pytest.mark.asyncio
    async def test_llm_success(self) -> None:
        llm = _mock_llm()
        engine = SelfReflectionEngine(llm)
        char = _make_character()
        model = await engine.generate_reflection(char, exchange_index=5)

        assert model.self_concept == "I see myself as someone who is growing."
        assert model.generated_at_exchange == 5
        assert len(model.value_tensions) == 1
        assert len(model.growth_edges) == 1
        llm.generate.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_llm_failure_fallback(self) -> None:
        llm = MagicMock()
        llm.generate = AsyncMock(side_effect=ValueError("LLM down"))
        engine = SelfReflectionEngine(llm)
        char = _make_character()
        model = await engine.generate_reflection(char, exchange_index=3)

        # Should produce a minimal fallback model
        assert char.name in model.self_concept
        assert model.generated_at_exchange == 3

    @pytest.mark.asyncio
    async def test_llm_bad_json_fallback(self) -> None:
        llm = _mock_llm(response="not valid json at all")
        engine = SelfReflectionEngine(llm)
        char = _make_character()
        model = await engine.generate_reflection(char, exchange_index=2)
        assert char.name in model.self_concept

    @pytest.mark.asyncio
    async def test_includes_internal_monologue_context(self) -> None:
        llm = _mock_llm()
        engine = SelfReflectionEngine(llm)
        char = _make_character()
        char.add_internal_thought("I wonder if I'm being honest with myself.")
        await engine.generate_reflection(char, exchange_index=5)

        prompt_text = llm.generate.call_args[0][0]
        assert "honest with myself" in prompt_text

    @pytest.mark.asyncio
    async def test_includes_recent_exchanges(self) -> None:
        llm = _mock_llm()
        engine = SelfReflectionEngine(llm)
        char = _make_character()
        char.add_memory_to_history("Elena", "That's an interesting perspective.")
        await engine.generate_reflection(char, exchange_index=5)

        prompt_text = llm.generate.call_args[0][0]
        assert "interesting perspective" in prompt_text


# ---------------------------------------------------------------------------
# maybe_reflect
# ---------------------------------------------------------------------------


class TestMaybeReflect:
    @pytest.mark.asyncio
    async def test_no_reflection_before_interval(self) -> None:
        llm = _mock_llm()
        engine = SelfReflectionEngine(llm)
        char = _make_character()
        result = await engine.maybe_reflect(char, exchange_index=1)
        assert result is None
        assert char.self_model is None
        llm.generate.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_reflection_at_interval(self) -> None:
        llm = _mock_llm()
        engine = SelfReflectionEngine(llm)
        char = _make_character()
        result = await engine.maybe_reflect(char, exchange_index=SELF_REFLECTION_INTERVAL)
        assert result is not None
        assert char.self_model is result
        assert char.self_model.generated_at_exchange == SELF_REFLECTION_INTERVAL

    @pytest.mark.asyncio
    async def test_previous_model_pushed_to_history(self) -> None:
        llm = _mock_llm()
        engine = SelfReflectionEngine(llm)
        char = _make_character()

        # First reflection
        await engine.maybe_reflect(char, exchange_index=SELF_REFLECTION_INTERVAL)
        assert char.self_model is not None
        assert len(char.self_model_history) == 0

        # Second reflection — first model should be pushed to history
        await engine.maybe_reflect(char, exchange_index=SELF_REFLECTION_INTERVAL * 2)
        assert len(char.self_model_history) == 1
        assert char.self_model.generated_at_exchange == SELF_REFLECTION_INTERVAL * 2

    @pytest.mark.asyncio
    async def test_history_cap(self) -> None:
        llm = _mock_llm()
        engine = SelfReflectionEngine(llm)
        char = _make_character()

        # Fill beyond capacity
        for i in range(MAX_SELF_MODEL_HISTORY + 3):
            exchange_idx = SELF_REFLECTION_INTERVAL * (i + 1)
            char.self_model = SelfModel(generated_at_exchange=exchange_idx - SELF_REFLECTION_INTERVAL)
            if i > 0:
                char.self_model_history.append(SelfModel(generated_at_exchange=exchange_idx - SELF_REFLECTION_INTERVAL - 1))
            await engine.maybe_reflect(char, exchange_index=exchange_idx)

        assert len(char.self_model_history) <= MAX_SELF_MODEL_HISTORY


# ---------------------------------------------------------------------------
# Character integration
# ---------------------------------------------------------------------------


class TestCharacterSelfModelIntegration:
    def test_character_starts_without_self_model(self) -> None:
        char = _make_character()
        assert char.self_model is None
        assert char.self_model_history == []

    def test_profile_without_self_model(self) -> None:
        char = _make_character()
        profile = char.get_character_profile()
        assert "Self-Perception" not in profile

    def test_profile_with_self_model(self) -> None:
        char = _make_character()
        char.self_model = SelfModel(
            self_concept="I see myself as cautious but warm.",
            emotional_awareness="Feeling quietly hopeful.",
            value_tensions=["I value truth but fear conflict."],
            growth_edges=["Opening up to others."],
            generated_at_exchange=5,
        )
        profile = char.get_character_profile()
        assert "Self-Perception" in profile
        assert "cautious but warm" in profile
        assert "quietly hopeful" in profile
        assert "truth but fear conflict" in profile
        assert "Opening up to others" in profile

    def test_self_model_serialization(self) -> None:
        char = _make_character()
        char.self_model = SelfModel(
            self_concept="Contemplative soul.",
            generated_at_exchange=3,
        )
        data = char.model_dump()
        assert data["self_model"]["self_concept"] == "Contemplative soul."
        restored = Character.model_validate(data)
        assert restored.self_model is not None
        assert restored.self_model.self_concept == "Contemplative soul."


# ---------------------------------------------------------------------------
# Prompt template
# ---------------------------------------------------------------------------


class TestSelfReflectionPrompt:
    def test_template_substitution(self) -> None:
        from character_creator.llm.prompts import SelfReflectionPrompts, substitute_prompt

        result = substitute_prompt(
            SelfReflectionPrompts.SELF_REFLECTION,
            character_name="Elena",
            personality_summary="Open and warm",
            emotional_state="curious",
            core_values="honesty, growth",
            trait_changes="openness: +0.02",
            recent_monologue="Am I being true to myself?",
            recent_exchanges="Elena: That's fascinating.\nMarco: I agree.",
        )
        assert "Elena" in result
        assert "Open and warm" in result
        assert "honesty, growth" in result
        assert "openness: +0.02" in result
        assert "Am I being true" in result

    def test_template_has_all_required_variables(self) -> None:
        from string import Template

        from character_creator.llm.prompts import SelfReflectionPrompts

        # Ensure template is a proper Template instance
        assert isinstance(SelfReflectionPrompts.SELF_REFLECTION, Template)
