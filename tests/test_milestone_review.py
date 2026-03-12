"""Tests for Phase 4 — LLM Milestone Review."""

from __future__ import annotations

import json

from unittest.mock import AsyncMock, MagicMock

import pytest

from character_creator.core.character import Character
from character_creator.core.constants import (
    MAX_MILESTONE_SHIFT,
    MILESTONE_CONFIDENCE_THRESHOLD,
    TRAIT_MAX,
)
from character_creator.core.personality import Personality, PersonalityTraits, Values
from character_creator.core.self_model import SelfModel
from character_creator.core.trait_evolution import (
    MilestoneReview,
    MilestoneReviewEngine,
    PersonalityShift,
    TraitDelta,
    TraitHistory,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_character(name: str = "Luna", **trait_overrides) -> Character:
    return Character(
        name=name,
        description="A bold test character",
        personality=Personality(
            traits=PersonalityTraits(**trait_overrides),
            values=Values(priority_keywords=["justice", "creativity"]),
        ),
    )


def _mock_llm(response: str | None = None) -> MagicMock:
    if response is None:
        response = json.dumps({
            "shifts": [
                {
                    "trait_name": "assertiveness",
                    "old_value": 0.5,
                    "new_value": 0.58,
                    "justification": "Luna stood her ground repeatedly.",
                    "confidence": 0.8,
                },
                {
                    "trait_name": "warmth",
                    "old_value": 0.5,
                    "new_value": 0.54,
                    "justification": "She showed empathy in a heated moment.",
                    "confidence": 0.6,
                },
            ],
            "narrative_summary": "Luna grew more assertive while discovering her softer side.",
        })
    provider = MagicMock()
    provider.generate = AsyncMock(return_value=response)
    return provider


def _snapshot() -> dict[str, float]:
    """Default trait snapshot at scene start (all 0.5)."""
    return PersonalityTraits().to_dict()


# ---------------------------------------------------------------------------
# PersonalityShift model
# ---------------------------------------------------------------------------


class TestPersonalityShift:
    def test_construction(self) -> None:
        s = PersonalityShift(
            trait_name="warmth",
            old_value=0.5,
            new_value=0.6,
            justification="Opened up during conflict.",
            confidence=0.75,
        )
        assert s.trait_name == "warmth"
        assert s.new_value == 0.6
        assert s.confidence == 0.75

    def test_defaults(self) -> None:
        s = PersonalityShift(trait_name="openness", old_value=0.3, new_value=0.35)
        assert s.justification == ""
        assert s.confidence == 0.5

    def test_round_trip(self) -> None:
        s = PersonalityShift(
            trait_name="assertiveness",
            old_value=0.4,
            new_value=0.5,
            justification="Test",
            confidence=0.9,
        )
        data = s.model_dump()
        restored = PersonalityShift.model_validate(data)
        assert restored.trait_name == s.trait_name
        assert restored.confidence == s.confidence


# ---------------------------------------------------------------------------
# MilestoneReview model
# ---------------------------------------------------------------------------


class TestMilestoneReview:
    def test_construction(self) -> None:
        review = MilestoneReview(
            character_name="Luna",
            shifts=[
                PersonalityShift(trait_name="warmth", old_value=0.5, new_value=0.6),
            ],
            narrative_summary="Luna evolved.",
            scene_description="A tense negotiation.",
            exchange_count=10,
        )
        assert review.character_name == "Luna"
        assert len(review.shifts) == 1
        assert review.exchange_count == 10

    def test_defaults(self) -> None:
        review = MilestoneReview(character_name="Test")
        assert review.shifts == []
        assert review.narrative_summary == ""
        assert review.exchange_count == 0
        assert review.timestamp is not None

    def test_round_trip(self) -> None:
        review = MilestoneReview(
            character_name="Luna",
            shifts=[
                PersonalityShift(trait_name="openness", old_value=0.5, new_value=0.55, confidence=0.7),
            ],
            narrative_summary="Growth occurred.",
            scene_description="Scene 1",
            exchange_count=5,
        )
        data = review.model_dump()
        restored = MilestoneReview.model_validate(data)
        assert len(restored.shifts) == 1
        assert restored.narrative_summary == "Growth occurred."


# ---------------------------------------------------------------------------
# MilestoneReviewEngine — LLM success path
# ---------------------------------------------------------------------------


class TestMilestoneReviewEngine:
    @pytest.mark.asyncio
    async def test_happy_path(self) -> None:
        llm = _mock_llm()
        engine = MilestoneReviewEngine(llm)
        char = _make_character()
        review = await engine.review_scene(
            character=char,
            scene_description="A heated debate",
            exchange_count=8,
            trait_snapshot_start=_snapshot(),
        )
        assert review.character_name == "Luna"
        assert len(review.shifts) == 2
        assert review.narrative_summary != ""
        assert review.exchange_count == 8
        llm.generate.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_shifts_clamped_to_max(self) -> None:
        """Shifts exceeding MAX_MILESTONE_SHIFT are clamped."""
        response = json.dumps({
            "shifts": [
                {
                    "trait_name": "assertiveness",
                    "old_value": 0.5,
                    "new_value": 0.95,
                    "justification": "Huge change",
                    "confidence": 0.9,
                },
            ],
            "narrative_summary": "Big shift.",
        })
        llm = _mock_llm(response)
        engine = MilestoneReviewEngine(llm)
        char = _make_character()
        review = await engine.review_scene(char, "scene", 5, _snapshot())
        assert len(review.shifts) == 1
        delta = review.shifts[0].new_value - review.shifts[0].old_value
        assert abs(delta) <= MAX_MILESTONE_SHIFT + 1e-9

    @pytest.mark.asyncio
    async def test_negative_shift_clamped(self) -> None:
        """Negative shifts are also clamped."""
        response = json.dumps({
            "shifts": [
                {
                    "trait_name": "warmth",
                    "old_value": 0.5,
                    "new_value": 0.1,
                    "justification": "Withdrew completely",
                    "confidence": 0.8,
                },
            ],
            "narrative_summary": "Retreat.",
        })
        llm = _mock_llm(response)
        engine = MilestoneReviewEngine(llm)
        review = await engine.review_scene(_make_character(), "scene", 5, _snapshot())
        delta = review.shifts[0].new_value - review.shifts[0].old_value
        assert abs(delta) <= MAX_MILESTONE_SHIFT + 1e-9

    @pytest.mark.asyncio
    async def test_low_confidence_shifts_rejected(self) -> None:
        """Shifts below MILESTONE_CONFIDENCE_THRESHOLD are filtered out."""
        response = json.dumps({
            "shifts": [
                {
                    "trait_name": "openness",
                    "old_value": 0.5,
                    "new_value": 0.55,
                    "justification": "Slight opening",
                    "confidence": MILESTONE_CONFIDENCE_THRESHOLD - 0.01,
                },
                {
                    "trait_name": "warmth",
                    "old_value": 0.5,
                    "new_value": 0.56,
                    "justification": "Genuine warmth",
                    "confidence": MILESTONE_CONFIDENCE_THRESHOLD + 0.1,
                },
            ],
            "narrative_summary": "Mixed results.",
        })
        llm = _mock_llm(response)
        engine = MilestoneReviewEngine(llm)
        review = await engine.review_scene(_make_character(), "scene", 5, _snapshot())
        assert len(review.shifts) == 1
        assert review.shifts[0].trait_name == "warmth"

    @pytest.mark.asyncio
    async def test_llm_failure_returns_empty_review(self) -> None:
        llm = MagicMock()
        llm.generate = AsyncMock(side_effect=ValueError("LLM down"))
        engine = MilestoneReviewEngine(llm)
        review = await engine.review_scene(_make_character(), "scene", 5, _snapshot())
        assert review.shifts == []
        assert review.narrative_summary != ""

    @pytest.mark.asyncio
    async def test_bad_json_returns_empty_review(self) -> None:
        llm = _mock_llm("not valid json")
        engine = MilestoneReviewEngine(llm)
        review = await engine.review_scene(_make_character(), "scene", 5, _snapshot())
        assert review.shifts == []

    @pytest.mark.asyncio
    async def test_trait_bounds_enforced(self) -> None:
        """new_value is clamped to [TRAIT_MIN, TRAIT_MAX]."""
        response = json.dumps({
            "shifts": [
                {
                    "trait_name": "warmth",
                    "old_value": 0.95,
                    "new_value": 1.5,
                    "justification": "Overflows",
                    "confidence": 0.9,
                },
            ],
            "narrative_summary": "Edge case.",
        })
        llm = _mock_llm(response)
        engine = MilestoneReviewEngine(llm)
        review = await engine.review_scene(_make_character(), "scene", 5, _snapshot())
        assert review.shifts[0].new_value <= TRAIT_MAX

    @pytest.mark.asyncio
    async def test_includes_self_model_in_prompt(self) -> None:
        llm = _mock_llm()
        engine = MilestoneReviewEngine(llm)
        char = _make_character()
        char.self_model = SelfModel(
            self_concept="I see myself as a fighter for justice.",
            generated_at_exchange=5,
        )
        await engine.review_scene(char, "scene", 5, _snapshot())
        prompt_text = llm.generate.call_args[0][0]
        assert "fighter for justice" in prompt_text

    @pytest.mark.asyncio
    async def test_includes_trait_history_in_prompt(self) -> None:
        llm = _mock_llm()
        engine = MilestoneReviewEngine(llm)
        char = _make_character()
        char.trait_history = TraitHistory(
            character_name="Luna",
            deltas=[
                TraitDelta(trait_name="assertiveness", delta=0.02, source="conflict", exchange_index=1),
            ],
        )
        await engine.review_scene(char, "scene", 5, _snapshot())
        prompt_text = llm.generate.call_args[0][0]
        assert "assertiveness" in prompt_text
        assert "+0.020" in prompt_text


# ---------------------------------------------------------------------------
# apply_review
# ---------------------------------------------------------------------------


class TestApplyReview:
    def test_applies_shifts(self) -> None:
        char = _make_character()
        original_warmth = char.personality.traits.warmth
        review = MilestoneReview(
            character_name="Luna",
            shifts=[
                PersonalityShift(
                    trait_name="warmth",
                    old_value=original_warmth,
                    new_value=original_warmth + 0.05,
                    confidence=0.8,
                ),
            ],
        )
        applied = MilestoneReviewEngine.apply_review(char, review)
        assert "warmth" in applied
        assert char.personality.traits.warmth == pytest.approx(original_warmth + 0.05)

    def test_ignores_unknown_traits(self) -> None:
        char = _make_character()
        review = MilestoneReview(
            character_name="Luna",
            shifts=[
                PersonalityShift(
                    trait_name="nonexistent_trait",
                    old_value=0.5,
                    new_value=0.6,
                    confidence=0.9,
                ),
            ],
        )
        applied = MilestoneReviewEngine.apply_review(char, review)
        assert applied == {}

    def test_clamps_to_bounds(self) -> None:
        char = _make_character(warmth=0.98)
        review = MilestoneReview(
            character_name="Luna",
            shifts=[
                PersonalityShift(
                    trait_name="warmth",
                    old_value=0.98,
                    new_value=1.5,
                    confidence=0.9,
                ),
            ],
        )
        MilestoneReviewEngine.apply_review(char, review)
        assert char.personality.traits.warmth <= TRAIT_MAX

    def test_no_change_when_same(self) -> None:
        char = _make_character()
        review = MilestoneReview(
            character_name="Luna",
            shifts=[
                PersonalityShift(
                    trait_name="warmth",
                    old_value=0.5,
                    new_value=0.5,
                    confidence=0.9,
                ),
            ],
        )
        applied = MilestoneReviewEngine.apply_review(char, review)
        assert applied == {}

    def test_multiple_shifts_applied(self) -> None:
        char = _make_character()
        review = MilestoneReview(
            character_name="Luna",
            shifts=[
                PersonalityShift(trait_name="warmth", old_value=0.5, new_value=0.55, confidence=0.8),
                PersonalityShift(trait_name="openness", old_value=0.5, new_value=0.45, confidence=0.7),
            ],
        )
        applied = MilestoneReviewEngine.apply_review(char, review)
        assert len(applied) == 2
        assert char.personality.traits.warmth == pytest.approx(0.55)
        assert char.personality.traits.openness == pytest.approx(0.45)


# ---------------------------------------------------------------------------
# Prompt template
# ---------------------------------------------------------------------------


class TestMilestonePrompt:
    def test_template_substitution(self) -> None:
        from character_creator.llm.prompts import TraitEvolutionPrompts, substitute_prompt

        result = substitute_prompt(
            TraitEvolutionPrompts.MILESTONE_REVIEW,
            character_name="Luna",
            character_profile="Profile text",
            self_model="Self-concept: fighter",
            scene_description="A tense debate",
            exchange_count="10",
            traits_start="warmth: 0.500",
            traits_current="warmth: 0.520",
            trait_deltas="warmth: +0.020 (from conflict at exchange 3)",
        )
        assert "Luna" in result
        assert "tense debate" in result
        assert "warmth: 0.500" in result
        assert "warmth: 0.520" in result
        assert "+0.020" in result

    def test_template_is_template_instance(self) -> None:
        from string import Template

        from character_creator.llm.prompts import TraitEvolutionPrompts

        assert isinstance(TraitEvolutionPrompts.MILESTONE_REVIEW, Template)
