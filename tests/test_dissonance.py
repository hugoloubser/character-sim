"""Tests for Phase 5 — Dissonance Detection (behaviour vs values)."""

from __future__ import annotations

import json

from unittest.mock import AsyncMock, MagicMock

import pytest

from character_creator.core.character import Character
from character_creator.core.constants import (
    BEHAVIOUR_EXTRACTION_INTERVAL,
    DISSONANCE_SEVERITY_THRESHOLD,
    MAX_ACTIVE_DISSONANCES,
)
from character_creator.core.personality import Personality, PersonalityTraits, Values
from character_creator.core.self_model import (
    BehaviourTheme,
    CognitiveDissonance,
    DissonanceDetector,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_character(name: str = "Kira", **trait_overrides) -> Character:
    return Character(
        name=name,
        description="A principled test character",
        personality=Personality(
            traits=PersonalityTraits(**trait_overrides),
            values=Values(priority_keywords=["honesty", "loyalty", "courage"]),
        ),
    )


def _mock_llm(response: str | None = None) -> MagicMock:
    provider = MagicMock()
    provider.generate = AsyncMock(return_value=response or "[]")
    return provider


def _themes_response() -> str:
    return json.dumps([
        {"theme": "Avoided direct confrontation", "confidence": 0.8},
        {"theme": "Deflected with humour", "confidence": 0.6},
    ])


def _dissonance_response() -> str:
    return json.dumps([
        {
            "value": "honesty",
            "behaviour": "deflected questions instead of answering directly",
            "severity": 0.7,
        },
    ])


# ---------------------------------------------------------------------------
# BehaviourTheme model
# ---------------------------------------------------------------------------


class TestBehaviourTheme:
    def test_construction(self) -> None:
        t = BehaviourTheme(theme="Avoided eye contact", confidence=0.7, source_exchanges=[1, 2])
        assert t.theme == "Avoided eye contact"
        assert t.confidence == 0.7
        assert t.source_exchanges == [1, 2]

    def test_defaults(self) -> None:
        t = BehaviourTheme(theme="Test")
        assert t.confidence == 0.5
        assert t.source_exchanges == []

    def test_round_trip(self) -> None:
        t = BehaviourTheme(theme="Was evasive", confidence=0.85)
        data = t.model_dump()
        restored = BehaviourTheme.model_validate(data)
        assert restored.theme == t.theme
        assert restored.confidence == t.confidence


# ---------------------------------------------------------------------------
# CognitiveDissonance model
# ---------------------------------------------------------------------------


class TestCognitiveDissonance:
    def test_construction(self) -> None:
        d = CognitiveDissonance(
            value="honesty",
            behaviour="lied about their past",
            severity=0.8,
            detected_at_exchange=5,
        )
        assert d.value == "honesty"
        assert d.behaviour == "lied about their past"
        assert d.severity == 0.8
        assert not d.resolved
        assert d.resolution is None

    def test_defaults(self) -> None:
        d = CognitiveDissonance(value="loyalty", behaviour="abandoned ally")
        assert d.severity == 0.5
        assert d.detected_at_exchange == 0
        assert not d.resolved

    def test_round_trip(self) -> None:
        d = CognitiveDissonance(
            value="courage",
            behaviour="ran from confrontation",
            severity=0.6,
            resolved=True,
            resolution="Returned to face the situation.",
        )
        data = d.model_dump()
        restored = CognitiveDissonance.model_validate(data)
        assert restored.resolved is True
        assert restored.resolution == "Returned to face the situation."


# ---------------------------------------------------------------------------
# DissonanceDetector.should_check
# ---------------------------------------------------------------------------


class TestShouldCheck:
    def test_not_yet_first_time(self) -> None:
        assert not DissonanceDetector.should_check(BEHAVIOUR_EXTRACTION_INTERVAL - 1, None)

    def test_triggers_first_time(self) -> None:
        assert DissonanceDetector.should_check(BEHAVIOUR_EXTRACTION_INTERVAL, None)

    def test_not_yet_subsequent(self) -> None:
        assert not DissonanceDetector.should_check(
            BEHAVIOUR_EXTRACTION_INTERVAL + 1,
            BEHAVIOUR_EXTRACTION_INTERVAL,
        )

    def test_triggers_subsequent(self) -> None:
        assert DissonanceDetector.should_check(
            BEHAVIOUR_EXTRACTION_INTERVAL * 2,
            BEHAVIOUR_EXTRACTION_INTERVAL,
        )

    def test_zero_never_triggers(self) -> None:
        assert not DissonanceDetector.should_check(0, None)


# ---------------------------------------------------------------------------
# Extract behaviour themes
# ---------------------------------------------------------------------------


class TestExtractBehaviourThemes:
    @pytest.mark.asyncio
    async def test_happy_path(self) -> None:
        llm = _mock_llm(_themes_response())
        detector = DissonanceDetector(llm)
        char = _make_character()
        exchanges = [{"speaker": "Kira", "text": "I don't want to talk about it."}]
        themes = await detector.extract_behaviour_themes(char, exchanges)
        assert len(themes) == 2
        assert themes[0].theme == "Avoided direct confrontation"
        llm.generate.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_llm_failure_returns_empty(self) -> None:
        llm = MagicMock()
        llm.generate = AsyncMock(side_effect=ValueError("LLM down"))
        detector = DissonanceDetector(llm)
        themes = await detector.extract_behaviour_themes(_make_character(), [])
        assert themes == []

    @pytest.mark.asyncio
    async def test_bad_json_returns_empty(self) -> None:
        llm = _mock_llm("not json at all")
        detector = DissonanceDetector(llm)
        themes = await detector.extract_behaviour_themes(_make_character(), [])
        assert themes == []

    @pytest.mark.asyncio
    async def test_dict_with_themes_key(self) -> None:
        """LLM wraps output in a {"themes": [...]} object."""
        wrapped = json.dumps({
            "themes": [{"theme": "Was confrontational", "confidence": 0.9}]
        })
        llm = _mock_llm(wrapped)
        detector = DissonanceDetector(llm)
        themes = await detector.extract_behaviour_themes(_make_character(), [])
        assert len(themes) == 1


# ---------------------------------------------------------------------------
# Detect dissonance
# ---------------------------------------------------------------------------


class TestDetectDissonance:
    @pytest.mark.asyncio
    async def test_happy_path(self) -> None:
        llm = _mock_llm(_dissonance_response())
        detector = DissonanceDetector(llm)
        char = _make_character()
        themes = [BehaviourTheme(theme="Deflected questions")]
        dissonances = await detector.detect_dissonance(char, themes)
        assert len(dissonances) == 1
        assert dissonances[0].value == "honesty"
        assert dissonances[0].severity == 0.7

    @pytest.mark.asyncio
    async def test_low_severity_filtered(self) -> None:
        response = json.dumps([
            {"value": "honesty", "behaviour": "slight evasion", "severity": DISSONANCE_SEVERITY_THRESHOLD - 0.01},
        ])
        llm = _mock_llm(response)
        detector = DissonanceDetector(llm)
        themes = [BehaviourTheme(theme="Was evasive")]
        dissonances = await detector.detect_dissonance(_make_character(), themes)
        assert dissonances == []

    @pytest.mark.asyncio
    async def test_empty_themes_returns_empty(self) -> None:
        llm = _mock_llm()
        detector = DissonanceDetector(llm)
        dissonances = await detector.detect_dissonance(_make_character(), [])
        assert dissonances == []
        llm.generate.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_no_values_returns_empty(self) -> None:
        llm = _mock_llm()
        detector = DissonanceDetector(llm)
        char = Character(
            name="Empty",
            description="No values",
            personality=Personality(values=Values(priority_keywords=[])),
        )
        dissonances = await detector.detect_dissonance(char, [BehaviourTheme(theme="X")])
        assert dissonances == []
        llm.generate.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_llm_failure_returns_empty(self) -> None:
        llm = MagicMock()
        llm.generate = AsyncMock(side_effect=OSError("network"))
        detector = DissonanceDetector(llm)
        dissonances = await detector.detect_dissonance(
            _make_character(), [BehaviourTheme(theme="X")]
        )
        assert dissonances == []

    @pytest.mark.asyncio
    async def test_dict_with_dissonances_key(self) -> None:
        wrapped = json.dumps({
            "dissonances": [{"value": "loyalty", "behaviour": "betrayed trust", "severity": 0.8}]
        })
        llm = _mock_llm(wrapped)
        detector = DissonanceDetector(llm)
        dissonances = await detector.detect_dissonance(
            _make_character(), [BehaviourTheme(theme="Betrayal")]
        )
        assert len(dissonances) == 1


# ---------------------------------------------------------------------------
# Full detection pipeline
# ---------------------------------------------------------------------------


class TestMaybeDetect:
    @pytest.mark.asyncio
    async def test_no_detection_before_interval(self) -> None:
        llm = _mock_llm()
        detector = DissonanceDetector(llm)
        result = await detector.maybe_detect(_make_character(), exchange_index=0)
        assert result == []
        llm.generate.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_detection_at_interval(self) -> None:
        # Two sequential LLM calls: themes then dissonances
        llm = MagicMock()
        llm.generate = AsyncMock(
            side_effect=[_themes_response(), _dissonance_response()]
        )
        detector = DissonanceDetector(llm)
        char = _make_character()
        # Add some conversation history so there are exchanges to analyse
        for i in range(6):
            char.add_memory_to_history("Kira", f"Exchange {i}")

        result = await detector.maybe_detect(char, BEHAVIOUR_EXTRACTION_INTERVAL)
        assert len(result) == 1
        assert result[0].value == "honesty"
        assert result[0].detected_at_exchange == BEHAVIOUR_EXTRACTION_INTERVAL
        assert len(char.active_dissonances) == 1

    @pytest.mark.asyncio
    async def test_cap_active_dissonances(self) -> None:
        """New dissonances evict lowest-severity when at capacity."""
        char = _make_character()
        # Pre-fill to capacity
        for i in range(MAX_ACTIVE_DISSONANCES):
            char.active_dissonances.append(
                CognitiveDissonance(
                    value=f"value_{i}",
                    behaviour=f"behaviour_{i}",
                    severity=0.3 + i * 0.1,
                )
            )

        high_severity_response = json.dumps([
            {"value": "honesty", "behaviour": "lied", "severity": 0.95},
        ])
        llm = MagicMock()
        llm.generate = AsyncMock(
            side_effect=[_themes_response(), high_severity_response]
        )
        detector = DissonanceDetector(llm)
        for i in range(6):
            char.add_memory_to_history("Kira", f"Exchange {i}")

        await detector.maybe_detect(char, BEHAVIOUR_EXTRACTION_INTERVAL)
        assert len(char.active_dissonances) <= MAX_ACTIVE_DISSONANCES
        # The new high-severity one should be present
        assert any(d.severity == 0.95 for d in char.active_dissonances)

    @pytest.mark.asyncio
    async def test_low_severity_not_evicting(self) -> None:
        """New dissonance that's weaker than all existing ones is dropped."""
        char = _make_character()
        for i in range(MAX_ACTIVE_DISSONANCES):
            char.active_dissonances.append(
                CognitiveDissonance(
                    value=f"value_{i}",
                    behaviour=f"behaviour_{i}",
                    severity=0.9,
                )
            )

        weak_response = json.dumps([
            {"value": "honesty", "behaviour": "minor evasion", "severity": 0.45},
        ])
        llm = MagicMock()
        llm.generate = AsyncMock(
            side_effect=[_themes_response(), weak_response]
        )
        detector = DissonanceDetector(llm)
        for i in range(6):
            char.add_memory_to_history("Kira", f"Exchange {i}")

        await detector.maybe_detect(char, BEHAVIOUR_EXTRACTION_INTERVAL)
        assert len(char.active_dissonances) == MAX_ACTIVE_DISSONANCES
        # The weak one should NOT be present
        assert not any(d.value == "honesty" for d in char.active_dissonances)


# ---------------------------------------------------------------------------
# Character integration
# ---------------------------------------------------------------------------


class TestCharacterDissonanceIntegration:
    def test_character_starts_without_dissonances(self) -> None:
        char = _make_character()
        assert char.active_dissonances == []

    def test_profile_without_dissonances(self) -> None:
        char = _make_character()
        profile = char.get_character_profile()
        assert "Internal Tensions" not in profile

    def test_profile_with_dissonances(self) -> None:
        char = _make_character()
        char.active_dissonances.append(
            CognitiveDissonance(
                value="honesty",
                behaviour="deflected a direct question",
                severity=0.7,
            )
        )
        profile = char.get_character_profile()
        assert "Internal Tensions" in profile
        assert "honesty" in profile
        assert "deflected" in profile

    def test_resolved_dissonances_hidden(self) -> None:
        char = _make_character()
        char.active_dissonances.append(
            CognitiveDissonance(
                value="courage",
                behaviour="ran away",
                severity=0.8,
                resolved=True,
                resolution="Came back and faced the threat.",
            )
        )
        profile = char.get_character_profile()
        # Resolved dissonances should not show in the profile tensions
        assert "Internal Tensions" not in profile

    def test_dissonance_serialization(self) -> None:
        char = _make_character()
        char.active_dissonances.append(
            CognitiveDissonance(value="loyalty", behaviour="betrayed ally", severity=0.6)
        )
        data = char.model_dump()
        assert len(data["active_dissonances"]) == 1
        restored = Character.model_validate(data)
        assert restored.active_dissonances[0].value == "loyalty"


# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------


class TestDissonancePrompts:
    def test_behaviour_extraction_substitution(self) -> None:
        from character_creator.llm.prompts import DissonancePrompts, substitute_prompt

        result = substitute_prompt(
            DissonancePrompts.BEHAVIOUR_EXTRACTION,
            character_name="Kira",
            recent_exchanges="Kira: I don't want to discuss it.",
        )
        assert "Kira" in result
        assert "I don't want to discuss it" in result

    def test_dissonance_detection_substitution(self) -> None:
        from character_creator.llm.prompts import DissonancePrompts, substitute_prompt

        result = substitute_prompt(
            DissonancePrompts.DISSONANCE_DETECTION,
            character_name="Kira",
            core_values="honesty, loyalty",
            behaviour_themes="- Avoided confrontation (confidence: 0.80)",
        )
        assert "Kira" in result
        assert "honesty, loyalty" in result
        assert "Avoided confrontation" in result

    def test_monologue_prompt_includes_tensions(self) -> None:
        from character_creator.llm.prompts import DialoguePrompts, substitute_prompt

        result = substitute_prompt(
            DialoguePrompts.POST_EXCHANGE_MONOLOGUE,
            character_name="Kira",
            character_profile="Profile",
            mbti_type="INFJ",
            mbti_archetype="Advocate",
            communication_style="empathetic",
            active_tensions="- I value honesty but I've been deflecting questions",
            conversation_history="Kira: Fine. Whatever.",
            own_dialogue="Fine. Whatever.",
        )
        assert "honesty" in result
        assert "deflecting" in result
        assert "tensions" in result.lower() or "spoken" in result.lower()
