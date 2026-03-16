"""Tests for the DialogueSystem and prompt generation.

Covers speaker selection, prompt generation, emotional inference
(both LLM and heuristic), and end-to-end sequential dialogue
with a mock LLM provider.
"""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from character_creator.core.character import Character
from character_creator.core.dialogue import DialogueContext, DialogueSystem
from character_creator.core.memory import Background
from character_creator.core.personality import Personality, PersonalityTraits, Values

# ---------------------------------------------------------------------------
# Helpers / Fixtures
# ---------------------------------------------------------------------------


def _make_character(
    name: str,
    *,
    assertiveness: float = 0.5,
    warmth: float = 0.5,
    openness: float = 0.5,
) -> Character:
    """Build a Character with configurable personality for testing."""
    return Character(
        name=name,
        description=f"Test character {name}",
        personality=Personality(
            traits=PersonalityTraits(
                assertiveness=assertiveness,
                warmth=warmth,
                openness=openness,
            ),
            values=Values(priority_keywords=["testing"]),
            speech_patterns=["Speaks plainly"],
            quirks=["None"],
        ),
        background=Background(age=30, origin="Testland", occupation="Tester"),
    )


@pytest.fixture
def alice() -> Character:
    return _make_character("Alice", assertiveness=0.9, warmth=0.6, openness=0.8)


@pytest.fixture
def bob() -> Character:
    return _make_character("Bob", assertiveness=0.3, warmth=0.8, openness=0.4)


@pytest.fixture
def scene(alice: Character, bob: Character) -> DialogueContext:
    return DialogueContext(
        characters=[alice, bob],
        scene_description="A quiet park bench",
        topic="philosophy",
    )


@pytest.fixture
def mock_provider() -> AsyncMock:
    """An AsyncMock that mimics LLMProvider.generate / close."""
    provider = AsyncMock()
    provider.generate = AsyncMock(return_value="Hello from the LLM.")
    provider.close = AsyncMock()
    return provider


@pytest.fixture
def system(mock_provider: AsyncMock) -> DialogueSystem:
    return DialogueSystem(mock_provider)


# ---------------------------------------------------------------------------
# generate_next_speaker
# ---------------------------------------------------------------------------


class TestGenerateNextSpeaker:
    """Tests for personality-based speaker selection."""

    def test_most_assertive_starts(
        self, system: DialogueSystem, scene: DialogueContext, alice: Character
    ) -> None:
        """With no history, the most assertive character is chosen."""
        speaker = system.generate_next_speaker(scene)
        assert speaker.name == alice.name

    def test_turn_taking_penalty(
        self,
        system: DialogueSystem,
        scene: DialogueContext,
        alice: Character,
        bob: Character,
    ) -> None:
        """A character who just spoke gets a penalty, allowing others a turn."""
        scene.add_exchange(speaker=alice, text="I'll go first")
        speaker = system.generate_next_speaker(scene)
        assert speaker.name == bob.name

    def test_empty_characters_raises(self, system: DialogueSystem) -> None:
        """ValueError when no characters are in the context."""
        empty = DialogueContext(characters=[], scene_description="void")
        with pytest.raises(ValueError, match="No characters"):
            system.generate_next_speaker(empty)

    def test_single_character(self, system: DialogueSystem, alice: Character) -> None:
        """A single character is always selected."""
        solo = DialogueContext(
            characters=[alice],
            scene_description="Alone",
        )
        assert system.generate_next_speaker(solo).name == alice.name


# ---------------------------------------------------------------------------
# Prompt generation
# ---------------------------------------------------------------------------


class TestPromptGeneration:
    """Tests for dialogue and monologue prompt building."""

    def test_dialogue_prompt_contains_character(
        self, system: DialogueSystem, scene: DialogueContext, alice: Character
    ) -> None:
        prompt = system.generate_dialogue_prompt(scene, alice)
        assert "Alice" in prompt
        assert "philosophy" in prompt
        assert "quiet park bench" in prompt

    def test_dialogue_prompt_lists_other_characters(
        self, system: DialogueSystem, scene: DialogueContext, alice: Character
    ) -> None:
        prompt = system.generate_dialogue_prompt(scene, alice)
        assert "Bob" in prompt

    def test_monologue_prompt_contains_character(
        self, system: DialogueSystem, scene: DialogueContext, alice: Character
    ) -> None:
        prompt = system.generate_post_exchange_monologue_prompt(scene, alice, own_dialogue="")
        assert "Alice" in prompt

    def test_pre_exchange_monologue_prompt_contains_character(
        self, system: DialogueSystem, scene: DialogueContext, alice: Character
    ) -> None:
        prompt = system.generate_pre_exchange_monologue_prompt(scene, alice)
        assert "Alice" in prompt

    def test_pre_exchange_monologue_prompt_is_prospective(
        self, system: DialogueSystem, scene: DialogueContext, alice: Character
    ) -> None:
        """Pre-exchange prompt should frame the monologue as before speaking."""
        prompt = system.generate_pre_exchange_monologue_prompt(scene, alice)
        assert "pre-exchange" in prompt.lower() or "before" in prompt.lower() or "deliberation" in prompt.lower()

    def test_post_exchange_monologue_prompt_includes_own_dialogue(
        self, system: DialogueSystem, scene: DialogueContext, alice: Character
    ) -> None:
        """Post-exchange prompt should include the character's own dialogue text."""
        own_dialogue = "I think therefore I am."
        prompt = system.generate_post_exchange_monologue_prompt(scene, alice, own_dialogue=own_dialogue)
        assert own_dialogue in prompt


# ---------------------------------------------------------------------------
# Emotional inference — heuristic
# ---------------------------------------------------------------------------


class TestEmotionalHeuristic:
    """Tests for the static keyword-based fallback."""

    def test_enthusiastic(self, alice: Character) -> None:
        assert DialogueSystem._infer_emotional_context_heuristic(alice, "That's amazing!") == "enthusiastic"

    def test_apologetic(self, alice: Character) -> None:
        assert DialogueSystem._infer_emotional_context_heuristic(alice, "I'm sorry about that") == "apologetic"

    def test_curious(self, alice: Character) -> None:
        assert DialogueSystem._infer_emotional_context_heuristic(alice, "I wonder what that means?") == "curious"

    def test_resistant(self, alice: Character) -> None:
        assert DialogueSystem._infer_emotional_context_heuristic(alice, "No, I won't do that") == "resistant"

    def test_thoughtful(self, alice: Character) -> None:
        assert DialogueSystem._infer_emotional_context_heuristic(alice, "Hmm, let me think...") == "thoughtful"

    def test_neutral_fallback(self, alice: Character) -> None:
        result = DialogueSystem._infer_emotional_context_heuristic(alice, "The sky is blue")
        assert result == "neutral"


# ---------------------------------------------------------------------------
# Emotional inference — LLM path
# ---------------------------------------------------------------------------


class TestEmotionalInferenceLLM:
    """Tests for the async LLM-based emotion inference."""

    @pytest.mark.asyncio
    async def test_returns_llm_word(
        self, system: DialogueSystem, alice: Character, mock_provider: AsyncMock
    ) -> None:
        mock_provider.generate.return_value = "melancholy"
        result = await system.infer_emotional_context(alice, "The rain falls softly.")
        assert result == "melancholy"

    @pytest.mark.asyncio
    async def test_falls_back_on_multi_word(
        self, system: DialogueSystem, alice: Character, mock_provider: AsyncMock
    ) -> None:
        """Multi-word LLM replies fall back to the heuristic."""
        mock_provider.generate.return_value = "deeply sad"
        result = await system.infer_emotional_context(alice, "The rain falls softly.")
        # Heuristic fallback — no keywords matched → neutral
        assert result == "neutral"

    @pytest.mark.asyncio
    async def test_falls_back_on_provider_error(
        self, system: DialogueSystem, alice: Character, mock_provider: AsyncMock
    ) -> None:
        """Provider exception falls back to the heuristic."""
        mock_provider.generate.side_effect = RuntimeError("boom")
        result = await system.infer_emotional_context(alice, "That's amazing!")
        assert result == "enthusiastic"  # from heuristic


# ---------------------------------------------------------------------------
# End-to-end generate_response
# ---------------------------------------------------------------------------


class TestGenerateResponse:
    """Tests for the full async generate_response pipeline."""

    @pytest.mark.asyncio
    async def test_returns_pre_thought_dialogue_and_post_thought(
        self, system: DialogueSystem, scene: DialogueContext, alice: Character, mock_provider: AsyncMock
    ) -> None:
        mock_provider.generate.side_effect = [
            "I wonder what they mean by that.",     # pre-exchange thought
            "A beautiful day for philosophy.",       # dialogue
            "I hope that landed the right way.",     # post-exchange thought
        ]
        pre_thought, dialogue, post_thought = await system.generate_response(scene, alice)
        assert pre_thought == "I wonder what they mean by that."
        assert dialogue == "A beautiful day for philosophy."
        assert post_thought == "I hope that landed the right way."
        assert mock_provider.generate.call_count == 3

    @pytest.mark.asyncio
    async def test_sequential_dialogue(
        self, system: DialogueSystem, scene: DialogueContext, mock_provider: AsyncMock
    ) -> None:
        """generate_sequential_dialogue adds exchanges to the context."""
        mock_provider.generate.return_value = "Test line."
        await system.generate_sequential_dialogue(scene, num_exchanges=2)
        assert len(scene.exchanges) == 2

    @pytest.mark.asyncio
    async def test_exchange_stores_both_thoughts(
        self, system: DialogueSystem, scene: DialogueContext, alice: Character, mock_provider: AsyncMock
    ) -> None:
        """Both pre- and post-exchange thoughts are stored on the DialogueExchange."""
        mock_provider.generate.side_effect = [
            "pre-thought here",
            "public dialogue here",
            "post-thought here",
            "neutral",  # emotion inference
        ]
        pre_thought, dialogue, post_thought = await system.generate_response(scene, alice)
        emotional = await system.infer_emotional_context(alice, dialogue)
        scene.add_exchange(
            speaker=alice,
            text=dialogue,
            emotional_context=emotional,
            pre_exchange_thought=pre_thought,
            internal_thought=post_thought,
        )
        assert scene.exchanges[0].pre_exchange_thought == "pre-thought here"
        assert scene.exchanges[0].internal_thought == "post-thought here"
