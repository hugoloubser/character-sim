"""End-to-end integration test for the full dialogue pipeline.

Exercises character creation → dialogue generation → trait shifts →
self-reflection → dissonance detection → milestone reviews, all wired
together with a deterministic mock LLM that returns structured responses.

Logs every step to ``local/logs/tests/`` for post-mortem troubleshooting.
"""

from __future__ import annotations

import json
import logging

from unittest.mock import AsyncMock

import pytest

from character_creator.core.character import Character
from character_creator.core.constants import (
    BEHAVIOUR_EXTRACTION_INTERVAL,
    SELF_REFLECTION_INTERVAL,
)
from character_creator.core.dialogue import DialogueContext, DialogueSystem
from character_creator.core.memory import Background, Memory
from character_creator.core.personality import Personality, PersonalityTraits, Values

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Deterministic mock LLM
# ---------------------------------------------------------------------------

# Pre-canned responses keyed by prompt substring detection.
# The mock scans each prompt for these markers and returns the matching value.
_RESPONSES: list[tuple[str, str]] = [
    # Experience classification (trait shift engine)
    ("classify this exchange", "connection"),
    # Self-reflection
    (
        "Reflect on yourself",
        json.dumps({
            "self_concept": "I see myself as someone who values deep connections.",
            "emotional_awareness": "I feel a quiet warmth growing.",
            "value_tensions": ["I want honesty but sometimes hold back"],
            "growth_edges": ["Learning to be more direct"],
        }),
    ),
    # Behaviour extraction (dissonance detector)
    (
        "behavioural patterns",
        json.dumps([
            {"theme": "Avoided giving direct opinions", "confidence": 0.75},
            {"theme": "Deferred to the other person", "confidence": 0.6},
        ]),
    ),
    # Dissonance detection
    (
        "contradict",
        json.dumps([
            {
                "value": "honesty",
                "behaviour": "avoided giving direct opinions",
                "severity": 0.65,
            },
        ]),
    ),
    # Milestone review
    (
        "milestone",
        json.dumps({
            "shifts": [
                {
                    "trait_name": "warmth",
                    "old_value": 0.7,
                    "new_value": 0.75,
                    "justification": "Grew closer through conversation",
                    "confidence": 0.6,
                },
            ],
            "narrative_summary": "The conversation deepened their mutual respect.",
        }),
    ),
    # Emotion inference (single word expected)
    ("classify the emotion", "reflective"),
    # Internal monologue
    ("inner voice", "I wonder if I'm being genuine enough with them."),
]

# Default fallback for dialogue generation and anything else
_DEFAULT_RESPONSE = "That's an interesting perspective. I'd love to hear more."


def _smart_mock_generate(prompt: str, **_kwargs: object) -> str:
    """Route a prompt to the appropriate canned response."""
    prompt_lower = prompt.lower()
    for marker, response in _RESPONSES:
        if marker.lower() in prompt_lower:
            logger.debug("Mock LLM matched marker %r → %.80s", marker, response)
            return response
    logger.debug("Mock LLM using default response for prompt: %.120s", prompt_lower)
    return _DEFAULT_RESPONSE


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_elena() -> Character:
    return Character(
        name="Elena",
        description="A thoughtful librarian with kind eyes",
        personality=Personality(
            traits=PersonalityTraits(
                openness=0.85,
                conscientiousness=0.7,
                extraversion=0.4,
                agreeableness=0.8,
                emotional_stability=0.6,
                warmth=0.75,
                assertiveness=0.35,
                sensitivity=0.8,
            ),
            values=Values(priority_keywords=["honesty", "knowledge", "empathy"]),
            speech_patterns=["Speaks in measured, thoughtful sentences"],
            quirks=["Quotes books mid-conversation"],
        ),
        background=Background(
            age=34,
            origin="Prague",
            occupation="Librarian",
            memories=[
                Memory(
                    title="Childhood library",
                    description="Grew up in a house full of books",
                    impact="Developed a deep love for knowledge and quiet reflection",
                    emotional_weight=0.8,
                ),
            ],
        ),
    )


def _make_marcus() -> Character:
    return Character(
        name="Marcus",
        description="A confident street musician with a warm smile",
        personality=Personality(
            traits=PersonalityTraits(
                openness=0.9,
                conscientiousness=0.4,
                extraversion=0.85,
                agreeableness=0.65,
                emotional_stability=0.5,
                warmth=0.7,
                assertiveness=0.8,
                sensitivity=0.55,
            ),
            values=Values(priority_keywords=["freedom", "authenticity", "joy"]),
            speech_patterns=["Uses musical metaphors", "Talks with his hands"],
            quirks=["Hums between sentences"],
        ),
        background=Background(
            age=28,
            origin="New Orleans",
            occupation="Street musician",
            memories=[
                Memory(
                    title="Leaving home",
                    description="Left home at 18 to travel with a jazz band",
                    impact="Learned to value freedom and living in the moment",
                    emotional_weight=0.9,
                ),
            ],
        ),
    )


# ---------------------------------------------------------------------------
# Test class
# ---------------------------------------------------------------------------


class TestEndToEnd:
    """Full pipeline integration test with file-based logging."""

    @pytest.mark.asyncio
    async def test_full_dialogue_run(self, file_logger: logging.Logger) -> None:
        """Run 8 exchanges through the entire pipeline and verify every subsystem fires."""
        # -- Arrange -------------------------------------------------------
        llm = AsyncMock()
        llm.generate = AsyncMock(side_effect=_smart_mock_generate)
        llm.close = AsyncMock()

        elena = _make_elena()
        marcus = _make_marcus()
        system = DialogueSystem(llm)

        context = DialogueContext(
            characters=[elena, marcus],
            scene_description="A rainy evening at a small café near the Charles Bridge",
            topic="what makes a life feel meaningful",
        )

        # Snapshot starting traits for milestone review
        trait_snapshots = {
            c.name: c.personality.traits.to_dict() for c in context.characters
        }

        num_exchanges = 8
        file_logger.info("=== E2E test starting: %d exchanges ===", num_exchanges)
        file_logger.info("Characters: %s", [c.name for c in context.characters])
        file_logger.info("Scene: %s", context.scene_description)

        # -- Act -----------------------------------------------------------
        await system.generate_sequential_dialogue(
            context, num_exchanges=num_exchanges
        )

        # Run milestone reviews post-scene
        reviews = await system.run_milestone_reviews(context, trait_snapshots)

        # -- Assert: dialogue generated ------------------------------------
        assert len(context.exchanges) == num_exchanges
        speakers = {e.speaker.name for e in context.exchanges}
        assert speakers == {"Elena", "Marcus"}, "Both characters should have spoken"

        for ex in context.exchanges:
            assert ex.text, "Every exchange should have dialogue text"
            assert ex.emotional_context, "Every exchange should have an emotion label"
            file_logger.info(
                "[%s] (%s) %s",
                ex.speaker.name,
                ex.emotional_context,
                ex.text[:100],
            )

        # -- Assert: conversation history persisted on characters ----------
        assert len(elena.conversation_history) > 0
        assert len(marcus.conversation_history) > 0

        # -- Assert: trait shifts occurred ---------------------------------
        # The engine runs every exchange with 'connection' classification,
        # so at least one character should have trait deltas recorded.
        engine_histories = system.trait_shift_engine._histories
        assert len(engine_histories) > 0, "Trait shift engine should have recorded history"
        for name, history in engine_histories.items():
            file_logger.info(
                "Trait history for %s: %d deltas, net=%s",
                name,
                len(history.deltas),
                history.net_deltas(),
            )

        # -- Assert: self-reflection fired ---------------------------------
        # SELF_REFLECTION_INTERVAL=5, so with 8 exchanges at least one
        # character should have gotten a reflection.
        reflected = [
            c.name for c in [elena, marcus] if c.self_model is not None
        ]
        file_logger.info("Characters with self-model: %s", reflected)
        assert len(reflected) > 0, (
            f"At least one character should have a self-model after {num_exchanges} exchanges "
            f"(interval={SELF_REFLECTION_INTERVAL})"
        )

        # -- Assert: dissonance detection fired ----------------------------
        # BEHAVIOUR_EXTRACTION_INTERVAL=3, mock returns severity 0.65 > threshold
        all_dissonances = elena.active_dissonances + marcus.active_dissonances
        file_logger.info(
            "Active dissonances: Elena=%d, Marcus=%d",
            len(elena.active_dissonances),
            len(marcus.active_dissonances),
        )
        assert len(all_dissonances) > 0, (
            f"At least one dissonance should be detected after {num_exchanges} exchanges "
            f"(interval={BEHAVIOUR_EXTRACTION_INTERVAL})"
        )

        # -- Assert: milestone reviews produced ----------------------------
        assert len(reviews) == 2, "One review per character"
        for review in reviews:
            file_logger.info(
                "Milestone review for %s: %d shifts, summary=%.80s",
                review.character_name,
                len(review.shifts),
                review.narrative_summary,
            )
            assert review.narrative_summary

        # -- Assert: character profiles render without error ----------------
        for c in [elena, marcus]:
            profile = c.get_character_profile()
            assert c.name in profile
            file_logger.debug("Profile for %s:\n%s", c.name, profile)

        file_logger.info("=== E2E test complete ===")

    @pytest.mark.asyncio
    async def test_single_exchange_no_crash(self, file_logger: logging.Logger) -> None:
        """Smoke test: even a single exchange should complete without error."""
        llm = AsyncMock()
        llm.generate = AsyncMock(side_effect=_smart_mock_generate)
        llm.close = AsyncMock()

        elena = _make_elena()
        marcus = _make_marcus()
        system = DialogueSystem(llm)
        context = DialogueContext(
            characters=[elena, marcus],
            scene_description="Quick hello in a hallway",
            topic="the weather",
        )

        await system.generate_sequential_dialogue(context, num_exchanges=1)
        assert len(context.exchanges) == 1
        file_logger.info("Single-exchange smoke test passed")

    @pytest.mark.asyncio
    async def test_llm_failure_mid_dialogue_raises(self, file_logger: logging.Logger) -> None:
        """A hard LLM failure mid-dialogue should propagate as RuntimeError."""
        call_count = 0

        async def _failing_after_two(prompt: str, **_kw: object) -> str:
            nonlocal call_count
            call_count += 1
            if call_count > 4:
                msg = "Simulated LLM outage"
                raise OSError(msg)
            return _smart_mock_generate(prompt)

        llm = AsyncMock()
        llm.generate = AsyncMock(side_effect=_failing_after_two)
        llm.close = AsyncMock()

        system = DialogueSystem(llm)
        context = DialogueContext(
            characters=[_make_elena(), _make_marcus()],
            scene_description="A tense moment",
            topic="secrets",
        )

        with pytest.raises(RuntimeError, match="Failed to generate exchange"):
            await system.generate_sequential_dialogue(context, num_exchanges=5)
        file_logger.info("LLM failure test passed (raised as expected)")
