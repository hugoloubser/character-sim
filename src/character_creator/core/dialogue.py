"""Dialogue generation and multi-character interaction system."""

import asyncio
import logging

from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field

from character_creator.core.character import Character
from character_creator.core.constants import (
    DEFAULT_EMOTIONAL_STATE,
    DEFAULT_TOPIC,
    DIALOGUE_TEMPERATURE,
    EMOTION_INFERENCE_TEMPERATURE,
    MONOLOGUE_TEMPERATURE,
    RECENT_SPEAKER_DECAY,
    RECENT_SPEAKER_FLOOR,
    SAME_SPEAKER_PENALTY,
    SPEAKER_ASSERTIVENESS_WEIGHT,
    SPEAKER_COMPATIBILITY_WEIGHT,
    SPEAKER_EXTRAVERSION_WEIGHT,
    SPEAKER_OPENNESS_WEIGHT,
    SPEAKER_WARMTH_WEIGHT,
)
from character_creator.core.memory_tiered import MemoryCondenser
from character_creator.core.personality import mbti_compatibility
from character_creator.core.self_model import DissonanceDetector, SelfReflectionEngine
from character_creator.core.trait_evolution import MilestoneReviewEngine, TraitShiftEngine
from character_creator.llm.metrics import llm_context
from character_creator.llm.prompts import DialoguePrompts, substitute_prompt

if TYPE_CHECKING:
    from character_creator.llm.providers import LLMProvider

logger = logging.getLogger(__name__)


class DialogueExchange(BaseModel):
    """A single exchange in a dialogue between characters."""

    model_config = {"arbitrary_types_allowed": True}

    speaker: Character
    text: str
    emotional_context: str  # How they felt saying this
    internal_thought: str  # What they were thinking


class DialogueContext(BaseModel):
    """Context for a dialogue scene.

    Maintains state across multiple exchanges to ensure coherent
    multi-character conversations.
    """

    model_config = {"arbitrary_types_allowed": True}

    characters: list[Character]
    scene_description: str
    topic: str = DEFAULT_TOPIC
    exchanges: list[DialogueExchange] = Field(default_factory=list)

    def add_exchange(
        self,
        speaker: Character,
        text: str,
        emotional_context: str = DEFAULT_EMOTIONAL_STATE,
        internal_thought: str = "",
    ) -> None:
        """Add a dialogue exchange to the scene."""
        exchange = DialogueExchange(
            speaker=speaker,
            text=text,
            emotional_context=emotional_context,
            internal_thought=internal_thought,
        )
        self.exchanges.append(exchange)
        # Update character's conversation history
        speaker.add_memory_to_history(speaker.name, text)

    def get_conversation_history(self) -> str:
        """Get formatted conversation history for context."""
        return "\n".join(
            f"{exchange.speaker.name}: {exchange.text}" for exchange in self.exchanges
        )

    def get_other_characters(self, speaker: Character) -> list[Character]:
        """Get all characters in scene except the speaker."""
        return [c for c in self.characters if c.name != speaker.name]

    def to_dict(self) -> dict[str, Any]:
        """Convert dialogue context to dictionary format."""
        return {
            "characters": [c.name for c in self.characters],
            "scene_description": self.scene_description,
            "topic": self.topic,
            "exchanges": [
                {
                    "speaker": e.speaker.name,
                    "text": e.text,
                    "emotional_context": e.emotional_context,
                    "internal_thought": e.internal_thought,
                }
                for e in self.exchanges
            ],
        }


class DialogueSystem:
    """Manages dialogue generation and multi-character interactions.

    This system uses LLMs to generate natural dialogue while maintaining
    character consistency, personality traits, and background knowledge.
    """

    def __init__(self, llm_provider: "LLMProvider") -> None:
        """Initialize dialogue system with an LLM provider.

        Args:
            llm_provider: LLM service provider for generating dialogue.

        """
        self.llm_provider = llm_provider
        self.memory_condenser = MemoryCondenser(llm_provider)
        self.trait_shift_engine = TraitShiftEngine(llm_provider)
        self.self_reflection_engine = SelfReflectionEngine(llm_provider)
        self.milestone_review_engine = MilestoneReviewEngine(llm_provider)
        self.dissonance_detector = DissonanceDetector(llm_provider)

    def generate_dialogue_prompt(
        self, context: DialogueContext, character: Character, dialogue_history_length: int = 5
    ) -> str:
        """Generate a prompt for LLM to produce character dialogue.

        Args:
            context: Current dialogue context and scene information.
            character: Character who should speak next.
            dialogue_history_length: How many recent exchanges to include.

        Returns:
            Formatted prompt for LLM dialogue generation.

        """
        other_chars = context.get_other_characters(character)
        other_chars_desc = "\n".join(
            [f"- {c.name}: {c.description}" for c in other_chars]
        )

        recent_history = context.get_conversation_history().split("\n")[-dialogue_history_length:]
        history_str = "\n".join(recent_history)

        return substitute_prompt(
            DialoguePrompts.CHARACTER_RESPONSE,
            character_name=character.name,
            character_profile=character.get_character_profile(),
            mbti_type=character.personality.mbti_type.value,
            mbti_archetype=character.personality.mbti_archetype,
            communication_style=character.personality.communication_style,
            scene_description=context.scene_description,
            other_characters=other_chars_desc,
            topic=context.topic,
            conversation_history=history_str,
        )

    def generate_internal_monologue_prompt(
        self, context: DialogueContext, character: Character
    ) -> str:
        """Generate a prompt for character's internal thoughts.

        Args:
            context: Current dialogue context.
            character: Character whose thoughts to generate.

        Returns:
            Formatted prompt for LLM.

        """
        recent_dialogue = context.get_conversation_history().split("\n")[-3:]
        dialogue_str = "\n".join(recent_dialogue)

        # Format active dissonances for internal monologue
        tensions_str = "(none)"
        if character.active_dissonances:
            tension_lines = [
                f"- I value {d.value} but I've been {d.behaviour}"
                for d in character.active_dissonances
                if not d.resolved
            ]
            if tension_lines:
                tensions_str = "\n".join(tension_lines)

        return substitute_prompt(
            DialoguePrompts.INTERNAL_MONOLOGUE,
            character_name=character.name,
            character_profile=character.get_character_profile(),
            mbti_type=character.personality.mbti_type.value,
            mbti_archetype=character.personality.mbti_archetype,
            communication_style=character.personality.communication_style,
            active_tensions=tensions_str,
            conversation_history=dialogue_str,
        )

    async def generate_response(
        self, context: DialogueContext, character: Character
    ) -> tuple[str, str]:
        """Generate dialogue response and internal monologue for a character.

        Args:
            context: Current dialogue context.
            character: Character who should respond.

        Returns:
            Tuple of (dialogue_response, internal_monologue).

        """
        dialogue_prompt = self.generate_dialogue_prompt(context, character)
        with llm_context("dialogue"):
            dialogue_response = await self.llm_provider.generate(
                dialogue_prompt, temperature=DIALOGUE_TEMPERATURE
            )

        # Small pause between API calls to avoid back-to-back rate-limit hits
        await asyncio.sleep(0.5)

        monologue_prompt = self.generate_internal_monologue_prompt(context, character)
        with llm_context("internal_monologue"):
            internal_monologue = await self.llm_provider.generate(
                monologue_prompt, temperature=MONOLOGUE_TEMPERATURE
            )

        return dialogue_response.strip(), internal_monologue.strip()

    def generate_next_speaker(
        self, context: DialogueContext
    ) -> Character:
        """Determine which character should speak next based on personality.

        Uses personality traits to select the next speaker:
        - Assertiveness: Higher assertiveness → more likely to jump into conversation
        - Warmth: Higher warmth → more responsive, likely to engage
        - Openness: Higher openness → more likely to start new topics
        - Recent participation: Characters less recently heard from get priority

        Args:
            context: Current dialogue context.

        Returns:
            Character who should speak next.

        """
        if not context.characters:
            msg = "No characters available in dialogue context"
            raise ValueError(msg)

        # If no exchanges yet, pick the most extraverted character
        if not context.exchanges:
            return max(
                context.characters,
                key=lambda c: c.personality.traits.extraversion,
            )

        # Build speaker scores based on personality and recency
        speaker_scores: list[tuple[Character, float]] = []
        last_speaker = context.exchanges[-1].speaker

        for character in context.characters:
            t = character.personality.traits
            # Base score from personality traits
            extraversion_factor = t.extraversion * SPEAKER_EXTRAVERSION_WEIGHT
            assertiveness_factor = t.assertiveness * SPEAKER_ASSERTIVENESS_WEIGHT
            warmth_factor = t.warmth * SPEAKER_WARMTH_WEIGHT
            openness_factor = t.openness * SPEAKER_OPENNESS_WEIGHT

            score = extraversion_factor + assertiveness_factor + warmth_factor + openness_factor

            # MBTI compatibility bonus: characters who click with the last
            # speaker are more likely to jump in and engage.
            compat = mbti_compatibility(
                character.personality.mbti_type,
                last_speaker.personality.mbti_type,
            )
            score += compat * SPEAKER_COMPATIBILITY_WEIGHT

            # Reduce score if character just spoke (encourage turn-taking)
            if last_speaker.name == character.name:
                score *= SAME_SPEAKER_PENALTY

            # Slightly reduce score if character has spoken recently
            recent_speakers = [e.speaker.name for e in context.exchanges[-3:]]
            recent_count = sum(1 for s in recent_speakers if s == character.name)
            score *= max(RECENT_SPEAKER_FLOOR, 1.0 - (recent_count * RECENT_SPEAKER_DECAY))

            speaker_scores.append((character, score))

        # Pick character with highest score
        return max(speaker_scores, key=lambda x: x[1])[0]

    async def generate_sequential_dialogue(
        self,
        context: DialogueContext,
        num_exchanges: int = 5,
        include_internal_thoughts: bool = True,
    ) -> None:
        """Generate a sequence of dialogue exchanges with turn-taking.

        Automatically determines speaker order based on personality, generates
        natural dialogue, and maintains conversation history for coherence.

        Args:
            context: Dialogue context (modified in place).
            num_exchanges: Number of dialogue exchanges to generate.
            include_internal_thoughts: Whether to include internal monologue.

        Raises:
            RuntimeError: If LLM generation fails.

        """
        logger.info(
            "Starting dialogue: %d exchanges, %d characters, scene=%r",
            num_exchanges,
            len(context.characters),
            context.scene_description[:80],
        )
        for i in range(num_exchanges):
            try:
                # Determine who speaks next
                speaker = self.generate_next_speaker(context)
                logger.info("Exchange %d/%d — speaker: %s", i + 1, num_exchanges, speaker.name)

                # Generate dialogue and internal state
                dialogue, internal_thought = await self.generate_response(context, speaker)
                logger.debug("Dialogue: %.120s", dialogue)

                # Determine emotional context via LLM
                emotional_context = await self.infer_emotional_context(speaker, dialogue)
                logger.debug("%s emotion: %s", speaker.name, emotional_context)

                # Add to conversation history
                context.add_exchange(
                    speaker=speaker,
                    text=dialogue,
                    emotional_context=emotional_context,
                    internal_thought=internal_thought if include_internal_thoughts else "",
                )

                # Trigger memory condensation for the speaker if needed
                if speaker.memory_store.working_is_full:
                    logger.info("Memory condensation triggered for %s", speaker.name)
                    await self.memory_condenser.trigger_condensation(
                        speaker.memory_store, speaker
                    )

                # Apply personality micro-shifts based on experience
                await self.trait_shift_engine.process_exchange(
                    speaker, dialogue, emotional_context, i
                )

                # Periodic self-reflection
                reflection = await self.self_reflection_engine.maybe_reflect(speaker, i)
                if reflection:
                    logger.info("Self-reflection generated for %s at exchange %d", speaker.name, i)

                # Periodic dissonance detection
                dissonances = await self.dissonance_detector.maybe_detect(speaker, i)
                if dissonances:
                    logger.info(
                        "Dissonance detected for %s: %s",
                        speaker.name,
                        [(d.value, d.severity) for d in dissonances],
                    )
            except Exception as e:
                msg = f"Failed to generate exchange {i + 1}: {e}"
                raise RuntimeError(msg) from e
        logger.info("Dialogue complete: %d exchanges generated", len(context.exchanges))

    async def run_milestone_reviews(
        self,
        context: DialogueContext,
        trait_snapshots_start: dict[str, dict[str, float]],
    ) -> list:
        """Run milestone reviews for all characters after a scene ends.

        Args:
            context: The completed dialogue context.
            trait_snapshots_start: Mapping of character name → trait dict at scene start.

        Returns:
            List of ``MilestoneReview`` objects (one per character).

        """
        from character_creator.core.trait_evolution import MilestoneReview  # noqa: PLC0415

        reviews: list[MilestoneReview] = []
        for character in context.characters:
            snapshot = trait_snapshots_start.get(
                character.name,
                character.personality.traits.to_dict(),
            )
            review = await self.milestone_review_engine.review_scene(
                character=character,
                scene_description=context.scene_description,
                exchange_count=len(context.exchanges),
                trait_snapshot_start=snapshot,
            )
            # Apply the validated shifts
            MilestoneReviewEngine.apply_review(character, review)
            reviews.append(review)
        return reviews

    async def infer_emotional_context(self, character: Character, dialogue: str) -> str:
        """Infer emotional context using the LLM with a keyword-based fallback.

        Asks the LLM to classify emotion from the dialogue line in a single
        word.  Falls back to the fast heuristic if the LLM call fails.

        Args:
            character: Speaking character.
            dialogue: Generated dialogue text.

        Returns:
            Single-word description of emotional context.

        """
        prompt = substitute_prompt(
            DialoguePrompts.EMOTION_INFERENCE,
            character_name=character.name,
            emotional_stability=f"{character.personality.traits.emotional_stability:.2f}",
            current_emotional_state=character.current_emotional_state,
            dialogue_text=dialogue,
        )
        try:
            with llm_context("emotion"):
                result = await self.llm_provider.generate(prompt, temperature=EMOTION_INFERENCE_TEMPERATURE)
            word = result.strip().lower().rstrip(".")
            # Accept only single-word results that look like an emotion label
            if word and " " not in word:
                return word
        except Exception:  # noqa: BLE001
            logger.debug("LLM emotion inference failed, falling back to heuristic")
        # Fallback to heuristic
        return self._infer_emotional_context_heuristic(character, dialogue)

    @staticmethod
    def _infer_emotional_context_heuristic(character: Character, dialogue: str) -> str:
        """Fast keyword-based fallback for emotion inference.

        Args:
            character: Speaking character.
            dialogue: Generated dialogue text.

        Returns:
            Description of emotional context.

        """
        dialogue_lower = dialogue.lower()

        base_emotion = character.current_emotional_state.base

        if any(word in dialogue_lower for word in ["!", "excited", "amazing", "great", "love"]):
            return "enthusiastic"
        if any(word in dialogue_lower for word in ["sorry", "apologize", "regret", "wrong"]):
            return "apologetic"
        if any(word in dialogue_lower for word in ["?", "wonder", "curious", "interesting"]):
            return "curious"
        if any(word in dialogue_lower for word in ["no", "don't", "won't", "refuse"]):
            return "resistant"
        if any(word in dialogue_lower for word in ["...", "hmm", "well", "seem"]):
            return "thoughtful"

        return base_emotion or DEFAULT_EMOTIONAL_STATE
