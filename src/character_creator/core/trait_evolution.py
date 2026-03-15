"""Experience-triggered personality micro-shifts.

After each dialogue exchange the system classifies the *experience type*
(conflict, vulnerability, connection, …) and applies a small, deterministic
trait delta drawn from predefined influence vectors.  Over many exchanges
these micro-shifts accumulate into genuine character arcs.
"""

from __future__ import annotations

import json
import logging

from datetime import UTC, datetime
from enum import StrEnum
from typing import TYPE_CHECKING

from pydantic import BaseModel, Field

from character_creator.core.constants import (
    EXPERIENCE_CLASSIFICATION_TEMPERATURE,
    MAX_MILESTONE_SHIFT,
    MAX_TRAIT_DELTA_PER_EXCHANGE,
    MILESTONE_CONFIDENCE_THRESHOLD,
    MILESTONE_REVIEW_TEMPERATURE,
    TRAIT_MAX,
    TRAIT_MIN,
)
from character_creator.llm.metrics import llm_context
from character_creator.llm.prompts import TraitEvolutionPrompts, substitute_prompt

if TYPE_CHECKING:
    from character_creator.core.character import Character
    from character_creator.llm.providers import LLMProvider

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Enums & influence vectors
# ---------------------------------------------------------------------------


class ExperienceType(StrEnum):
    """Canonical experience types that can be extracted from dialogue."""

    CONFLICT = "conflict"
    VULNERABILITY = "vulnerability"
    SUCCESS = "success"
    REJECTION = "rejection"
    CONNECTION = "connection"
    BETRAYAL = "betrayal"
    DISCOVERY = "discovery"
    LOSS = "loss"
    HUMOR = "humor"
    NEUTRAL = "neutral"


# Mapping from experience type → {trait_name: delta}.
# Only traits that are affected are listed; unlisted traits get 0.
EXPERIENCE_INFLUENCE_VECTORS: dict[ExperienceType, dict[str, float]] = {
    ExperienceType.CONFLICT: {
        "assertiveness": 0.02,
        "agreeableness": -0.01,
        "emotional_stability": -0.01,
    },
    ExperienceType.VULNERABILITY: {
        "warmth": 0.02,
        "openness": 0.01,
        "emotional_stability": -0.02,
    },
    ExperienceType.SUCCESS: {
        "assertiveness": 0.01,
        "emotional_stability": 0.02,
    },
    ExperienceType.REJECTION: {
        "warmth": -0.01,
        "emotional_stability": -0.02,
        "openness": -0.01,
    },
    ExperienceType.CONNECTION: {
        "warmth": 0.02,
        "agreeableness": 0.01,
        "extraversion": 0.01,
    },
    ExperienceType.BETRAYAL: {
        "warmth": -0.02,
        "agreeableness": -0.02,
        "emotional_stability": -0.01,
    },
    ExperienceType.DISCOVERY: {
        "openness": 0.02,
        "humor_inclination": 0.01,
    },
    ExperienceType.LOSS: {
        "emotional_stability": -0.02,
        "warmth": 0.01,
    },
    ExperienceType.HUMOR: {
        "humor_inclination": 0.02,
        "warmth": 0.01,
        "formality": -0.01,
    },
    ExperienceType.NEUTRAL: {},
}


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


class TraitDelta(BaseModel):
    """A single recorded trait change with provenance."""

    trait_name: str
    delta: float
    source: ExperienceType
    exchange_index: int
    timestamp: datetime = Field(default_factory=lambda: datetime.now(tz=UTC))


class TraitHistory(BaseModel):
    """Full audit trail of personality shifts for one character."""

    character_name: str
    deltas: list[TraitDelta] = Field(default_factory=list)
    snapshots: list[dict[str, float]] = Field(default_factory=list)

    def record_delta(self, delta: TraitDelta) -> None:
        """Append a trait delta to the history."""
        self.deltas.append(delta)

    def record_snapshot(self, traits_dict: dict[str, float]) -> None:
        """Store a full trait snapshot for charting / diffing."""
        self.snapshots.append(traits_dict.copy())

    def net_deltas(self) -> dict[str, float]:
        """Sum all deltas per trait."""
        totals: dict[str, float] = {}
        for d in self.deltas:
            totals[d.trait_name] = totals.get(d.trait_name, 0.0) + d.delta
        return totals


# ---------------------------------------------------------------------------
# Experience classifier
# ---------------------------------------------------------------------------

# Keywords used by the heuristic fallback
_EXPERIENCE_KEYWORDS: dict[ExperienceType, list[str]] = {
    ExperienceType.CONFLICT: ["argue", "disagree", "fight", "angry", "furious", "clash", "confront"],
    ExperienceType.VULNERABILITY: ["afraid", "scared", "admit", "confess", "open up", "trust", "vulnerable"],
    ExperienceType.SUCCESS: ["succeed", "accomplish", "win", "proud", "achieve", "well done", "great job"],
    ExperienceType.REJECTION: ["reject", "dismiss", "ignore", "unwanted", "alone", "excluded"],
    ExperienceType.CONNECTION: ["connect", "bond", "friend", "together", "understand", "close", "love"],
    ExperienceType.BETRAYAL: ["betray", "lie", "cheat", "deceive", "stab", "backstab", "broke my trust"],
    ExperienceType.DISCOVERY: ["discover", "learn", "realise", "realize", "understand", "insight", "eureka"],
    ExperienceType.LOSS: ["loss", "lost", "gone", "miss", "grief", "mourn", "death"],
    ExperienceType.HUMOR: ["laugh", "funny", "joke", "hilarious", "haha", "lol", "amuse"],
}


class ExperienceClassifier:
    """Classifies dialogue exchanges by experience type."""

    def __init__(self, llm_provider: LLMProvider) -> None:
        self.llm_provider = llm_provider

    async def classify(
        self,
        character: Character,
        exchange_text: str,
        emotional_context: str,
    ) -> ExperienceType:
        """Classify an exchange via LLM, falling back to heuristic."""
        prompt = substitute_prompt(
            TraitEvolutionPrompts.EXPERIENCE_CLASSIFICATION,
            character_name=character.name,
            emotional_context=emotional_context,
            exchange_text=exchange_text,
            experience_types=", ".join(ExperienceType),
        )
        try:
            with llm_context("experience_classification"):
                raw = await self.llm_provider.generate(
                    prompt,
                    temperature=EXPERIENCE_CLASSIFICATION_TEMPERATURE,
                )
            label = raw.strip().lower().rstrip(".")
            return ExperienceType(label)
        except (ValueError, OSError, TypeError):
            logger.debug("LLM experience classification failed; using heuristic")
        result = self.classify_heuristic(exchange_text, emotional_context)
        logger.debug("Experience classified as %s for %s", result.value, character.name)
        return result

    @staticmethod
    def classify_heuristic(
        exchange_text: str,
        emotional_context: str,
    ) -> ExperienceType:
        """Keyword-based fallback classifier."""
        text = f"{exchange_text} {emotional_context}".lower()
        best: ExperienceType = ExperienceType.NEUTRAL
        best_count = 0
        for exp_type, keywords in _EXPERIENCE_KEYWORDS.items():
            count = sum(1 for kw in keywords if kw in text)
            if count > best_count:
                best = exp_type
                best_count = count
        return best


# ---------------------------------------------------------------------------
# Trait shift engine
# ---------------------------------------------------------------------------


def apply_trait_delta(
    traits_dict: dict[str, float],
    deltas: dict[str, float],
) -> dict[str, float]:
    """Apply clamped deltas to a traits dict, returning actual changes made.

    Mutates *traits_dict* in-place and returns a dict of the actual deltas
    applied (after clamping).
    """
    actual: dict[str, float] = {}
    for trait, raw_delta in deltas.items():
        if trait not in traits_dict:
            continue
        clamped = max(-MAX_TRAIT_DELTA_PER_EXCHANGE, min(raw_delta, MAX_TRAIT_DELTA_PER_EXCHANGE))
        old = traits_dict[trait]
        new = max(TRAIT_MIN, min(old + clamped, TRAIT_MAX))
        if new != old:
            traits_dict[trait] = new
            actual[trait] = new - old
    return actual


class TraitShiftEngine:
    """Orchestrates experience classification and trait micro-shifts."""

    def __init__(self, llm_provider: LLMProvider) -> None:
        self.classifier = ExperienceClassifier(llm_provider)
        self._histories: dict[str, TraitHistory] = {}

    def _get_history(self, character_name: str) -> TraitHistory:
        if character_name not in self._histories:
            self._histories[character_name] = TraitHistory(character_name=character_name)
        return self._histories[character_name]

    def get_history(self, character_name: str) -> TraitHistory:
        """Return (or create) the trait history for a character."""
        return self._get_history(character_name)

    async def process_exchange(
        self,
        character: Character,
        exchange_text: str,
        emotional_context: str,
        exchange_index: int,
    ) -> list[TraitDelta]:
        """Classify an exchange and apply trait micro-shifts.

        Returns the list of recorded ``TraitDelta`` objects.
        """
        exp_type = await self.classifier.classify(
            character, exchange_text, emotional_context
        )
        influence = EXPERIENCE_INFLUENCE_VECTORS.get(exp_type, {})
        if not influence:
            return []

        # Apply deltas to the character's traits
        traits_dict = character.personality.traits.to_dict()
        actual = apply_trait_delta(traits_dict, influence)

        # Write back mutated values
        for trait_name, new_val in traits_dict.items():
            if hasattr(character.personality.traits, trait_name):
                setattr(character.personality.traits, trait_name, new_val)

        # Record
        history = self._get_history(character.name)
        recorded: list[TraitDelta] = []
        for trait_name, delta_val in actual.items():
            td = TraitDelta(
                trait_name=trait_name,
                delta=delta_val,
                source=exp_type,
                exchange_index=exchange_index,
            )
            history.record_delta(td)
            recorded.append(td)

        return recorded


# ---------------------------------------------------------------------------
# Milestone review models
# ---------------------------------------------------------------------------


class PersonalityShift(BaseModel):
    """A single trait shift proposed (and validated) by a milestone review."""

    trait_name: str
    old_value: float
    new_value: float
    justification: str = ""
    confidence: float = 0.5


class MilestoneReview(BaseModel):
    """Result of an LLM-adjudicated scene-end personality review."""

    character_name: str
    shifts: list[PersonalityShift] = Field(default_factory=list)
    narrative_summary: str = ""
    scene_description: str = ""
    exchange_count: int = 0
    timestamp: datetime = Field(default_factory=lambda: datetime.now(tz=UTC))


# ---------------------------------------------------------------------------
# Milestone review engine
# ---------------------------------------------------------------------------


class MilestoneReviewEngine:
    """Runs an LLM review at scene boundaries to ratify/amplify/reverse shifts."""

    def __init__(self, llm_provider: LLMProvider) -> None:
        self.llm_provider = llm_provider

    async def review_scene(
        self,
        character: Character,
        scene_description: str,
        exchange_count: int,
        trait_snapshot_start: dict[str, float],
    ) -> MilestoneReview:
        """Run a milestone review for one character after a scene.

        The LLM sees the trait snapshot at scene start, the current traits,
        the accumulated micro-shift history, and the character's self-model.
        It proposes final shifts — which are validated and clamped here.
        """
        current_traits = character.personality.traits.to_dict()

        # Format trait deltas
        delta_lines: list[str] = []
        if character.trait_history:
            delta_lines.extend(
                f"  {d.trait_name}: {d.delta:+.3f} (from {d.source.value} at exchange {d.exchange_index})"
                for d in character.trait_history.deltas
            )
        deltas_text = "\n".join(delta_lines) or "(no micro-shifts recorded)"

        # Self-model text
        self_model_text = "(no self-model generated yet)"
        if character.self_model:
            sm = character.self_model
            parts = [f"Self-concept: {sm.self_concept}"]
            if sm.emotional_awareness:
                parts.append(f"Emotional awareness: {sm.emotional_awareness}")
            if sm.value_tensions:
                parts.append(f"Value tensions: {', '.join(sm.value_tensions)}")
            if sm.growth_edges:
                parts.append(f"Growth edges: {', '.join(sm.growth_edges)}")
            self_model_text = "\n".join(parts)

        # Format trait snapshots
        def _fmt_traits(d: dict[str, float]) -> str:
            return ", ".join(f"{k}: {v:.3f}" for k, v in sorted(d.items()))

        prompt = substitute_prompt(
            TraitEvolutionPrompts.MILESTONE_REVIEW,
            character_name=character.name,
            character_profile=character.get_character_profile(),
            self_model=self_model_text,
            scene_description=scene_description,
            exchange_count=str(exchange_count),
            traits_start=_fmt_traits(trait_snapshot_start),
            traits_current=_fmt_traits(current_traits),
            trait_deltas=deltas_text,
        )

        try:
            with llm_context("milestone_review"):
                raw = await self.llm_provider.generate(
                    prompt,
                    temperature=MILESTONE_REVIEW_TEMPERATURE,
                )
            data = json.loads(raw)
        except (json.JSONDecodeError, OSError, ValueError, TypeError):
            logger.warning("Milestone review LLM call failed; returning empty review")
            return MilestoneReview(
                character_name=character.name,
                scene_description=scene_description,
                exchange_count=exchange_count,
                narrative_summary="The scene concluded without a clear personality arc.",
            )

        # Parse and validate shifts
        raw_shifts = data.get("shifts", [])
        validated: list[PersonalityShift] = []
        for s in raw_shifts:
            try:
                shift = PersonalityShift(
                    trait_name=str(s.get("trait_name", "")),
                    old_value=float(s.get("old_value", 0.0)),
                    new_value=float(s.get("new_value", 0.0)),
                    justification=str(s.get("justification", "")),
                    confidence=float(s.get("confidence", 0.0)),
                )
            except (ValueError, TypeError):
                continue

            # Reject low-confidence shifts
            if shift.confidence < MILESTONE_CONFIDENCE_THRESHOLD:
                continue

            # Clamp shift magnitude
            delta = shift.new_value - shift.old_value
            if abs(delta) > MAX_MILESTONE_SHIFT:
                clamped = MAX_MILESTONE_SHIFT if delta > 0 else -MAX_MILESTONE_SHIFT
                shift.new_value = shift.old_value + clamped

            # Clamp to trait bounds
            shift.new_value = max(TRAIT_MIN, min(shift.new_value, TRAIT_MAX))

            validated.append(shift)

        return MilestoneReview(
            character_name=character.name,
            shifts=validated,
            narrative_summary=str(data.get("narrative_summary", "")),
            scene_description=scene_description,
            exchange_count=exchange_count,
        )

    @staticmethod
    def apply_review(
        character: Character,
        review: MilestoneReview,
    ) -> dict[str, float]:
        """Apply validated shifts to a character's traits.

        Returns a dict of {trait_name: actual_delta_applied}.
        """
        applied: dict[str, float] = {}
        for shift in review.shifts:
            if not hasattr(character.personality.traits, shift.trait_name):
                continue
            old = getattr(character.personality.traits, shift.trait_name)
            new = max(TRAIT_MIN, min(shift.new_value, TRAIT_MAX))
            if new != old:
                setattr(character.personality.traits, shift.trait_name, new)
                applied[shift.trait_name] = new - old
        return applied
