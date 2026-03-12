"""Reflective self-model for character self-awareness.

Every N exchanges, a character reflects on themselves — what they feel,
what tensions they notice between their values and recent behaviour,
how they see themselves evolving.  The resulting ``SelfModel`` is injected
into the dialogue prompt so the character can reference its own
self-awareness during conversation.
"""

from __future__ import annotations

import json
import logging

from datetime import UTC, datetime
from typing import TYPE_CHECKING

from pydantic import BaseModel, Field

from character_creator.core.constants import (
    BEHAVIOUR_EXTRACTION_INTERVAL,
    BEHAVIOUR_EXTRACTION_TEMPERATURE,
    DISSONANCE_DETECTION_TEMPERATURE,
    DISSONANCE_SEVERITY_THRESHOLD,
    MAX_ACTIVE_DISSONANCES,
    MAX_SELF_MODEL_HISTORY,
    SELF_REFLECTION_INTERVAL,
    SELF_REFLECTION_TEMPERATURE,
)
from character_creator.llm.metrics import llm_context
from character_creator.llm.prompts import (
    DissonancePrompts,
    SelfReflectionPrompts,
    substitute_prompt,
)

if TYPE_CHECKING:
    from character_creator.core.character import Character
    from character_creator.llm.providers import LLMProvider

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


class SelfModel(BaseModel):
    """A character's current self-concept produced by reflection."""

    self_concept: str = ""
    emotional_awareness: str = ""
    value_tensions: list[str] = Field(default_factory=list)
    growth_edges: list[str] = Field(default_factory=list)
    generated_at_exchange: int = 0
    timestamp: datetime = Field(default_factory=lambda: datetime.now(tz=UTC))


# ---------------------------------------------------------------------------
# Reflection engine
# ---------------------------------------------------------------------------


class SelfReflectionEngine:
    """Generates periodic self-reflections for characters."""

    def __init__(self, llm_provider: LLMProvider) -> None:
        self.llm_provider = llm_provider

    @staticmethod
    def should_reflect(
        exchange_index: int,
        last_reflection_exchange: int | None,
    ) -> bool:
        """Return True when it's time for a new self-reflection."""
        if last_reflection_exchange is None:
            return exchange_index >= SELF_REFLECTION_INTERVAL
        return (exchange_index - last_reflection_exchange) >= SELF_REFLECTION_INTERVAL

    async def generate_reflection(
        self,
        character: Character,
        exchange_index: int,
    ) -> SelfModel:
        """Ask the LLM to produce a self-model for the character."""
        # Gather context
        recent_monologue = character.internal_monologue[-5:] if character.internal_monologue else []
        recent_exchanges = character.memory_store.get_working_text() or "(no recent exchanges)"
        trait_summary = character.personality.describe_briefly()
        values_str = ", ".join(character.personality.values.priority_keywords) or "none specified"

        # Trait history summary (if available)
        trait_history_str = ""
        if character.trait_history and character.trait_history.deltas:
            net = character.trait_history.net_deltas()
            changes = [f"{k}: {v:+.3f}" for k, v in net.items() if abs(v) > 0.001]
            trait_history_str = ", ".join(changes) if changes else "no significant changes"
        else:
            trait_history_str = "no changes recorded yet"

        prompt = substitute_prompt(
            SelfReflectionPrompts.SELF_REFLECTION,
            character_name=character.name,
            personality_summary=trait_summary,
            emotional_state=str(character.current_emotional_state),
            core_values=values_str,
            trait_changes=trait_history_str,
            recent_monologue="\n".join(recent_monologue) or "(no internal thoughts yet)",
            recent_exchanges=recent_exchanges,
        )

        try:
            with llm_context("self_reflection"):
                raw = await self.llm_provider.generate(
                    prompt,
                    temperature=SELF_REFLECTION_TEMPERATURE,
                )
            data = json.loads(raw)
        except (json.JSONDecodeError, OSError, ValueError, TypeError):
            logger.warning("Self-reflection LLM call failed; using minimal model")
            data = {
                "self_concept": f"{character.name} is still forming their self-image.",
                "emotional_awareness": f"Currently feeling {character.current_emotional_state.base}.",
                "value_tensions": [],
                "growth_edges": [],
            }

        return SelfModel(
            self_concept=str(data.get("self_concept", "")),
            emotional_awareness=str(data.get("emotional_awareness", "")),
            value_tensions=data.get("value_tensions", []),
            growth_edges=data.get("growth_edges", []),
            generated_at_exchange=exchange_index,
        )

    async def maybe_reflect(
        self,
        character: Character,
        exchange_index: int,
    ) -> SelfModel | None:
        """Generate a reflection if it's time, updating the character in place.

        Returns the new ``SelfModel`` if one was generated, else ``None``.
        """
        last = (
            character.self_model.generated_at_exchange
            if character.self_model
            else None
        )
        if not self.should_reflect(exchange_index, last):
            return None

        model = await self.generate_reflection(character, exchange_index)

        # Rotate history
        if character.self_model is not None:
            character.self_model_history.append(character.self_model)
            if len(character.self_model_history) > MAX_SELF_MODEL_HISTORY:
                character.self_model_history = character.self_model_history[
                    -MAX_SELF_MODEL_HISTORY:
                ]

        character.self_model = model
        return model


# ---------------------------------------------------------------------------
# Dissonance detection models
# ---------------------------------------------------------------------------


class BehaviourTheme(BaseModel):
    """A behavioural pattern extracted from recent dialogue."""

    theme: str
    confidence: float = 0.5
    source_exchanges: list[int] = Field(default_factory=list)


class CognitiveDissonance(BaseModel):
    """A detected conflict between a character's values and behaviour."""

    value: str
    behaviour: str
    severity: float = 0.5
    detected_at_exchange: int = 0
    resolved: bool = False
    resolution: str | None = None


# ---------------------------------------------------------------------------
# Dissonance detector
# ---------------------------------------------------------------------------


class DissonanceDetector:
    """Periodically extracts behaviour themes and detects value conflicts."""

    def __init__(self, llm_provider: LLMProvider) -> None:
        self.llm_provider = llm_provider
        self._last_check: dict[str, int] = {}

    @staticmethod
    def should_check(
        exchange_index: int,
        last_check: int | None,
    ) -> bool:
        """Return True when it's time to run dissonance detection."""
        if last_check is None:
            return exchange_index >= BEHAVIOUR_EXTRACTION_INTERVAL
        return (exchange_index - last_check) >= BEHAVIOUR_EXTRACTION_INTERVAL

    async def extract_behaviour_themes(
        self,
        character: Character,
        recent_exchanges: list[dict[str, str]],
    ) -> list[BehaviourTheme]:
        """Ask the LLM to identify behavioural patterns in recent dialogue."""
        exchanges_text = "\n".join(
            f"{e.get('speaker', '?')}: {e.get('text', '')}" for e in recent_exchanges
        )
        prompt = substitute_prompt(
            DissonancePrompts.BEHAVIOUR_EXTRACTION,
            character_name=character.name,
            recent_exchanges=exchanges_text,
        )
        try:
            with llm_context("behaviour_extraction"):
                raw = await self.llm_provider.generate(
                    prompt,
                    temperature=BEHAVIOUR_EXTRACTION_TEMPERATURE,
                )
            data = json.loads(raw)
            if not isinstance(data, list):
                data = data.get("themes", [])
        except (json.JSONDecodeError, OSError, ValueError, TypeError):
            logger.debug("Behaviour extraction LLM call failed; returning empty")
            return []

        themes: list[BehaviourTheme] = []
        for item in data:
            try:
                themes.append(
                    BehaviourTheme(
                        theme=str(item.get("theme", "")),
                        confidence=float(item.get("confidence", 0.5)),
                    )
                )
            except (ValueError, TypeError):
                continue
        return themes

    async def detect_dissonance(
        self,
        character: Character,
        themes: list[BehaviourTheme],
    ) -> list[CognitiveDissonance]:
        """Compare behaviour themes against the character's stated values."""
        if not themes or not character.personality.values.priority_keywords:
            return []

        themes_text = "\n".join(
            f"- {t.theme} (confidence: {t.confidence:.2f})" for t in themes
        )
        values_text = ", ".join(character.personality.values.priority_keywords)

        prompt = substitute_prompt(
            DissonancePrompts.DISSONANCE_DETECTION,
            character_name=character.name,
            core_values=values_text,
            behaviour_themes=themes_text,
        )
        try:
            with llm_context("dissonance_detection"):
                raw = await self.llm_provider.generate(
                    prompt,
                    temperature=DISSONANCE_DETECTION_TEMPERATURE,
                )
            data = json.loads(raw)
            if not isinstance(data, list):
                data = data.get("dissonances", [])
        except (json.JSONDecodeError, OSError, ValueError, TypeError):
            logger.debug("Dissonance detection LLM call failed; returning empty")
            return []

        dissonances: list[CognitiveDissonance] = []
        for item in data:
            try:
                severity = float(item.get("severity", 0.0))
            except (ValueError, TypeError):
                continue
            if severity < DISSONANCE_SEVERITY_THRESHOLD:
                continue
            dissonances.append(
                CognitiveDissonance(
                    value=str(item.get("value", "")),
                    behaviour=str(item.get("behaviour", "")),
                    severity=severity,
                )
            )
        return dissonances

    async def maybe_detect(
        self,
        character: Character,
        exchange_index: int,
    ) -> list[CognitiveDissonance]:
        """Run detection if it's time, updating character's active_dissonances.

        Returns newly detected dissonances (may be empty).
        """
        last = self._last_check.get(character.name)
        if not self.should_check(exchange_index, last):
            return []

        logger.debug("Running dissonance detection for %s at exchange %d", character.name, exchange_index)
        self._last_check[character.name] = exchange_index

        recent = character.conversation_history[-6:]
        themes = await self.extract_behaviour_themes(character, recent)
        new_dissonances = await self.detect_dissonance(character, themes)

        # Tag exchange index on each
        for d in new_dissonances:
            d.detected_at_exchange = exchange_index

        # Merge into character's active list, respecting the cap
        for d in new_dissonances:
            if len(character.active_dissonances) >= MAX_ACTIVE_DISSONANCES:
                # Evict lowest-severity existing dissonance if new one is more severe
                min_existing = min(character.active_dissonances, key=lambda x: x.severity)
                if d.severity > min_existing.severity:
                    character.active_dissonances.remove(min_existing)
                else:
                    continue
            character.active_dissonances.append(d)

        return new_dissonances
