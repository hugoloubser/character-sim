"""Core character class integrating personality, memory, and identity."""

from typing import Any

from pydantic import BaseModel, Field, field_validator

from character_creator.core.constants import DEFAULT_AGE
from character_creator.core.emotional_state import EmotionalState
from character_creator.core.memory import Background
from character_creator.core.memory_tiered import MemoryStore
from character_creator.core.personality import Personality
from character_creator.core.self_model import CognitiveDissonance, SelfModel
from character_creator.core.trait_evolution import TraitHistory


class Character(BaseModel):
    """A complete character with personality, background, and identity.

    Characters are the core entities in the system. Each has a rich personality,
    detailed background, and the ability to interact through dialogue while
    maintaining consistent behavior and emotional reactions based on their
    lived experience.
    """

    name: str
    description: str  # Brief physical/visual description
    personality: Personality = Field(default_factory=Personality)
    background: Background = Field(
        default_factory=lambda: Background(age=DEFAULT_AGE, origin="", occupation="")
    )
    internal_monologue: list[str] = Field(default_factory=list)
    current_emotional_state: EmotionalState = Field(default_factory=EmotionalState.neutral)
    memory_store: MemoryStore = Field(default_factory=MemoryStore)
    trait_history: TraitHistory | None = Field(default=None)
    self_model: SelfModel | None = Field(default=None)
    self_model_history: list[SelfModel] = Field(default_factory=list)
    active_dissonances: list[CognitiveDissonance] = Field(default_factory=list)
    conversation_history: list[dict[str, str]] = Field(default_factory=list)

    @field_validator("current_emotional_state", mode="before")
    @classmethod
    def _coerce_emotional_state(cls, v: Any) -> EmotionalState:
        """Accept bare strings for backward compatibility."""
        if isinstance(v, str):
            return EmotionalState.from_string(v)
        if isinstance(v, dict):
            return EmotionalState.model_validate(v)
        return v

    def to_dict(self) -> dict[str, Any]:
        """Convert character to dictionary format."""
        d = self.model_dump()
        d["personality"] = self.personality.to_dict()
        d["background"] = self.background.to_dict()
        # Store emotional state as string for DB/JSON backward compat
        d["current_emotional_state"] = str(self.current_emotional_state)
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Character":
        """Create character from dictionary."""
        return cls.model_validate(data)

    def add_memory_to_history(self, speaker: str, text: str) -> None:
        """Add a conversation exchange to both working memory and legacy history."""
        self.conversation_history.append({"speaker": speaker, "text": text})
        self.memory_store.add_exchange(speaker, text)

    def get_character_profile(self) -> str:
        """Generate a comprehensive character profile for LLM context."""
        mbti = self.personality.mbti_type
        profile_parts: list[str] = [
            f"# {self.name}",
            f"\n## Physical Description\n{self.description}",
            f"\n## Personality\n{self.personality.describe_briefly()}",
            f"\n## MBTI Profile — {mbti.value} ({self.personality.mbti_archetype})"
            f"\nCommunication style: {self.personality.communication_style}",
        ]

        if self.personality.values.priority_keywords:
            profile_parts.append(
                f"\n## Core Values\n{', '.join(self.personality.values.priority_keywords)}"
            )

        profile_parts.append(f"\n## Background\n{self.background.get_context_summary()}")

        # Include tiered memory context when available
        long_term_ctx = self.memory_store.get_long_term_summaries()
        if long_term_ctx:
            profile_parts.append(f"\n## Long-Term Memories\n{long_term_ctx}")

        short_term_ctx = self.memory_store.get_short_term_summaries()
        if short_term_ctx:
            profile_parts.append(f"\n## Recent Memories\n{short_term_ctx}")

        # Working memory (verbatim recent exchanges)
        working_text = self.memory_store.get_working_text()
        if working_text:
            profile_parts.append(f"\n## Recent Conversation\n{working_text}")
        elif self.conversation_history:
            # Fallback for characters without tiered memory populated
            history_str = "\n".join(
                [f"{h['speaker']}: {h['text']}" for h in self.conversation_history[-5:]]
            )
            profile_parts.append(f"\n## Recent Conversation\n{history_str}")

        profile_parts.append(
            f"\n## Current Emotional State\n{self.current_emotional_state}"
        )

        # Self-model section (reflective self-awareness)
        if self.self_model:
            sm = self.self_model
            self_section = f"\n## Self-Perception\n{sm.self_concept}"
            if sm.emotional_awareness:
                self_section += f"\nEmotional awareness: {sm.emotional_awareness}"
            if sm.value_tensions:
                self_section += f"\nValue tensions: {', '.join(sm.value_tensions)}"
            if sm.growth_edges:
                self_section += f"\nGrowth edges: {', '.join(sm.growth_edges)}"
            profile_parts.append(self_section)

        # Active cognitive dissonances (internal tensions)
        if self.active_dissonances:
            tensions = "\n".join(
                f"- Tension: values {d.value} but {d.behaviour} (severity {d.severity:.1f})"
                for d in self.active_dissonances
                if not d.resolved
            )
            if tensions:
                profile_parts.append(f"\n## Internal Tensions\n{tensions}")

        return "".join(profile_parts)

    def get_character_self_perception(self) -> str:
        """Generate dynamic self-perception reflecting current emotional state.

        This is the character's present-moment self-understanding, influenced by
        their emotional state, recent conversations, and core values. Unlike the
        static profile, this evolves moment-to-moment, affecting how they interpret
        situations and respond to others.

        Returns:
            Dynamic self-perception from character's perspective.

        """
        recent_events = [h.get("text", "") for h in self.conversation_history[-5:]]
        return self.personality.describe_self(
            emotional_state=self.current_emotional_state.base,
            recent_events=recent_events if recent_events else None,
        )

    def update_emotional_state(self, new_state: str | EmotionalState) -> None:
        """Update the character's current emotional state.

        Args:
            new_state: A base-emotion string or an ``EmotionalState`` instance.

        """
        if isinstance(new_state, str):
            new_state = EmotionalState.from_string(new_state)
        self.current_emotional_state = new_state

    def add_internal_thought(self, thought: str) -> None:
        """Add an internal thought reflecting the character's mind."""
        self.internal_monologue.append(thought)
