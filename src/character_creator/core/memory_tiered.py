"""Tiered memory system with decay and perspective-biased condensation.

Replaces unbounded conversation_history with three memory tiers:
- **Working memory**: verbatim recent exchanges
- **Short-term memory**: LLM-summarised condensed memories
- **Long-term memory**: further-condensed thematic patterns

Memories decay over time unless anchored by emotional weight or
relevance to core values. Summarisation is biased by the character's
current emotional state and personality, modelling perspective warping.
"""

from __future__ import annotations

import uuid

from datetime import UTC, datetime
from enum import StrEnum
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field

from character_creator.core.constants import (
    CONDENSATION_TEMPERATURE,
    CORE_MEMORY_FLOOR,
    CORE_MEMORY_SALIENCE,
    DECAY_EMOTIONAL_ANCHOR,
    DECAY_VALUE_ANCHOR_MULTIPLIER,
    DEFAULT_DECAY_RATE,
    LONG_TERM_CAPACITY,
    MAX_DISTORTION_BUDGET,
    MIN_SALIENCE_THRESHOLD,
    SHORT_TERM_CAPACITY,
    WORKING_MEMORY_CAPACITY,
)
from character_creator.llm.metrics import llm_context

if TYPE_CHECKING:
    from character_creator.core.character import Character
    from character_creator.llm.providers import LLMProvider


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class MemoryTier(StrEnum):
    """The three tiers of the memory system."""

    WORKING = "working"
    SHORT_TERM = "short_term"
    LONG_TERM = "long_term"


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


class CondensedMemory(BaseModel):
    """A summarised memory produced by condensation.

    Carries metadata about its perspective bias, emotional tone, and
    decay state so the system can manage salience over time.
    """

    memory_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    tier: MemoryTier
    summary: str
    emotional_tone: str = "neutral"
    salience: float = Field(default=0.5, ge=0.0, le=1.0)
    perspective_bias: str | None = None
    original_exchange_count: int = 0
    source_speakers: list[str] = Field(default_factory=list)
    topics: list[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=lambda: datetime.now(tz=UTC))
    last_accessed: datetime = Field(default_factory=lambda: datetime.now(tz=UTC))
    decay_rate: float = Field(default=DEFAULT_DECAY_RATE)
    current_salience: float = Field(default=0.5, ge=0.0, le=1.0)
    distortion_budget: float = Field(default=0.0, ge=0.0)
    protected: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a JSON-friendly dict."""
        d = self.model_dump()
        d["created_at"] = self.created_at.isoformat()
        d["last_accessed"] = self.last_accessed.isoformat()
        return d


class MemoryStore(BaseModel):
    """Three-tier memory container for a character.

    *working* holds verbatim exchanges (most recent).
    *short_term* and *long_term* hold ``CondensedMemory`` objects at
    increasing levels of abstraction.
    *ground_truth_log* is an append-only, never-distorted record of
    every exchange for debugging and audit.
    """

    working: list[dict[str, str]] = Field(default_factory=list)
    short_term: list[CondensedMemory] = Field(default_factory=list)
    long_term: list[CondensedMemory] = Field(default_factory=list)
    ground_truth_log: list[dict[str, str]] = Field(default_factory=list)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def add_exchange(self, speaker: str, text: str) -> None:
        """Record a single exchange and a ground-truth copy."""
        entry = {"speaker": speaker, "text": text}
        self.working.append(entry)
        self.ground_truth_log.append(entry.copy())

    @property
    def working_is_full(self) -> bool:
        """True when working memory has reached capacity."""
        return len(self.working) >= WORKING_MEMORY_CAPACITY

    @property
    def short_term_is_full(self) -> bool:
        """True when short-term tier has reached capacity."""
        return len(self.short_term) >= SHORT_TERM_CAPACITY

    def get_working_text(self) -> str:
        """Return working memory formatted as dialogue lines."""
        return "\n".join(f"{e['speaker']}: {e['text']}" for e in self.working)

    def get_short_term_summaries(self) -> str:
        """Return short-term memory summaries as a block of text."""
        if not self.short_term:
            return ""
        return "\n".join(
            f"- {m.summary} (tone: {m.emotional_tone})" for m in self.short_term
        )

    def get_long_term_summaries(self) -> str:
        """Return long-term memory summaries as a block of text."""
        if not self.long_term:
            return ""
        return "\n".join(
            f"- {m.summary} (tone: {m.emotional_tone})" for m in self.long_term
        )


# ---------------------------------------------------------------------------
# Condensation engine
# ---------------------------------------------------------------------------

import json  # noqa: E402
import logging  # noqa: E402

from character_creator.llm.prompts import MemoryPrompts, substitute_prompt  # noqa: E402

logger = logging.getLogger(__name__)


class MemoryCondenser:
    """Orchestrates memory condensation and decay across tiers.

    Uses an LLM to summarise exchanges through the character's current
    perspective, producing perspective-biased condensed memories.
    """

    def __init__(self, llm_provider: LLMProvider) -> None:
        self.llm_provider = llm_provider

    # ------------------------------------------------------------------
    # Condensation
    # ------------------------------------------------------------------

    async def condense_working_to_short_term(
        self,
        character: Character,
        exchanges: list[dict[str, str]],
    ) -> CondensedMemory:
        """Summarise verbatim exchanges into one short-term memory.

        The summary is coloured by the character's current personality
        and emotional state (perspective bias).
        """
        exchanges_text = "\n".join(f"{e['speaker']}: {e['text']}" for e in exchanges)
        speakers = sorted({e["speaker"] for e in exchanges})

        prompt = substitute_prompt(
            MemoryPrompts.WORKING_TO_SHORT_TERM,
            character_name=character.name,
            personality_summary=character.personality.describe_briefly(),
            emotional_state=str(character.current_emotional_state),
            core_values=", ".join(character.personality.values.priority_keywords) or "none specified",
            exchanges=exchanges_text,
        )

        try:
            with llm_context("memory_condensation"):
                raw = await self.llm_provider.generate(
                    prompt,
                    temperature=CONDENSATION_TEMPERATURE,
                )
            data = json.loads(raw)
        except (json.JSONDecodeError, OSError, ValueError, TypeError):
            logger.warning("LLM condensation failed; using heuristic summary")
            data = self._heuristic_summary(exchanges, character)

        return CondensedMemory(
            tier=MemoryTier.SHORT_TERM,
            summary=str(data.get("summary", "Exchange summary unavailable.")),
            emotional_tone=str(data.get("emotional_tone", "neutral")),
            salience=float(data.get("salience", 0.5)),
            perspective_bias=data.get("perspective_bias"),
            original_exchange_count=len(exchanges),
            source_speakers=speakers,
            topics=data.get("topics", []),
            current_salience=float(data.get("salience", 0.5)),
        )

    async def condense_short_to_long_term(
        self,
        character: Character,
        memories: list[CondensedMemory],
    ) -> CondensedMemory:
        """Merge several short-term memories into one long-term memory."""
        memories_text = "\n".join(
            f"- {m.summary} (tone: {m.emotional_tone}, salience: {m.current_salience:.2f})"
            for m in memories
        )
        speakers = sorted({s for m in memories for s in m.source_speakers})
        total_exchanges = sum(m.original_exchange_count for m in memories)
        # Accumulate distortion: average existing budgets + 1 for this condensation step
        avg_distortion = (
            sum(m.distortion_budget for m in memories) / len(memories) if memories else 0.0
        )
        new_distortion = min(avg_distortion + 1.0, MAX_DISTORTION_BUDGET)

        prompt = substitute_prompt(
            MemoryPrompts.SHORT_TO_LONG_TERM,
            character_name=character.name,
            personality_summary=character.personality.describe_briefly(),
            emotional_state=str(character.current_emotional_state),
            core_values=", ".join(character.personality.values.priority_keywords) or "none specified",
            memories=memories_text,
        )

        try:
            with llm_context("memory_promotion"):
                raw = await self.llm_provider.generate(
                    prompt,
                    temperature=CONDENSATION_TEMPERATURE,
                )
            data = json.loads(raw)
        except (json.JSONDecodeError, OSError, ValueError, TypeError):
            logger.warning("LLM long-term condensation failed; using heuristic merge")
            data = self._heuristic_merge(memories, character)

        return CondensedMemory(
            tier=MemoryTier.LONG_TERM,
            summary=str(data.get("summary", "Merged memory summary.")),
            emotional_tone=str(data.get("emotional_tone", "neutral")),
            salience=float(data.get("salience", 0.5)),
            perspective_bias=data.get("perspective_bias"),
            original_exchange_count=total_exchanges,
            source_speakers=speakers,
            topics=data.get("topics", []),
            current_salience=float(data.get("salience", 0.5)),
            distortion_budget=new_distortion,
        )

    # ------------------------------------------------------------------
    # Decay
    # ------------------------------------------------------------------

    @staticmethod
    def apply_decay(
        memories: list[CondensedMemory],
        character: Character,
    ) -> list[CondensedMemory]:
        """Reduce salience on each memory, honouring anchors.

        Returns only the memories that survive (above threshold).
        """
        value_keywords = {
            kw.lower() for kw in character.personality.values.priority_keywords
        }
        surviving: list[CondensedMemory] = []

        for mem in memories:
            rate = mem.decay_rate

            # Emotional anchor: high-weight memories decay slower
            if mem.salience >= DECAY_EMOTIONAL_ANCHOR:
                rate *= 0.5

            # Value anchor: memories whose topics overlap core values decay slower
            mem_topics = {t.lower() for t in mem.topics}
            if mem_topics & value_keywords:
                rate *= DECAY_VALUE_ANCHOR_MULTIPLIER

            new_salience = mem.current_salience - rate

            # Core memory protection
            if mem.salience >= CORE_MEMORY_SALIENCE or mem.protected:
                new_salience = max(new_salience, CORE_MEMORY_FLOOR)
                mem.protected = True

            mem.current_salience = max(new_salience, 0.0)

            if mem.current_salience >= MIN_SALIENCE_THRESHOLD:
                surviving.append(mem)
            else:
                logger.debug("Memory decayed below threshold: %s", mem.memory_id)

        return surviving

    # ------------------------------------------------------------------
    # Orchestrator
    # ------------------------------------------------------------------

    async def trigger_condensation(
        self,
        memory_store: MemoryStore,
        character: Character,
    ) -> MemoryStore:
        """Check capacities and promote / condense as needed.

        Call this after exchanges are added.  It will:
        1. Condense working → short-term if working is over capacity.
        2. Condense short-term → long-term if short-term is over capacity.
        3. Apply decay to short-term and long-term tiers.

        Returns the (mutated) memory store.
        """
        # Step 1: working → short-term
        if memory_store.working_is_full:
            condensed = await self.condense_working_to_short_term(
                character, memory_store.working
            )
            memory_store.short_term.append(condensed)
            memory_store.working.clear()

        # Step 2: short-term → long-term
        if memory_store.short_term_is_full:
            # Take the oldest half to merge
            split = len(memory_store.short_term) // 2
            to_merge = memory_store.short_term[:split]
            memory_store.short_term = memory_store.short_term[split:]

            merged = await self.condense_short_to_long_term(character, to_merge)
            memory_store.long_term.append(merged)

        # Step 3: cap long-term
        if len(memory_store.long_term) > LONG_TERM_CAPACITY:
            # Remove lowest-salience entries
            memory_store.long_term.sort(
                key=lambda m: m.current_salience, reverse=True
            )
            memory_store.long_term = memory_store.long_term[:LONG_TERM_CAPACITY]

        # Step 4: decay
        memory_store.short_term = self.apply_decay(
            memory_store.short_term, character
        )
        memory_store.long_term = self.apply_decay(
            memory_store.long_term, character
        )

        return memory_store

    # ------------------------------------------------------------------
    # Heuristic fallbacks (no LLM needed)
    # ------------------------------------------------------------------

    @staticmethod
    def _heuristic_summary(
        exchanges: list[dict[str, str]],
        character: Character,
    ) -> dict[str, Any]:
        """Produce a simple summary dict without an LLM."""
        speakers = sorted({e["speaker"] for e in exchanges})
        first_text = exchanges[0]["text"][:80] if exchanges else ""
        return {
            "summary": (
                f"Conversation between {', '.join(speakers)}: \"{first_text}…\" "
                f"({len(exchanges)} exchanges)"
            ),
            "emotional_tone": character.current_emotional_state.base,
            "salience": 0.4,
            "topics": [],
            "perspective_bias": None,
        }

    @staticmethod
    def _heuristic_merge(
        memories: list[CondensedMemory],
        character: Character,
    ) -> dict[str, Any]:
        """Produce a simple merged summary dict without an LLM."""
        all_topics = sorted({t for m in memories for t in m.topics})
        return {
            "summary": (
                f"Series of {len(memories)} interactions touching on "
                f"{', '.join(all_topics) if all_topics else 'various topics'}."
            ),
            "emotional_tone": character.current_emotional_state.base,
            "salience": 0.4,
            "topics": all_topics[:5],
            "perspective_bias": None,
        }
