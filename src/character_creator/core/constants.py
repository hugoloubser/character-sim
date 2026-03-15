"""Shared constants for the character creator system.

All magic literals and repeated values are centralised here to ensure
a single source of truth across the codebase.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Trait defaults & bounds
# ---------------------------------------------------------------------------

TRAIT_DEFAULT: float = 0.5
"""Default value for all personality trait axes."""

TRAIT_MIN: float = 0.0
"""Minimum allowed value for a personality trait."""

TRAIT_MAX: float = 1.0
"""Maximum allowed value for a personality trait."""

# ---------------------------------------------------------------------------
# Emotional state
# ---------------------------------------------------------------------------

DEFAULT_EMOTIONAL_STATE: str = "neutral"
"""Starting emotional state for every character."""

EMOTIONAL_STATES: tuple[str, ...] = (
    "neutral",
    "happy",
    "angry",
    "sad",
    "anxious",
    "confident",
    "vulnerable",
)
"""Canonical list of recognised emotional states."""

EMOTIONAL_MODIFIERS: dict[str, str | None] = {
    "happy": "Right now, I'm feeling uplifted and optimistic.",
    "angry": "At the moment, I'm frustrated and sharp.",
    "sad": "Currently, I'm feeling contemplative and withdrawn.",
    "anxious": "I'm feeling a bit tense and uncertain right now.",
    "confident": "I'm feeling capable and assured at this moment.",
    "vulnerable": "Right now, I'm feeling exposed and careful.",
    "neutral": None,
}
"""First-person self-perception modifiers keyed by emotional state."""

# ---------------------------------------------------------------------------
# Dialogue defaults
# ---------------------------------------------------------------------------

DEFAULT_TOPIC: str = "casual conversation"
"""Fallback topic when none is supplied."""

DEFAULT_MUTATION_RATE: float = 0.05
"""Default genetic mutation rate for heredity crossover."""

# ---------------------------------------------------------------------------
# Speaker-selection scoring weights
# ---------------------------------------------------------------------------

SPEAKER_EXTRAVERSION_WEIGHT: float = 0.35
SPEAKER_ASSERTIVENESS_WEIGHT: float = 0.20
SPEAKER_WARMTH_WEIGHT: float = 0.15
SPEAKER_OPENNESS_WEIGHT: float = 0.10
SPEAKER_COMPATIBILITY_WEIGHT: float = 0.20
SAME_SPEAKER_PENALTY: float = 0.2
RECENT_SPEAKER_DECAY: float = 0.15
RECENT_SPEAKER_FLOOR: float = 0.5

# ---------------------------------------------------------------------------
# LLM temperature presets
# ---------------------------------------------------------------------------

DIALOGUE_TEMPERATURE: float = 0.7
"""Temperature for generating character dialogue."""

MONOLOGUE_TEMPERATURE: float = 0.6
"""Temperature for generating internal monologue."""

EMOTION_INFERENCE_TEMPERATURE: float = 0.2
"""Temperature for inferring emotional context."""

# ---------------------------------------------------------------------------
# MBTI compatibility scoring
# ---------------------------------------------------------------------------

MBTI_COMPATIBILITY_SCORES: dict[int, float] = {
    4: 0.60,
    3: 0.85,
    2: 0.70,
    1: 0.50,
    0: 0.40,
}
"""Compatibility score by number of shared MBTI preference axes.

4/4 shared → 0.60 (too similar — less dynamic)
3/4 shared → 0.85 (high rapport, slight difference creates depth)
2/4 shared → 0.70 (balanced — good complement)
1/4 shared → 0.50 (mostly different — tension possible)
0/4 shared → 0.40 (maximally different — fascination but friction)
"""

# ---------------------------------------------------------------------------
# Background defaults
# ---------------------------------------------------------------------------

DEFAULT_AGE: int = 30
"""Default character age when not specified."""

MAX_EXCHANGES: int = 30
"""Default upper bound for dialogue exchanges per scene."""

# ---------------------------------------------------------------------------
# Tiered memory system
# ---------------------------------------------------------------------------

WORKING_MEMORY_CAPACITY: int = 10
"""Max verbatim exchanges kept in working memory before condensation."""

SHORT_TERM_CAPACITY: int = 20
"""Max condensed memories in the short-term tier."""

LONG_TERM_CAPACITY: int = 50
"""Max condensed memories in the long-term tier."""

DEFAULT_DECAY_RATE: float = 0.02
"""Salience reduction applied to each condensed memory per scene."""

DECAY_EMOTIONAL_ANCHOR: float = 0.5
"""Memories with emotional_weight above this value decay at half rate."""

DECAY_VALUE_ANCHOR_MULTIPLIER: float = 0.5
"""Multiplier for decay rate when memory topics overlap core values."""

MIN_SALIENCE_THRESHOLD: float = 0.1
"""Memories below this salience are eligible for removal."""

CORE_MEMORY_SALIENCE: float = 0.9
"""Memories at or above this salience are protected from deep decay."""

CORE_MEMORY_FLOOR: float = 0.5
"""Protected memories never decay below this salience."""

MAX_DISTORTION_BUDGET: float = 3.0
"""Maximum cumulative perspective distortion a memory may accumulate."""

CONDENSATION_TEMPERATURE: float = 0.4
"""LLM temperature for memory summarisation calls."""

# ---------------------------------------------------------------------------
# Trait evolution (personality micro-shifts)
# ---------------------------------------------------------------------------

MAX_TRAIT_DELTA_PER_EXCHANGE: float = 0.03
"""Cap on any single trait delta from one exchange."""

TRAIT_SHIFT_MOMENTUM: float = 0.7
"""Weight toward recent shifts when multiple push the same direction."""

EXPERIENCE_CLASSIFICATION_TEMPERATURE: float = 0.2
"""LLM temperature for classifying exchange experience types."""

# ---------------------------------------------------------------------------
# Self-reflection / self-model
# ---------------------------------------------------------------------------

SELF_REFLECTION_INTERVAL: int = 5
"""Generate a self-reflection every N exchanges."""

SELF_REFLECTION_TEMPERATURE: float = 0.5
"""LLM temperature for self-reflection generation."""

MAX_SELF_MODEL_HISTORY: int = 10
"""Keep at most this many past self-models for tracking evolution."""

# ---------------------------------------------------------------------------
# Milestone review (LLM-adjudicated scene-end shifts)
# ---------------------------------------------------------------------------

MILESTONE_REVIEW_TEMPERATURE: float = 0.5
"""LLM temperature for milestone personality review."""

MAX_MILESTONE_SHIFT: float = 0.10
"""Maximum absolute trait shift the LLM may propose in a milestone review."""

MILESTONE_CONFIDENCE_THRESHOLD: float = 0.3
"""Proposed shifts with confidence below this are discarded."""

# ---------------------------------------------------------------------------
# Dissonance detection (behaviour vs values)
# ---------------------------------------------------------------------------

DISSONANCE_SEVERITY_THRESHOLD: float = 0.4
"""Dissonances below this severity are not surfaced to the character."""

BEHAVIOUR_EXTRACTION_INTERVAL: int = 3
"""Extract behaviour themes every N exchanges."""

MAX_ACTIVE_DISSONANCES: int = 3
"""Maximum simultaneous cognitive dissonances a character can hold."""

BEHAVIOUR_EXTRACTION_TEMPERATURE: float = 0.3
"""LLM temperature for behaviour theme extraction."""

DISSONANCE_DETECTION_TEMPERATURE: float = 0.3
"""LLM temperature for dissonance detection."""
