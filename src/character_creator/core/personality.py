"""Personality system for characters with configurable traits and behaviors."""

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field, model_validator

from character_creator.core.constants import (
    DEFAULT_EMOTIONAL_STATE,
    EMOTIONAL_MODIFIERS,
    MBTI_COMPATIBILITY_SCORES,
    TRAIT_DEFAULT,
    TRAIT_MAX,
    TRAIT_MIN,
)

# ---------------------------------------------------------------------------
# MBTI — derived from OCEAN traits, not assigned manually
# ---------------------------------------------------------------------------


class MBTIType(StrEnum):
    """Myers-Briggs personality types derived from OCEAN dimensions."""

    INTJ = "INTJ"
    INTP = "INTP"
    ENTJ = "ENTJ"
    ENTP = "ENTP"
    INFJ = "INFJ"
    INFP = "INFP"
    ENFJ = "ENFJ"
    ENFP = "ENFP"
    ISTJ = "ISTJ"
    ISFJ = "ISFJ"
    ESTJ = "ESTJ"
    ESFJ = "ESFJ"
    ISTP = "ISTP"
    ISFP = "ISFP"
    ESTP = "ESTP"
    ESFP = "ESFP"


# Descriptive archetype label for each MBTI code
MBTI_ARCHETYPE: dict[MBTIType, str] = {
    MBTIType.INTJ: "Architect",
    MBTIType.INTP: "Logician",
    MBTIType.ENTJ: "Commander",
    MBTIType.ENTP: "Debater",
    MBTIType.INFJ: "Advocate",
    MBTIType.INFP: "Mediator",
    MBTIType.ENFJ: "Protagonist",
    MBTIType.ENFP: "Campaigner",
    MBTIType.ISTJ: "Logistician",
    MBTIType.ISFJ: "Defender",
    MBTIType.ESTJ: "Executive",
    MBTIType.ESFJ: "Consul",
    MBTIType.ISTP: "Virtuoso",
    MBTIType.ISFP: "Adventurer",
    MBTIType.ESTP: "Entrepreneur",
    MBTIType.ESFP: "Entertainer",
}

# Communication-style hints the LLM can use when generating dialogue.
MBTI_COMMUNICATION_STYLE: dict[MBTIType, str] = {
    MBTIType.INTJ: "Strategic and concise; prefers logic over pleasantries",
    MBTIType.INTP: "Analytical and questioning; explores ideas tangentially",
    MBTIType.ENTJ: "Commanding and direct; drives toward actionable outcomes",
    MBTIType.ENTP: "Provocative and witty; plays devil's advocate eagerly",
    MBTIType.INFJ: "Insightful and measured; reads between the lines",
    MBTIType.INFP: "Idealistic and metaphorical; speaks from deeply held values",
    MBTIType.ENFJ: "Charismatic and encouraging; focuses on others' potential",
    MBTIType.ENFP: "Enthusiastic and tangential; connects disparate ideas",
    MBTIType.ISTJ: "Methodical and factual; references precedent and duty",
    MBTIType.ISFJ: "Supportive and practical; remembers personal details",
    MBTIType.ESTJ: "Structured and decisive; establishes rules and expectations",
    MBTIType.ESFJ: "Warm and harmonising; smooths social friction",
    MBTIType.ISTP: "Laconic and observant; speaks only when it matters",
    MBTIType.ISFP: "Gentle and experiential; communicates through actions",
    MBTIType.ESTP: "Bold and pragmatic; focuses on present-moment action",
    MBTIType.ESFP: "Expressive and spontaneous; lights up the room",
}


def mbti_compatibility(a: "MBTIType", b: "MBTIType") -> float:
    """Return a 0-1 compatibility score between two MBTI types.

    Uses the classic complementary-function theory: types that share the
    middle two letters (perception + judging functions) but differ on E/I
    tend to complement each other best.
    """
    code_a, code_b = a.value, b.value
    shared = sum(ca == cb for ca, cb in zip(code_a, code_b, strict=True))
    # 4/4 identical → 0.6 (same type — high harmony but no growth tension)
    # 3/4 shared   → 0.85 (near-complementary)
    # 2/4 shared   → 0.7
    # 1/4 shared   → 0.5
    # 0/4 shared   → 0.4 (maximally different — fascination but friction)
    return MBTI_COMPATIBILITY_SCORES[shared]


class PersonalityAxis(StrEnum):
    """Personality axes for character dimension."""

    ASSERTIVENESS = "assertiveness"
    WARMTH = "warmth"
    OPENNESS = "openness"
    CONSCIENTIOUSNESS = "conscientiousness"
    EMOTIONAL_STABILITY = "emotional_stability"
    EXTRAVERSION = "extraversion"
    AGREEABLENESS = "agreeableness"


class PersonalityTraits(BaseModel):
    """Quantified personality traits for a character.

    The seven original axes plus the two OCEAN dimensions (extraversion,
    agreeableness) needed to derive a full Big-Five / MBTI profile.

    Attributes:
        assertiveness: How direct and confident (0-1.0).
        warmth: How friendly and empathetic (0-1.0).
        openness: How open to new experiences (0-1.0).
        conscientiousness: How organized and disciplined (0-1.0).
        emotional_stability: How resilient to stress (0-1.0).
        humor_inclination: How likely to use humor (0-1.0).
        formality: Speech formality level (0-1.0).
        extraversion: How energised by social interaction (0-1.0).
        agreeableness: How cooperative and empathetic (0-1.0).

    """

    assertiveness: float = Field(TRAIT_DEFAULT, ge=TRAIT_MIN, le=TRAIT_MAX)
    warmth: float = Field(TRAIT_DEFAULT, ge=TRAIT_MIN, le=TRAIT_MAX)
    openness: float = Field(TRAIT_DEFAULT, ge=TRAIT_MIN, le=TRAIT_MAX)
    conscientiousness: float = Field(TRAIT_DEFAULT, ge=TRAIT_MIN, le=TRAIT_MAX)
    emotional_stability: float = Field(TRAIT_DEFAULT, ge=TRAIT_MIN, le=TRAIT_MAX)
    humor_inclination: float = Field(TRAIT_DEFAULT, ge=TRAIT_MIN, le=TRAIT_MAX)
    formality: float = Field(TRAIT_DEFAULT, ge=TRAIT_MIN, le=TRAIT_MAX)
    extraversion: float = Field(TRAIT_DEFAULT, ge=TRAIT_MIN, le=TRAIT_MAX)
    agreeableness: float = Field(TRAIT_DEFAULT, ge=TRAIT_MIN, le=TRAIT_MAX)

    @model_validator(mode="before")
    @classmethod
    def _backfill_ocean(cls, data: Any) -> Any:
        """Back-fill extraversion/agreeableness from legacy dicts."""
        if isinstance(data, dict):
            data.setdefault("extraversion", data.get("assertiveness", TRAIT_DEFAULT))
            data.setdefault("agreeableness", data.get("warmth", TRAIT_DEFAULT))
        return data

    def to_dict(self) -> dict[str, float]:
        """Convert traits to dictionary format."""
        return self.model_dump()

    @classmethod
    def from_dict(cls, data: dict[str, float]) -> "PersonalityTraits":
        """Create traits from dictionary."""
        return cls.model_validate(data)


class Values(BaseModel):
    """Core values that influence character decisions and reactions."""

    priority_keywords: list[str] = Field(default_factory=list)
    beliefs: list[str] = Field(default_factory=list)
    dislikes: list[str] = Field(default_factory=list)
    strengths: list[str] = Field(default_factory=list)
    weaknesses: list[str] = Field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert values to dictionary format."""
        return self.model_dump()

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Values":
        """Create values from dictionary."""
        return cls.model_validate(data)


class Personality(BaseModel):
    """Complete personality profile for a character.

    Combines measurable traits with values and behavioral patterns.
    """

    traits: PersonalityTraits = Field(default_factory=PersonalityTraits)
    values: Values = Field(default_factory=Values)
    speech_patterns: list[str] = Field(default_factory=list)
    quirks: list[str] = Field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert personality to dictionary format."""
        d = self.model_dump()
        d["mbti_type"] = self.mbti_type.value
        d["mbti_archetype"] = MBTI_ARCHETYPE[self.mbti_type]
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Personality":
        """Create personality from dictionary."""
        return cls.model_validate(data)

    @property
    def mbti_type(self) -> MBTIType:
        """Derive MBTI type from OCEAN traits.

        Mapping (matches the scratchpad's ``PersonalityProfile.mbti``):
        - E/I ← extraversion
        - N/S ← openness
        - F/T ← agreeableness
        - J/P ← conscientiousness
        """
        t = self.traits
        code = (
            ("E" if t.extraversion >= 0.5 else "I")
            + ("N" if t.openness >= 0.5 else "S")
            + ("F" if t.agreeableness >= 0.5 else "T")
            + ("J" if t.conscientiousness >= 0.5 else "P")
        )
        return MBTIType[code]

    @property
    def mbti_archetype(self) -> str:
        """Human-readable archetype label (e.g. 'Architect')."""
        return MBTI_ARCHETYPE[self.mbti_type]

    @property
    def communication_style(self) -> str:
        """Communication style hint for LLM prompt injection."""
        return MBTI_COMMUNICATION_STYLE[self.mbti_type]

    @property
    def diction_style(self) -> str:
        """Derive concrete diction guidance from the formality trait.

        Eloquence is taught — vocabulary level, sentence structure and word
        choice should reflect the character's background, not default to
        polished LLM output.  This is injected into every dialogue prompt so
        the LLM writes speech that sounds like *this* character.

        Returns:
            A short, actionable description of the character's diction.

        """
        f = self.traits.formality

        if f >= 0.75:
            vocab = "sophisticated, precise vocabulary"
            sentences = "well-structured, complete sentences"
            contractions = "rarely uses contractions"
        elif f >= 0.50:
            vocab = "clear, everyday vocabulary with occasional formal terms"
            sentences = "natural mix of complete and casual sentences"
            contractions = "uses contractions naturally"
        elif f >= 0.25:
            vocab = "conversational, informal vocabulary"
            sentences = "short, relaxed sentences; may trail off"
            contractions = "freely uses contractions and filler words (like, you know, yeah)"
        else:
            vocab = "blunt, colloquial vocabulary; may use slang"
            sentences = "very short sentences or fragments; may interrupt themselves"
            contractions = "always uses contractions, drops word endings (gonna, wanna, dunno)"

        return f"Vocabulary: {vocab}. Sentences: {sentences}. {contractions}."

    def get_trait(self, axis: PersonalityAxis) -> float:
        """Get a specific trait value by axis."""
        return getattr(self.traits, axis.value)

    def describe_briefly(self) -> str:
        """Generate a brief static description of core personality traits.

        Includes the derived MBTI archetype as a grounding label.

        Returns:
            Brief personality description based on stable traits.

        """
        traits = self.traits
        descriptions: list[str] = [
            f"{self.mbti_type.value} ({self.mbti_archetype})",
        ]

        if traits.assertiveness > 0.7:
            descriptions.append("assertive and confident")
        elif traits.assertiveness < 0.3:
            descriptions.append("reserved and cautious")
        else:
            descriptions.append("balanced in approach")

        if traits.warmth > 0.7:
            descriptions.append("warm and empathetic")
        elif traits.warmth < 0.3:
            descriptions.append("formal and professional")
        else:
            descriptions.append("socially adaptable")

        if traits.openness > 0.7:
            descriptions.append("open to new ideas")
        elif traits.openness < 0.3:
            descriptions.append("values tradition and proven methods")

        if traits.conscientiousness > 0.7:
            descriptions.append("conscientious and organized")
        elif traits.conscientiousness < 0.3:
            descriptions.append("spontaneous and flexible")

        if traits.emotional_stability > 0.7:
            descriptions.append("emotionally resilient")
        elif traits.emotional_stability < 0.3:
            descriptions.append("emotionally reactive")

        return ", ".join(descriptions)

    def describe_self(
        self,
        emotional_state: str = DEFAULT_EMOTIONAL_STATE,
        recent_events: list[str] | None = None,
    ) -> str:
        """Generate a dynamic self-description from the character's perspective.

        This is the "temporal slice"—how the character sees themselves RIGHT NOW,
        influenced by their emotional state, recent events, and core values.
        This evolves moment-to-moment rather than being static, allowing for
        more nuanced and authentic character behavior.

        Addresses TODO: Imagine this as a 4D being's 3D cross-section at this moment.
        The character's self-perception directly influences their reactions and
        dialogue, creating dynamic and believable personality manifestation.

        Args:
            emotional_state: Current emotional state (changes perception).
            recent_events: Recent conversation/events that influence self-view.

        Returns:
            Dynamic self-description influenced by current context.

        """
        base_description = self.describe_briefly()

        # Start with core personality
        perception: list[str] = [f"Fundamentally, I am {base_description}."]

        # Add emotional modifier
        if EMOTIONAL_MODIFIERS.get(emotional_state):
            perception.append(EMOTIONAL_MODIFIERS[emotional_state])  # type: ignore[arg-type]

        # Add values-based reflection
        if self.values.priority_keywords:
            key_values = ", ".join(self.values.priority_keywords[:3])
            perception.append(f"What matters most to me is: {key_values}.")

        # Add recent events impact
        if recent_events:
            # Most recent event shapes current self-perception
            latest_event = recent_events[-1] if recent_events else None
            if latest_event:
                if any(
                    word in latest_event.lower()
                    for word in ["disagree", "conflict", "wrong", "opposed"]
                ):
                    perception.append("Recent interactions have challenged my perspective.")
                elif any(
                    word in latest_event.lower()
                    for word in ["agree", "alignment", "understood", "connected"]
                ):
                    perception.append("I've felt validated and understood recently.")
                elif any(
                    word in latest_event.lower()
                    for word in ["question", "wonder", "uncertain", "unclear"]
                ):
                    perception.append("I'm questioning some things I took for granted.")

        return " ".join(perception)
