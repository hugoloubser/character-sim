"""Pydantic models for API requests and responses."""

from typing import Any

from pydantic import BaseModel, Field

from character_creator.core.constants import (
    DEFAULT_AGE,
    DEFAULT_MUTATION_RATE,
    DEFAULT_TOPIC,
    TRAIT_DEFAULT,
    TRAIT_MAX,
    TRAIT_MIN,
)


class PersonalityTraitsRequest(BaseModel):
    """Request model for personality traits."""

    assertiveness: float = Field(TRAIT_DEFAULT, ge=TRAIT_MIN, le=TRAIT_MAX)
    warmth: float = Field(TRAIT_DEFAULT, ge=TRAIT_MIN, le=TRAIT_MAX)
    openness: float = Field(TRAIT_DEFAULT, ge=TRAIT_MIN, le=TRAIT_MAX)
    conscientiousness: float = Field(TRAIT_DEFAULT, ge=TRAIT_MIN, le=TRAIT_MAX)
    emotional_stability: float = Field(TRAIT_DEFAULT, ge=TRAIT_MIN, le=TRAIT_MAX)
    humor_inclination: float = Field(TRAIT_DEFAULT, ge=TRAIT_MIN, le=TRAIT_MAX)
    formality: float = Field(TRAIT_DEFAULT, ge=TRAIT_MIN, le=TRAIT_MAX)
    extraversion: float = Field(TRAIT_DEFAULT, ge=TRAIT_MIN, le=TRAIT_MAX)
    agreeableness: float = Field(TRAIT_DEFAULT, ge=TRAIT_MIN, le=TRAIT_MAX)


class ValuesRequest(BaseModel):
    """Request model for character values."""

    priority_keywords: list[str] = Field(default_factory=list)
    beliefs: list[str] = Field(default_factory=list)
    dislikes: list[str] = Field(default_factory=list)
    strengths: list[str] = Field(default_factory=list)
    weaknesses: list[str] = Field(default_factory=list)


class PersonalityRequest(BaseModel):
    """Request model for complete personality."""

    traits: PersonalityTraitsRequest = Field(default_factory=PersonalityTraitsRequest)
    values: ValuesRequest = Field(default_factory=ValuesRequest)
    speech_patterns: list[str] = Field(default_factory=list)
    quirks: list[str] = Field(default_factory=list)


class BackgroundRequest(BaseModel):
    """Request model for character background."""

    age: int = Field(DEFAULT_AGE, ge=1, le=150)
    origin: str
    occupation: str
    relationships: dict[str, str] = Field(default_factory=dict)
    motivations: list[str] = Field(default_factory=list)
    fears: list[str] = Field(default_factory=list)
    desires: list[str] = Field(default_factory=list)


class CharacterCreateRequest(BaseModel):
    """Request model for creating a new character."""

    name: str = Field(..., min_length=1, max_length=100)
    description: str = Field(..., min_length=10, max_length=1000)
    personality: PersonalityRequest = Field(default_factory=PersonalityRequest)
    background: BackgroundRequest = Field(default_factory=BackgroundRequest)


class CharacterResponse(BaseModel):
    """Response model for character data."""

    name: str
    description: str
    personality: dict[str, Any]
    background: dict[str, Any]
    current_emotional_state: str
    internal_monologue: list[str]


class DialogueRequest(BaseModel):
    """Request model for generating dialogue."""

    character_name: str
    context: str
    topic: str = DEFAULT_TOPIC
    other_characters: list[str] = Field(default_factory=list)


class DialogueResponse(BaseModel):
    """Response model for dialogue generation."""

    character_name: str
    dialogue: str
    pre_exchange_thought: str
    internal_thought: str
    emotional_state: str


class SceneSetupRequest(BaseModel):
    """Request model for setting up a dialogue scene."""

    scene_description: str
    topic: str
    character_names: list[str] = Field(..., min_length=1)


class CharacterGenerationRequest(BaseModel):
    """Request model for LLM-assisted character generation."""

    name: str = Field(..., min_length=1, max_length=100)
    concept: str = Field(..., min_length=10, max_length=500)
    include_background: bool = True
    include_appearance: bool = True


class HealthResponse(BaseModel):
    """Response model for health check."""

    status: str
    version: str


# ---------------------------------------------------------------------------
# Heredity / Evolution endpoints
# ---------------------------------------------------------------------------


class ReproduceRequest(BaseModel):
    """Request model for breeding two characters."""

    parent1_name: str = Field(..., min_length=1)
    parent2_name: str = Field(..., min_length=1)
    child_name: str = Field(..., min_length=1, max_length=100)
    mutation_rate: float = Field(DEFAULT_MUTATION_RATE, ge=0.0, le=0.5)


class MBTIProfileResponse(BaseModel):
    """Compact MBTI summary for a character."""

    name: str
    mbti_type: str
    archetype: str
    communication_style: str
    extraversion: float
    agreeableness: float
    openness: float
    conscientiousness: float
    emotional_stability: float


class CompatibilityRequest(BaseModel):
    """Request model for MBTI compatibility check between two characters."""

    character1_name: str = Field(..., min_length=1)
    character2_name: str = Field(..., min_length=1)


class CompatibilityResponse(BaseModel):
    """Response model for MBTI compatibility score."""

    character1: str
    character2: str
    mbti1: str
    mbti2: str
    compatibility: float
