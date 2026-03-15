"""Heredity and evolution API routes.

Endpoints for:
- Breeding two existing characters to produce offspring via genetic crossover.
- Querying MBTI profiles and inter-character compatibility.
"""

import logging

from typing import Any

from fastapi import APIRouter, HTTPException, status

from character_creator.api.models import (
    CompatibilityRequest,
    CompatibilityResponse,
    MBTIProfileResponse,
    ReproduceRequest,
)
from character_creator.api.routes.characters import character_repository
from character_creator.core.heredity import reproduce
from character_creator.core.personality import mbti_compatibility

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/evolution", tags=["evolution"])


@router.post("/reproduce", status_code=status.HTTP_201_CREATED)
async def breed_characters(request: ReproduceRequest) -> dict[str, Any]:
    """Breed two existing characters to create offspring.

    The child inherits personality traits via single-gene crossover with
    Gaussian mutation, values sampled from the union of both parents, and
    speech patterns / quirks randomly drawn from each parent.

    Args:
        request: Parent names, child name, and optional mutation rate.

    Returns:
        The newly created child character data.

    """
    parent1 = character_repository.read(request.parent1_name)
    if parent1 is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Parent character '{request.parent1_name}' not found",
        )

    parent2 = character_repository.read(request.parent2_name)
    if parent2 is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Parent character '{request.parent2_name}' not found",
        )

    if character_repository.exists(request.child_name):
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Character '{request.child_name}' already exists",
        )

    child = reproduce(
        parent1,
        parent2,
        request.child_name,
        mutation_rate=request.mutation_rate,
    )
    character_repository.create(child)
    logger.info(
        "Bred child '%s' from parents '%s' + '%s'",
        child.name,
        parent1.name,
        parent2.name,
    )
    return child.to_dict()


@router.get("/mbti/{character_name}", response_model=MBTIProfileResponse)
async def get_mbti_profile(character_name: str) -> dict[str, Any]:
    """Return the MBTI personality profile for a character.

    Args:
        character_name: Name of the character to inspect.

    Returns:
        MBTI type, archetype label, communication style, and OCEAN scores.

    """
    character = character_repository.read(character_name)
    if character is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Character '{character_name}' not found",
        )

    p = character.personality
    t = p.traits
    return {
        "name": character.name,
        "mbti_type": p.mbti_type.value,
        "archetype": p.mbti_archetype,
        "communication_style": p.communication_style,
        "extraversion": t.extraversion,
        "agreeableness": t.agreeableness,
        "openness": t.openness,
        "conscientiousness": t.conscientiousness,
        "emotional_stability": t.emotional_stability,
    }


@router.post("/compatibility", response_model=CompatibilityResponse)
async def check_compatibility(request: CompatibilityRequest) -> dict[str, Any]:
    """Compute MBTI compatibility between two characters.

    Args:
        request: Names of the two characters to compare.

    Returns:
        MBTI types and a 0-1 compatibility score.

    """
    c1 = character_repository.read(request.character1_name)
    if c1 is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Character '{request.character1_name}' not found",
        )

    c2 = character_repository.read(request.character2_name)
    if c2 is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Character '{request.character2_name}' not found",
        )

    score = mbti_compatibility(c1.personality.mbti_type, c2.personality.mbti_type)
    return {
        "character1": c1.name,
        "character2": c2.name,
        "mbti1": c1.personality.mbti_type.value,
        "mbti2": c2.personality.mbti_type.value,
        "compatibility": score,
    }
