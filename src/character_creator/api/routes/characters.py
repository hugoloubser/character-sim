"""Character-related API routes.

Supports multiple storage backends via the CharacterRepository pattern:
- InMemoryRepository: Fast, volatile storage for development/testing
- SQLiteRepository: Persistent local storage (default for production)

Startup automatically initializes database and loads/creates default characters.
"""

import json
import logging

from typing import Any

from fastapi import APIRouter, HTTPException, status

from character_creator.api.models import (
    CharacterCreateRequest,
    CharacterGenerationRequest,
    CharacterResponse,
)
from character_creator.core.character import Character
from character_creator.core.database import (
    CharacterRepository,
    SQLiteRepository,
    create_default_characters,
)
from character_creator.core.memory import Background
from character_creator.core.personality import Personality
from character_creator.llm.prompts import CharacterCreationPrompts, substitute_prompt
from character_creator.llm.providers import LLMError, get_llm_provider
from character_creator.utils.config import settings

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/characters", tags=["characters"])

# Character repository (configuration: use SQLiteRepository for production,
# InMemoryRepository for development. Can be overridden via environment variable.)
character_repository: CharacterRepository = SQLiteRepository(
    db_path=settings.characters_db_path,
)


def initialize_default_characters() -> None:
    """Initialize database with default characters if empty.

    Creates a cast of 5 distinct character archetypes for testing and
    demonstration purposes on first run. Subsequent runs preserve existing
    characters. To reset to defaults, delete the database in local/.
    """
    existing_characters = character_repository.list_all()

    if not existing_characters:
        logger.info("Initializing database with default character cast...")
        default_chars = create_default_characters()
        for character in default_chars:
            try:
                character_repository.create(character)
                logger.info(f"Created default character: {character.name}")
            except ValueError as e:
                logger.warning(f"Failed to create default character: {e}")
    else:
        logger.info(f"Database already initialized with {len(existing_characters)} characters")


@router.post("/", response_model=CharacterResponse, status_code=status.HTTP_201_CREATED)
async def create_character(request: CharacterCreateRequest) -> dict[str, Any]:
    """Create a new character with provided details.

    Persists to database (SQLite by default). Character names must be unique.

    Args:
        request: Character creation request with name, description, and traits.

    Returns:
        Created character data.

    Raises:
        HTTPException: If character name already exists.

    """
    if character_repository.exists(request.name):
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Character '{request.name}' already exists",
        )

    personality = Personality.from_dict(request.personality.model_dump())
    background = Background.from_dict(request.background.model_dump())

    character = Character(
        name=request.name,
        description=request.description,
        personality=personality,
        background=background,
    )

    try:
        character_repository.create(character)
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=str(e),
        ) from e

    return character.to_dict()


@router.get("/{character_name}", response_model=CharacterResponse)
async def get_character(character_name: str) -> dict[str, Any]:
    """Retrieve a character by name.

    Args:
        character_name: Name of the character.

    Returns:
        Character data.

    Raises:
        HTTPException: If character not found.

    """
    character = character_repository.read(character_name)
    if not character:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Character '{character_name}' not found",
        )

    return character.to_dict()


@router.get("/", response_model=list[dict[str, Any]])
async def list_characters() -> list[dict[str, Any]]:
    """List all available characters.

    Retrieves all characters from database, sorted by name. Thread-safe
    with concurrent access protection.

    Returns:
        List of character data dictionaries.

    """
    return [character.to_dict() for character in character_repository.list_all()]


@router.put("/{character_name}", response_model=CharacterResponse)
async def update_character(
    character_name: str, request: CharacterCreateRequest
) -> dict[str, Any]:
    """Update an existing character's data.

    Updates all character attributes (description, personality, background)
    in the database.

    Args:
        character_name: Name of character to update.
        request: Updated character data.

    Returns:
        Updated character data.

    Raises:
        HTTPException: If character not found.

    """
    character = character_repository.read(character_name)
    if not character:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Character '{character_name}' not found",
        )

    personality = Personality.from_dict(request.personality.model_dump())
    background = Background.from_dict(request.background.model_dump())

    character.name = request.name
    character.description = request.description
    character.personality = personality
    character.background = background

    try:
        character_repository.update(character)
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        ) from e

    return character.to_dict()


@router.delete("/{character_name}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_character(character_name: str) -> None:
    """Delete a character from the database.

    Args:
        character_name: Name of character to delete.

    Raises:
        HTTPException: If character not found.

    """
    if not character_repository.delete(character_name):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Character '{character_name}' not found",
        )


@router.post("/generate", response_model=CharacterResponse)
async def generate_character(request: CharacterGenerationRequest) -> dict[str, Any]:
    """Generate a character using LLM assistance.

    Calls the configured LLM provider to generate personality traits,
    background, and description based on the supplied concept.

    Args:
        request: Character generation parameters (name, concept).

    Returns:
        Generated character data.

    Raises:
        HTTPException: If generation fails or character already exists.

    """
    if character_repository.exists(request.name):
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Character '{request.name}' already exists",
        )

    provider = get_llm_provider(
        settings.llm_provider,
        api_key=settings.api_key_for(settings.llm_provider),
    )

    try:
        # --- 1) Generate personality ---
        personality_prompt = substitute_prompt(
            CharacterCreationPrompts.GENERATE_PERSONALITY,
            name=request.name,
            concept=request.concept,
        )
        personality_raw = await provider.generate(personality_prompt, temperature=0.7)
        personality_data = json.loads(_extract_json(personality_raw))

        # Map flat LLM output into the nested structure Personality.from_dict expects
        personality = Personality.from_dict({
            "traits": {
                "assertiveness": personality_data.get("assertiveness", 0.5),
                "warmth": personality_data.get("warmth", 0.5),
                "openness": personality_data.get("openness", 0.5),
                "conscientiousness": personality_data.get("conscientiousness", 0.5),
                "emotional_stability": personality_data.get("emotional_stability", 0.5),
                "humor_inclination": personality_data.get("humor_inclination", 0.5),
                "formality": personality_data.get("formality", 0.5),
            },
            "values": {
                "priority_keywords": personality_data.get("priority_keywords", []),
                "beliefs": personality_data.get("beliefs", []),
                "strengths": personality_data.get("strengths", []),
                "weaknesses": personality_data.get("weaknesses", []),
            },
            "speech_patterns": personality_data.get("speech_patterns", []),
            "quirks": personality_data.get("quirks", []),
        })
        personality_summary = personality.describe_briefly()

        # --- 2) Generate background (optional) ---
        background: Background | None = None
        if request.include_background:
            background_prompt = substitute_prompt(
                CharacterCreationPrompts.GENERATE_BACKGROUND,
                name=request.name,
                personality_summary=personality_summary,
            )
            background_raw = await provider.generate(background_prompt, temperature=0.7)
            background_data = json.loads(_extract_json(background_raw))
            background = Background.from_dict(background_data)

        # --- 3) Generate description (optional) ---
        description = f"A character based on: {request.concept}"
        if request.include_appearance:
            desc_prompt = substitute_prompt(
                CharacterCreationPrompts.GENERATE_DESCRIPTION,
                name=request.name,
                personality_summary=personality_summary,
                background_summary=(
                    background.get_context_summary() if background else "No background yet."
                ),
            )
            description = await provider.generate(desc_prompt, temperature=0.7)
            description = description.strip()

        # --- Assemble character ---
        character = Character(
            name=request.name,
            description=description,
            personality=personality,
            background=background or Background(age=30, origin="Unknown", occupation="Unknown"),
        )

        character_repository.create(character)

    except (json.JSONDecodeError, KeyError, TypeError) as exc:
        logger.exception("Failed to parse LLM output for character '%s'", request.name)
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Could not parse LLM response into a valid character: {exc}",
        ) from exc
    except LLMError as exc:
        logger.exception("LLM generation failed for character '%s'", request.name)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"LLM generation failed: {exc}",
        ) from exc
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=str(exc),
        ) from exc
    finally:
        await provider.close()

    return character.to_dict()


def _extract_json(text: str) -> str:
    """Extract the first JSON object from an LLM response string.

    Handles responses wrapped in markdown code fences or leading prose.
    """
    # Strip markdown fences (```json ... ``` or ``` ... ```)
    if "```" in text:
        parts = text.split("```")
        for part in parts:
            stripped = part.strip()
            if stripped.startswith("json"):
                stripped = stripped[4:].strip()
            if stripped.startswith("{"):
                return stripped

    # Fall back to finding the first { ... } block
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        return text[start : end + 1]

    return text
