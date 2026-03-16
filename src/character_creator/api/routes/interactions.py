"""Dialogue and interaction API routes.

Wires incoming requests to the real ``DialogueSystem`` for LLM-powered
dialogue generation.  Scenes are backed by an ``InteractionRepository``
for SQLite persistence and kept in an in-memory cache for active dialogue
contexts that hold live ``Character`` objects.
"""

import logging

from typing import Any

from fastapi import APIRouter, HTTPException, status

from character_creator.api.models import (
    DialogueRequest,
    DialogueResponse,
    SceneSetupRequest,
)
from character_creator.api.routes.characters import character_repository
from character_creator.core.dialogue import DialogueContext, DialogueSystem
from character_creator.core.interaction import (
    InteractionRecord,
    SQLiteInteractionRepository,
)
from character_creator.llm.providers import LLMError, get_llm_provider
from character_creator.utils.config import settings

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/interactions", tags=["interactions"])

# Persistent interaction store (survives restarts)
interaction_repository = SQLiteInteractionRepository(
    db_path=settings.interactions_db_path,
)

# In-memory cache of *active* scenes (holds live Character objects needed
# for dialogue generation — rebuilt on demand from the DB if missing).
_active_scenes: dict[str, DialogueContext] = {}


@router.post("/scenes", status_code=status.HTTP_201_CREATED)
async def create_scene(request: SceneSetupRequest) -> dict[str, Any]:
    """Create a new dialogue scene with specified characters.

    Saves an ``InteractionRecord`` to SQLite and caches the live
    ``DialogueContext`` for subsequent exchanges.

    Args:
        request: Scene setup parameters.

    Returns:
        Scene ID and initial state.

    Raises:
        HTTPException: If characters not found.

    """
    # Validate all characters exist
    characters = []
    for char_name in request.character_names:
        character = character_repository.read(char_name)
        if not character:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Character '{char_name}' not found",
            )
        characters.append(character)

    # Create persistent record
    record = InteractionRecord(
        scene_description=request.scene_description,
        topic=request.topic,
        characters=request.character_names,
        provider=settings.llm_provider,
    )
    interaction_repository.save(record)

    # Cache a live DialogueContext for generation endpoints
    scene = DialogueContext(
        characters=characters,
        scene_description=request.scene_description,
        topic=request.topic,
    )
    _active_scenes[record.interaction_id] = scene

    return {
        "scene_id": record.interaction_id,
        "scene_description": request.scene_description,
        "topic": request.topic,
        "characters": request.character_names,
    }


def _get_or_rebuild_scene(scene_id: str) -> DialogueContext | None:
    """Return the cached DialogueContext, rebuilding from the DB if needed."""
    if scene_id in _active_scenes:
        return _active_scenes[scene_id]

    record = interaction_repository.get(scene_id)
    if record is None:
        return None

    # Rebuild a live DialogueContext from the persisted record
    characters = []
    for name in record.characters:
        char = character_repository.read(name)
        if char:
            characters.append(char)
    if not characters:
        return None

    scene = DialogueContext(
        characters=characters,
        scene_description=record.scene_description,
        topic=record.topic,
    )
    # Replay persisted exchanges into the context
    for ex in record.exchanges:
        speaker = character_repository.read(ex["speaker"])
        if speaker:
            scene.add_exchange(
                speaker=speaker,
                text=ex["text"],
                emotional_context=ex.get("emotional_context", "neutral"),
                internal_thought=ex.get("internal_thought", ""),
            )
    _active_scenes[scene_id] = scene
    return scene


@router.get("/scenes/{scene_id}")
async def get_scene(scene_id: str) -> dict[str, Any]:
    """Get scene state and conversation history.

    Args:
        scene_id: ID of the scene.

    Returns:
        Scene data and conversation state.

    Raises:
        HTTPException: If scene not found.

    """
    scene = _get_or_rebuild_scene(scene_id)
    if scene is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Scene '{scene_id}' not found",
        )

    return scene.to_dict()


@router.post("/generate-response", response_model=DialogueResponse)
async def generate_response(request: DialogueRequest) -> dict[str, Any]:
    """Generate an LLM-powered dialogue response for a character.

    Args:
        request: Dialogue generation request.

    Returns:
        Generated dialogue and internal state.

    Raises:
        HTTPException: If character not found or LLM call fails.

    """
    character = character_repository.read(request.character_name)
    if not character:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Character '{request.character_name}' not found",
        )

    # Resolve other characters referenced in the request
    others = []
    for name in request.other_characters:
        other = character_repository.read(name)
        if other:
            others.append(other)

    # Build a temporary scene context
    context = DialogueContext(
        characters=[character, *others],
        scene_description=request.context,
        topic=request.topic,
    )

    provider = get_llm_provider(
        settings.llm_provider,
        api_key=settings.api_key_for(settings.llm_provider),
    )
    try:
        system = DialogueSystem(provider)
        pre_exchange_thought, dialogue_text, internal_thought = await system.generate_response(context, character)
        emotional_state = await system.infer_emotional_context(character, dialogue_text)
    except LLMError as exc:
        logger.exception("LLM generation failed for %s", request.character_name)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"LLM generation failed: {exc}",
        ) from exc
    finally:
        await provider.close()

    return {
        "character_name": request.character_name,
        "dialogue": dialogue_text,
        "pre_exchange_thought": pre_exchange_thought,
        "internal_thought": internal_thought,
        "emotional_state": emotional_state,
    }


@router.post("/scenes/{scene_id}/add-exchange")
async def add_exchange(scene_id: str, request: DialogueRequest) -> dict[str, Any]:
    """Generate and append an LLM-powered exchange to an active scene.

    Args:
        scene_id: ID of the scene.
        request: Dialogue exchange to add.

    Returns:
        Updated scene state.

    Raises:
        HTTPException: If scene or character not found, or LLM call fails.

    """
    scene = _get_or_rebuild_scene(scene_id)
    if scene is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Scene '{scene_id}' not found",
        )

    character = character_repository.read(request.character_name)
    if not character:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Character '{request.character_name}' not found",
        )

    provider = get_llm_provider(
        settings.llm_provider,
        api_key=settings.api_key_for(settings.llm_provider),
    )
    try:
        system = DialogueSystem(provider)
        pre_exchange_thought, dialogue_text, internal_thought = await system.generate_response(scene, character)
        emotional_state = await system.infer_emotional_context(character, dialogue_text)
    except LLMError as exc:
        logger.exception("LLM generation failed for scene %s", scene_id)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"LLM generation failed: {exc}",
        ) from exc
    finally:
        await provider.close()

    # Update live context
    scene.add_exchange(
        speaker=character,
        text=dialogue_text,
        emotional_context=emotional_state,
        pre_exchange_thought=pre_exchange_thought,
        internal_thought=internal_thought,
    )

    # Persist to SQLite
    record = interaction_repository.get(scene_id)
    if record:
        record.add_exchange(
            speaker=character.name,
            text=dialogue_text,
            emotional_context=emotional_state,
            pre_exchange_thought=pre_exchange_thought,
            internal_thought=internal_thought,
        )
        interaction_repository.save(record)

    return scene.to_dict()


@router.get("/history")
async def list_interactions() -> list[dict[str, Any]]:
    """List all persisted interaction records (newest first)."""
    return [r.to_dict() for r in interaction_repository.list_all()]


@router.get("/history/{interaction_id}")
async def get_interaction(interaction_id: str) -> dict[str, Any]:
    """Retrieve a single persisted interaction record.

    Args:
        interaction_id: UUID of the interaction.

    Raises:
        HTTPException: If not found.

    """
    record = interaction_repository.get(interaction_id)
    if record is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Interaction '{interaction_id}' not found",
        )
    return record.to_dict()
