"""Tests for FastAPI endpoints (characters + interactions routes).

Uses ``httpx.AsyncClient`` with FastAPI's test transport so no real
server is started.  The character repository and interaction repository
are swapped to in-memory backends for isolation.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from httpx import ASGITransport, AsyncClient

from character_creator.api.main import create_app
from character_creator.core.database import InMemoryRepository

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def _patch_repos(tmp_path):
    """Replace production repos with in-memory / temp-file variants."""
    from character_creator.core.interaction import InMemoryInteractionRepository

    mem_char_repo = InMemoryRepository()
    mem_int_repo = InMemoryInteractionRepository()

    with (
        patch(
            "character_creator.api.routes.characters.character_repository",
            mem_char_repo,
        ),
        patch(
            "character_creator.api.routes.interactions.character_repository",
            mem_char_repo,
        ),
        patch(
            "character_creator.api.routes.interactions.interaction_repository",
            mem_int_repo,
        ),
        patch(
            "character_creator.api.routes.evolution.character_repository",
            mem_char_repo,
        ),
    ):
        yield mem_char_repo


@pytest.fixture
async def client(_patch_repos) -> AsyncClient:
    """Yield an httpx AsyncClient bound to the FastAPI app."""
    app = create_app()
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


# ---------------------------------------------------------------------------
# Sample payloads
# ---------------------------------------------------------------------------

_MINIMAL_CREATE = {
    "name": "TestHero",
    "description": "A brave test subject",
    "personality": {
        "traits": {
            "assertiveness": 0.7,
            "warmth": 0.5,
            "openness": 0.6,
            "conscientiousness": 0.5,
            "emotional_stability": 0.5,
            "humor_inclination": 0.5,
            "formality": 0.5,
        },
        "values": {
            "priority_keywords": ["courage"],
            "beliefs": ["Never give up"],
            "strengths": ["bravery"],
            "weaknesses": ["stubborn"],
        },
        "speech_patterns": ["Speaks boldly"],
        "quirks": ["Cracks knuckles"],
    },
    "background": {
        "age": 30,
        "origin": "Testville",
        "occupation": "Hero",
        "motivations": ["Save the world"],
        "fears": ["Spiders"],
        "desires": ["Peace"],
        "relationships": {},
        "memories": [],
    },
}


# ---------------------------------------------------------------------------
# Health / root
# ---------------------------------------------------------------------------


class TestHealthAndRoot:
    async def test_health(self, client: AsyncClient) -> None:
        resp = await client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "healthy"

    async def test_root(self, client: AsyncClient) -> None:
        resp = await client.get("/")
        assert resp.status_code == 200
        assert "message" in resp.json()


# ---------------------------------------------------------------------------
# Character CRUD
# ---------------------------------------------------------------------------


class TestCharacterCRUD:
    async def test_create_and_get(self, client: AsyncClient) -> None:
        resp = await client.post("/characters/", json=_MINIMAL_CREATE)
        assert resp.status_code == 201
        data = resp.json()
        assert data["name"] == "TestHero"

        resp2 = await client.get("/characters/TestHero")
        assert resp2.status_code == 200
        assert resp2.json()["name"] == "TestHero"

    async def test_create_duplicate(self, client: AsyncClient) -> None:
        await client.post("/characters/", json=_MINIMAL_CREATE)
        resp = await client.post("/characters/", json=_MINIMAL_CREATE)
        assert resp.status_code == 409

    async def test_list_characters(self, client: AsyncClient) -> None:
        await client.post("/characters/", json=_MINIMAL_CREATE)
        resp = await client.get("/characters/")
        assert resp.status_code == 200
        assert len(resp.json()) >= 1

    async def test_get_missing(self, client: AsyncClient) -> None:
        resp = await client.get("/characters/Nobody")
        assert resp.status_code == 404

    async def test_delete(self, client: AsyncClient) -> None:
        await client.post("/characters/", json=_MINIMAL_CREATE)
        resp = await client.delete("/characters/TestHero")
        assert resp.status_code == 204

        resp2 = await client.get("/characters/TestHero")
        assert resp2.status_code == 404

    async def test_update(self, client: AsyncClient) -> None:
        await client.post("/characters/", json=_MINIMAL_CREATE)
        updated = {**_MINIMAL_CREATE, "description": "An updated hero"}
        resp = await client.put("/characters/TestHero", json=updated)
        assert resp.status_code == 200
        assert resp.json()["description"] == "An updated hero"


# ---------------------------------------------------------------------------
# Scenes
# ---------------------------------------------------------------------------


class TestScenes:
    async def _seed_characters(self, client: AsyncClient) -> None:
        await client.post("/characters/", json=_MINIMAL_CREATE)
        other = {**_MINIMAL_CREATE, "name": "Sidekick", "description": "A loyal friend"}
        await client.post("/characters/", json=other)

    async def test_create_scene(self, client: AsyncClient) -> None:
        await self._seed_characters(client)
        resp = await client.post(
            "/interactions/scenes",
            json={
                "scene_description": "A tavern",
                "topic": "adventure",
                "character_names": ["TestHero", "Sidekick"],
            },
        )
        assert resp.status_code == 201
        data = resp.json()
        assert "scene_id" in data
        assert data["topic"] == "adventure"

    async def test_create_scene_missing_character(self, client: AsyncClient) -> None:
        resp = await client.post(
            "/interactions/scenes",
            json={
                "scene_description": "void",
                "topic": "nothing",
                "character_names": ["Ghost"],
            },
        )
        assert resp.status_code == 404

    async def test_get_scene(self, client: AsyncClient) -> None:
        await self._seed_characters(client)
        create_resp = await client.post(
            "/interactions/scenes",
            json={
                "scene_description": "A tavern",
                "topic": "adventure",
                "character_names": ["TestHero"],
            },
        )
        scene_id = create_resp.json()["scene_id"]
        resp = await client.get(f"/interactions/scenes/{scene_id}")
        assert resp.status_code == 200
        assert resp.json()["topic"] == "adventure"

    async def test_get_missing_scene(self, client: AsyncClient) -> None:
        resp = await client.get("/interactions/scenes/nonexistent")
        assert resp.status_code == 404


# ---------------------------------------------------------------------------
# Generate-response (mocked LLM)
# ---------------------------------------------------------------------------


class TestGenerateResponse:
    async def test_generate_response_with_mock_llm(self, client: AsyncClient) -> None:
        await client.post("/characters/", json=_MINIMAL_CREATE)

        mock_provider = AsyncMock()
        mock_provider.generate = AsyncMock(side_effect=[
            "A brave reply!",       # dialogue
            "I must press on.",     # internal monologue
            "determined",           # emotion inference
        ])
        mock_provider.close = AsyncMock()

        with patch(
            "character_creator.api.routes.interactions.get_llm_provider",
            return_value=mock_provider,
        ):
            resp = await client.post(
                "/interactions/generate-response",
                json={
                    "character_name": "TestHero",
                    "context": "A dark cave",
                    "topic": "courage",
                },
            )

        assert resp.status_code == 200
        data = resp.json()
        assert data["character_name"] == "TestHero"
        assert data["dialogue"] == "A brave reply!"
        assert data["internal_thought"] == "I must press on."
        assert data["emotional_state"] == "determined"

    async def test_generate_response_missing_character(self, client: AsyncClient) -> None:
        resp = await client.post(
            "/interactions/generate-response",
            json={
                "character_name": "Nobody",
                "context": "void",
                "topic": "nothing",
            },
        )
        assert resp.status_code == 404


# ---------------------------------------------------------------------------
# History endpoints
# ---------------------------------------------------------------------------


class TestHistory:
    async def test_list_empty(self, client: AsyncClient) -> None:
        resp = await client.get("/interactions/history")
        assert resp.status_code == 200
        assert resp.json() == []

    async def test_history_after_scene_creation(self, client: AsyncClient) -> None:
        await client.post("/characters/", json=_MINIMAL_CREATE)
        await client.post(
            "/interactions/scenes",
            json={
                "scene_description": "Park",
                "topic": "nature",
                "character_names": ["TestHero"],
            },
        )
        resp = await client.get("/interactions/history")
        assert resp.status_code == 200
        assert len(resp.json()) == 1


# ---------------------------------------------------------------------------
# Evolution / Heredity endpoints
# ---------------------------------------------------------------------------

_PARENT_A = {
    "name": "ParentA",
    "description": "First parent test character",
    "personality": {
        "traits": {
            "assertiveness": 0.8,
            "warmth": 0.6,
            "openness": 0.9,
            "conscientiousness": 0.7,
            "emotional_stability": 0.5,
            "humor_inclination": 0.5,
            "formality": 0.5,
            "extraversion": 0.8,
            "agreeableness": 0.3,
        },
    },
    "background": {"age": 1, "origin": "TestLand", "occupation": "Tester"},
}

_PARENT_B = {
    "name": "ParentB",
    "description": "Second parent test character",
    "personality": {
        "traits": {
            "assertiveness": 0.3,
            "warmth": 0.9,
            "openness": 0.2,
            "conscientiousness": 0.4,
            "emotional_stability": 0.8,
            "humor_inclination": 0.5,
            "formality": 0.5,
            "extraversion": 0.3,
            "agreeableness": 0.8,
        },
    },
    "background": {"age": 1, "origin": "TestWorld", "occupation": "Tester"},
}


class TestEvolutionEndpoints:
    """Tests for /evolution/* routes."""

    async def _create_parents(self, client: AsyncClient) -> None:
        resp = await client.post("/characters/", json=_PARENT_A)
        assert resp.status_code == 201
        resp = await client.post("/characters/", json=_PARENT_B)
        assert resp.status_code == 201

    async def test_reproduce_success(self, client: AsyncClient) -> None:
        await self._create_parents(client)
        resp = await client.post(
            "/evolution/reproduce",
            json={
                "parent1_name": "ParentA",
                "parent2_name": "ParentB",
                "child_name": "ChildC",
            },
        )
        assert resp.status_code == 201
        data = resp.json()
        assert data["name"] == "ChildC"
        assert "ParentA" in data["background"]["origin"]
        assert "ParentB" in data["background"]["origin"]

    async def test_reproduce_missing_parent(self, client: AsyncClient) -> None:
        resp = await client.post(
            "/evolution/reproduce",
            json={
                "parent1_name": "Ghost",
                "parent2_name": "Shadow",
                "child_name": "Orphan",
            },
        )
        assert resp.status_code == 404

    async def test_reproduce_duplicate_child(self, client: AsyncClient) -> None:
        await self._create_parents(client)
        # First breed succeeds
        resp = await client.post(
            "/evolution/reproduce",
            json={
                "parent1_name": "ParentA",
                "parent2_name": "ParentB",
                "child_name": "DupeChild",
            },
        )
        assert resp.status_code == 201
        # Second breed with same child name fails
        resp = await client.post(
            "/evolution/reproduce",
            json={
                "parent1_name": "ParentA",
                "parent2_name": "ParentB",
                "child_name": "DupeChild",
            },
        )
        assert resp.status_code == 409

    async def test_mbti_profile(self, client: AsyncClient) -> None:
        await self._create_parents(client)
        resp = await client.get("/evolution/mbti/ParentA")
        assert resp.status_code == 200
        data = resp.json()
        assert data["name"] == "ParentA"
        assert len(data["mbti_type"]) == 4
        assert "archetype" in data
        assert "communication_style" in data

    async def test_mbti_profile_not_found(self, client: AsyncClient) -> None:
        resp = await client.get("/evolution/mbti/Nobody")
        assert resp.status_code == 404

    async def test_compatibility(self, client: AsyncClient) -> None:
        await self._create_parents(client)
        resp = await client.post(
            "/evolution/compatibility",
            json={
                "character1_name": "ParentA",
                "character2_name": "ParentB",
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert 0.0 <= data["compatibility"] <= 1.0
        assert data["character1"] == "ParentA"
        assert data["character2"] == "ParentB"

    async def test_compatibility_not_found(self, client: AsyncClient) -> None:
        resp = await client.post(
            "/evolution/compatibility",
            json={
                "character1_name": "Nobody",
                "character2_name": "Also Nobody",
            },
        )
        assert resp.status_code == 404
