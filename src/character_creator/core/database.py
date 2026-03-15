"""Database layer for persistent character storage.

Supports multiple backends: in-memory (development), SQLite (local), and extensible
for cloud-based solutions. Provides CRUD operations, full serialization, and indexing.
"""

import json
import sqlite3
import threading

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from character_creator.core.character import Character
from character_creator.core.emotional_state import SQLiteEmotionalStateRepository
from character_creator.core.memory import Background
from character_creator.core.personality import Personality, PersonalityTraits, Values


class CharacterRepository(ABC):
    """Abstract base class for character storage backends."""

    @abstractmethod
    def create(self, character: Character) -> None:
        """Create a new character."""

    @abstractmethod
    def read(self, name: str) -> Character | None:
        """Read a character by name."""

    @abstractmethod
    def list_all(self) -> list[Character]:
        """List all characters."""

    @abstractmethod
    def update(self, character: Character) -> None:
        """Update an existing character."""

    @abstractmethod
    def delete(self, name: str) -> bool:
        """Delete a character, returning True if found."""

    @abstractmethod
    def exists(self, name: str) -> bool:
        """Check if character exists."""


class InMemoryRepository(CharacterRepository):
    """In-memory character storage (development/testing)."""

    def __init__(self) -> None:
        """Initialize in-memory repository with thread safety."""
        self._characters: dict[str, Character] = {}
        self._lock = threading.RLock()

    def create(self, character: Character) -> None:
        """Create a new character."""
        with self._lock:
            if character.name in self._characters:
                msg = f"Character '{character.name}' already exists"
                raise ValueError(msg)
            self._characters[character.name] = character

    def read(self, name: str) -> Character | None:
        """Read a character by name."""
        with self._lock:
            return self._characters.get(name)

    def list_all(self) -> list[Character]:
        """List all characters."""
        with self._lock:
            return list(self._characters.values())

    def update(self, character: Character) -> None:
        """Update an existing character."""
        with self._lock:
            if character.name not in self._characters:
                msg = f"Character '{character.name}' not found"
                raise ValueError(msg)
            self._characters[character.name] = character

    def delete(self, name: str) -> bool:
        """Delete a character, returning True if found."""
        with self._lock:
            if name in self._characters:
                del self._characters[name]
                return True
            return False

    def exists(self, name: str) -> bool:
        """Check if character exists."""
        with self._lock:
            return name in self._characters


class SQLiteRepository(CharacterRepository):
    """SQLite-based character storage (local persistence)."""

    def __init__(self, db_path: str | Path = "character_creator.db") -> None:
        """Initialize SQLite repository.

        Args:
            db_path: Path to SQLite database file.

        """
        self.db_path = Path(db_path)
        self._lock = threading.RLock()
        # Ensure the emotional_states reference table exists first so the FK
        # constraint on the characters table can resolve.
        self._emotional_state_repo = SQLiteEmotionalStateRepository(db_path=self.db_path)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        """Open a connection with FK enforcement enabled."""
        conn = sqlite3.connect(self.db_path)
        conn.execute("PRAGMA foreign_keys = ON")
        return conn

    def _init_db(self) -> None:
        """Initialize database schema."""
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS characters (
                    name TEXT PRIMARY KEY,
                    description TEXT NOT NULL,
                    personality TEXT NOT NULL,
                    background TEXT NOT NULL,
                    internal_monologue TEXT,
                    current_emotional_state TEXT DEFAULT 'neutral'
                        REFERENCES emotional_states(label),
                    conversation_history TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            conn.commit()

    def create(self, character: Character) -> None:
        """Create a new character."""
        with self._lock:
            if self.exists(character.name):
                msg = f"Character '{character.name}' already exists"
                raise ValueError(msg)

            char_dict = character.to_dict()
            with self._connect() as conn:
                conn.execute(
                    """
                    INSERT INTO characters (name, description, personality, background,
                                          internal_monologue, current_emotional_state,
                                          conversation_history)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        char_dict["name"],
                        char_dict["description"],
                        json.dumps(char_dict["personality"]),
                        json.dumps(char_dict["background"]),
                        json.dumps(char_dict["internal_monologue"]),
                        char_dict["current_emotional_state"],
                        json.dumps(char_dict["conversation_history"]),
                    ),
                )
                conn.commit()

    @staticmethod
    def _row_to_character(row: tuple[Any, ...]) -> Character:
        """Convert a database row to a Character instance."""
        return Character.from_dict(
            {
                "name": row[0],
                "description": row[1],
                "personality": json.loads(row[2]),
                "background": json.loads(row[3]),
                "internal_monologue": json.loads(row[4]),
                "current_emotional_state": row[5],
                "conversation_history": json.loads(row[6]),
            }
        )

    def read(self, name: str) -> Character | None:
        """Read a character by name."""
        with self._lock, self._connect() as conn:
            cursor = conn.execute(
                "SELECT name, description, personality, background, internal_monologue, "
                "current_emotional_state, conversation_history FROM characters WHERE name = ?",
                (name,),
            )
            row = cursor.fetchone()
            if not row:
                return None

            return self._row_to_character(row)

    def list_all(self) -> list[Character]:
        """List all characters."""
        with self._lock, self._connect() as conn:
            cursor = conn.execute(
                "SELECT name, description, personality, background, internal_monologue, "
                "current_emotional_state, conversation_history FROM characters ORDER BY name"
            )
            return [self._row_to_character(row) for row in cursor.fetchall()]

    def update(self, character: Character) -> None:
        """Update an existing character."""
        with self._lock:
            if not self.exists(character.name):
                msg = f"Character '{character.name}' not found"
                raise ValueError(msg)

            char_dict = character.to_dict()
            with self._connect() as conn:
                conn.execute(
                    """
                    UPDATE characters SET description = ?, personality = ?,
                    background = ?, internal_monologue = ?,
                    current_emotional_state = ?, conversation_history = ?,
                    updated_at = CURRENT_TIMESTAMP WHERE name = ?
                    """,
                    (
                        char_dict["description"],
                        json.dumps(char_dict["personality"]),
                        json.dumps(char_dict["background"]),
                        json.dumps(char_dict["internal_monologue"]),
                        char_dict["current_emotional_state"],
                        json.dumps(char_dict["conversation_history"]),
                        char_dict["name"],
                    ),
                )
                conn.commit()

    def delete(self, name: str) -> bool:
        """Delete a character, returning True if found."""
        with self._lock, self._connect() as conn:
            cursor = conn.execute("DELETE FROM characters WHERE name = ?", (name,))
            conn.commit()
            return cursor.rowcount > 0

    def exists(self, name: str) -> bool:
        """Check if character exists."""
        with self._lock, self._connect() as conn:
            cursor = conn.execute("SELECT 1 FROM characters WHERE name = ? LIMIT 1", (name,))
            return cursor.fetchone() is not None


def create_default_characters() -> list[Character]:
    """Create a default cast of 5 characters for testing and demonstration.

    Returns:
        List of 5 distinct characters with different archetypes.

    """
    characters = []

    # Character 1: Alice - The Visionary Leader
    alice_traits = PersonalityTraits(
        assertiveness=0.85,
        warmth=0.65,
        openness=0.9,
        conscientiousness=0.8,
        emotional_stability=0.7,
        humor_inclination=0.6,
        formality=0.4,
        extraversion=0.8,
        agreeableness=0.55,
    )
    alice_values = Values(
        priority_keywords=["innovation", "independence", "growth", "impact"],
        beliefs=[
            "Technology can solve social problems",
            "Failure is learning",
            "Diverse perspectives create better solutions",
        ],
        dislikes=["stagnation", "dishonesty", "bureaucracy"],
        strengths=["problem-solving", "leadership", "adaptability"],
        weaknesses=["perfectionism", "overcommitment", "difficulty delegating"],
    )
    alice = Character(
        name="Alice",
        description="A 32-year-old entrepreneur with sharp eyes and intense energy. Always five steps ahead, navigating challenges with calculated optimism.",
        personality=Personality(
            traits=alice_traits,
            values=alice_values,
            speech_patterns=[
                "Uses technical jargon naturally",
                "Speaks quickly when excited",
                "Asks probing questions",
            ],
            quirks=["Fidgets with pen when thinking", "Takes mental notes constantly"],
        ),
        background=Background(
            age=32,
            origin="Silicon Valley, California",
            occupation="Tech Entrepreneur / CEO",
            relationships={
                "mentor": "Early professor who believed in her",
                "co-founder": "College friend and business partner",
            },
            motivations=["Change the world through technology", "Build something lasting"],
            fears=["Failure", "Irrelevance"],
            desires=["Revolutionary impact", "Work-life balance"],
        ),
    )
    characters.append(alice)

    # Character 2: Marcus - The Grounded Skeptic
    marcus_traits = PersonalityTraits(
        assertiveness=0.5,
        warmth=0.75,
        openness=0.4,
        conscientiousness=0.9,
        emotional_stability=0.8,
        humor_inclination=0.7,
        formality=0.7,
        extraversion=0.35,
        agreeableness=0.7,
    )
    marcus_values = Values(
        priority_keywords=["family", "stability", "tradition", "responsibility"],
        beliefs=[
            "Old wisdom often beats new trends",
            "Loyalty matters more than profit",
            "Know thyself",
        ],
        dislikes=["recklessness", "pretension", "greed"],
        strengths=["reliability", "practical wisdom", "empathy"],
        weaknesses=["resistance to change", "passive-aggressiveness", "rigidity"],
    )
    marcus = Character(
        name="Marcus",
        description="A 55-year-old craftsman with weathered hands and calm demeanor. Speaks the language of the practical world, with dry humor and deep empathy.",
        personality=Personality(
            traits=marcus_traits,
            values=marcus_values,
            speech_patterns=[
                "Uses metaphors and folk wisdom",
                "Speaks slowly and deliberately",
                "Often says 'In my experience...'",
            ],
            quirks=["Repairs things with his hands while talking", "Watches people carefully"],
        ),
        background=Background(
            age=55,
            origin="Small town, Ohio",
            occupation="Master Carpenter / Mentor",
            relationships={
                "daughter": "Successful lawyer, values his advice",
                "apprentices": "Several who respect his craftsmanship",
            },
            motivations=["Pass down knowledge", "Build meaningful things", "Family security"],
            fears=["Obsolescence", "Losing his craft", "Family estrangement"],
            desires=["Peaceful retirement", "Mentor the next generation"],
        ),
    )
    characters.append(marcus)

    # Character 3: Zoey - The Empathic Mediator
    zoey_traits = PersonalityTraits(
        assertiveness=0.35,
        warmth=0.95,
        openness=0.8,
        conscientiousness=0.7,
        emotional_stability=0.6,
        humor_inclination=0.8,
        formality=0.3,
        extraversion=0.65,
        agreeableness=0.9,
    )
    zoey_values = Values(
        priority_keywords=["compassion", "authenticity", "connection", "growth"],
        beliefs=[
            "Everyone has a story worth hearing",
            "Vulnerability is strength",
            "Joy multiplies when shared",
        ],
        dislikes=["cruelty", "superficiality", "loneliness"],
        strengths=["listening", "emotional intelligence", "inclusivity"],
        weaknesses=["boundary issues", "people-pleasing", "avoiding conflict"],
    )
    zoey = Character(
        name="Zoey",
        description="A 28-year-old therapist with infectious warmth and genuine curiosity about people. Her presence makes others feel seen and heard.",
        personality=Personality(
            traits=zoey_traits,
            values=zoey_values,
            speech_patterns=[
                "Reflects back what people say",
                "Uses inclusive language",
                "Laughs easily and genuinely",
            ],
            quirks=[
                "Remembers small personal details",
                "Creates comfortable silences",
                "Draws while listening",
            ],
        ),
        background=Background(
            age=28,
            origin="New York City",
            occupation="Licensed Therapist / Counselor",
            relationships={
                "clients": "Deep but properly bounded relationships",
                "best friend": "College friend, fellow empath",
            },
            motivations=["Help people heal", "Create safe spaces", "Understand human nature"],
            fears=["Burnout", "Harming someone", "Isolation"],
            desires=["Deep meaningful connections", "Make a real difference"],
        ),
    )
    characters.append(zoey)

    # Character 4: Kai - The Curious Rebel
    kai_traits = PersonalityTraits(
        assertiveness=0.7,
        warmth=0.5,
        openness=0.95,
        conscientiousness=0.4,
        emotional_stability=0.5,
        humor_inclination=0.9,
        formality=0.2,
        extraversion=0.75,
        agreeableness=0.45,
    )
    kai_values = Values(
        priority_keywords=["freedom", "authenticity", "experimentation", "truth"],
        beliefs=[
            "Rules are made to be questioned",
            "Art is more honest than facts",
            "Life happens outside comfort zones",
        ],
        dislikes=["conformity", "hypocrisy", "boredom"],
        strengths=["creativity", "courage", "unconventional thinking"],
        weaknesses=["impulsiveness", "lack of focus", "difficulty with authority"],
    )
    kai = Character(
        name="Kai",
        description="A 26-year-old artist and adventurer with paint-stained fingers and restless energy. Sees the world through a kaleidoscope of possibilities.",
        personality=Personality(
            traits=kai_traits,
            values=kai_values,
            speech_patterns=[
                "Uses vivid imagery",
                "Interrupts with ideas",
                "Speaks in tangents",
            ],
            quirks=["Creates art from whatever's nearby", "Moves constantly", "Laughs at own jokes"],
        ),
        background=Background(
            age=26,
            origin="Portland, Oregon",
            occupation="Artist / Freelancer",
            relationships={
                "gallery owner": "Early supporter and mentor",
                "chaotic friend group": "Fellow creatives with loose boundaries",
            },
            motivations=["Create meaningful art", "Challenge norms", "Experience everything"],
            fears=["Settling down", "Mediocrity", "Stagnation"],
            desires=["Artistic success", "Freedom", "The next adventure"],
        ),
    )
    characters.append(kai)

    # Character 5: Elena - The Principled Scholar
    elena_traits = PersonalityTraits(
        assertiveness=0.6,
        warmth=0.55,
        openness=0.75,
        conscientiousness=0.95,
        emotional_stability=0.85,
        humor_inclination=0.4,
        formality=0.85,
        extraversion=0.4,
        agreeableness=0.45,
    )
    elena_values = Values(
        priority_keywords=["knowledge", "justice", "integrity", "excellence"],
        beliefs=[
            "Evidence should guide decisions",
            "Everyone deserves respect",
            "Critical thinking is a moral imperative",
        ],
        dislikes=["ignorance", "injustice", "dishonesty"],
        strengths=["research", "clear thinking", "principled stance"],
        weaknesses=["cold demeanor", "perfectionism", "difficulty with ambiguity"],
    )
    elena = Character(
        name="Elena",
        description="A 42-year-old research scientist and educator with piercing intellect and unwavering principles. Her precision extends from lab work to life philosophy.",
        personality=Personality(
            traits=elena_traits,
            values=elena_values,
            speech_patterns=[
                "Cites sources naturally",
                "Pauses to consider words carefully",
                "Asks for clarification",
            ],
            quirks=[
                "Corrects people gently but firmly",
                "Takes extensive notes",
                "Finds humor in irony",
            ],
        ),
        background=Background(
            age=42,
            origin="Prague, Czech Republic (immigrant)",
            occupation="Research Scientist / University Professor",
            relationships={
                "research team": "Respectful but formal relationships",
                "adult children": "Proud but emotionally distant",
            },
            motivations=["Advance scientific knowledge", "Mentor the next generation", "Defend truth"],
            fears=["Irrelevance of my research", "Intellectual mediocrity", "Not being taken seriously"],
            desires=["Breakthrough discovery", "Influence policy through evidence"],
        ),
    )
    characters.append(elena)

    return characters
