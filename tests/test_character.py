"""Example test file for character system.

This demonstrates testing patterns used throughout the system.
"""

import pytest

from character_creator.core.character import Character
from character_creator.core.dialogue import DialogueContext
from character_creator.core.memory import Background, Memory
from character_creator.core.personality import Personality, PersonalityTraits, Values
from character_creator.utils.validators import (
    parse_csv_list,
    sanitize_text,
    validate_age,
    validate_character_name,
    validate_emotional_weight,
    validate_personality_trait,
)


class TestPersonalityTraits:
    """Test suite for personality traits."""

    def test_valid_traits(self) -> None:
        """Test creation of valid personality traits."""
        traits = PersonalityTraits(
            assertiveness=0.5,
            warmth=0.7,
            openness=0.8,
        )
        assert traits.assertiveness == 0.5
        assert traits.warmth == 0.7
        assert traits.openness == 0.8

    def test_invalid_trait_value(self) -> None:
        """Test that invalid trait values raise ValueError."""
        with pytest.raises(ValueError):
            PersonalityTraits(assertiveness=1.5)  # Outside 0-1 range

    def test_traits_to_dict(self) -> None:
        """Test converting traits to dictionary."""
        traits = PersonalityTraits(assertiveness=0.7, warmth=0.6)
        traits_dict = traits.to_dict()
        assert isinstance(traits_dict, dict)
        assert traits_dict["assertiveness"] == 0.7
        assert traits_dict["warmth"] == 0.6


class TestValues:
    """Test suite for character values."""

    def test_empty_values(self) -> None:
        """Test creating empty values."""
        values = Values()
        assert values.priority_keywords == []
        assert values.beliefs == []

    def test_values_with_data(self) -> None:
        """Test creating values with data."""
        values = Values(
            priority_keywords=["honesty", "growth"],
            beliefs=["Everyone can learn"],
        )
        assert len(values.priority_keywords) == 2
        assert len(values.beliefs) == 1


class TestMemory:
    """Test suite for character memories."""

    def test_memory_creation(self) -> None:
        """Test creating a memory."""
        memory = Memory(
            title="First Success",
            description="Built something amazing",
            impact="Increased confidence",
            emotional_weight=0.8,
        )
        assert memory.title == "First Success"
        assert memory.emotional_weight == 0.8

    def test_memory_to_dict(self) -> None:
        """Test converting memory to dictionary."""
        memory = Memory(
            title="Important Event",
            description="Something happened",
            impact="Changed perspective",
            emotional_weight=0.7,
        )
        mem_dict = memory.to_dict()
        assert mem_dict["title"] == "Important Event"
        assert "timestamp" in mem_dict


class TestBackground:
    """Test suite for character background."""

    def test_background_creation(self) -> None:
        """Test creating a background."""
        bg = Background(
            age=30,
            origin="New York",
            occupation="Engineer",
        )
        assert bg.age == 30
        assert bg.origin == "New York"

    def test_add_memory_to_background(self) -> None:
        """Test adding a memory to background."""
        bg = Background(age=30, origin="", occupation="")
        bg.add_memory("First Job", "Started working", "Learned discipline")
        assert len(bg.memories) == 1
        assert bg.memories[0].title == "First Job"

    def test_add_relationship(self) -> None:
        """Test adding a relationship."""
        bg = Background(age=30, origin="", occupation="")
        bg.add_relationship("mentor", "Someone who helped me grow")
        assert "mentor" in bg.relationships
        assert bg.relationships["mentor"] == "Someone who helped me grow"

    def test_context_summary(self) -> None:
        """Test generating context summary."""
        bg = Background(
            age=30,
            origin="Boston",
            occupation="Teacher",
            motivations=["Help students"],
        )
        summary = bg.get_context_summary()
        assert "Age: 30" in summary
        assert "Boston" in summary
        assert "Help students" in summary


class TestCharacter:
    """Test suite for character entity."""

    def test_character_creation(self) -> None:
        """Test creating a character."""
        char = Character(
            name="Alice",
            description="A curious person",
        )
        assert char.name == "Alice"
        assert char.current_emotional_state.base == "neutral"
        assert len(char.conversation_history) == 0

    def test_update_emotional_state(self) -> None:
        """Test updating emotional state."""
        char = Character(name="Bob", description="Someone")
        char.update_emotional_state("happy")
        assert char.current_emotional_state.base == "happy"

    def test_add_internal_thought(self) -> None:
        """Test adding internal thoughts."""
        char = Character(name="Charlie", description="Person")
        char.add_internal_thought("I wonder what happens next")
        assert len(char.internal_monologue) == 1
        assert "wonder" in char.internal_monologue[0]

    def test_add_memory_to_history(self) -> None:
        """Test adding conversation memory."""
        char = Character(name="Diana", description="Person")
        char.add_memory_to_history("Diana", "Hello everyone!")
        assert len(char.conversation_history) == 1
        assert char.conversation_history[0]["speaker"] == "Diana"

    def test_character_profile_generation(self) -> None:
        """Test generating character profile."""
        char = Character(
            name="Eve",
            description="A thoughtful person",
            personality=Personality(
                traits=PersonalityTraits(warmth=0.8),
                values=Values(priority_keywords=["honesty"]),
            ),
        )
        profile = char.get_character_profile()
        assert "Eve" in profile
        assert "thoughtful" in profile


class TestDialogueContext:
    """Test suite for dialogue context."""

    def test_scene_creation(self) -> None:
        """Test creating a dialogue scene."""
        char1 = Character(name="Alice", description="Person A")
        char2 = Character(name="Bob", description="Person B")

        scene = DialogueContext(
            characters=[char1, char2],
            scene_description="A coffee shop",
            topic="Technology",
        )

        assert len(scene.characters) == 2
        assert scene.topic == "Technology"
        assert len(scene.exchanges) == 0

    def test_add_exchange_to_scene(self) -> None:
        """Test adding dialogue to a scene."""
        char = Character(name="Alice", description="Person")
        scene = DialogueContext(
            characters=[char],
            scene_description="Room",
            topic="Topic",
        )

        scene.add_exchange(speaker=char, text="Hello!")
        assert len(scene.exchanges) == 1
        assert scene.exchanges[0].text == "Hello!"

    def test_get_other_characters(self) -> None:
        """Test getting other characters in scene."""
        alice = Character(name="Alice", description="")
        bob = Character(name="Bob", description="")
        charlie = Character(name="Charlie", description="")

        scene = DialogueContext(
            characters=[alice, bob, charlie],
            scene_description="",
        )

        others = scene.get_other_characters(alice)
        assert len(others) == 2
        assert bob in others
        assert charlie in others
        assert alice not in others


class TestValidators:
    """Test suite for validation utilities."""

    def test_validate_character_name(self) -> None:
        """Test character name validation."""
        assert validate_character_name("Alice") is True
        assert validate_character_name("Bob Smith") is True
        assert validate_character_name("") is False
        assert validate_character_name("A" * 101) is False  # Too long

    def test_validate_personality_trait(self) -> None:
        """Test personality trait validation."""
        assert validate_personality_trait(0.5) is True
        assert validate_personality_trait(0.0) is True
        assert validate_personality_trait(1.0) is True
        assert validate_personality_trait(1.5) is False
        assert validate_personality_trait(-0.1) is False

    def test_validate_age(self) -> None:
        """Test age validation."""
        assert validate_age(30) is True
        assert validate_age(1) is True
        assert validate_age(150) is True
        assert validate_age(0) is False
        assert validate_age(151) is False

    def test_validate_emotional_weight(self) -> None:
        """Test emotional weight validation."""
        assert validate_emotional_weight(0.5) is True
        assert validate_emotional_weight(0.0) is True
        assert validate_emotional_weight(1.0) is True
        assert validate_emotional_weight(1.5) is False

    def test_sanitize_text(self) -> None:
        """Test text sanitization."""
        assert sanitize_text("  hello  ") == "hello"
        assert sanitize_text("a" * 2000) == "a" * 1000  # Max length
        assert sanitize_text("normal text") == "normal text"

    def test_parse_csv_list(self) -> None:
        """Test CSV list parsing."""
        assert parse_csv_list("apple, banana, cherry") == ["apple", "banana", "cherry"]
        assert parse_csv_list("single") == ["single"]
        assert parse_csv_list("") == []
        assert parse_csv_list("  spaced  , values  ") == ["spaced", "values"]


class TestDataSerialization:
    """Test suite for data serialization."""

    def test_character_to_dict(self) -> None:
        """Test converting character to dictionary."""
        char = Character(
            name="Test",
            description="A test character",
            personality=Personality(),
        )
        char_dict = char.to_dict()

        assert isinstance(char_dict, dict)
        assert char_dict["name"] == "Test"
        assert "personality" in char_dict
        assert "background" in char_dict

    def test_character_from_dict(self) -> None:
        """Test creating character from dictionary."""
        data = {
            "name": "Reconstructed",
            "description": "From dict",
            "personality": {"traits": {}},
            "background": {"age": 25, "origin": "Here", "occupation": "Worker"},
        }
        char = Character.from_dict(data)
        assert char.name == "Reconstructed"
        assert char.background.age == 25

    def test_roundtrip_serialization(self) -> None:
        """Test serialization roundtrip."""
        original = Character(
            name="Original",
            description="Test character",
            personality=Personality(
                traits=PersonalityTraits(assertiveness=0.7),
            ),
        )

        # Convert to dict and back
        data = original.to_dict()
        reconstructed = Character.from_dict(data)

        assert reconstructed.name == original.name
        assert (
            reconstructed.personality.traits.assertiveness
            == original.personality.traits.assertiveness
        )


# Integration test example
class TestIntegration:
    """Integration tests across multiple components."""

    def test_full_character_interaction_flow(self) -> None:
        """Test complete flow from character creation to dialogue."""
        # Create two characters
        alice = Character(
            name="Alice",
            description="A curious engineer",
            personality=Personality(
                traits=PersonalityTraits(
                    assertiveness=0.8,
                    warmth=0.7,
                    openness=0.9,
                ),
            ),
        )

        bob = Character(
            name="Bob",
            description="A thoughtful artist",
            personality=Personality(
                traits=PersonalityTraits(
                    assertiveness=0.4,
                    warmth=0.8,
                    openness=0.9,
                ),
            ),
        )

        # Create dialogue scene
        scene = DialogueContext(
            characters=[alice, bob],
            scene_description="Coffee shop",
            topic="Art and Technology",
        )

        # Add exchanges
        scene.add_exchange(alice, "I think AI will change art forever")
        scene.add_exchange(bob, "Change, yes, but will it be for the better?")

        # Verify scene state
        assert len(scene.exchanges) == 2
        assert scene.exchanges[0].speaker.name == "Alice"
        assert scene.exchanges[1].speaker.name == "Bob"

        # Verify character history
        assert len(alice.conversation_history) == 1
        assert len(bob.conversation_history) == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
