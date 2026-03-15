"""Basic character creation and interaction demo.

This demo shows how to:
1. Create characters with detailed personalities and backgrounds
2. Set up a dialogue scene
3. Generate interactions between characters
"""

import asyncio

# Ensure imports work regardless of working directory
from character_creator.utils.path import ensure_src_in_path

ensure_src_in_path()

from character_creator.core.character import Character
from character_creator.core.dialogue import DialogueContext
from character_creator.core.memory import Background
from character_creator.core.personality import Personality, PersonalityTraits, Values


async def create_demo_characters() -> tuple[Character, Character]:
    """Create two example characters for demonstration.

    Returns:
        Tuple of two created characters.
    """
    # Create Character 1: Alice - An ambitious entrepreneur
    alice_traits = PersonalityTraits(
        assertiveness=0.85,
        warmth=0.65,
        openness=0.9,
        conscientiousness=0.8,
        emotional_stability=0.7,
        humor_inclination=0.6,
        formality=0.4,
    )

    alice_values = Values(
        priority_keywords=["innovation", "independence", "growth", "impact"],
        beliefs=[
            "Technology can solve social problems",
            "Failure is just learning",
            "Diverse perspectives create better solutions",
        ],
        dislikes=["stagnation", "dishonesty", "bureaucracy"],
        strengths=["problem-solving", "leadership", "adaptability"],
        weaknesses=["perfectionism", "workaholic tendencies", "difficulty delegating"],
    )

    alice = Character(
        name="Alice",
        description="A 32-year-old entrepreneur with sharp eyes and an energetic presence. Always seems to be thinking three steps ahead.",
        personality=Personality(
            traits=alice_traits,
            values=alice_values,
            speech_patterns=[
                "Uses technical jargon naturally",
                "Speaks quickly when excited",
                "Asks probing questions",
            ],
            quirks=[
                "Fidgets with pen when thinking",
                "Constantly takes mental notes",
                "Laughs at her own jokes",
            ],
        ),
    )

    # Add memories to Alice
    alice.background = Background(
        age=32,
        origin="Silicon Valley, California",
        occupation="Tech Entrepreneur / CEO",
        relationships={
            "mentor": "Early professor who encouraged her tech dreams",
            "co-founder": "College friend who built first startup with her",
        },
        motivations=["Change the world through technology", "Build something lasting"],
        fears=["Failure", "Irrelevance"],
        desires=["Successfully scale her company", "Inspire other women in tech"],
    )

    # Add formative memories
    alice.background.add_memory(
        title="First Coding Experience",
        description="At age 12, she built her first website and watched people use it. That moment changed everything.",
        impact="Solidified her belief that technology could solve real problems",
        emotional_weight=0.9,
    )

    alice.background.add_memory(
        title="First Business Failure",
        description="Her first startup failed spectacularly after 18 months. Lost savings, disappointed investors.",
        impact="Taught her resilience and the importance of learning from failure",
        emotional_weight=0.85,
    )

    alice.background.add_memory(
        title="Major Success",
        description="Her second company was acquired for $50M. Validated her vision.",
        impact="Increased confidence but also awareness of responsibility",
        emotional_weight=0.8,
    )

    # Create Character 2: Marcus - A thoughtful artist
    marcus_traits = PersonalityTraits(
        assertiveness=0.35,
        warmth=0.85,
        openness=0.95,
        conscientiousness=0.5,
        emotional_stability=0.6,
        humor_inclination=0.7,
        formality=0.3,
    )

    marcus_values = Values(
        priority_keywords=["beauty", "authenticity", "connection", "meaning"],
        beliefs=[
            "Art reflects the human condition",
            "Authenticity matters more than perfection",
            "Everyone has a story to tell",
        ],
        dislikes=["commercialization of art", "pretentiousness", "dishonesty"],
        strengths=["empathy", "creativity", "emotional intelligence"],
        weaknesses=["indecisiveness", "struggle with self-promotion", "perfectionism"],
    )

    marcus = Character(
        name="Marcus",
        description="A 28-year-old artist with paint-stained hands and thoughtful eyes. Radiates a calm, creative energy.",
        personality=Personality(
            traits=marcus_traits,
            values=marcus_values,
            speech_patterns=[
                "Speaks in metaphors and observations",
                "Often pauses to think",
                "Uses hand gestures expressively",
            ],
            quirks=[
                "Sketches while listening",
                "Gets lost in conversation",
                "Hums while thinking",
            ],
        ),
    )

    # Add memories to Marcus
    marcus.background = Background(
        age=28,
        origin="Portland, Oregon",
        occupation="Visual Artist / Art Teacher",
        relationships={
            "art teacher": "Who first believed in his talent",
            "artist friend": "Collaborator and source of artistic inspiration",
        },
        motivations=["Create meaningful art", "Inspire creativity in others"],
        fears=["Becoming commercially corrupted", "Running out of ideas"],
        desires=["Have a major gallery exhibition", "Help underprivileged kids access art education"],
    )

    # Add formative memories
    marcus.background.add_memory(
        title="Art School Rejection",
        description="Applied to prestigious art school, was rejected. Devastated but determined to prove doubters wrong.",
        impact="Made him question but ultimately strengthen his artistic voice",
        emotional_weight=0.75,
    )

    marcus.background.add_memory(
        title="First Positive Feedback",
        description="An established artist saw his work at a gallery and said it had real depth and authenticity.",
        impact="Validated his artistic direction and boosted confidence",
        emotional_weight=0.8,
    )

    return alice, marcus


async def demo_character_creation() -> None:
    """Demonstrate character creation and profile generation."""
    print("\n" + "=" * 70)
    print("CHARACTER CREATION DEMO")
    print("=" * 70)

    alice, marcus = await create_demo_characters()

    print("\n--- Character 1: Alice ---")
    print(alice.get_character_profile())

    print("\n--- Character 2: Marcus ---")
    print(marcus.get_character_profile())

    return alice, marcus


async def demo_dialogue_setup(alice: Character, marcus: Character) -> None:
    """Demonstrate dialogue scene setup.

    Args:
        alice: First character.
        marcus: Second character.
    """
    print("\n" + "=" * 70)
    print("DIALOGUE SCENE SETUP")
    print("=" * 70)

    scene = DialogueContext(
        characters=[alice, marcus],
        scene_description="A coffee shop on a sunny afternoon. Alice is sitting at a table with her laptop, Marcus walks in.",
        topic="The balance between technology and art",
    )

    print(f"\nScene: {scene.scene_description}")
    print(f"Topic: {scene.topic}")
    print(f"Characters: {', '.join([c.name for c in scene.characters])}")

    return scene


async def demo_character_interaction(alice: Character, marcus: Character) -> None:
    """Demonstrate character interactions without LLM (using placeholders).

    Args:
        alice: First character.
        marcus: Second character.
    """
    print("\n" + "=" * 70)
    print("CHARACTER INTERACTIONS")
    print("=" * 70)

    scene = DialogueContext(
        characters=[alice, marcus],
        scene_description="A coffee shop on a sunny afternoon",
        topic="The relationship between technology and art",
    )

    # Simulate interactions (without actual LLM calls)
    print("\n--- Simulated Dialogue ---\n")

    # Alice speaks
    alice_dialogue = (
        "You know, I've been thinking about how tech could democratize art creation. "
        "Imagine AI tools that help artists rather than replace them."
    )
    alice.add_internal_thought("I hope he doesn't think I'm trying to commercialize everything...")
    scene.add_exchange(
        speaker=alice,
        text=alice_dialogue,
        emotional_context="thoughtful and hopeful",
        internal_thought="I wonder if he'll understand my perspective",
    )
    print(f"Alice: {alice_dialogue}")
    print("  [Thinking: I wonder if he'll understand my perspective]")

    # Marcus speaks
    marcus_dialogue = (
        "That's... actually interesting. I worry about authenticity getting lost, "
        "but maybe it's just the tools evolving, like when photography emerged."
    )
    scene.add_exchange(
        speaker=marcus,
        text=marcus_dialogue,
        emotional_context="intrigued but cautious",
        internal_thought="She's not just about making money... there's depth here",
    )
    print(f"\nMarcus: {marcus_dialogue}")
    print("  [Thinking: She's not just about making money... there's depth here]")

    # Alice responds
    alice_dialogue2 = (
        "Exactly! Photography didn't kill painting—it freed artists to explore expression beyond representation. "
        "AI could do the same."
    )
    scene.add_exchange(
        speaker=alice,
        text=alice_dialogue2,
        emotional_context="energized",
        internal_thought="I think he gets it!",
    )
    print(f"\nAlice: {alice_dialogue2}")
    print("  [Thinking: I think he gets it!]")

    # Marcus final response
    marcus_dialogue2 = (
        "You might be right... Maybe I've been afraid of change instead of embracing it. "
        "I'd like to explore this more."
    )
    scene.add_exchange(
        speaker=marcus,
        text=marcus_dialogue2,
        emotional_context="thoughtful openness",
        internal_thought="This conversation is changing how I see things",
    )
    print(f"\nMarcus: {marcus_dialogue2}")
    print("  [Thinking: This conversation is changing how I see things]")

    # Display conversation summary
    print("\n--- Conversation Analysis ---")
    print(f"Total exchanges: {len(scene.exchanges)}")
    print(f"Characters participating: {', '.join([c.name for c in scene.characters])}")
    print(f"Topic discussed: {scene.topic}")

    # Character development
    print("\n--- Character Development During Scene ---")
    print(f"Alice emotional state: {alice.current_emotional_state}")
    print(f"Marcus emotional state: {marcus.current_emotional_state}")
    print(f"\nAlice's conversation history ({len(alice.conversation_history)} entries)")
    print(f"Marcus's conversation history ({len(marcus.conversation_history)} entries)")


async def main() -> None:
    """Run all demonstrations."""
    print("\n🎭 CHARACTER CREATOR - BASIC INTERACTION DEMO\n")

    # Create characters
    alice, marcus = await demo_character_creation()

    # Set up a scene
    await demo_dialogue_setup(alice, marcus)

    # Run interactions
    await demo_character_interaction(alice, marcus)

    print("\n" + "=" * 70)
    print("DEMO COMPLETE")
    print("=" * 70)
    print("\nKey Features Demonstrated:")
    print("✓ Rich character personality traits with quantified dimensions")
    print("✓ Detailed background with formative memories")
    print("✓ Multi-character dialogue scene management")
    print("✓ Character emotional state tracking")
    print("✓ Internal monologue/thought processes")
    print("✓ Conversation history maintenance")
    print("\nNext Steps:")
    print("- Integrate LLM provider for actual dialogue generation")
    print("- Build interactive environment definitions")
    print("- Create persistent character storage system")


if __name__ == "__main__":
    asyncio.run(main())
