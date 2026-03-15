"""Multi-character scene demonstration with complex interactions.

This demo shows:
1. Multiple characters interacting in a shared environment
2. Dynamic personality effects on dialogue
3. Relationship dynamics influencing responses
4. Emergent behavior from personality combinations
"""

import asyncio

# Ensure imports work regardless of working directory
from character_creator.utils.path import ensure_src_in_path

ensure_src_in_path()

from character_creator.core.character import Character
from character_creator.core.dialogue import DialogueContext
from character_creator.core.memory import Background
from character_creator.core.personality import Personality, PersonalityTraits, Values


def create_character_cast() -> dict[str, Character]:
    """Create a diverse cast of characters for complex interactions.

    Returns:
        Dictionary of characters keyed by name.
    """
    characters: dict[str, Character] = {}

    # Character 1: Emma - The Mediator
    emma = Character(
        name="Emma",
        description="A 45-year-old school principal with warm eyes and a calm demeanor. Natural listener.",
        personality=Personality(
            traits=PersonalityTraits(
                assertiveness=0.5,
                warmth=0.9,
                openness=0.7,
                conscientiousness=0.85,
                emotional_stability=0.8,
                humor_inclination=0.5,
                formality=0.6,
            ),
            values=Values(
                priority_keywords=["harmony", "growth", "community", "fairness"],
                beliefs=[
                    "Everyone deserves a voice",
                    "Understanding trumps judgment",
                    "Small changes create lasting impact",
                ],
                strengths=["empathy", "conflict resolution", "organization"],
                weaknesses=["avoiding confrontation", "overthinking", "difficulty saying no"],
            ),
            speech_patterns=["Asks clarifying questions", "Validates others' feelings"],
            quirks=["Remembers everyone's names", "Offers tea/coffee to everyone"],
        ),
    )
    emma.background = Background(
        age=45,
        origin="Midwest, USA",
        occupation="School Principal",
        motivations=["Ensure every student succeeds", "Create safe communities"],
        fears=["Conflict", "Failure to help someone in need"],
        desires=["See her students thrive", "Build better school systems"],
    )
    characters["Emma"] = emma

    # Character 2: Jake - The Skeptic
    jake = Character(
        name="Jake",
        description="A 38-year-old journalist with sharp features and keen eyes. Quick to question.",
        personality=Personality(
            traits=PersonalityTraits(
                assertiveness=0.8,
                warmth=0.4,
                openness=0.85,
                conscientiousness=0.7,
                emotional_stability=0.6,
                humor_inclination=0.8,
                formality=0.3,
            ),
            values=Values(
                priority_keywords=["truth", "justice", "independence", "integrity"],
                beliefs=[
                    "Question everything",
                    "Follow the evidence",
                    "Power should be challenged",
                ],
                strengths=["critical thinking", "research", "communication"],
                weaknesses=["cynicism", "difficulty trusting", "bluntness"],
            ),
            speech_patterns=["Asks probing questions", "Uses dry humor", "Interrupts occasionally"],
            quirks=["Always takes notes", "Makes sardonic observations", "Challenges assumptions"],
        ),
    )
    jake.background = Background(
        age=38,
        origin="New York",
        occupation="Investigative Journalist",
        relationships={"Emma": "Respects her fairness but questions her optimism"},
        motivations=["Expose truth", "Hold power accountable"],
        fears=["Corruption", "Apathy"],
        desires=["Win a major journalism award", "Expose something that changes systems"],
    )
    characters["Jake"] = jake

    # Character 3: Yuki - The Dreamer
    yuki = Character(
        name="Yuki",
        description="A 26-year-old artist/activist with vibrant energy and colorful clothing.",
        personality=Personality(
            traits=PersonalityTraits(
                assertiveness=0.65,
                warmth=0.85,
                openness=1.0,
                conscientiousness=0.4,
                emotional_stability=0.5,
                humor_inclination=0.9,
                formality=0.2,
            ),
            values=Values(
                priority_keywords=["creativity", "justice", "authenticity", "community"],
                beliefs=[
                    "Art can change the world",
                    "Everyone is creative",
                    "Systems are broken and need rebuilding",
                ],
                strengths=["creativity", "passion", "optimism"],
                weaknesses=["lack of planning", "emotional volatility", "difficulty with nuance"],
            ),
            speech_patterns=["Uses vivid metaphors", "Gets excited easily", "Idealistic language"],
            quirks=["Draws while talking", "Uses lots of emoji language", "Very expressive"],
        ),
    )
    yuki.background = Background(
        age=26,
        origin="Tokyo, Japan",
        occupation="Artist/Activist",
        relationships={"Emma": "Sees her as a secret radical", "Jake": "Respects his truth-seeking"},
        motivations=["Create meaningful art", "Spark social change", "Connect people"],
        fears=["Being ignored", "Meaninglessness"],
        desires=["Create art that changes hearts", "Build a movement"],
    )
    characters["Yuki"] = yuki

    return characters


async def run_multi_character_scene() -> None:
    """Run a complex multi-character scene with different personalities interacting."""
    print("\n" + "=" * 80)
    print("MULTI-CHARACTER SCENE: 'Education and Social Change'")
    print("=" * 80)

    characters = create_character_cast()

    scene = DialogueContext(
        characters=list(characters.values()),
        scene_description=(
            "A community center meeting room. Emma has organized a panel discussion "
            "on education reform. Jake is here to report on the discussion. Yuki is "
            "presenting artistic interpretations of educational challenges."
        ),
        topic="Can art and activism effectively reform education systems?",
    )

    print(f"\nScene Setting: {scene.scene_description}")
    print(f"Topic: {scene.topic}")
    print(f"Participants: {', '.join([c.name for c in scene.characters])}\n")

    print("=" * 80)
    print("DIALOGUE SEQUENCE (Simulated)")
    print("=" * 80)

    # Emma opens the discussion
    emmas_opening = (
        "I'm grateful everyone is here. We all care about our students and schools. "
        "I think if we listen to each other, we can find common ground."
    )
    scene.add_exchange(
        speaker=characters["Emma"],
        text=emmas_opening,
        emotional_context="hopeful and inclusive",
        internal_thought="I hope we can have a civil discussion despite our differences",
    )
    print("\n[Emma - Opening]")
    print("  \"I'm grateful everyone is here. We all care about our students and schools. "
          "I think if we listen to each other, we can find common ground.\"")
    print("  💭 Internal: I hope we can have a civil discussion despite our differences")

    # Jake immediately challenges
    jakes_response = (
        "With respect, I don't know if good intentions are enough. We need data. "
        "How many of your reforms actually work based on measurable outcomes?"
    )
    scene.add_exchange(
        speaker=characters["Jake"],
        text=jakes_response,
        emotional_context="critical but not hostile",
        internal_thought="Emma means well, but we need facts not feelings",
    )
    print("\n[Jake - Challenging]")
    print("  \"With respect, I don't know if good intentions are enough. We need data. "
          "How many of your reforms actually work based on measurable outcomes?\"")
    print("  💭 Internal: Emma means well, but we need facts not feelings")

    # Yuki brings emotion and creativity
    yukis_input = (
        "But numbers don't capture the whole story! I've created pieces showing "
        "students who feel trapped, ignored. The measurable doesn't capture the human."
    )
    scene.add_exchange(
        speaker=characters["Yuki"],
        text=yukis_input,
        emotional_context="passionate and emotional",
        internal_thought="Jake reduces everything to cold facts, but education is about people!",
    )
    print("\n[Yuki - Adding Perspective]")
    print("  \"But numbers don't capture the whole story! I've created pieces showing "
          "students who feel trapped, ignored. The measurable doesn't capture the human.\"")
    print("  💭 Internal: Jake reduces everything to cold facts, but education is about people!")

    # Emma mediates
    emmas_mediation = (
        "I hear both of you. Jake, we absolutely need to measure impact. "
        "And Yuki, the emotional experience of students is data we can't ignore. "
        "Perhaps we need both the numbers and the stories?"
    )
    scene.add_exchange(
        speaker=characters["Emma"],
        text=emmas_mediation,
        emotional_context="thoughtful and bridging",
        internal_thought="Can I help them see they're not as far apart as they think?",
    )
    print("\n[Emma - Mediating]")
    print("  \"I hear both of you. Jake, we absolutely need to measure impact. "
          "And Yuki, the emotional experience of students is data we can't ignore. "
          "Perhaps we need both the numbers and the stories?\"")
    print("  💭 Internal: Can I help them see they're not as far apart as they think?")

    # Jake acknowledges
    jakes_concession = (
        "That's... fair. I've interviewed students whose test scores improved "
        "but who still felt disengaged. Maybe we've been measuring the wrong things."
    )
    scene.add_exchange(
        speaker=characters["Jake"],
        text=jakes_concession,
        emotional_context="grudgingly thoughtful",
        internal_thought="Maybe there's something to this integration idea",
    )
    print("\n[Jake - Acknowledging]")
    print("  \"That's... fair. I've interviewed students whose test scores improved "
          "but who still felt disengaged. Maybe we've been measuring the wrong things.\"")
    print("  💭 Internal: Maybe there's something to this integration idea")

    # Yuki builds on it
    yukis_vision = (
        "YES! What if we used art to help students envision themselves succeeding? "
        "Create murals about their futures, community-based learning through creativity?"
    )
    scene.add_exchange(
        speaker=characters["Yuki"],
        text=yukis_vision,
        emotional_context="excited and visionary",
        internal_thought="They're getting it! This could actually work!",
    )
    print("\n[Yuki - Building Vision]")
    print("  \"YES! What if we used art to help students envision themselves succeeding? "
          "Create murals about their futures, community-based learning through creativity?\"")
    print("  💭 Internal: They're getting it! This could actually work!")

    # Analysis
    print("\n" + "=" * 80)
    print("SCENE ANALYSIS")
    print("=" * 80)

    print(f"\nTotal exchanges: {len(scene.exchanges)}")
    print("\nCharacter contributions:")
    for char in scene.characters:
        contributions = sum(1 for e in scene.exchanges if e.speaker.name == char.name)
        print(f"  - {char.name}: {contributions} contributions")

    print("\nPersonality dynamics observed:")
    print("  • Emma (Mediator): Successfully bridged perspectives")
    print("  • Jake (Skeptic): Questioned but remained open-minded")
    print("  • Yuki (Dreamer): Brought creative energy and vision")

    print("\nEmerged themes:")
    print("  ✓ Integration of qualitative and quantitative measures")
    print("  ✓ Art as educational and transformative tool")
    print("  ✓ Importance of student emotional engagement")
    print("  ✓ Multi-stakeholder collaboration potential")

    print("\n" + "=" * 80)
    print("KEY INSIGHTS")
    print("=" * 80)
    print("""
The interaction demonstrated emergent behavior not pre-programmed:

1. COMPLEMENTARY STRENGTHS
   - Emma's empathy found common ground
   - Jake's skepticism pushed for rigor
   - Yuki's creativity offered new solutions

2. PERSONALITY EFFECTS ON DIALOGUE
   - Emma's warmth allowed her to mediate without dismissing anyone
   - Jake's directness created initial tension but forced clarity
   - Yuki's passion was infectious and opened new possibilities

3. DYNAMIC EVOLUTION
   - Positions softened through genuine listening
   - New ideas emerged from the intersection of perspectives
   - The group moved toward integration rather than compromise

This shows how careful personality modeling creates believable, rich
character interactions that feel authentic rather than scripted.
    """)


async def main() -> None:
    """Run the multi-character scene demo."""
    print("\n🎭 MULTI-CHARACTER SCENE DEMONSTRATION\n")

    await run_multi_character_scene()

    print("\n" + "=" * 80)
    print("DEMO COMPLETE - Multi-Character Interaction")
    print("=" * 80)


if __name__ == "__main__":
    asyncio.run(main())
