"""Prompt templates and generation utilities for LLM interactions."""

from string import Template
from typing import Any


class CharacterCreationPrompts:
    """Templates for character creation assistance."""

    GENERATE_PERSONALITY = Template("""Create a detailed personality profile for a character named $name with this concept: $concept

Generate the response as a JSON object with these exact fields:
{
    "assertiveness": <float 0-1>,
    "warmth": <float 0-1>,
    "openness": <float 0-1>,
    "conscientiousness": <float 0-1>,
    "emotional_stability": <float 0-1>,
    "humor_inclination": <float 0-1>,
    "formality": <float 0-1>,
    "extraversion": <float 0-1, how energised by social interaction>,
    "agreeableness": <float 0-1, how cooperative and empathetic>,
    "speech_patterns": [<list of strings describing speech patterns>],
    "quirks": [<list of character quirks>],
    "priority_keywords": [<list of core values>],
    "beliefs": [<list of core beliefs>],
    "strengths": [<list of strengths>],
    "weaknesses": [<list of weaknesses>]
}

The nine numeric traits map to the OCEAN Big-Five model:
- Openness = openness
- Conscientiousness = conscientiousness
- Extraversion = extraversion
- Agreeableness = agreeableness
- Neuroticism ≈ inverse of emotional_stability

Be creative but consistent. The traits should feel authentic and interconnected.""")

    GENERATE_BACKGROUND = Template("""Create a compelling background for a character named $name.

Personality traits: $personality_summary

Generate the response as a JSON object with these exact fields:
{
    "age": <integer>,
    "origin": "<string describing where they're from>",
    "occupation": "<string>",
    "motivations": [<list of what drives them>],
    "fears": [<list of their fears>],
    "desires": [<list of what they want>],
    "memories": [
        {
            "title": "<string>",
            "description": "<detailed description of the memory>",
            "impact": "<how this shaped them>",
            "emotional_weight": <float 0-1>
        }
    ],
    "relationships": {
        "<name>": "<relationship description>"
    }
}

Create 3-5 significant memories that would shape someone with this personality. Make the background feel lived-in and authentic.""")

    GENERATE_DESCRIPTION = Template("""Create a vivid physical and presence description for $name.

Character: $personality_summary
Background: $background_summary

Provide a 2-3 sentence description that captures their appearance and presence, reflecting their personality and background. Be specific and evocative.""")


class DialoguePrompts:
    """Templates for dialogue generation."""

    CHARACTER_RESPONSE = Template("""You are roleplaying as $character_name. Stay in character and respond naturally.

## Your Character Profile
$character_profile

## MBTI Communication Guidance
Your personality type is $mbti_type ($mbti_archetype).
Communication style: $communication_style

## Scene
$scene_description

## Other Characters in the Scene
$other_characters

## Current Topic
$topic

## Recent Conversation
$conversation_history

## Instructions
1. Respond as $character_name would, honouring their MBTI communication style.
2. Keep the response concise (1-5 sentences typically).
3. Reflect their emotional state, values, and personality type in the response.
4. Be authentic to their experiences and beliefs.
5. Respond with ONLY the dialogue text, no narration or meta-commentary.

$character_name's response:""")

    PRE_EXCHANGE_MONOLOGUE = Template("""You are generating $character_name's immediate inner reaction after hearing the previous speaker, as they prepare to respond.

## Character Profile
$character_profile

## MBTI Context
Personality type: $mbti_type ($mbti_archetype).
Inner world filter: $communication_style

## Active Internal Tensions
$active_tensions

## Recent Dialogue (ending with what $character_name just heard)
$conversation_history

## Instructions
This is the unguarded thought that surfaces the instant the previous speaker finishes — before $character_name chooses to speak.
1. Capture their raw, instinctive reaction to what was just said.
2. Show any gap between their private feelings and the composed face they will present publicly.
3. Let them weigh what to say next, shaped by their MBTI type, values, and active tensions.
4. The thought should feel live and immediate — this is the moment of deliberation before they open their mouth.
5. Generate a brief internal thought (1-3 sentences).
6. Respond with ONLY the internal thought text, no narration or meta-commentary.

$character_name's pre-exchange thought:""")

    POST_EXCHANGE_MONOLOGUE = Template("""You are generating $character_name's immediate inner afterthought in the moments after they have just spoken.

## Character Profile
$character_profile

## MBTI Context
Personality type: $mbti_type ($mbti_archetype).
Inner world filter: $communication_style

## Active Internal Tensions
$active_tensions

## Recent Dialogue
$conversation_history

## What $character_name Just Said
"$own_dialogue"

## Instructions
This is the private thought that follows the act of speaking — an unfiltered reaction to their own words.
1. Generate a brief internal thought (1-2 sentences) that reflects their mental state in the immediate aftermath.
2. They may replay what they said, feel relief or regret, notice something they left out, or simply let it go — whatever fits their personality.
3. Consider their MBTI type — it shapes how they internally process their own actions.
4. If there are active internal tensions, the character may feel them more sharply now that they have spoken.
5. Respond with ONLY the internal thought text, no narration or meta-commentary.

$character_name's post-exchange thought:""")

    EMOTION_INFERENCE = Template("""Given this dialogue line and the speaker's personality, identify the emotional context in ONE word.

Speaker: $character_name (emotional stability: $emotional_stability)
Current emotional state: $current_emotional_state
Dialogue: "$dialogue_text"

Choose one word from: enthusiastic, curious, thoughtful, apologetic, resistant, confident, vulnerable, playful, frustrated, neutral.
Respond with ONLY the single word.""")

    MULTI_CHARACTER_SCENE = Template("""Generate a brief multi-character dialogue exchange for this scene.

Scene: $scene_description
Topic: $topic

Characters:
$characters_list

Recent exchanges:
$previous_exchanges

Generate the next exchange as "CHARACTER_NAME: dialogue text" format. One character speaks. Keep it natural and true to character personalities.

Next exchange:""")


class MemoryPrompts:
    """Templates for tiered memory condensation."""

    WORKING_TO_SHORT_TERM = Template("""You are summarising a block of conversation exchanges from the perspective of $character_name.

## Character Context
Personality: $personality_summary
Current emotional state: $emotional_state
Core values: $core_values

## Exchanges to Summarise
$exchanges

## Instructions
Summarise these exchanges into a single memory from $character_name's subjective perspective.
Your summary should be coloured by the character's personality and current emotional state — this is how THEY would remember these events, not an objective account.

Respond with ONLY a JSON object (no markdown fencing):
{
    "summary": "<1-3 sentence subjective summary>",
    "emotional_tone": "<single word: the dominant emotion of this memory>",
    "salience": <float 0.0-1.0: how important this feels to the character>,
    "topics": ["<key theme 1>", "<key theme 2>"],
    "perspective_bias": "<brief note on how personality/emotion coloured this memory, or null>"
}""")

    SHORT_TO_LONG_TERM = Template("""You are condensing several memories for $character_name into a broader pattern or theme.

## Character Context
Personality: $personality_summary
Current emotional state: $emotional_state
Core values: $core_values

## Memories to Condense
$memories

## Instructions
Merge these memories into a single long-term memory — a general impression, pattern, or theme rather than specific details. This is how $character_name would recall this period of their life after time has passed.

Respond with ONLY a JSON object (no markdown fencing):
{
    "summary": "<1-3 sentence thematic summary>",
    "emotional_tone": "<single word: overall emotional impression>",
    "salience": <float 0.0-1.0: lasting importance to the character>,
    "topics": ["<theme 1>", "<theme 2>"],
    "perspective_bias": "<brief note on how memory has been coloured by personality, or null>"
}""")


class EnvironmentPrompts:
    """Templates for environment and scene creation."""

    GENERATE_ENVIRONMENT = Template("""Create a detailed environment description for a scene where characters will interact.

Setting concept: $setting_concept
Purpose: $purpose

Describe the environment in these categories:
- Physical Layout: (describe the space, dimensions, key features)
- Atmosphere: (mood, lighting, ambiance)
- Interactive Elements: (objects or features characters can interact with)
- Sensory Details: (sounds, smells, tactile sensations)
- Emotional Impact: (how does this space make someone feel?)

Make it immersive and detailed enough for AI characters to naturally reference in dialogue.""")

    GENERATE_SCENARIO = Template("""Create an interaction scenario for characters in this setting.

Environment: $environment_description
Characters involved: $character_names
Setting: $setting_description

Generate:
- Opening situation: (how the scene begins)
- Initial tensions or dynamics: (what creates interest)
- Potential turning points: (key moments that could shift the scene)
- Environmental challenges: (how the setting affects interaction)

Make it realistic and spark natural character interaction.""")


class SelfReflectionPrompts:
    """Templates for character self-reflection and self-model generation."""

    SELF_REFLECTION = Template("""You are generating a self-reflection for $character_name — as if they are pausing to take stock of themselves.

## Character Context
Personality: $personality_summary
Current emotional state: $emotional_state
Core values: $core_values
Recent trait changes: $trait_changes

## Internal Monologue
$recent_monologue

## Recent Exchanges
$recent_exchanges

## Instructions
Generate a reflective self-model — how $character_name sees themselves RIGHT NOW.
Consider how their recent experiences, personality evolution, and emotional state colour their self-perception.
Be authentic to the character voice.

Respond with ONLY a JSON object (no markdown fencing):
{
    "self_concept": "<2-3 sentence description of how they see themselves right now>",
    "emotional_awareness": "<1-2 sentences on what they're feeling and why>",
    "value_tensions": ["<tension 1 between beliefs and recent actions>", "<tension 2>"],
    "growth_edges": ["<area where they sense they are changing>", "<another area>"]
}""")


class DissonancePrompts:
    """Templates for behaviour extraction and cognitive dissonance detection."""

    BEHAVIOUR_EXTRACTION = Template("""Analyse $character_name's recent dialogue and identify their behavioural patterns.

## Recent Exchanges
$recent_exchanges

## Instructions
Extract 2-4 behavioural themes — what is $character_name actually DOING (not saying they believe)?
Focus on actions, tone, and interpersonal dynamics rather than stated intentions.

Respond with ONLY a JSON array (no markdown fencing):
[
    {"theme": "<behavioural pattern described in one sentence>", "confidence": <float 0.0-1.0>}
]""")

    DISSONANCE_DETECTION = Template("""Detect conflicts between $character_name's stated values and their observed behaviour.

## Core Values
$core_values

## Observed Behaviour Themes
$behaviour_themes

## Instructions
For each value, check whether any behaviour theme contradicts it. Only report genuine contradictions — minor inconsistencies should be ignored.

Respond with ONLY a JSON array (no markdown fencing). Return an empty array [] if no dissonances are found:
[
    {
        "value": "<the value being contradicted>",
        "behaviour": "<what the character actually did that contradicts it>",
        "severity": <float 0.0-1.0, how stark the contradiction is>
    }
]""")


class TraitEvolutionPrompts:
    """Templates for experience classification and trait evolution."""

    EXPERIENCE_CLASSIFICATION = Template("""Classify the emotional experience in this dialogue exchange for $character_name.

Emotional context: $emotional_context
Dialogue: "$exchange_text"

Choose exactly ONE experience type from: $experience_types
Respond with ONLY the single lowercase word.""")

    MILESTONE_REVIEW = Template("""You are reviewing how a scene has affected $character_name's personality.

## Character Profile
$character_profile

## Self-Model
$self_model

## Scene Context
Scene: $scene_description
Total exchanges: $exchange_count

## Personality at Scene Start
$traits_start

## Personality Now (after micro-shifts)
$traits_current

## Accumulated Micro-Shifts
$trait_deltas

## Instructions
Review the accumulated personality changes from this scene. For each trait that has meaningfully shifted, decide whether to:
- **Ratify** the shift (the scene justified it)
- **Amplify** it (the scene's impact was stronger than the micro-shifts captured)
- **Reverse** it (on reflection, the character would push back against this change)

For traits with no meaningful change, you may propose a new shift if the scene clearly warranted one.

Respond with ONLY a JSON object (no markdown fencing):
{
    "shifts": [
        {
            "trait_name": "<trait name>",
            "old_value": <float, value at scene start>,
            "new_value": <float, your proposed final value>,
            "justification": "<1-2 sentence narrative justification>",
            "confidence": <float 0.0-1.0, how certain you are>
        }
    ],
    "narrative_summary": "<2-4 sentence summary of this character's arc during the scene>"
}""")


def substitute_prompt(template: Template, **kwargs: Any) -> str:
    """Safely substitute variables in a prompt template.

    Args:
        template: String template with $variable placeholders.
        **kwargs: Variables to substitute.

    Returns:
        Substituted prompt string.

    Raises:
        KeyError: If required variable is missing.

    """
    return template.substitute(**kwargs)
