"""Heredity engine for evolutionary personality crossover and mutation.

Handles genetic-style crossover of personality traits between two parent
characters to produce offspring with inherited (but slightly mutated)
personality profiles.  This drives the anthropological simulation layer
where populations of characters can evolve over generations.
"""

from __future__ import annotations

import random

from typing import TYPE_CHECKING

from character_creator.core.constants import DEFAULT_MUTATION_RATE, TRAIT_MAX, TRAIT_MIN
from character_creator.core.memory import Background
from character_creator.core.personality import Personality, PersonalityTraits, Values

if TYPE_CHECKING:
    from character_creator.core.character import Character


# Derive heritable traits directly from the model schema — never drifts.
_HERITABLE_TRAITS: list[str] = list(PersonalityTraits.model_fields.keys())


def cross_traits(
    p1: PersonalityTraits,
    p2: PersonalityTraits,
    *,
    mutation_rate: float = DEFAULT_MUTATION_RATE,
) -> PersonalityTraits:
    """Create a child trait set via single-gene crossover + Gaussian mutation.

    For each trait the child inherits from a randomly chosen parent, then
    a small Gaussian mutation is applied to simulate developmental drift.

    Args:
        p1: First parent's traits.
        p2: Second parent's traits.
        mutation_rate: Standard deviation of the Gaussian mutation noise.

    Returns:
        A new ``PersonalityTraits`` instance for the child.

    """
    parent_dicts = [p1.to_dict(), p2.to_dict()]
    child: dict[str, float] = {}
    for trait in _HERITABLE_TRAITS:
        base = random.choice(parent_dicts)[trait]  # noqa: S311
        mutation = random.gauss(0, mutation_rate)
        child[trait] = max(TRAIT_MIN, min(TRAIT_MAX, base + mutation))
    return PersonalityTraits.from_dict(child)


def cross_values(v1: Values, v2: Values) -> Values:
    """Merge parent value systems, sampling from both.

    Each list field takes a random subset of the union of both parents'
    entries (up to a reasonable cap) — children don't inherit *everything*
    but carry echoes of both lineages.
    """

    def _sample_union(a: list[str], b: list[str], cap: int = 5) -> list[str]:
        pool = list(dict.fromkeys(a + b))  # deduplicated, order-preserved
        k = min(len(pool), cap)
        return random.sample(pool, k) if pool else []

    return Values(
        priority_keywords=_sample_union(v1.priority_keywords, v2.priority_keywords),
        beliefs=_sample_union(v1.beliefs, v2.beliefs),
        dislikes=_sample_union(v1.dislikes, v2.dislikes),
        strengths=_sample_union(v1.strengths, v2.strengths),
        weaknesses=_sample_union(v1.weaknesses, v2.weaknesses),
    )


def reproduce(
    parent1: Character,
    parent2: Character,
    child_name: str,
    *,
    mutation_rate: float = DEFAULT_MUTATION_RATE,
) -> Character:
    """Create a new character from two parents via personality crossover.

    The child inherits:
    - Personality traits via single-gene crossover + Gaussian mutation.
    - Values sampled from the union of both parents.
    - Speech patterns / quirks randomly drawn from both parents.
    - A background reflecting their parentage.

    Args:
        parent1: First parent character.
        parent2: Second parent character.
        child_name: Name for the child character.
        mutation_rate: Trait mutation strength.

    Returns:
        A new ``Character`` with inherited personality.

    """
    # Deferred import to break circular dependency (character -> personality -> heredity)
    from character_creator.core.character import Character  # noqa: PLC0415

    child_traits = cross_traits(
        parent1.personality.traits,
        parent2.personality.traits,
        mutation_rate=mutation_rate,
    )
    child_values = cross_values(
        parent1.personality.values,
        parent2.personality.values,
    )

    # Randomly inherit some speech patterns & quirks from each parent
    all_patterns = parent1.personality.speech_patterns + parent2.personality.speech_patterns
    all_quirks = parent1.personality.quirks + parent2.personality.quirks
    child_patterns = random.sample(all_patterns, min(len(all_patterns), 3)) if all_patterns else []
    child_quirks = random.sample(all_quirks, min(len(all_quirks), 3)) if all_quirks else []

    child_personality = Personality(
        traits=child_traits,
        values=child_values,
        speech_patterns=child_patterns,
        quirks=child_quirks,
    )

    # Build a lineage-aware background
    p1_gen = parent1.background.age  # repurpose age as proxy during sim
    p2_gen = parent2.background.age
    child_gen = max(p1_gen, p2_gen) + 1

    child_background = Background(
        age=child_gen,
        origin=f"Child of {parent1.name} ({parent1.background.origin}) "
        f"and {parent2.name} ({parent2.background.origin})",
        occupation="Emerging",
        relationships={
            parent1.name: "parent",
            parent2.name: "parent",
        },
        motivations=random.sample(
            parent1.background.motivations + parent2.background.motivations,
            min(2, len(parent1.background.motivations + parent2.background.motivations)),
        )
        if (parent1.background.motivations or parent2.background.motivations)
        else [],
    )

    mbti = child_personality.mbti_type
    return Character(
        name=child_name,
        description=(
            f"Generation-{child_gen} descendant of {parent1.name} & {parent2.name}. "
            f"Personality type: {mbti.value} ({child_personality.mbti_archetype})."
        ),
        personality=child_personality,
        background=child_background,
    )
