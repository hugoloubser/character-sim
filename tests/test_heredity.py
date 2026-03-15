"""Tests for MBTI derivation, compatibility scoring, and heredity engine."""

from character_creator.core.character import Character
from character_creator.core.heredity import cross_traits, cross_values, reproduce
from character_creator.core.memory import Background
from character_creator.core.personality import (
    MBTI_ARCHETYPE,
    MBTI_COMMUNICATION_STYLE,
    MBTIType,
    Personality,
    PersonalityTraits,
    Values,
    mbti_compatibility,
)

# ---------------------------------------------------------------------------
# MBTI derivation
# ---------------------------------------------------------------------------

class TestMBTIDerivation:
    """Verify MBTI type is correctly derived from OCEAN traits."""

    def test_high_all_gives_enfj(self) -> None:
        """E(xtraversion>=0.5), N(openness>=0.5), F(agreeableness>=0.5), J(conscientiousness>=0.5)."""
        traits = PersonalityTraits(extraversion=0.8, openness=0.7, agreeableness=0.6, conscientiousness=0.9)
        p = Personality(traits=traits)
        assert p.mbti_type == MBTIType.ENFJ

    def test_low_all_gives_istp(self) -> None:
        """I, S, T, P when all relevant traits < 0.5."""
        traits = PersonalityTraits(extraversion=0.2, openness=0.3, agreeableness=0.1, conscientiousness=0.2)
        p = Personality(traits=traits)
        assert p.mbti_type == MBTIType.ISTP

    def test_intj(self) -> None:
        """Classic architect: low E, high N, low F, high J."""
        traits = PersonalityTraits(extraversion=0.3, openness=0.9, agreeableness=0.2, conscientiousness=0.8)
        p = Personality(traits=traits)
        assert p.mbti_type == MBTIType.INTJ
        assert p.mbti_archetype == "Architect"

    def test_esfp(self) -> None:
        """Classic entertainer: high E, low N, high F, low J."""
        traits = PersonalityTraits(extraversion=0.9, openness=0.3, agreeableness=0.8, conscientiousness=0.2)
        p = Personality(traits=traits)
        assert p.mbti_type == MBTIType.ESFP
        assert p.mbti_archetype == "Entertainer"

    def test_boundary_at_half(self) -> None:
        """Exactly 0.5 should map to E, N, F, J (>= 0.5)."""
        traits = PersonalityTraits(extraversion=0.5, openness=0.5, agreeableness=0.5, conscientiousness=0.5)
        p = Personality(traits=traits)
        assert p.mbti_type == MBTIType.ENFJ

    def test_all_16_types_reachable(self) -> None:
        """Every MBTIType can be derived from some trait combination."""
        achieved: set[MBTIType] = set()
        for e in (0.2, 0.8):
            for o in (0.2, 0.8):
                for a in (0.2, 0.8):
                    for c in (0.2, 0.8):
                        t = PersonalityTraits(extraversion=e, openness=o, agreeableness=a, conscientiousness=c)
                        achieved.add(Personality(traits=t).mbti_type)
        assert achieved == set(MBTIType)


class TestMBTIMetadata:
    """Verify archetype and communication style mappings are complete."""

    def test_archetype_for_every_type(self) -> None:
        assert set(MBTI_ARCHETYPE.keys()) == set(MBTIType)

    def test_communication_style_for_every_type(self) -> None:
        assert set(MBTI_COMMUNICATION_STYLE.keys()) == set(MBTIType)

    def test_communication_style_property(self) -> None:
        traits = PersonalityTraits(extraversion=0.3, openness=0.9, agreeableness=0.2, conscientiousness=0.8)
        p = Personality(traits=traits)
        assert p.communication_style == MBTI_COMMUNICATION_STYLE[MBTIType.INTJ]


# ---------------------------------------------------------------------------
# MBTI compatibility
# ---------------------------------------------------------------------------

class TestMBTICompatibility:
    """Verify compatibility scoring between types."""

    def test_identical_types(self) -> None:
        assert mbti_compatibility(MBTIType.INTJ, MBTIType.INTJ) == 0.60

    def test_one_letter_different(self) -> None:
        # INTJ vs ENTJ differ only on E/I
        assert mbti_compatibility(MBTIType.INTJ, MBTIType.ENTJ) == 0.85

    def test_two_letters_different(self) -> None:
        # INTJ vs ENTP differ on E/I and J/P
        assert mbti_compatibility(MBTIType.INTJ, MBTIType.ENTP) == 0.70

    def test_all_different(self) -> None:
        # INTJ vs ESFP differ on all 4
        assert mbti_compatibility(MBTIType.INTJ, MBTIType.ESFP) == 0.40

    def test_symmetric(self) -> None:
        assert mbti_compatibility(MBTIType.ENFP, MBTIType.ISTJ) == mbti_compatibility(MBTIType.ISTJ, MBTIType.ENFP)


# ---------------------------------------------------------------------------
# Personality serialization with new fields
# ---------------------------------------------------------------------------

class TestPersonalitySerialisation:
    """Ensure new fields round-trip through to_dict / from_dict."""

    def test_traits_round_trip(self) -> None:
        original = PersonalityTraits(extraversion=0.7, agreeableness=0.3)
        restored = PersonalityTraits.from_dict(original.to_dict())
        assert restored.extraversion == original.extraversion
        assert restored.agreeableness == original.agreeableness

    def test_personality_to_dict_includes_mbti(self) -> None:
        traits = PersonalityTraits(extraversion=0.3, openness=0.9, agreeableness=0.2, conscientiousness=0.8)
        p = Personality(traits=traits)
        d = p.to_dict()
        assert d["mbti_type"] == "INTJ"
        assert d["mbti_archetype"] == "Architect"

    def test_backwards_compat_defaults(self) -> None:
        """Old dicts without extraversion/agreeableness should still load."""
        old_data: dict = {
            "assertiveness": 0.6,
            "warmth": 0.7,
            "openness": 0.5,
            "conscientiousness": 0.5,
            "emotional_stability": 0.5,
            "humor_inclination": 0.5,
            "formality": 0.5,
        }
        traits = PersonalityTraits.from_dict(old_data)
        # Extraversion defaults to assertiveness when missing
        assert traits.extraversion == 0.6
        # Agreeableness defaults to warmth when missing
        assert traits.agreeableness == 0.7


# ---------------------------------------------------------------------------
# Heredity engine
# ---------------------------------------------------------------------------

def _make_parent(
    name: str, *, extraversion: float = 0.5, agreeableness: float = 0.5, openness: float = 0.5
) -> Character:
    """Helper to build a minimal parent character."""
    return Character(
        name=name,
        description=f"Test character {name}",
        personality=Personality(
            traits=PersonalityTraits(extraversion=extraversion, agreeableness=agreeableness, openness=openness),
            values=Values(
                priority_keywords=[f"{name}_value1", f"{name}_value2"],
                beliefs=[f"{name}_belief"],
                strengths=[f"{name}_strength"],
            ),
            speech_patterns=[f"{name} pattern"],
            quirks=[f"{name} quirk"],
        ),
        background=Background(age=1, origin=f"{name}-land", occupation="Tester"),
    )


class TestCrossTraits:
    """Verify personality trait crossover and mutation."""

    def test_child_traits_in_range(self) -> None:
        p1 = PersonalityTraits(assertiveness=0.0, warmth=1.0, extraversion=0.1, agreeableness=0.9)
        p2 = PersonalityTraits(assertiveness=1.0, warmth=0.0, extraversion=0.9, agreeableness=0.1)
        for _ in range(50):  # run many times — stochastic
            child = cross_traits(p1, p2, mutation_rate=0.1)
            for attr in child.to_dict():
                assert 0.0 <= child.to_dict()[attr] <= 1.0

    def test_zero_mutation_inherits_parent(self) -> None:
        """With zero mutation, child should exactly match one parent per trait."""
        p1 = PersonalityTraits(assertiveness=0.2, warmth=0.8)
        p2 = PersonalityTraits(assertiveness=0.9, warmth=0.1)
        child = cross_traits(p1, p2, mutation_rate=0.0)
        for trait in child.to_dict():
            val = child.to_dict()[trait]
            assert val in (p1.to_dict()[trait], p2.to_dict()[trait])


class TestCrossValues:
    """Verify value blending from two parents."""

    def test_child_values_from_parents(self) -> None:
        v1 = Values(priority_keywords=["a", "b"], beliefs=["x"])
        v2 = Values(priority_keywords=["c", "d"], beliefs=["y"])
        child = cross_values(v1, v2)
        # All child keywords should come from parent pools
        assert all(k in {"a", "b", "c", "d"} for k in child.priority_keywords)
        assert all(b in {"x", "y"} for b in child.beliefs)


class TestReproduce:
    """Integration test for the full reproduce() pipeline."""

    def test_child_has_lineage_background(self) -> None:
        p1 = _make_parent("Hera", extraversion=0.8, agreeableness=0.9)
        p2 = _make_parent("Zeus", extraversion=0.6, agreeableness=0.4)
        child = reproduce(p1, p2, "Athena")
        assert "Hera" in child.background.origin
        assert "Zeus" in child.background.origin
        assert child.background.relationships.get("Hera") == "parent"
        assert child.background.relationships.get("Zeus") == "parent"

    def test_child_generation_increments(self) -> None:
        p1 = _make_parent("A")
        p2 = _make_parent("B")
        child = reproduce(p1, p2, "C")
        assert child.background.age == 2  # parents are gen 1, child is gen 2

    def test_child_has_valid_mbti(self) -> None:
        p1 = _make_parent("X", extraversion=0.1, agreeableness=0.9, openness=0.9)
        p2 = _make_parent("Y", extraversion=0.9, agreeableness=0.1, openness=0.1)
        child = reproduce(p1, p2, "Z")
        assert child.personality.mbti_type in MBTIType

    def test_child_description_mentions_parents(self) -> None:
        p1 = _make_parent("Alpha")
        p2 = _make_parent("Beta")
        child = reproduce(p1, p2, "Gamma")
        assert "Alpha" in child.description
        assert "Beta" in child.description

    def test_high_mutation_stays_bounded(self) -> None:
        p1 = _make_parent("Lo", extraversion=0.0)
        p2 = _make_parent("Hi", extraversion=1.0)
        for _ in range(30):
            child = reproduce(p1, p2, "Mid", mutation_rate=0.5)
            for val in child.personality.traits.to_dict().values():
                assert 0.0 <= val <= 1.0


# ---------------------------------------------------------------------------
# Character profile includes MBTI
# ---------------------------------------------------------------------------

class TestCharacterProfileMBTI:
    """Ensure get_character_profile() emits MBTI information."""

    def test_profile_contains_mbti(self) -> None:
        char = _make_parent("TestChar", extraversion=0.3, openness=0.9)
        profile = char.get_character_profile()
        assert "MBTI Profile" in profile
        assert char.personality.mbti_type.value in profile
        assert char.personality.mbti_archetype in profile
        assert "Communication style" in profile
