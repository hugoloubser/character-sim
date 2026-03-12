"""Validation utilities for character data and configurations."""

import re


def validate_character_name(name: str) -> bool:
    """Validate a character name.

    Args:
        name: Character name to validate.

    Returns:
        True if valid, False otherwise.

    """
    if not name or len(name) < 1 or len(name) > 100:
        return False
    # Allow alphanumeric, spaces, and some special characters
    return bool(re.match(r"^[a-zA-Z0-9\s\-']+$", name))


def validate_personality_trait(value: float) -> bool:
    """Validate a personality trait value.

    Args:
        value: Trait value to validate (should be 0-1).

    Returns:
        True if valid, False otherwise.

    """
    return isinstance(value, (int, float)) and 0.0 <= value <= 1.0


def validate_age(age: int) -> bool:
    """Validate a character age.

    Args:
        age: Age to validate.

    Returns:
        True if valid, False otherwise.

    """
    return isinstance(age, int) and 1 <= age <= 150


def validate_emotional_weight(weight: float) -> bool:
    """Validate emotional weight of a memory.

    Args:
        weight: Weight to validate (should be 0-1).

    Returns:
        True if valid, False otherwise.

    """
    return isinstance(weight, (int, float)) and 0.0 <= weight <= 1.0


def sanitize_text(text: str, max_length: int = 1000) -> str:
    """Sanitize user input text.

    Args:
        text: Text to sanitize.
        max_length: Maximum allowed length.

    Returns:
        Sanitized text.

    """
    return text.strip()[:max_length]


def parse_csv_list(csv_string: str) -> list[str]:
    """Parse a comma-separated list into individual items.

    Args:
        csv_string: Comma-separated string.

    Returns:
        List of individual items, stripped and non-empty.

    """
    if not csv_string:
        return []
    return [item.strip() for item in csv_string.split(",") if item.strip()]
