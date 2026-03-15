"""Memory and background system for characters."""

from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel, Field


class Memory(BaseModel):
    """A single memory or experience in a character's backstory."""

    title: str
    description: str
    impact: str  # How this shaped the character
    emotional_weight: float  # 0-1.0, importance to character identity
    timestamp: datetime = Field(default_factory=lambda: datetime.now(tz=UTC))
    related_topics: list[str] = Field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert memory to dictionary format."""
        d = self.model_dump()
        d["timestamp"] = self.timestamp.isoformat()
        return d


class Background(BaseModel):
    """Character backstory and life history.

    Contains formative experiences, relationships, and events that shape
    the character's identity, motivations, and reactions.
    """

    age: int
    origin: str  # Where they're from/their cultural background
    occupation: str
    relationships: dict[str, str] = Field(default_factory=dict)  # Name -> description
    memories: list[Memory] = Field(default_factory=list)
    motivations: list[str] = Field(default_factory=list)
    fears: list[str] = Field(default_factory=list)
    desires: list[str] = Field(default_factory=list)

    def add_memory(
        self,
        title: str,
        description: str,
        impact: str,
        emotional_weight: float = 0.5,
        related_topics: list[str] | None = None,
    ) -> None:
        """Add a memory to the character's background."""
        memory = Memory(
            title=title,
            description=description,
            impact=impact,
            emotional_weight=emotional_weight,
            related_topics=related_topics or [],
        )
        self.memories.append(memory)

    def add_relationship(self, name: str, description: str) -> None:
        """Add or update a relationship."""
        self.relationships[name] = description

    def get_context_summary(self) -> str:
        """Generate a summary of the background for LLM context."""
        parts: list[str] = [
            f"Age: {self.age}",
            f"Origin: {self.origin}",
            f"Occupation: {self.occupation}",
        ]

        if self.motivations:
            parts.append(f"Motivated by: {', '.join(self.motivations)}")

        if self.fears:
            parts.append(f"Fears: {', '.join(self.fears)}")

        if self.desires:
            parts.append(f"Desires: {', '.join(self.desires)}")

        if self.relationships:
            rel_str = "; ".join([f"{name}: {desc}" for name, desc in self.relationships.items()])
            parts.append(f"Relationships: {rel_str}")

        if self.memories:
            # Include top 3 most emotionally impactful memories
            top_memories = sorted(
                self.memories, key=lambda m: m.emotional_weight, reverse=True
            )[:3]
            memory_str = "; ".join([f"{m.title}: {m.impact}" for m in top_memories])
            parts.append(f"Key experiences: {memory_str}")

        return "\n".join(parts)

    def to_dict(self) -> dict[str, Any]:
        """Convert background to dictionary format."""
        d = self.model_dump()
        d["memories"] = [m.to_dict() for m in self.memories]
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Background":
        """Create background from dictionary."""
        return cls.model_validate(data)
