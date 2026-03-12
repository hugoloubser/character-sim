# Developer Guide

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Core Concepts](#core-concepts)
3. [Code Quality Standards](#code-quality-standards)
4. [Adding New Features](#adding-new-features)
5. [Extending the LLM System](#extending-the-llm-system)
6. [Extending the Evolution Pipeline](#extending-the-evolution-pipeline)
7. [Testing](#testing)
8. [Code Standards](#code-standards)

---

## Architecture Overview

See [ARCHITECTURE.md](ARCHITECTURE.md) for the full system diagram, data flow,
and module responsibility table. The summary below provides the developer
perspective.

### System Layers

```
UI Layer (Streamlit)  →  API Layer (FastAPI)  →  Core Layer
    ↓                                               ↓
Dashboard (metrics)                           Evolution Pipeline
                                                    ↓
                                              LLM Layer (providers)
                                                    ↓
                                              Data Layer (SQLite / JSONL)
```

### Key Design Patterns

**Repository Pattern** — `CharacterRepository` abstracts storage:
```python
class CharacterRepository:
    def create(character: Character) -> None
    def read(name: str) -> Character
    def update(character: Character) -> None
    def delete(name: str) -> None
    def list_all() -> list[Character]
    def exists(name: str) -> bool
```

**Provider Pattern** — `LLMProvider` abstracts LLM calls:
```python
class LLMProvider:
    async def generate(prompt: str, temperature: float) -> str
    async def generate_with_format(prompt: str, response_format: type[T]) -> T
```

Implemented by `OpenAIProvider`, `AnthropicProvider`, `GoogleProvider`.

**Pydantic Everywhere** — all domain objects are `BaseModel` subclasses:
```python
class Character(BaseModel):
    name: str
    description: str
    personality: Personality
    background: Background
    memory_store: MemoryStore
    trait_history: TraitHistory
    self_model: SelfModel
    active_dissonances: list[CognitiveDissonance]
    ...
```

---

## Core Concepts

### Character

A `Character` is the central entity combining:
- **Identity** — name, description
- **Personality** — 9-axis traits, values, speech patterns, quirks, MBTI type
- **Background** — age, origin, occupation, memories, relationships, fears, desires
- **Memory** — tiered memory store (working → short-term → long-term)
- **Evolution** — trait history, self-model, cognitive dissonances
- **State** — emotional state, conversation history, internal monologue

Key methods:
```python
character.get_character_profile()          # Full profile for LLM context
character.get_character_self_perception()  # Dynamic self-view
character.add_memory_to_history()          # Track conversation
character.update_emotional_state()         # Update current emotion
```

### PersonalityTraits (9 axes)

| Trait | Range | Pole (0.0) | Pole (1.0) |
|-------|-------|------------|------------|
| assertiveness | 0.0–1.0 | passive | dominant |
| warmth | 0.0–1.0 | cold | friendly |
| openness | 0.0–1.0 | rigid | creative |
| conscientiousness | 0.0–1.0 | spontaneous | organized |
| emotional_stability | 0.0–1.0 | volatile | calm |
| humor_inclination | 0.0–1.0 | serious | joking |
| formality | 0.0–1.0 | casual | formal |
| extraversion | 0.0–1.0 | introverted | extraverted |
| agreeableness | 0.0–1.0 | combative | cooperative |

### MBTI Integration

`MBTIType` is a 16-member `StrEnum`. Each type carries an archetype label
(e.g., INTJ → "Architect"). `mbti_compatibility()` returns a `float` score
used in speaker selection.

### Tiered Memory

```
Working Memory (cap 10)
    ↓ condense (LLM)
Short-Term Memory (cap 20)
    ↓ promote (salience-based)
Long-Term Memory (cap 50)
```

Memories decay by `DEFAULT_DECAY_RATE` per scene. Anchored memories
(emotional or value-aligned) decay more slowly. Below
`MIN_SALIENCE_THRESHOLD`, memories are dropped.

### Evolution Pipeline

After every dialogue exchange the system runs:

1. **Experience Classification** — LLM classifies the exchange (conflict,
   vulnerability, success, etc.) → `ExperienceType`
2. **Trait Micro-Shift** — influence vectors nudge personality traits by up to
   `MAX_TRAIT_DELTA_PER_EXCHANGE` per axis
3. **Self-Reflection** (every 5 exchanges) — LLM generates a `SelfModel`
4. **Behaviour Extraction + Dissonance Detection** (every 3 exchanges) — LLM
   identifies behavioural themes and contradictions between stated values and
   observed behaviour

At scene end:
5. **Milestone Review** — LLM evaluates the full scene and proposes larger
   trait shifts with confidence scores

### DialogueSystem

Orchestrates a multi-character scene:
```python
dialogue_system = DialogueSystem(llm_provider)
speaker = dialogue_system.generate_next_speaker(context)
response, thought = await dialogue_system.generate_response(context, character)
```

### Metrics & Instrumentation

`InstrumentedProvider` wraps any `LLMProvider` transparently. Every LLM call is
recorded as an `LLMCallRecord` with call type, token estimate, latency, and
subsystem tag. Use `llm_context("subsystem_name")` to tag calls.

`MetricsCollector` aggregates records in memory and persists to JSONL.

---

## Code Quality Standards

### No Magic Literals

Every constant lives in `core/constants.py`. Import by name — never inline:

```python
# Wrong
score += extraversion * 0.35

# Right
from character_creator.core.constants import SPEAKER_EXTRAVERSION_WEIGHT
score += extraversion * SPEAKER_EXTRAVERSION_WEIGHT
```

When adding a constant:
1. Define it in `core/constants.py` with a descriptive `ALL_CAPS` name.
2. Import it everywhere it's used.
3. Add a brief inline comment if the meaning isn't obvious.

### Data Models — Pydantic Only

All domain objects must be `BaseModel` subclasses. Use `Field(...)` with `ge`,
`le`, `min_length`, etc. Reference constants for default values and bounds.
Prefer `model_dump()` over hand-written serialization.

### DRY

- Extract shared logic into private helpers.
- Derive lists from model metadata (`model_fields.keys()`) instead of
  maintaining parallel manual lists.
- Use `settings.api_key_for(provider)` for key lookup.

### Linting — Ruff

All rules live in `ruff.toml` at the project root. Do **not** add
`[tool.ruff]` sections to `pyproject.toml`.

Key enforced rule sets: `F`, `E/W`, `I`, `N`, `PL`, `UP`, `B`, `S`, `D`,
`ANN`, `RUF`, `ERA`, `T20`, `PT`.

```bash
python -m ruff check src/ tests/          # lint
python -m ruff check src/ tests/ --fix    # auto-fix
python -m ruff format src/ tests/         # format
```

### Pre-commit Hooks

Install once:
```bash
pip install pre-commit
pre-commit install --install-hooks
```

Hooks: trailing whitespace, EOF fixer, YAML/TOML/JSON checks, merge-conflict
detection, large-file guard, AST check, no-commit-to-branch (main/master),
Ruff lint+format, mypy with Pydantic plugin, Bandit security scan, Commitizen.

Run manually:
```bash
pre-commit run --all-files
```

### Commit Messages

Follow [Conventional Commits](https://www.conventionalcommits.org/), enforced
by Commitizen:
```
feat: add enneagram personality model
fix: clamp trait values on deserialization
refactor: extract speaker-weight constants
docs: update developer guide
test: add heredity crossover edge-case tests
```

---

## Adding New Features

### Add a New Personality Trait

1. Add the field to `PersonalityTraits` in `core/personality.py`:
   ```python
   ambition: float = Field(TRAIT_DEFAULT, ge=TRAIT_MIN, le=TRAIT_MAX)
   ```
2. Add an API request field in `api/models.py`.
3. Add a slider in the UI workshop page (`ui/app.py`).
4. If it influences speaker selection, add a weight constant to
   `core/constants.py` and update `_score_speaker()` in `core/dialogue.py`.
5. Add a test in `tests/test_character.py`.

### Add a New API Endpoint

Add the route to the appropriate router in `api/routes/`:
```python
@router.get("/characters/compare")
async def compare_characters(names: list[str] = Query(...)) -> dict:
    ...
```
Add a Pydantic response model in `api/models.py` and a test.

### Add a New UI Page

1. Add a page function in `ui/app.py`.
2. Add the page name to the sidebar radio list.
3. Wire it in the page-dispatch `if/elif` chain.

---

## Extending the LLM System

### Adding a New Provider

1. Subclass `LLMProvider` in `llm/providers.py`:
   ```python
   class NewProvider(LLMProvider):
       async def generate(self, prompt: str, temperature: float = 0.7) -> str:
           ...
       async def generate_with_format(
           self, prompt: str, response_format: type[T], temperature: float = 0.7
       ) -> T:
           ...
   ```
2. Add a `ProviderType` enum member.
3. Handle the new type in `ProviderFactory.create_provider()`.
4. Add the API key field to `Settings` in `utils/config.py`.
5. Add a test in `tests/test_llm_providers.py`.

### Adding a New Prompt Template

Add the template string to `llm/prompts.py` and use `substitute_prompt()` for
variable interpolation. Tag the call with an appropriate `llm_context()` label.

---

## Extending the Evolution Pipeline

### Adding a New Evolution Subsystem

1. Define any new Pydantic models in the appropriate `core/` module.
2. Create an engine class with an `async` method that takes the current
   `Character`, `DialogueContext`, and `LLMProvider`.
3. Wire it into the post-exchange hook chain in `ui/app.py`
   (`_run_post_exchange_hooks`) or scene-end hook (`_finish_scene`).
4. If it uses an LLM call, wrap it in `llm_context("your_subsystem")`.
5. Add a frequency constant to `core/constants.py` if it doesn't run every
   exchange.
6. Add tests using a `FakeLLMProvider`.

### Subsystem Frequencies (constants)

| Constant | Default | Controls |
|----------|---------|----------|
| `WORKING_MEMORY_CAPACITY` | 10 | Memory condensation trigger |
| `SHORT_TERM_CAPACITY` | 20 | Memory promotion trigger |
| `SELF_REFLECTION_INTERVAL` | 5 | Exchanges between reflections |
| `BEHAVIOUR_EXTRACTION_INTERVAL` | 3 | Exchanges between dissonance checks |
| `MAX_TRAIT_DELTA_PER_EXCHANGE` | 0.02 | Trait shift speed limit |
| `MAX_MILESTONE_SHIFT` | 0.05 | Scene-end shift limit |

---

## Testing

### Test Patterns

**Unit tests** — isolated, mocked LLM:
```python
def test_personality_trait_validation():
    traits = PersonalityTraits(assertiveness=1.5)
    assert traits.assertiveness <= TRAIT_MAX
```

**Integration tests** — cross-component with `FakeLLMProvider`:
```python
@pytest.mark.asyncio
async def test_full_evolution_cycle():
    provider = FakeLLMProvider()
    ...
```

**LLM integration tests** — live API, skipped by default:
```python
@pytest.mark.skipif(not os.getenv("OPENAI_API_KEY"), reason="No key")
@pytest.mark.asyncio
async def test_openai_generation():
    ...
```

### Running Tests

```bash
python -m pytest tests/ -v                 # all tests
python -m pytest tests/test_character.py   # single file
python -m pytest tests/ -k "test_trait"    # name filter
python -m pytest tests/ --cov=character_creator --cov-report=html  # coverage
```

---

## Code Standards

### Type Hints

All functions must have complete type annotations:
```python
def create_character(
    name: str,
    description: str,
    personality: Personality,
) -> Character: ...
```

### Docstrings

Google-style, enforced by pydocstyle `D` rules:
```python
def generate_dialogue(
    context: DialogueContext,
    character: Character,
) -> str:
    """Generate dialogue for a character in a scene.

    Args:
        context: The dialogue scene context.
        character: Character generating dialogue.

    Returns:
        Generated dialogue text.

    Raises:
        ValueError: If character not in scene.
    """
```

### Error Messages

```python
# Explicit, descriptive
msg = f"Character '{name}' not found in database"
raise ValueError(msg)
```

---

## Common Extension Points

| Extension | Where |
|-----------|-------|
| New personality trait | `core/personality.py` + `api/models.py` + UI |
| New LLM provider | `llm/providers.py` + `utils/config.py` |
| New database backend | `core/database.py` (implement `CharacterRepository`) |
| New evolution subsystem | `core/` module + wire into `ui/app.py` hooks |
| New dialogue feature | `core/dialogue.py` (`DialogueSystem`) |
| New API endpoint | `api/routes/` router |
| New UI page | `ui/app.py` + sidebar routing |
| New constant | `core/constants.py` |
| New metric tag | `llm_context("tag")` in calling code |

---

## Resources

- [FastAPI](https://fastapi.tiangolo.com/)
- [Streamlit](https://docs.streamlit.io/)
- [Pydantic v2](https://docs.pydantic.dev/)
- [structlog](https://www.structlog.org/)
- [pytest](https://docs.pytest.org/)
- [Ruff](https://docs.astral.sh/ruff/)
