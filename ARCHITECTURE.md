# Character Creator — Architecture & Design Guide

## System Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                      User Interfaces                         │
├────────────────────────────┬─────────────────────────────────┤
│  Streamlit UI              │      FastAPI REST API           │
│  (app.py + dashboard.py)   │      (Swagger at /docs)        │
└────────────┬───────────────┴──────────────┬──────────────────┘
             │                              │
             └──────────────┬───────────────┘
                            │
┌───────────────────────────▼──────────────────────────────────┐
│                    Core Layer                                 │
├──────────────────────────────────────────────────────────────┤
│  Character          │  Dialogue System                       │
│  ├─ Personality     │  ├─ Speaker selection                  │
│  ├─ Background      │  ├─ Response generation                │
│  ├─ MemoryStore     │  ├─ Internal monologue                 │
│  ├─ SelfModel       │  └─ Emotional context inference        │
│  └─ Dissonances     │                                        │
│                     │  Evolution Pipeline                     │
│                     │  ├─ ExperienceClassifier                │
│                     │  ├─ TraitShiftEngine                    │
│                     │  ├─ SelfReflectionEngine                │
│                     │  ├─ DissonanceDetector                  │
│                     │  ├─ MilestoneReviewEngine               │
│                     │  └─ MemoryCondenser                     │
├──────────────────────────────────────────────────────────────┤
│  Persistence                                                 │
│  ├─ CharacterRepository (SQLite / in-memory)                 │
│  ├─ InteractionRepository (SQLite)                           │
│  └─ MetricsCollector (JSONL)                                 │
└───────────────────────────┬──────────────────────────────────┘
                            │
┌───────────────────────────▼──────────────────────────────────┐
│                    LLM Layer                                  │
├──────────────────────────────────────────────────────────────┤
│  InstrumentedProvider (transparent metrics proxy)             │
│  ├─ OpenAIProvider    (async, official SDK)                   │
│  ├─ AnthropicProvider (async, official SDK)                   │
│  └─ GoogleProvider    (async, OpenAI-compatible endpoint)     │
│                                                               │
│  Prompt Templates (llm/prompts.py)                            │
│  Metrics & Instrumentation (llm/metrics.py)                   │
└──────────────────────────────────────────────────────────────┘
```

## Data Flow

### Dialogue Exchange (per turn)

```
Speaker Selection (personality-weighted scoring)
    ↓
Dialogue Generation  →  LLM call (call_type: dialogue)
    ↓
Internal Monologue   →  LLM call (call_type: internal_monologue)
    ↓
Emotion Inference    →  LLM call (call_type: emotion)
    ↓
Context Updated (DialogueContext + InteractionRecord)
    ↓
Post-Exchange Hooks
    ├─ Memory Condensation    (if working memory full)
    ├─ Experience Classification + Trait Shift (every exchange)
    ├─ Self-Reflection        (every 5 exchanges)
    └─ Dissonance Detection   (every 3 exchanges)
```

### Scene Lifecycle

```
_start_scene() → trait snapshots + DialogueSystem init
    ↓
Loop: _run_one_step() or _director_run_continuous()
    ├─ _generate_exchange()
    ├─ _add_exchange_to_state()
    └─ _run_post_exchange_hooks()
    ↓
_finish_scene()
    ├─ Milestone Reviews (LLM-adjudicated trait shifts)
    ├─ Character persistence (evolved traits saved to DB)
    ├─ InteractionRecord persisted to SQLite
    └─ Event loop cleanup
```

## Character Data Model

```
Character
├── name: str
├── description: str
├── personality: Personality
│   ├── traits: PersonalityTraits
│   │   ├── assertiveness       (0.0 – 1.0)
│   │   ├── warmth
│   │   ├── openness
│   │   ├── conscientiousness
│   │   ├── emotional_stability
│   │   ├── humor_inclination
│   │   ├── formality
│   │   ├── extraversion
│   │   └── agreeableness
│   ├── values: Values
│   │   ├── priority_keywords
│   │   ├── beliefs
│   │   ├── dislikes
│   │   ├── strengths
│   │   └── weaknesses
│   ├── speech_patterns: list[str]
│   ├── quirks: list[str]
│   ├── mbti_type: MBTIType (16 types)
│   └── mbti_archetype: str
├── background: Background
│   ├── age, origin, occupation
│   ├── relationships, memories
│   ├── motivations, fears, desires
│   └── formative_events
├── memory_store: MemoryStore
│   ├── working: list[CondensedMemory]    (cap: 10)
│   ├── short_term: list[CondensedMemory] (cap: 20)
│   ├── long_term: list[CondensedMemory]  (cap: 50)
│   └── ground_truth_log: list
├── trait_history: TraitHistory
│   ├── deltas: list[TraitDelta]
│   └── snapshots: list
├── self_model: SelfModel
│   ├── self_concept, emotional_awareness
│   ├── value_tensions, growth_edges
│   └── generated_at_exchange
├── self_model_history: list[SelfModel]
├── active_dissonances: list[CognitiveDissonance]
├── current_emotional_state: EmotionalState
├── internal_monologue: list[str]
└── conversation_history: list[dict]
```

## Evolution Subsystem Intervals

| Subsystem | Trigger | Threshold |
|-----------|---------|-----------|
| Memory Condensation | Working memory full | 10 exchanges |
| Memory Promotion | Short-term full | 20 condensed memories |
| Experience Classification | Every exchange | — |
| Trait Micro-Shifts | Every exchange (via influence vectors) | — |
| Self-Reflection | Periodic | Every 5 exchanges |
| Behaviour Extraction | Periodic | Every 3 exchanges |
| Dissonance Detection | Periodic | Every 3 exchanges |
| Milestone Review | End of scene | — |
| Memory Decay | Per scene | 0.02 salience/scene |

## Module Responsibilities

| Module | Responsibility |
|--------|---------------|
| `core/character.py` | Character entity, profile generation, emotional state |
| `core/constants.py` | All named constants (single source of truth) |
| `core/database.py` | `CharacterRepository` interface + SQLite/in-memory backends |
| `core/dialogue.py` | `DialogueSystem` — speaker selection, LLM generation, scene orchestration |
| `core/heredity.py` | `cross_traits()`, `cross_values()`, `reproduce()` |
| `core/interaction.py` | `InteractionRecord`, `SQLiteInteractionRepository` |
| `core/memory.py` | `Background`, `Memory`, `Values` Pydantic models |
| `core/memory_tiered.py` | `MemoryStore`, `MemoryCondenser`, salience decay |
| `core/personality.py` | `PersonalityTraits`, `Personality`, `MBTIType`, compatibility |
| `core/self_model.py` | `SelfReflectionEngine`, `DissonanceDetector`, `SelfModel` |
| `core/trait_evolution.py` | `ExperienceClassifier`, `TraitShiftEngine`, `MilestoneReviewEngine` |
| `llm/metrics.py` | `MetricsCollector`, `InstrumentedProvider`, `llm_context()` |
| `llm/prompts.py` | Prompt templates with `substitute_prompt()` |
| `llm/providers.py` | `LLMProvider` ABC + OpenAI, Anthropic, Google implementations |
| `ui/app.py` | Streamlit application (onboarding → workshop → scenes → archive → settings) |
| `ui/dashboard.py` | Metrics dashboard with charts, KPIs, and CSV export |
| `utils/config.py` | `Settings` via pydantic-settings, path management |
| `utils/logging.py` | `structlog` console renderer setup |
| `utils/validators.py` | Input validation helpers |

## Design Principles

### Separation of Concerns
Each module has a single responsibility. The dialogue engine doesn't know about
persistence; the LLM layer doesn't know about characters.

### Pydantic Everywhere
All domain objects are `BaseModel` subclasses with `Field` validation. No raw
dicts or plain dataclasses for domain models.

### No Magic Literals
Every numeric constant, threshold, and default lives in `core/constants.py`.
Modules import by name — never inline the value.

### Provider Pattern
LLM operations are abstracted behind `LLMProvider`. Adding a new provider
requires implementing `generate()` and `generate_with_format()`.

### Repository Pattern
Database operations are abstracted by `CharacterRepository`. Swap between
`SQLiteRepository` and `InMemoryRepository` with one line.

### Composition Over Inheritance
`Character` is composed of `Personality` + `Background` + `MemoryStore` +
`SelfModel`. Evolution engines are injected into `DialogueSystem`, not
inherited.

### Async I/O
All LLM calls use `async`/`await` for non-blocking execution. The Streamlit UI
bridges to async via a dedicated event loop per scene.

### Transparent Instrumentation
`InstrumentedProvider` wraps any `LLMProvider` and records every call without
modifying provider behaviour. The `llm_context()` context manager tags calls by
subsystem (`dialogue`, `emotion`, `self_reflection`, etc.).

## Persistence Layout

```
local/                          # git-ignored
├── .env                        # API keys and config
├── character_creator.db        # SQLite character database
├── interactions.db             # SQLite interaction archive
├── metrics.jsonl               # LLM call metrics (append-only)
├── user_profile.json           # Streamlit user profile
└── logs/                       # Structured log files
```

## Security Considerations

- All user inputs validated via Pydantic with type checking and range
  constraints.
- Prompts constructed from templates — user input is interpolated into
  validated placeholders, not concatenated.
- API keys stored in environment variables or `local/.env` (git-ignored).
- CORS configured in FastAPI middleware.
- Bandit security scanning in pre-commit hooks.

## Testing Strategy

- **Unit tests** — isolated component testing with mocked LLM calls
- **Integration tests** — end-to-end scene flow with fake providers
- **LLM integration tests** — skipped by default (require live API keys)
- **Coverage** — `pytest --cov=character_creator --cov-report=html`
