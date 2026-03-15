# Streamlit User Guide

## Table of Contents

1. [Getting Started](#getting-started)
2. [Onboarding (My Profile)](#onboarding-my-profile)
3. [Workshop — Creating Characters](#workshop--creating-characters)
4. [Cast — Browsing Characters](#cast--browsing-characters)
5. [Scene — Dialogue](#scene--dialogue)
6. [Archive — Past Interactions](#archive--past-interactions)
7. [Dashboard — LLM Metrics](#dashboard--llm-metrics)
8. [Settings](#settings)
9. [Tips & Tricks](#tips--tricks)

---

## Getting Started

### Launching the App

```bash
streamlit run src/character_creator/ui/app.py
```

The app opens at `http://localhost:8501`.

### Navigation

The sidebar contains these pages:

| Page | Purpose |
|------|---------|
| **My Profile** | Onboarding and user profile |
| **Cast** | Browse and manage characters |
| **Scene** | Run multi-character dialogue scenes |
| **Workshop** | Create new characters |
| **Archive** | Review past interaction records |
| **Dashboard** | LLM call metrics and charts |
| **Settings** | Provider, model, and behaviour configuration |

### Session State

- **Active Character** persists across page navigation.
- **Active Scene** stays loaded until you close it.
- **Settings** are maintained during the session.
- **Database** persists across sessions (characters and interactions survive
  browser refresh).

---

## Onboarding (My Profile)

On first launch you see the **My Profile** page. Enter your display name and
optional preferences. This profile is saved to `local/user_profile.json` and
remembered on subsequent visits.

After onboarding, the sidebar navigation becomes available.

---

## Workshop — Creating Characters

Navigate to **Workshop** to create a new character.

### Step 1: Basic Information

- **Character Name** — unique identifier (e.g. "Alice", "Marcus")
- **Physical Description** — appearance and presentation
- **Character Concept** — core essence in a short phrase

### Step 2: Personality Traits

Set each trait on a 0.0–1.0 slider:

| Trait | Low (0.0) | High (1.0) |
|-------|-----------|------------|
| Assertiveness | passive | dominant |
| Warmth | cold | friendly |
| Openness | rigid | creative |
| Conscientiousness | spontaneous | organized |
| Emotional Stability | volatile | calm |
| Humor Inclination | serious | joking |
| Formality | casual | formal |
| Extraversion | introverted | extraverted |
| Agreeableness | combative | cooperative |

### Step 3: MBTI Type

Select from the 16 Myers-Briggs types (e.g. INTJ, ENFP). The archetype label
updates automatically (e.g. INTJ → "Architect"). MBTI influences compatibility
scoring in dialogue.

### Step 4: Values & Beliefs

- **Core Values** — comma-separated keywords
- **Core Beliefs** — statements the character holds true
- **Dislikes** — things they oppose
- **Strengths / Weaknesses** — capabilities and limitations

### Step 5: Speech & Behaviour

- **Speech Patterns** — how they typically speak
- **Character Quirks** — distinctive behaviours

### Step 6: Background

- **Age, Origin, Occupation**
- **Motivations, Fears, Desires**
- **Formative Events, Relationships, Memories**

### Step 7: Preview & Create

Click **Preview Character** to review the full profile, then **Create
Character** to save. The character is immediately available in Cast and Scene.

---

## Cast — Browsing Characters

Navigate to **Cast** to see all characters (including the 5 defaults: Alice,
Elena, Kai, Marcus, Zoey).

### Character Detail Tabs

Select a character from the dropdown to see:

**Profile** — full character profile text.

**Personality** — 9 trait values, core values and beliefs, speech patterns and
quirks.

**Background** — age, origin, occupation, motivations, fears, desires,
relationships, and memories.

**Self-Perception** — how the character currently sees themselves. This view
updates dynamically based on emotional state, recent events, and the evolution
pipeline (self-model, dissonances, trait shifts).

### Actions

- **Load for Dialogue** — sets this character as active for the Scene page.
- **Delete** — permanently removes the character from the database.

---

## Scene — Dialogue

### Creating a Scene

1. **Select Characters** — pick 2 or more from the multi-select.
2. **Scene Description** — environment context (e.g. "A rooftop bar at
   sunset").
3. **Conversation Topic** — what they are discussing.
4. Click **Create Scene**.

### Running Dialogue

**Director Mode (continuous)** — the system generates multiple exchanges
automatically with personality-weighted speaker selection.

**Single Exchange** — generates one dialogue turn at a time.

Each exchange shows:
- Speaker name
- Dialogue text
- Emotional context badge
- Expandable internal monologue

### Behind the Scenes

After every exchange the system runs **post-exchange hooks**:
- Memory condensation (if working memory is full)
- Experience classification and trait micro-shifts
- Self-reflection (every 5 exchanges)
- Behaviour extraction and dissonance detection (every 3 exchanges)

At scene end, **milestone reviews** evaluate the full scene and may apply
larger trait shifts. Evolved characters are persisted back to the database.

### Dialogue Tabs

**Dialogue Generation** — the conversation itself.

**Character Dynamics** — personality trait comparison and predicted speaker
order with weighted scores.

**Scene Simulation** — character motivations, desires, and current emotional
states.

### Finishing a Scene

Click **Close Scene** to trigger milestone reviews and save the interaction
record to the archive.

---

## Archive — Past Interactions

Navigate to **Archive** to browse saved interaction records. Each record
includes:
- Scene description and topic
- Participating characters
- Full exchange history
- Timestamps

---

## Dashboard — LLM Metrics

Navigate to **Dashboard** to see aggregated LLM usage metrics:

- **KPIs** — total calls, total estimated tokens, average latency
- **Call-Type Breakdown** — pie chart of calls by type (dialogue,
  internal_monologue, emotion, self_reflection, experience_classification,
  trait_shift, condensation, milestone_review, dissonance_detection, etc.)
- **Subsystem Charts** — bar charts of calls and tokens per subsystem
- **CSV Export** — download the raw metrics data

Metrics are persisted to `local/metrics.jsonl` and survive app restarts.

---

## Settings

### LLM Configuration

- **Provider** — openai, anthropic, or google
- **Model** — e.g. `gpt-4-turbo-preview`, `claude-3-sonnet`, `gemini-1.5-pro`
- **Creativity (Temperature)** — 0.0 (deterministic) to 2.0 (highly creative)

### Character Behaviour

- **Conversation Memory Tokens** — how much history characters retain
- **Emotional Sensitivity** — how dramatically emotions swing
- **Turn-Taking Politeness** — balance between dominant and equal turns

---

## Tips & Tricks

### Creating Compelling Characters

- **Be specific with values** — "intellectual honesty" is stronger than
  "honesty".
- **Balance trait extremes** — a high-assertiveness + low-warmth character
  creates natural tension.
- **Use memorable quirks** — they make dialogue distinctive.
- **Leverage MBTI** — pair compatible or clashing types for interesting
  dynamics.

### Setting Up Great Scenes

- **Specific topics** produce better dialogue than vague ones.
  - "Whether to leave their job" beats "casual conversation".
- **Environmental context matters** — formal settings shift speech style.
- **Mixed personalities** create the most realistic exchanges.

### Observing Character Evolution

- Watch the **Self-Perception** tab after several exchanges — the self-model
  updates as the character reflects on their behaviour.
- Check **trait values** before and after a scene to see micro-shifts.
- Look for **cognitive dissonances** — contradictions between stated values and
  observed behaviour that the system detects automatically.

### Working Without LLM Keys

Without API keys you can still:
- Create and configure characters
- Explore the 5 defaults
- Set up dialogue scenes
- Analyse personality dynamics and speaker predictions
- Browse the archive

Dialogue generation and evolution hooks require a configured LLM provider.

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| Can't create character | Ensure name and description are filled in |
| Character doesn't appear | Check the Cast page; refresh if needed |
| Dialogue button does nothing | Verify LLM provider and API key in Settings |
| Speaker predictions seem random | Differentiate personality traits between characters |

---

## Next Steps

- [DEVELOPER_GUIDE.md](DEVELOPER_GUIDE.md) — extend the system
- [API_DOCUMENTATION.md](API_DOCUMENTATION.md) — REST API reference
- [ARCHITECTURE.md](ARCHITECTURE.md) — system design
