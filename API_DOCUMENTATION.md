# API Documentation

## Overview

The Character Creator REST API is built with FastAPI. Interactive
documentation is available at `/docs` (Swagger UI) and `/redoc` (ReDoc) when
the server is running.

## Base URL

```
http://localhost:8000
```

## Starting the Server

```bash
uvicorn character_creator.api.main:create_app --reload
```

## Common Response Codes

| Code | Meaning |
|------|---------|
| 200 | Success |
| 201 | Resource created |
| 204 | Deleted (no content) |
| 400 | Invalid request |
| 404 | Resource not found |
| 409 | Resource already exists |
| 500 | Server error |
| 503 | LLM provider unavailable |

---

## Health Check

### GET /health

```bash
curl http://localhost:8000/health
```

**Response** (200):
```json
{
  "status": "healthy",
  "version": "0.1.0"
}
```

---

## Characters

### POST /characters/

Create a new character.

**Request Body**:
```json
{
  "name": "Sarah",
  "description": "A 28-year-old marketing executive",
  "personality": {
    "traits": {
      "assertiveness": 0.8,
      "warmth": 0.7,
      "openness": 0.9,
      "conscientiousness": 0.8,
      "emotional_stability": 0.6,
      "humor_inclination": 0.7,
      "formality": 0.6,
      "extraversion": 0.75,
      "agreeableness": 0.65
    },
    "values": {
      "priority_keywords": ["ambition", "innovation"],
      "beliefs": ["Great teams achieve great things"],
      "dislikes": ["mediocrity"],
      "strengths": ["strategic thinking", "communication"],
      "weaknesses": ["perfectionism", "impatience"]
    },
    "speech_patterns": ["uses metaphors", "speaks quickly when excited"],
    "quirks": ["takes notes constantly"]
  },
  "background": {
    "age": 28,
    "origin": "Boston, Massachusetts",
    "occupation": "Marketing Executive",
    "motivations": ["lead a successful company"],
    "fears": ["failure"],
    "desires": ["build something lasting"]
  }
}
```

**Response** (201): Full character object.

**Errors**: `400` invalid data, `409` name already exists.

---

### GET /characters/

List all characters.

```bash
curl http://localhost:8000/characters/
```

**Response** (200): Array of character objects.

---

### GET /characters/{character_name}

Get a character by name.

```bash
curl http://localhost:8000/characters/Alice
```

**Response** (200): Character object. **Error**: `404` not found.

---

### PUT /characters/{character_name}

Update an existing character. Request body has the same shape as POST (all
fields optional for partial update).

```bash
curl -X PUT http://localhost:8000/characters/Alice \
  -H "Content-Type: application/json" \
  -d '{"description": "Updated description"}'
```

**Response** (200): Updated character object. **Error**: `404` not found.

---

### DELETE /characters/{character_name}

Delete a character.

```bash
curl -X DELETE http://localhost:8000/characters/Alice
```

**Response** (204): No content. **Error**: `404` not found.

---

### POST /characters/generate

Generate a character via LLM from a concept prompt.

**Request Body**:
```json
{
  "concept": "A cynical coffee-shop owner learning to trust again"
}
```

**Response** (201): Full character object created by the LLM.

---

## Interactions

### POST /interactions/scenes

Create a dialogue scene.

**Request Body**:
```json
{
  "character_names": ["Alice", "Marcus"],
  "scene_description": "Coffee shop at afternoon",
  "topic": "Whether to start a startup"
}
```

**Response** (201): Scene object with `scene_id`.

---

### GET /interactions/scenes/{scene_id}

Retrieve a scene and its exchanges.

**Response** (200): Scene object including exchange history.

---

### POST /interactions/generate-response

Generate a dialogue response for a character within a scene.

**Request Body**:
```json
{
  "scene_id": "scene_1",
  "character_names": ["Alice", "Marcus"],
  "character_responding": "Alice",
  "scene_context": "Coffee shop discussing startups",
  "topic": "Should we take investment?",
  "recent_dialogue": [
    {"speaker": "Marcus", "text": "I think we should wait..."}
  ]
}
```

**Response** (200):
```json
{
  "response": "I agree the timing is crucial, but we need to be smart about investor selection.",
  "emotional_context": "determined",
  "internal_thought": "Marcus makes a good point, but speed matters here."
}
```

---

### POST /interactions/scenes/{scene_id}/add-exchange

Manually add an exchange to a scene.

**Request Body**:
```json
{
  "speaker": "Alice",
  "text": "Let me think about that.",
  "emotional_context": "contemplative"
}
```

**Response** (200): Updated scene object.

---

### GET /interactions/history

List all saved interaction records.

**Response** (200): Array of interaction summaries.

---

### GET /interactions/history/{interaction_id}

Get a single interaction record.

**Response** (200): Full interaction record with all exchanges.

---

## Evolution

### POST /evolution/reproduce

Breed two existing characters to create offspring via genetic crossover.

**Request Body**:
```json
{
  "parent1_name": "Alice",
  "parent2_name": "Marcus",
  "child_name": "Alex",
  "mutation_rate": 0.1
}
```

**Response** (201): Full child character object.

**Errors**: `404` parent not found, `409` child name already exists.

---

### GET /evolution/mbti/{character_name}

Get the MBTI personality profile for a character.

**Response** (200):
```json
{
  "name": "Alice",
  "mbti_type": "ENTJ",
  "archetype": "Commander",
  "communication_style": "Direct and strategic",
  "extraversion": 0.75,
  "agreeableness": 0.65,
  "openness": 0.9,
  "conscientiousness": 0.8,
  "emotional_stability": 0.6
}
```

---

### POST /evolution/compatibility

Compute MBTI compatibility between two characters.

**Request Body**:
```json
{
  "character1_name": "Alice",
  "character2_name": "Marcus"
}
```

**Response** (200):
```json
{
  "character1_name": "Alice",
  "character1_mbti": "ENTJ",
  "character2_name": "Marcus",
  "character2_mbti": "ISTJ",
  "compatibility_score": 0.75
}
```

---

## Data Models

### Character

```json
{
  "name": "string",
  "description": "string",
  "personality": {
    "traits": {
      "assertiveness": 0.0-1.0,
      "warmth": 0.0-1.0,
      "openness": 0.0-1.0,
      "conscientiousness": 0.0-1.0,
      "emotional_stability": 0.0-1.0,
      "humor_inclination": 0.0-1.0,
      "formality": 0.0-1.0,
      "extraversion": 0.0-1.0,
      "agreeableness": 0.0-1.0
    },
    "values": {
      "priority_keywords": ["string"],
      "beliefs": ["string"],
      "dislikes": ["string"],
      "strengths": ["string"],
      "weaknesses": ["string"]
    },
    "speech_patterns": ["string"],
    "quirks": ["string"],
    "mbti_type": "INTJ",
    "mbti_archetype": "Architect"
  },
  "background": {
    "age": 30,
    "origin": "string",
    "occupation": "string",
    "motivations": ["string"],
    "fears": ["string"],
    "desires": ["string"],
    "formative_events": ["string"],
    "relationships": ["string"],
    "memories": [{"content": "string", "emotional_weight": 0.8}]
  }
}
```

### DialogueResponse

```json
{
  "response": "string",
  "emotional_context": "string",
  "internal_thought": "string"
}
```

---

## Interactive Documentation

Once the server is running:

- **Swagger UI**: http://localhost:8000/docs — try endpoints interactively.
- **ReDoc**: http://localhost:8000/redoc — browsable reference.

---

## Error Handling

All errors return a JSON body:
```json
{
  "detail": "Character 'Unknown' not found"
}
```

Use the `detail` field for diagnostic messages.

---

## See Also

- [USER_GUIDE.md](USER_GUIDE.md) — Streamlit app guide
- [DEVELOPER_GUIDE.md](DEVELOPER_GUIDE.md) — extending the system
- [ARCHITECTURE.md](ARCHITECTURE.md) — system design
