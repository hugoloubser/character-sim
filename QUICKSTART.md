# Quick Start Guide

## What You Get

An LLM-backed character creation and dialogue system featuring:

- **9-axis personality model** with MBTI integration and heredity
- **Tiered memory** (working → short-term → long-term) with salience decay
- **Dynamic character evolution** — trait shifts, self-reflection, dissonance detection
- **Multi-character dialogue** with personality-weighted speaker selection
- **Streamlit dashboard** — character workshop, scene runner, metrics, archive
- **FastAPI REST API** with Swagger documentation
- **LLM metrics instrumentation** — call tracking, token estimation, JSONL persistence

## Getting Started

### 1. Install

```bash
# Python 3.11+ required
python --version

# Install uv (if not already installed)
pip install uv

# Create and activate virtual environment
uv venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

# Install the package
uv pip install -e .
```

### 2. Configure API Keys

```bash
# Copy the example and add at least one API key
copy .env.example local/.env
```

Supported providers:
- `OPENAI_API_KEY` — OpenAI (GPT-4, etc.)
- `ANTHROPIC_API_KEY` — Anthropic (Claude)
- `GOOGLE_API_KEY` — Google (Gemini via OpenAI-compatible endpoint)

### 3. Launch the Streamlit UI

```bash
streamlit run src/character_creator/ui/app.py
# Opens at http://localhost:8501
```

### 4. Start the REST API (optional)

```bash
uvicorn character_creator.api.main:create_app --reload
# Swagger docs at http://localhost:8000/docs
```

### 5. Run Demo Scripts

```bash
python -m character_creator.demos.basic_interaction
python -m character_creator.demos.multi_character_scene
```

## Configuration

Create `local/.env` (git-ignored) with:

```ini
LLM_PROVIDER=openai
OPENAI_API_KEY=sk-...
DEFAULT_MODEL=gpt-4-turbo-preview
```

See [INSTALLATION_SETUP.md](INSTALLATION_SETUP.md) for full configuration options.

## Next Steps

| Guide | Purpose |
|-------|---------|
| [USER_GUIDE.md](USER_GUIDE.md) | Streamlit app walkthrough |
| [API_DOCUMENTATION.md](API_DOCUMENTATION.md) | REST API reference |
| [DEVELOPER_GUIDE.md](DEVELOPER_GUIDE.md) | Extending the system |
| [ARCHITECTURE.md](ARCHITECTURE.md) | System design and data flow |
