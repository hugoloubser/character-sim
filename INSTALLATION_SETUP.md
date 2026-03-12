# Installation & Setup Guide

## Prerequisites

- **Python** 3.11+ (tested on 3.11.9)
- **pip** (latest) or **uv** (recommended)
- **Git**

## Installation

### 1. Clone the Repository

```bash
git clone <repository-url>
cd character_creator
```

### 2. Create a Virtual Environment

Using **uv** (recommended):
```bash
pip install uv
uv venv
```

Or plain venv:
```bash
python -m venv .venv
```

Activate:
- **Windows (PowerShell)**: `.\.venv\Scripts\Activate.ps1`
- **Windows (CMD)**: `.venv\Scripts\activate.bat`
- **macOS / Linux**: `source .venv/bin/activate`

### 3. Install the Package

```bash
uv pip install -e .
```

Or with pip:
```bash
pip install --upgrade pip
pip install -e .
```

### 4. Verify

```bash
python -m pytest tests/ -q
```

Expected: **391 tests passing** (8 skipped — LLM integration tests that
require live API keys).

---

## Configuration

### Environment Variables

Copy the example file and edit:
```bash
copy .env.example local/.env
```

Minimum required — one API key:
```ini
LLM_PROVIDER=openai
OPENAI_API_KEY=sk-...
```

All supported variables:
```ini
# Provider (openai | anthropic | google)
LLM_PROVIDER=openai

# API keys (set at least one)
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
GOOGLE_API_KEY=...

# Model selection
DEFAULT_MODEL=gpt-4-turbo-preview

# Logging
LOG_LEVEL=INFO
```

### Provider-Specific Notes

**OpenAI** — uses the official `openai` SDK.

**Anthropic** — uses the official `anthropic` SDK.

**Google (Gemini)** — connects via an OpenAI-compatible endpoint using the
`openai` SDK. No separate Google package is needed.

### Database

- **Type**: SQLite (file-based, zero configuration)
- **Locations**: `local/character_creator.db` and `local/interactions.db`
- **Auto-initialization**: schema created on first run
- **Default characters**: 5 pre-populated on first Streamlit launch

### Metrics

- LLM call metrics written to `local/metrics.jsonl` (append-only JSONL).
- Viewable in the Streamlit Dashboard page.

---

## Running the Application

### Streamlit Web Application

```bash
streamlit run src/character_creator/ui/app.py
```

Opens at `http://localhost:8501`. Pages: My Profile, Cast, Scene, Workshop,
Archive, Dashboard, Settings.

### FastAPI REST API

```bash
uvicorn character_creator.api.main:create_app --reload
```

- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

---

## Development Setup

### Code Quality

```bash
# Lint
python -m ruff check src/ tests/

# Auto-fix
python -m ruff check src/ tests/ --fix

# Format
python -m ruff format src/ tests/
```

### Pre-commit Hooks

```bash
pip install pre-commit
pre-commit install --install-hooks
pre-commit run --all-files
```

### Tests

```bash
python -m pytest tests/ -v                 # verbose
python -m pytest tests/test_character.py   # single file
python -m pytest tests/ --cov=character_creator --cov-report=html  # coverage
```

---

## Project Dependencies

Core (from `pyproject.toml`):

| Package | Purpose |
|---------|---------|
| fastapi | REST API framework |
| uvicorn | ASGI server |
| streamlit | Web UI |
| pydantic / pydantic-settings | Data validation and config |
| openai | OpenAI + Google (Gemini) LLM calls |
| anthropic | Anthropic LLM calls |
| httpx | Async HTTP client |
| tenacity | Retry logic |
| structlog | Structured logging |
| python-dotenv | Environment file loading |

Dev:

| Package | Purpose |
|---------|---------|
| pytest / pytest-asyncio / pytest-cov | Testing |
| ruff | Linting and formatting |
| mypy | Static type checking |

---

## Common Issues

### `ModuleNotFoundError: No module named 'character_creator'`

Install in editable mode:
```bash
pip install -e .
```

### Database locked or missing tables

Delete and restart — the app recreates the schema automatically:
```bash
del local\character_creator.db
streamlit run src/character_creator/ui/app.py
```

### LLM calls fail

1. Verify the API key is set in `local/.env`.
2. Check internet connectivity.
3. Confirm sufficient API quota.

---

## Next Steps

| Guide | Purpose |
|-------|---------|
| [USER_GUIDE.md](USER_GUIDE.md) | Streamlit app walkthrough |
| [API_DOCUMENTATION.md](API_DOCUMENTATION.md) | REST API reference |
| [DEVELOPER_GUIDE.md](DEVELOPER_GUIDE.md) | Extending the system |
| [ARCHITECTURE.md](ARCHITECTURE.md) | System design |
