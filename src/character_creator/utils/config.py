"""Configuration management for the character creator system.

Supports two environments controlled by the ``APP_ENV`` variable:

* **local** (default) — data files (databases, logs, user profile) live in
  ``<project_root>/local/``.  The ``.env`` file is loaded from that same
  directory.
* **prod** — placeholder paths pointing at AWS services (S3, DynamoDB, etc.).
  Nothing is wired up yet; swap the stubs when you're ready.
"""

from pathlib import Path
from typing import Any

from pydantic_settings import BaseSettings, SettingsConfigDict

from character_creator.utils.path import get_project_root

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _project_root() -> Path:
    try:
        return get_project_root()
    except RuntimeError:
        return Path.cwd()


def _local_dir() -> Path:
    """Return ``<project_root>/local/``, creating it if necessary."""
    d = _project_root() / "local"
    d.mkdir(exist_ok=True)
    return d


def _find_env_file() -> Path:
    """Locate the ``.env`` file.

    Checks ``local/.env`` first (the canonical location), then falls back
    to the project root for legacy setups.
    """
    local_env = _local_dir() / ".env"
    if local_env.exists():
        return local_env
    root_env = _project_root() / ".env"
    if root_env.exists():
        return root_env
    # Return the preferred path even if it doesn't exist yet — pydantic
    # will simply ignore a missing file.
    return local_env


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------


class Settings(BaseSettings):
    """Application settings loaded from environment variables.

    Set ``APP_ENV=prod`` to switch to production path resolution (currently
    placeholder stubs for AWS).
    """

    # Environment setting (local or prod)
    app_env: str = "local"

    # API Settings
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    debug: bool = False

    # LLM Settings
    llm_provider: str = "openai"
    openai_api_key: str | None = None
    anthropic_api_key: str | None = None
    google_api_key: str | None = None
    default_model: str = "gpt-4-turbo-preview"
    fast_model: str = "gpt-3.5-turbo"

    # Character Settings
    max_character_memory_tokens: int = 4000
    dialogue_temperature: float = 0.8
    creativity_weight: float = 0.7

    # Logging
    log_level: str = "INFO"

    # CORS
    cors_allowed_origins: list[str] = ["http://localhost:8501", "http://localhost:8000"]

    model_config = SettingsConfigDict(
        env_file=str(_find_env_file()),
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    # ------------------------------------------------------------------
    # Path helpers — local vs prod
    # ------------------------------------------------------------------

    @property
    def is_local(self) -> bool:
        """True when running in the local development environment."""
        return self.app_env != "prod"

    @property
    def data_dir(self) -> Path:
        """Root directory for mutable data files.

        * **local**: ``<project_root>/local/``
        * **prod**: placeholder — would typically be an EFS mount or temp dir
          backed by S3.
        """
        if self.is_local:
            return _local_dir()
        # PROD placeholder — replace with a real mount / temp path
        return Path("/mnt/data/character_creator")

    @property
    def characters_db_path(self) -> Path:
        """Path to the characters SQLite database."""
        if self.is_local:
            return self.data_dir / "character_creator.db"
        # PROD placeholder — swap for DynamoDB table name / connection string
        return self.data_dir / "character_creator.db"

    @property
    def interactions_db_path(self) -> Path:
        """Path to the interactions SQLite database."""
        if self.is_local:
            return self.data_dir / "interactions.db"
        # PROD placeholder
        return self.data_dir / "interactions.db"

    @property
    def user_profile_path(self) -> Path:
        """Path to the user profile JSON file."""
        if self.is_local:
            return self.data_dir / "user_profile.json"
        # PROD placeholder — swap for S3 key / DynamoDB item
        return self.data_dir / "user_profile.json"

    @property
    def log_dir(self) -> Path:
        """Directory for log files."""
        d = self.data_dir / "logs"
        if self.is_local:
            d.mkdir(parents=True, exist_ok=True)
        return d

    @property
    def metrics_path(self) -> Path:
        """Path to the JSONL metrics file."""
        return self.data_dir / "metrics.jsonl"

    # ------------------------------------------------------------------
    # Config dicts (unchanged API)
    # ------------------------------------------------------------------

    def get_llm_config(self) -> dict[str, Any]:
        """Get LLM configuration as dictionary.

        Returns:
            Dictionary with LLM settings.

        """
        return {
            "provider": self.llm_provider,
            "default_model": self.default_model,
            "fast_model": self.fast_model,
            "temperature": self.dialogue_temperature,
        }

    def get_character_config(self) -> dict[str, Any]:
        """Get character configuration as dictionary.

        Returns:
            Dictionary with character settings.

        """
        return {
            "max_memory_tokens": self.max_character_memory_tokens,
            "dialogue_temperature": self.dialogue_temperature,
            "creativity_weight": self.creativity_weight,
        }

    def api_key_for(self, provider_name: str) -> str | None:
        """Return the API key for the given provider name.

        Args:
            provider_name: One of ``"openai"``, ``"anthropic"``, ``"google"``.

        Returns:
            API key string, or ``None`` if not set.

        """
        key_map: dict[str, str | None] = {
            "openai": self.openai_api_key,
            "anthropic": self.anthropic_api_key,
            "google": self.google_api_key,
        }
        return key_map.get(provider_name.lower())


# Global settings instance
settings = Settings()
