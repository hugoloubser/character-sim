"""LLM provider abstraction and implementations.

Uses official SDKs (openai, anthropic) for robust, production-grade LLM access.
All providers share a common error hierarchy, structured logging, input validation,
and client lifecycle management.

Supported providers:
    - OpenAI (GPT-4, GPT-3.5-turbo, etc.) via ``openai`` SDK
    - Anthropic (Claude 3+) via ``anthropic`` SDK
    - Google (Gemini 2.0+) via ``openai`` SDK with OpenAI-compatible endpoint
"""

from __future__ import annotations

import os

from abc import ABC, abstractmethod
from enum import StrEnum
from typing import Any

import anthropic
import openai
import structlog

from pydantic import BaseModel, Field
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

logger = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Exception hierarchy
# ---------------------------------------------------------------------------


class LLMError(Exception):
    """Base exception for all LLM provider errors."""

    def __init__(self, message: str, *, provider: str = "unknown", model: str = "") -> None:
        self.provider = provider
        self.model = model
        super().__init__(message)


class LLMAuthenticationError(LLMError):
    """Raised when the API key is invalid or missing permissions."""


class LLMRateLimitError(LLMError):
    """Raised when the provider signals rate-limiting (HTTP 429)."""


class LLMContentFilterError(LLMError):
    """Raised when the response is blocked by content-safety filters."""


class LLMConnectionError(LLMError):
    """Raised on network-level failures (DNS, TCP, TLS)."""


class LLMTimeoutError(LLMError):
    """Raised when a request exceeds the configured timeout."""


class LLMResponseError(LLMError):
    """Raised when the response is malformed or missing expected data."""


class LLMInvalidRequestError(LLMError):
    """Raised for client-side validation failures (empty prompt, etc.)."""


# ---------------------------------------------------------------------------
# Provider type enum
# ---------------------------------------------------------------------------


class ProviderType(StrEnum):
    """Supported LLM provider types."""

    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GOOGLE = "google"


# ---------------------------------------------------------------------------
# Configuration models
# ---------------------------------------------------------------------------


class LLMConfig(BaseModel):
    """Configuration for LLM providers with validation."""

    api_key: str = Field(..., min_length=1)
    large_model: str = Field(..., min_length=1)
    small_model: str = Field(..., min_length=1)
    temperature: float = Field(0.7, ge=0.0, le=2.0)
    max_tokens: int = Field(500, ge=1, le=16384)
    timeout: float = Field(60.0, gt=0.0)
    max_retries: int = Field(3, ge=0, le=10)


class OpenAIConfig(LLMConfig):
    """OpenAI-specific configuration."""

    base_url: str = "https://api.openai.com/v1"
    large_model: str = "gpt-4-turbo-preview"
    small_model: str = "gpt-3.5-turbo"


class AnthropicConfig(LLMConfig):
    """Anthropic-specific configuration."""

    large_model: str = "claude-3-5-sonnet-20241022"
    small_model: str = "claude-3-5-haiku-20241022"


class GoogleConfig(LLMConfig):
    """Google Gemini configuration (uses OpenAI-compatible endpoint)."""

    base_url: str = "https://generativelanguage.googleapis.com/v1beta/openai"
    large_model: str = "gemini-3.1-flash-lite-preview"
    small_model: str = "gemini-3.1-flash-lite-preview"
    max_retries: int = Field(1, ge=0, le=10)  # keep low — free tier rate-limits aggressively
    timeout: float = Field(30.0, gt=0.0)  # shorter timeout for snappier UX


# ---------------------------------------------------------------------------
# Base provider
# ---------------------------------------------------------------------------

# Maximum prompt length in characters to prevent accidental huge requests
_MAX_PROMPT_CHARS = 200_000


def _extract_openai_compat_content(
    response: Any,
    *,
    provider: str,
    model: str,
    log: Any,
) -> str:
    """Extract text from an OpenAI-compatible ChatCompletion response.

    Shared by :class:`OpenAIProvider` and :class:`GoogleProvider` since
    both use the ``openai`` SDK response format.

    Raises:
        LLMContentFilterError: If the finish reason indicates filtering.
        LLMResponseError: If no content is returned.

    """
    choice = response.choices[0] if response.choices else None
    if choice is None:
        raise LLMResponseError(
            f"{provider} returned no choices",
            provider=provider,
            model=model,
        )

    if choice.finish_reason == "content_filter":
        raise LLMContentFilterError(
            f"{provider} response blocked by content filter",
            provider=provider,
            model=model,
        )

    text = (choice.message.content or "").strip()
    if not text:
        raise LLMResponseError(
            f"{provider} returned empty content",
            provider=provider,
            model=model,
        )

    log.info(
        "llm.generate.ok",
        response_len=len(text),
        finish_reason=choice.finish_reason,
        usage_prompt=getattr(response.usage, "prompt_tokens", None),
        usage_completion=getattr(response.usage, "completion_tokens", None),
    )
    return text


class LLMProvider(ABC):
    """Abstract base class for LLM providers.

    Subclasses must implement ``generate`` and ``generate_with_format``.
    Common helpers for model/temperature resolution, input validation,
    and logging are provided here.
    """

    provider_name: str = "base"

    def __init__(self, config: LLMConfig) -> None:
        """Initialize LLM provider with configuration.

        Args:
            config: LLM configuration with API key and models.

        """
        self.config = config

    @abstractmethod
    async def generate(
        self,
        prompt: str,
        model: str | None = None,
        temperature: float | None = None,
    ) -> str:
        """Generate text using the LLM.

        Args:
            prompt: The prompt to send to the LLM.
            model: Model to use (defaults to large_model).
            temperature: Sampling temperature (0-2). Uses config default if None.

        Returns:
            Generated text response.

        Raises:
            LLMError: On any provider failure (see subclasses for specifics).

        """

    @abstractmethod
    async def generate_with_format(
        self,
        prompt: str,
        response_format: str,
        model: str | None = None,
        temperature: float | None = None,
    ) -> str:
        """Generate text with a specific format requirement.

        Args:
            prompt: The prompt to send to the LLM.
            response_format: Expected format (e.g., "json", "markdown").
            model: Model to use (defaults to large_model).
            temperature: Sampling temperature. Uses config default if None.

        Returns:
            Generated text in the specified format.

        Raises:
            LLMError: On any provider failure.

        """

    @abstractmethod
    async def close(self) -> None:
        """Release underlying HTTP resources.

        Override in subclasses that hold long-lived clients.
        """

    # -- helpers -------------------------------------------------------------

    def _resolve_model(self, model: str | None) -> str:
        """Resolve model, using large_model if not specified."""
        return model or self.config.large_model

    def _resolve_temperature(self, temperature: float | None) -> float:
        """Resolve temperature, using config default if not specified."""
        return temperature if temperature is not None else self.config.temperature

    def _validate_prompt(self, prompt: str) -> None:
        """Validate a prompt before sending to the API.

        Raises:
            LLMInvalidRequestError: If the prompt is empty or too long.

        """
        if not prompt or not prompt.strip():
            raise LLMInvalidRequestError(
                "Prompt must not be empty",
                provider=self.provider_name,
            )
        if len(prompt) > _MAX_PROMPT_CHARS:
            raise LLMInvalidRequestError(
                f"Prompt exceeds maximum length ({len(prompt):,} > {_MAX_PROMPT_CHARS:,} chars)",
                provider=self.provider_name,
            )


# ---------------------------------------------------------------------------
# OpenAI provider (official SDK)
# ---------------------------------------------------------------------------


class OpenAIProvider(LLMProvider):
    """OpenAI API provider using the official ``openai`` Python SDK."""

    provider_name: str = "openai"

    def __init__(
        self,
        api_key: str | None = None,
        config: OpenAIConfig | None = None,
    ) -> None:
        """Initialize OpenAI provider.

        Args:
            api_key: OpenAI API key (falls back to ``OPENAI_API_KEY`` env var).
            config: Optional custom configuration.

        Raises:
            ValueError: If no API key is available.

        """
        if config:
            super().__init__(config)
        else:
            key = api_key or os.getenv("OPENAI_API_KEY")
            if not key:
                msg = "OpenAI API key not provided and OPENAI_API_KEY not set"
                raise ValueError(msg)
            super().__init__(OpenAIConfig(api_key=key))

        cfg: OpenAIConfig = self.config  # type: ignore[assignment]
        self._client = openai.AsyncOpenAI(
            api_key=cfg.api_key,
            base_url=cfg.base_url,
            timeout=cfg.timeout,
            max_retries=cfg.max_retries,
        )

    async def close(self) -> None:
        """Close the underlying HTTP client."""
        await self._client.close()

    async def generate(
        self,
        prompt: str,
        model: str | None = None,
        temperature: float | None = None,
    ) -> str:
        """Generate text using the OpenAI chat completions API.

        Raises:
            LLMAuthenticationError: On invalid API key.
            LLMRateLimitError: On 429 responses.
            LLMConnectionError: On network failures.
            LLMTimeoutError: On request timeout.
            LLMContentFilterError: When content is filtered.
            LLMResponseError: On unexpected response shape.

        """
        self._validate_prompt(prompt)
        resolved_model = self._resolve_model(model)
        resolved_temp = self._resolve_temperature(temperature)

        log = logger.bind(provider="openai", model=resolved_model)
        log.info("llm.generate.start", prompt_len=len(prompt))

        try:
            response = await self._client.chat.completions.create(
                model=resolved_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=resolved_temp,
                max_tokens=self.config.max_tokens,
            )
        except openai.AuthenticationError as exc:
            log.error("llm.auth_error", error=str(exc))
            raise LLMAuthenticationError(
                f"OpenAI authentication failed: {exc}",
                provider="openai",
                model=resolved_model,
            ) from exc
        except openai.RateLimitError as exc:
            log.warning("llm.rate_limit", error=str(exc))
            raise LLMRateLimitError(
                f"OpenAI rate limit exceeded: {exc}",
                provider="openai",
                model=resolved_model,
            ) from exc
        except openai.APITimeoutError as exc:
            log.warning("llm.timeout", error=str(exc))
            raise LLMTimeoutError(
                f"OpenAI request timed out: {exc}",
                provider="openai",
                model=resolved_model,
            ) from exc
        except openai.APIConnectionError as exc:
            log.error("llm.connection_error", error=str(exc))
            raise LLMConnectionError(
                f"OpenAI connection failed: {exc}",
                provider="openai",
                model=resolved_model,
            ) from exc
        except openai.APIStatusError as exc:
            log.error("llm.api_error", status=exc.status_code, error=str(exc))
            raise LLMError(
                f"OpenAI API error (HTTP {exc.status_code}): {exc}",
                provider="openai",
                model=resolved_model,
            ) from exc

        return self._extract_openai_content(response, resolved_model, log)

    async def generate_with_format(
        self,
        prompt: str,
        response_format: str,
        model: str | None = None,
        temperature: float | None = None,
    ) -> str:
        """Generate text with a format hint appended to the prompt."""
        format_instructions = _format_instruction(response_format)
        return await self.generate(f"{prompt}\n\n{format_instructions}", model, temperature)

    # -- internal ------------------------------------------------------------

    def _extract_openai_content(
        self,
        response: Any,
        model: str,
        log: Any,
    ) -> str:
        """Extract text content from an OpenAI ChatCompletion response."""
        return _extract_openai_compat_content(
            response, provider=self.provider_name, model=model, log=log,
        )


# ---------------------------------------------------------------------------
# Google Gemini provider (OpenAI-compatible endpoint via ``openai`` SDK)
# ---------------------------------------------------------------------------


class GoogleProvider(LLMProvider):
    """Google Gemini provider using the OpenAI-compatible REST endpoint.

    Google exposes ``/v1beta/openai/chat/completions`` which is wire-compatible
    with the OpenAI SDK, keeping the implementation DRY and well-tested.
    """

    provider_name: str = "google"

    def __init__(
        self,
        api_key: str | None = None,
        config: GoogleConfig | None = None,
    ) -> None:
        """Initialize Google provider.

        Args:
            api_key: Google API key (falls back to ``GOOGLE_API_KEY`` env var).
            config: Optional custom configuration.

        Raises:
            ValueError: If no API key is available.

        """
        if config:
            super().__init__(config)
        else:
            key = api_key or os.getenv("GOOGLE_API_KEY")
            if not key:
                msg = "Google API key not provided and GOOGLE_API_KEY not set"
                raise ValueError(msg)
            super().__init__(GoogleConfig(api_key=key))

        cfg: GoogleConfig = self.config  # type: ignore[assignment]
        self._client = openai.AsyncOpenAI(
            api_key=cfg.api_key,
            base_url=cfg.base_url,
            timeout=cfg.timeout,
            max_retries=cfg.max_retries,
        )

    async def close(self) -> None:
        """Close the underlying HTTP client."""
        await self._client.close()

    async def generate(
        self,
        prompt: str,
        model: str | None = None,
        temperature: float | None = None,
    ) -> str:
        """Generate text using Google Gemini via the OpenAI-compatible endpoint.

        Error handling mirrors :class:`OpenAIProvider` since the wire protocol
        is identical.

        """
        self._validate_prompt(prompt)
        resolved_model = self._resolve_model(model)
        resolved_temp = self._resolve_temperature(temperature)

        log = logger.bind(provider="google", model=resolved_model)
        log.info("llm.generate.start", prompt_len=len(prompt))

        try:
            response = await self._client.chat.completions.create(
                model=resolved_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=resolved_temp,
                max_tokens=self.config.max_tokens,
            )
        except openai.AuthenticationError as exc:
            log.error("llm.auth_error", error=str(exc))
            raise LLMAuthenticationError(
                f"Google authentication failed: {exc}",
                provider="google",
                model=resolved_model,
            ) from exc
        except openai.RateLimitError as exc:
            log.warning("llm.rate_limit", error=str(exc))
            raise LLMRateLimitError(
                f"Google rate limit exceeded: {exc}",
                provider="google",
                model=resolved_model,
            ) from exc
        except openai.APITimeoutError as exc:
            log.warning("llm.timeout", error=str(exc))
            raise LLMTimeoutError(
                f"Google request timed out: {exc}",
                provider="google",
                model=resolved_model,
            ) from exc
        except openai.APIConnectionError as exc:
            log.error("llm.connection_error", error=str(exc))
            raise LLMConnectionError(
                f"Google connection failed: {exc}",
                provider="google",
                model=resolved_model,
            ) from exc
        except openai.APIStatusError as exc:
            log.error("llm.api_error", status=exc.status_code, error=str(exc))
            raise LLMError(
                f"Google API error (HTTP {exc.status_code}): {exc}",
                provider="google",
                model=resolved_model,
            ) from exc

        return self._extract_content(response, resolved_model, log)

    async def generate_with_format(
        self,
        prompt: str,
        response_format: str,
        model: str | None = None,
        temperature: float | None = None,
    ) -> str:
        """Generate text with a format hint appended to the prompt."""
        format_instructions = _format_instruction(response_format)
        return await self.generate(f"{prompt}\n\n{format_instructions}", model, temperature)

    # -- internal ------------------------------------------------------------

    def _extract_content(
        self,
        response: Any,
        model: str,
        log: Any,
    ) -> str:
        """Extract text content from a Gemini/OpenAI-compat response."""
        return _extract_openai_compat_content(
            response, provider=self.provider_name, model=model, log=log,
        )


# ---------------------------------------------------------------------------
# Anthropic provider (official SDK)
# ---------------------------------------------------------------------------


class AnthropicProvider(LLMProvider):
    """Anthropic Claude provider using the official ``anthropic`` Python SDK."""

    provider_name: str = "anthropic"

    def __init__(
        self,
        api_key: str | None = None,
        config: AnthropicConfig | None = None,
    ) -> None:
        """Initialize Anthropic provider.

        Args:
            api_key: Anthropic API key (falls back to ``ANTHROPIC_API_KEY`` env var).
            config: Optional custom configuration.

        Raises:
            ValueError: If no API key is available.

        """
        if config:
            super().__init__(config)
        else:
            key = api_key or os.getenv("ANTHROPIC_API_KEY")
            if not key:
                msg = "Anthropic API key not provided and ANTHROPIC_API_KEY not set"
                raise ValueError(msg)
            super().__init__(AnthropicConfig(api_key=key))

        cfg: AnthropicConfig = self.config  # type: ignore[assignment]
        self._client = anthropic.AsyncAnthropic(
            api_key=cfg.api_key,
            timeout=cfg.timeout,
            max_retries=cfg.max_retries,
        )

    async def close(self) -> None:
        """Close the underlying HTTP client."""
        await self._client.close()

    @retry(
        retry=retry_if_exception_type(anthropic.InternalServerError),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=20),
        reraise=True,
    )
    async def generate(
        self,
        prompt: str,
        model: str | None = None,
        temperature: float | None = None,
    ) -> str:
        """Generate text using the Anthropic messages API.

        Raises:
            LLMAuthenticationError: On invalid API key.
            LLMRateLimitError: On 429 responses.
            LLMConnectionError: On network failures.
            LLMTimeoutError: On request timeout.
            LLMContentFilterError: When content is blocked.
            LLMResponseError: On unexpected response shape.

        """
        self._validate_prompt(prompt)
        resolved_model = self._resolve_model(model)
        resolved_temp = self._resolve_temperature(temperature)

        log = logger.bind(provider="anthropic", model=resolved_model)
        log.info("llm.generate.start", prompt_len=len(prompt))

        try:
            response = await self._client.messages.create(
                model=resolved_model,
                max_tokens=self.config.max_tokens,
                messages=[{"role": "user", "content": prompt}],
                temperature=resolved_temp,
            )
        except anthropic.AuthenticationError as exc:
            log.error("llm.auth_error", error=str(exc))
            raise LLMAuthenticationError(
                f"Anthropic authentication failed: {exc}",
                provider="anthropic",
                model=resolved_model,
            ) from exc
        except anthropic.RateLimitError as exc:
            log.warning("llm.rate_limit", error=str(exc))
            raise LLMRateLimitError(
                f"Anthropic rate limit exceeded: {exc}",
                provider="anthropic",
                model=resolved_model,
            ) from exc
        except anthropic.APITimeoutError as exc:
            log.warning("llm.timeout", error=str(exc))
            raise LLMTimeoutError(
                f"Anthropic request timed out: {exc}",
                provider="anthropic",
                model=resolved_model,
            ) from exc
        except anthropic.APIConnectionError as exc:
            log.error("llm.connection_error", error=str(exc))
            raise LLMConnectionError(
                f"Anthropic connection failed: {exc}",
                provider="anthropic",
                model=resolved_model,
            ) from exc
        except anthropic.APIStatusError as exc:
            log.error("llm.api_error", status=exc.status_code, error=str(exc))
            raise LLMError(
                f"Anthropic API error (HTTP {exc.status_code}): {exc}",
                provider="anthropic",
                model=resolved_model,
            ) from exc

        return self._extract_content(response, resolved_model, log)

    async def generate_with_format(
        self,
        prompt: str,
        response_format: str,
        model: str | None = None,
        temperature: float | None = None,
    ) -> str:
        """Generate text with a format hint appended to the prompt."""
        format_instructions = _format_instruction(response_format)
        return await self.generate(f"{prompt}\n\n{format_instructions}", model, temperature)

    # -- internal ------------------------------------------------------------

    def _extract_content(
        self,
        response: Any,
        model: str,
        log: Any,
    ) -> str:
        """Extract text from an Anthropic Message response.

        Raises:
            LLMContentFilterError: If stop_reason indicates filtering.
            LLMResponseError: If no text content blocks are present.

        """
        # Anthropic signals content blocks may be empty on policy violations
        if response.stop_reason == "end_turn" and not response.content:
            raise LLMResponseError(
                "Anthropic returned an empty message",
                provider="anthropic",
                model=model,
            )

        # Extract all text blocks and join (handles multi-block responses)
        text_parts = [block.text for block in response.content if block.type == "text"]

        text = "\n".join(text_parts).strip()
        if not text:
            raise LLMResponseError(
                "Anthropic returned no text content",
                provider="anthropic",
                model=model,
            )

        log.info(
            "llm.generate.ok",
            response_len=len(text),
            stop_reason=response.stop_reason,
            usage_input=getattr(response.usage, "input_tokens", None),
            usage_output=getattr(response.usage, "output_tokens", None),
        )
        return text


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _format_instruction(response_format: str) -> str:
    """Return a prompt suffix instructing the LLM to use a specific format.

    Args:
        response_format: Desired format name.

    Returns:
        Instruction string to append to the prompt.

    """
    instructions: dict[str, str] = {
        "json": "Respond with valid JSON only. No markdown fences, no commentary.",
        "markdown": "Respond using well-structured Markdown formatting.",
    }
    return instructions.get(
        response_format.lower(),
        f"Respond in {response_format} format.",
    )


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def get_llm_provider(
    provider_name: str = "openai",
    *,
    api_key: str | None = None,
) -> LLMProvider:
    """Create an LLM provider by name.

    The returned provider holds an open HTTP client. Call ``await provider.close()``
    when finished, or use it as a context variable in an application lifespan.

    Args:
        provider_name: One of ``"openai"``, ``"anthropic"``, ``"google"``.
        api_key: Optional API key override.  When given, this takes precedence
            over the corresponding environment variable.

    Returns:
        Initialised LLM provider.

    Raises:
        ValueError: If the provider name is unknown.

    """
    name = provider_name.lower().strip()
    if name == "openai":
        return OpenAIProvider(api_key=api_key)
    if name == "anthropic":
        return AnthropicProvider(api_key=api_key)
    if name == "google":
        return GoogleProvider(api_key=api_key)
    msg = f"Unknown LLM provider: {provider_name!r}. Choose from: openai, anthropic, google."
    raise ValueError(msg)


__all__ = [
    "AnthropicConfig",
    "AnthropicProvider",
    "GoogleConfig",
    "GoogleProvider",
    "LLMAuthenticationError",
    "LLMConfig",
    "LLMConnectionError",
    "LLMContentFilterError",
    "LLMError",
    "LLMInvalidRequestError",
    "LLMProvider",
    "LLMRateLimitError",
    "LLMResponseError",
    "LLMTimeoutError",
    "OpenAIConfig",
    "OpenAIProvider",
    "ProviderType",
    "get_llm_provider",
]
