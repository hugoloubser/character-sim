"""Tests for LLM provider implementations."""

import os

import pytest

from pydantic_core import ValidationError

from character_creator.llm.providers import (
    AnthropicConfig,
    AnthropicProvider,
    GoogleConfig,
    GoogleProvider,
    LLMConfig,
    LLMError,
    LLMInvalidRequestError,
    OpenAIConfig,
    OpenAIProvider,
    ProviderType,
    get_llm_provider,
)


class TestProviderType:
    """Test suite for ProviderType enum."""

    def test_provider_type_values(self) -> None:
        """Test ProviderType enum values."""
        assert ProviderType.OPENAI == "openai"
        assert ProviderType.ANTHROPIC == "anthropic"
        assert ProviderType.GOOGLE == "google"

    def test_provider_type_is_string_enum(self) -> None:
        """Test ProviderType behaves as string enum."""
        assert isinstance(ProviderType.OPENAI.value, str)
        assert isinstance(ProviderType.OPENAI, str)


class TestLLMConfig:
    """Test suite for LLM configuration models."""

    def test_config_validation_valid(self) -> None:
        """Test valid configuration."""
        config = LLMConfig(
            api_key="test-key",
            large_model="gpt-4",
            small_model="gpt-3.5",
        )
        assert config.api_key == "test-key"
        assert config.large_model == "gpt-4"
        assert config.small_model == "gpt-3.5"
        assert config.temperature == 0.7
        assert config.max_tokens == 500

    def test_config_temperature_validation(self) -> None:
        """Test temperature validation."""
        with pytest.raises(ValidationError):
            LLMConfig(
                api_key="test-key",
                large_model="gpt-4",
                small_model="gpt-3.5",
                temperature=3.0,
            )

    def test_config_max_tokens_validation(self) -> None:
        """Test max_tokens validation."""
        with pytest.raises(ValidationError):
            LLMConfig(
                api_key="test-key",
                large_model="gpt-4",
                small_model="gpt-3.5",
                max_tokens=99999,
            )

    def test_openai_config_defaults(self) -> None:
        """Test OpenAI config has proper defaults."""
        config = OpenAIConfig(api_key="test-key")
        assert config.large_model == "gpt-4-turbo-preview"
        assert config.small_model == "gpt-3.5-turbo"
        assert config.base_url == "https://api.openai.com/v1"

    def test_anthropic_config_defaults(self) -> None:
        """Test Anthropic config has proper defaults."""
        config = AnthropicConfig(api_key="test-key")
        assert config.large_model == "claude-3-5-sonnet-20241022"
        assert config.small_model == "claude-3-5-haiku-20241022"

    def test_google_config_defaults(self) -> None:
        """Test Google config has proper defaults."""
        config = GoogleConfig(api_key="test-key")
        assert config.large_model == "gemini-3.1-flash-lite-preview"
        assert config.small_model == "gemini-3.1-flash-lite-preview"
        assert (
            config.base_url == "https://generativelanguage.googleapis.com/v1beta/openai"
        )


class TestOpenAIProvider:
    """Test suite for OpenAI provider."""

    def test_provider_with_api_key(self) -> None:
        """Test OpenAI provider initialization with explicit API key."""
        provider = OpenAIProvider(api_key="test-key-123")
        assert provider.config.api_key == "test-key-123"
        assert provider.config.large_model == "gpt-4-turbo-preview"
        assert provider.config.small_model == "gpt-3.5-turbo"

    def test_provider_without_key_raises_error(self) -> None:
        """Test OpenAI provider raises error without API key."""
        if "OPENAI_API_KEY" in os.environ:
            del os.environ["OPENAI_API_KEY"]

        with pytest.raises(ValueError, match="OpenAI API key"):
            OpenAIProvider()

    def test_provider_with_custom_config(self) -> None:
        """Test OpenAI provider with custom configuration."""
        config = OpenAIConfig(
            api_key="test-key", large_model="gpt-4", small_model="gpt-3.5"
        )
        provider = OpenAIProvider(config=config)
        assert provider.config.large_model == "gpt-4"
        assert provider.config.small_model == "gpt-3.5"

    def test_model_resolution(self) -> None:
        """Test model resolution in provider."""
        provider = OpenAIProvider(api_key="test-key")
        assert provider._resolve_model(None) == "gpt-4-turbo-preview"
        assert provider._resolve_model("custom-model") == "custom-model"

    def test_temperature_resolution(self) -> None:
        """Test temperature resolution in provider."""
        provider = OpenAIProvider(api_key="test-key")
        assert provider._resolve_temperature(None) == 0.7
        assert provider._resolve_temperature(0.5) == 0.5


class TestAnthropicProvider:
    """Test suite for Anthropic provider."""

    def test_provider_with_api_key(self) -> None:
        """Test Anthropic provider initialization with explicit API key."""
        provider = AnthropicProvider(api_key="test-key-456")
        assert provider.config.api_key == "test-key-456"
        assert provider.config.large_model == "claude-3-5-sonnet-20241022"
        assert provider.config.small_model == "claude-3-5-haiku-20241022"

    def test_provider_without_key_raises_error(self) -> None:
        """Test Anthropic provider raises error without API key."""
        if "ANTHROPIC_API_KEY" in os.environ:
            del os.environ["ANTHROPIC_API_KEY"]

        with pytest.raises(ValueError, match="Anthropic API key"):
            AnthropicProvider()

    def test_provider_with_custom_config(self) -> None:
        """Test Anthropic provider with custom configuration."""
        config = AnthropicConfig(
            api_key="test-key", large_model="custom-large", small_model="custom-small"
        )
        provider = AnthropicProvider(config=config)
        assert provider.config.large_model == "custom-large"
        assert provider.config.small_model == "custom-small"


class TestGoogleProvider:
    """Test suite for Google provider."""

    def test_provider_with_api_key(self) -> None:
        """Test Google provider initialization with explicit API key."""
        provider = GoogleProvider(api_key="test-key-789")
        assert provider.config.api_key == "test-key-789"
        assert provider.config.large_model == "gemini-3.1-flash-lite-preview"
        assert provider.config.small_model == "gemini-3.1-flash-lite-preview"

    def test_provider_without_key_raises_error(self) -> None:
        """Test Google provider raises error without API key."""
        if "GOOGLE_API_KEY" in os.environ:
            del os.environ["GOOGLE_API_KEY"]

        with pytest.raises(ValueError, match="Google API key"):
            GoogleProvider()

    def test_provider_with_custom_config(self) -> None:
        """Test Google provider with custom configuration."""
        config = GoogleConfig(
            api_key="test-key", large_model="custom-large", small_model="custom-small"
        )
        provider = GoogleProvider(config=config)
        assert provider.config.large_model == "custom-large"
        assert provider.config.small_model == "custom-small"


class TestProviderFactory:
    """Test suite for provider factory function."""

    def test_openai_provider_factory(self) -> None:
        """Test factory creates OpenAI provider."""
        os.environ["OPENAI_API_KEY"] = "test-key"
        provider = get_llm_provider("openai")
        assert isinstance(provider, OpenAIProvider)

    def test_anthropic_provider_factory(self) -> None:
        """Test factory creates Anthropic provider."""
        os.environ["ANTHROPIC_API_KEY"] = "test-key"
        provider = get_llm_provider("anthropic")
        assert isinstance(provider, AnthropicProvider)

    def test_google_provider_factory(self) -> None:
        """Test factory creates Google provider."""
        os.environ["GOOGLE_API_KEY"] = "test-key"
        provider = get_llm_provider("google")
        assert isinstance(provider, GoogleProvider)

    def test_case_insensitive_factory(self) -> None:
        """Test factory is case-insensitive."""
        os.environ["OPENAI_API_KEY"] = "test-key"
        provider = get_llm_provider("OpenAI")
        assert isinstance(provider, OpenAIProvider)

    def test_unknown_provider_raises_error(self) -> None:
        """Test factory raises error for unknown provider."""
        with pytest.raises(ValueError, match="Unknown LLM provider"):
            get_llm_provider("unknown-provider")

    def test_default_provider(self) -> None:
        """Test factory defaults to OpenAI."""
        os.environ["OPENAI_API_KEY"] = "test-key"
        provider = get_llm_provider()
        assert isinstance(provider, OpenAIProvider)


class TestProviderInterfaces:
    """Test suite for provider interface compliance."""

    def test_openai_has_generate_method(self) -> None:
        """Test OpenAI provider has generate method."""
        provider = OpenAIProvider(api_key="test-key")
        assert hasattr(provider, "generate")
        assert callable(provider.generate)

    def test_openai_has_generate_with_format_method(self) -> None:
        """Test OpenAI provider has generate_with_format method."""
        provider = OpenAIProvider(api_key="test-key")
        assert hasattr(provider, "generate_with_format")
        assert callable(provider.generate_with_format)

    def test_anthropic_has_required_methods(self) -> None:
        """Test Anthropic provider has required methods."""
        provider = AnthropicProvider(api_key="test-key")
        assert hasattr(provider, "generate")
        assert hasattr(provider, "generate_with_format")

    def test_google_has_required_methods(self) -> None:
        """Test Google provider has required methods."""
        provider = GoogleProvider(api_key="test-key")
        assert hasattr(provider, "generate")
        assert hasattr(provider, "generate_with_format")


class TestModelSelection:
    """Test suite for large/small model selection."""

    def test_openai_large_model(self) -> None:
        """Test OpenAI large model selection."""
        provider = OpenAIProvider(api_key="test-key")
        model = provider._resolve_model(None)
        assert model == "gpt-4-turbo-preview"

    def test_openai_small_model(self) -> None:
        """Test OpenAI small model selection."""
        provider = OpenAIProvider(api_key="test-key")
        model = provider._resolve_model(provider.config.small_model)
        assert model == "gpt-3.5-turbo"

    def test_anthropic_large_model(self) -> None:
        """Test Anthropic large model selection."""
        provider = AnthropicProvider(api_key="test-key")
        model = provider._resolve_model(None)
        assert model == "claude-3-5-sonnet-20241022"

    def test_anthropic_small_model(self) -> None:
        """Test Anthropic small model selection."""
        provider = AnthropicProvider(api_key="test-key")
        model = provider._resolve_model(provider.config.small_model)
        assert model == "claude-3-5-haiku-20241022"

    def test_google_large_model(self) -> None:
        """Test Google large model selection."""
        provider = GoogleProvider(api_key="test-key")
        model = provider._resolve_model(None)
        assert model == "gemini-3.1-flash-lite-preview"

    def test_google_small_model(self) -> None:
        """Test Google small model selection."""
        provider = GoogleProvider(api_key="test-key")
        model = provider._resolve_model(provider.config.small_model)
        assert model == "gemini-3.1-flash-lite-preview"


class TestTemperatureHandling:
    """Test suite for temperature parameter handling."""

    def test_temperature_default(self) -> None:
        """Test default temperature value."""
        provider = OpenAIProvider(api_key="test-key")
        assert provider.config.temperature == 0.7

    def test_temperature_custom(self) -> None:
        """Test custom temperature value."""
        config = OpenAIConfig(api_key="test-key", temperature=0.5)
        provider = OpenAIProvider(config=config)
        assert provider.config.temperature == 0.5

    def test_temperature_resolution_with_none(self) -> None:
        """Test temperature resolution with None uses config."""
        config = OpenAIConfig(api_key="test-key", temperature=0.9)
        provider = OpenAIProvider(config=config)
        assert provider._resolve_temperature(None) == 0.9

    def test_temperature_resolution_with_value(self) -> None:
        """Test temperature resolution with explicit value."""
        config = OpenAIConfig(api_key="test-key", temperature=0.9)
        provider = OpenAIProvider(config=config)
        assert provider._resolve_temperature(0.3) == 0.3

    def test_temperature_resolution_with_zero(self) -> None:
        """Test temperature resolution with zero (deterministic)."""
        config = OpenAIConfig(api_key="test-key", temperature=0.9)
        provider = OpenAIProvider(config=config)
        assert provider._resolve_temperature(0.0) == 0.0


class TestErrorHierarchy:
    """Test suite for the LLM error hierarchy."""

    def test_all_errors_inherit_from_llm_error(self) -> None:
        """Test all custom errors are subclasses of LLMError."""
        from character_creator.llm.providers import (
            LLMAuthenticationError,
            LLMConnectionError,
            LLMContentFilterError,
            LLMRateLimitError,
            LLMResponseError,
            LLMTimeoutError,
        )

        for exc_cls in [
            LLMAuthenticationError,
            LLMConnectionError,
            LLMContentFilterError,
            LLMInvalidRequestError,
            LLMRateLimitError,
            LLMResponseError,
            LLMTimeoutError,
        ]:
            assert issubclass(exc_cls, LLMError)

    def test_error_carries_provider_and_model(self) -> None:
        """Test LLMError stores provider and model metadata."""
        err = LLMError("test", provider="openai", model="gpt-4")
        assert err.provider == "openai"
        assert err.model == "gpt-4"
        assert str(err) == "test"


class TestInputValidation:
    """Test suite for prompt validation."""

    def test_empty_prompt_raises(self) -> None:
        """Test that an empty prompt raises LLMInvalidRequestError."""
        provider = OpenAIProvider(api_key="test-key")
        with pytest.raises(LLMInvalidRequestError):
            provider._validate_prompt("")

    def test_whitespace_prompt_raises(self) -> None:
        """Test that a whitespace-only prompt raises LLMInvalidRequestError."""
        provider = OpenAIProvider(api_key="test-key")
        with pytest.raises(LLMInvalidRequestError):
            provider._validate_prompt("   \n\t  ")

    def test_valid_prompt_passes(self) -> None:
        """Test that a valid prompt passes validation without error."""
        provider = OpenAIProvider(api_key="test-key")
        provider._validate_prompt("Hello, world!")  # Should not raise

    def test_close_method_exists(self) -> None:
        """Test all providers expose a close() coroutine."""
        import asyncio

        for provider in [
            OpenAIProvider(api_key="test-key"),
            AnthropicProvider(api_key="test-key"),
            GoogleProvider(api_key="test-key"),
        ]:
            assert asyncio.iscoroutinefunction(provider.close)

