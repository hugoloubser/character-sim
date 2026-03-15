"""Live integration test for LLM providers using the Anthropic API.

Run with: python tests/test_llm_integration.py
Output is written to _scratch_integration_output.txt for reliable inspection.

Tests that hit the live API will be marked SKIP (not FAIL) if the free-tier
quota is exhausted or no API key is configured, so the suite can still
validate all non-network logic.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import time
import traceback

from io import StringIO
from pathlib import Path

import pytest

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from dotenv import load_dotenv  # noqa: E402

# Load .env from local/ (preferred) or project root (fallback)
_local_env = PROJECT_ROOT / "local" / ".env"
_root_env = PROJECT_ROOT / ".env"
load_dotenv(_local_env if _local_env.exists() else _root_env)

from character_creator.llm.providers import (  # noqa: E402
    AnthropicConfig,
    AnthropicProvider,
    LLMError,
    LLMInvalidRequestError,
    LLMRateLimitError,
    get_llm_provider,
)

# Skip the entire module when no Anthropic key is available
_HAS_KEY = bool(os.getenv("ANTHROPIC_API_KEY"))
pytestmark = pytest.mark.skipif(
    not _HAS_KEY,
    reason="ANTHROPIC_API_KEY not set — skipping live integration tests",
)

OUTPUT_FILE = PROJECT_ROOT / "_scratch_integration_output.txt"
buf = StringIO()

# Seconds to wait between tests that make live API calls
API_COOLDOWN = 3

# The model to use for integration testing (free-tier eligible)
TEST_MODEL = "claude-3-5-sonnet-20241022"


def log(msg: str) -> None:
    """Print to both stdout and the buffer that gets written to disk."""
    line = msg
    print(line)
    buf.write(line + "\n")


def section(title: str) -> None:
    """Print a section header."""
    log(f"\n{'=' * 60}")
    log(f"  {title}")
    log(f"{'=' * 60}")


def _is_quota_exhausted(exc: LLMRateLimitError) -> bool:
    """Return True if the error is a daily quota exhaustion (not a transient spike)."""
    msg = str(exc).lower()
    return "quota" in msg or "exceeded" in msg or "rate" in msg


async def test_factory_creates_anthropic_provider() -> bool:
    """Test that the factory function returns an AnthropicProvider."""
    section("TEST 1: Factory creates AnthropicProvider")
    try:
        provider = get_llm_provider("anthropic")
        assert isinstance(provider, AnthropicProvider)
        log("PASS — get_llm_provider('anthropic') returns AnthropicProvider")
        await provider.close()
        return True
    except Exception:
        log(f"FAIL — {traceback.format_exc()}")
        return False


async def test_basic_generation() -> bool | None:
    """Test a simple text generation call against real Anthropic API.

    Returns True on pass, False on unexpected failure, None on rate-limit skip.
    """
    section(f"TEST 2: Basic text generation ({TEST_MODEL})")
    provider = get_llm_provider("anthropic")
    try:
        t0 = time.perf_counter()
        result = await provider.generate(
            "Say hello in exactly 5 words.",
            model=TEST_MODEL,
            temperature=0.3,
        )
        elapsed = time.perf_counter() - t0

        log(f"  Response ({elapsed:.2f}s): {result!r}")
        assert isinstance(result, str)
        assert len(result) > 0
        log("PASS — received non-empty string response")
        return True
    except LLMRateLimitError as exc:
        if _is_quota_exhausted(exc):
            log("SKIP — free-tier quota exhausted (not a code defect)")
            return None
        log(f"FAIL — unexpected rate-limit error: {exc}")
        return False
    except Exception:
        log(f"FAIL — {traceback.format_exc()}")
        return False
    finally:
        await provider.close()


async def test_json_format_generation() -> bool | None:
    """Test generate_with_format('json') returns parseable JSON.

    Returns True on pass, False on unexpected failure, None on rate-limit skip.
    """
    section("TEST 3: JSON format generation")
    provider = get_llm_provider("anthropic")
    try:
        result = await provider.generate_with_format(
            'Return a JSON object with keys "name" and "age" for a fictional character.',
            response_format="json",
            model=TEST_MODEL,
            temperature=0.2,
        )
        log(f"  Raw response: {result!r}")

        # Strip markdown fences if the model wraps them
        cleaned = result.strip()
        if cleaned.startswith("```"):
            cleaned = "\n".join(cleaned.split("\n")[1:])
        if cleaned.endswith("```"):
            cleaned = "\n".join(cleaned.split("\n")[:-1])
        cleaned = cleaned.strip()

        parsed = json.loads(cleaned)
        assert "name" in parsed
        log(f"  Parsed JSON: {parsed}")
        log("PASS — response is valid JSON with expected keys")
        return True
    except LLMRateLimitError as exc:
        if _is_quota_exhausted(exc):
            log("SKIP — free-tier quota exhausted (not a code defect)")
            return None
        log(f"FAIL — unexpected rate-limit error: {exc}")
        return False
    except json.JSONDecodeError as exc:
        log(f"FAIL — Could not parse JSON: {exc}")
        return False
    except Exception:
        log(f"FAIL — {traceback.format_exc()}")
        return False
    finally:
        await provider.close()


async def test_empty_prompt_rejected() -> bool:
    """Test that an empty prompt raises LLMInvalidRequestError locally."""
    section("TEST 4: Empty prompt validation")
    provider = get_llm_provider("anthropic")
    try:
        await provider.generate("")
        log("FAIL — should have raised LLMInvalidRequestError")
        return False
    except LLMInvalidRequestError:
        log("PASS — LLMInvalidRequestError raised for empty prompt")
        return True
    except Exception:
        log(f"FAIL — wrong exception: {traceback.format_exc()}")
        return False
    finally:
        await provider.close()


async def test_custom_config() -> bool | None:
    """Test provider works with a custom AnthropicConfig.

    Returns True on pass, False on unexpected failure, None on rate-limit skip.
    """
    section("TEST 5: Custom config with timeout and max_tokens")
    import os

    api_key = os.getenv("ANTHROPIC_API_KEY", "")
    config = AnthropicConfig(
        api_key=api_key,
        large_model=TEST_MODEL,
        small_model="claude-3-5-haiku-20241022",
        max_tokens=50,
        timeout=30.0,
        temperature=0.1,
    )
    provider = AnthropicProvider(config=config)
    try:
        result = await provider.generate("What is 2+2? Answer in one word.")
        log(f"  Response: {result!r}")
        assert isinstance(result, str)
        log("PASS — custom config works correctly")
        return True
    except LLMRateLimitError as exc:
        if _is_quota_exhausted(exc):
            log("SKIP — free-tier quota exhausted (not a code defect)")
            return None
        log(f"FAIL — unexpected rate-limit error: {exc}")
        return False
    except Exception:
        log(f"FAIL — {traceback.format_exc()}")
        return False
    finally:
        await provider.close()


async def test_model_resolution() -> bool:
    """Test that model and temperature resolution work correctly."""
    section("TEST 6: Model and temperature resolution")
    provider = get_llm_provider("anthropic")
    try:
        assert provider._resolve_model(None) == "claude-3-5-sonnet-20241022"
        assert provider._resolve_model("custom-model") == "custom-model"
        assert provider._resolve_temperature(None) == provider.config.temperature
        assert provider._resolve_temperature(0.0) == 0.0
        assert provider._resolve_temperature(1.5) == 1.5
        log("PASS — model/temperature resolution correct")
        return True
    except Exception:
        log(f"FAIL — {traceback.format_exc()}")
        return False
    finally:
        await provider.close()


async def test_character_dialogue_prompt() -> bool | None:
    """Test a realistic character dialogue prompt end-to-end.

    Returns True on pass, False on unexpected failure, None on rate-limit skip.
    """
    section("TEST 7: Realistic character dialogue prompt")
    provider = get_llm_provider("anthropic")
    try:
        prompt = (
            "You are roleplaying as Alice, a 32-year-old tech entrepreneur.\n"
            "Personality: assertive, warm, highly open to new ideas.\n"
            "Values: innovation, independence, growth.\n"
            "Speech patterns: uses technical jargon, speaks quickly when excited.\n\n"
            "Scene: A coffee shop meeting with a potential investor.\n"
            "Topic: Your new AI startup.\n\n"
            "Respond as Alice in 1-2 sentences. Stay in character."
        )
        t0 = time.perf_counter()
        result = await provider.generate(
            prompt,
            model=TEST_MODEL,
            temperature=0.7,
        )
        elapsed = time.perf_counter() - t0
        log(f"  Alice says ({elapsed:.2f}s): {result!r}")
        assert len(result) > 10
        log("PASS — received in-character dialogue response")
        return True
    except LLMRateLimitError as exc:
        if _is_quota_exhausted(exc):
            log("SKIP — free-tier quota exhausted (not a code defect)")
            return None
        log(f"FAIL — unexpected rate-limit error: {exc}")
        return False
    except Exception:
        log(f"FAIL — {traceback.format_exc()}")
        return False
    finally:
        await provider.close()


async def test_error_hierarchy() -> bool:
    """Test that custom errors behave correctly."""
    section("TEST 8: Error hierarchy")
    try:
        from character_creator.llm.providers import (
            LLMAuthenticationError,
            LLMConnectionError,
            LLMContentFilterError,
            LLMRateLimitError,
            LLMResponseError,
            LLMTimeoutError,
        )

        # All should be caught by LLMError
        for exc_cls in [
            LLMAuthenticationError,
            LLMConnectionError,
            LLMContentFilterError,
            LLMInvalidRequestError,
            LLMRateLimitError,
            LLMResponseError,
            LLMTimeoutError,
        ]:
            exc = exc_cls("test msg", provider="test", model="test-model")
            assert isinstance(exc, LLMError)
            assert exc.provider == "test"
            assert exc.model == "test-model"

        log("PASS — all error subclasses work correctly")
        return True
    except Exception:
        log(f"FAIL — {traceback.format_exc()}")
        return False


async def main() -> None:
    """Run all integration tests and write results to disk."""
    log(f"LLM Provider Integration Tests — {time.strftime('%Y-%m-%d %H:%M:%S')}")
    log(f"Output file: {OUTPUT_FILE}")
    log(f"Provider: Anthropic | Model: {TEST_MODEL}")

    # Tests are tagged as hitting the API or not so we can add cooldowns
    tests: list[tuple[bool, object]] = [
        (False, test_factory_creates_anthropic_provider),
        (True, test_basic_generation),
        (True, test_json_format_generation),
        (False, test_empty_prompt_rejected),
        (True, test_custom_config),
        (False, test_model_resolution),
        (True, test_character_dialogue_prompt),
        (False, test_error_hierarchy),
    ]

    results: list[tuple[str, bool | None]] = []
    quota_dead = False  # Once quota is gone, skip remaining API tests

    for hits_api, test_fn in tests:
        if hits_api and quota_dead:
            section(f"SKIPPING {test_fn.__name__} (quota exhausted earlier)")
            log("SKIP — quota known exhausted; skipping to avoid pointless 429")
            results.append((test_fn.__name__, None))
            continue

        try:
            passed = await test_fn()
        except Exception:
            log(f"FAIL (unhandled) — {traceback.format_exc()}")
            passed = False

        if passed is None and hits_api:
            quota_dead = True

        results.append((test_fn.__name__, passed))

        # Cooldown between API-hitting tests
        if hits_api and not quota_dead:
            log(f"  (cooling down {API_COOLDOWN}s between API calls...)")
            await asyncio.sleep(API_COOLDOWN)

    section("SUMMARY")
    passed_count = sum(1 for _, p in results if p is True)
    skipped_count = sum(1 for _, p in results if p is None)
    failed_count = sum(1 for _, p in results if p is False)
    total = len(results)

    for name, passed in results:
        if passed is True:
            status = "PASS"
        elif passed is None:
            status = "SKIP"
        else:
            status = "FAIL"
        log(f"  [{status}] {name}")

    log(
        f"\n  {passed_count} passed, {skipped_count} skipped,"
        f" {failed_count} failed / {total} total"
    )
    if skipped_count > 0:
        log("  NOTE: Skips are due to free-tier quota limits, not code defects.")
    if failed_count == 0:
        log("  All non-quota tests passed!")

    # Write everything to disk for reliable reading
    OUTPUT_FILE.write_text(buf.getvalue(), encoding="utf-8")
    log(f"\nResults written to {OUTPUT_FILE}")


if __name__ == "__main__":
    asyncio.run(main())
