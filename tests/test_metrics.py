"""Tests for the LLM metrics collector and instrumented provider."""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock

import pytest

from character_creator.llm.metrics import (
    InstrumentedProvider,
    LLMCallRecord,
    MetricsCollector,
    llm_context,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _mock_provider(response: str = "Hello world") -> MagicMock:
    provider = MagicMock()
    provider.provider_name = "test"
    provider.config = MagicMock()
    provider.config.large_model = "test-model"
    provider.config.temperature = 0.7
    provider.generate = AsyncMock(return_value=response)
    provider.generate_with_format = AsyncMock(return_value=response)
    provider.close = AsyncMock()
    return provider


# ---------------------------------------------------------------------------
# MetricsCollector basics
# ---------------------------------------------------------------------------


class TestMetricsCollector:
    def test_empty_state(self) -> None:
        c = MetricsCollector()
        assert c.total_calls == 0
        assert c.successful_calls == 0
        assert c.failed_calls == 0
        assert c.total_tokens == 0
        assert c.total_estimated_tokens == 0
        assert c.avg_latency_ms == 0.0
        assert c.p95_latency_ms == 0.0

    def test_wrap_returns_instrumented_provider(self) -> None:
        c = MetricsCollector()
        wrapped = c.wrap(_mock_provider())
        assert isinstance(wrapped, InstrumentedProvider)

    def test_calls_by_type(self) -> None:
        c = MetricsCollector()
        now = datetime.now(tz=UTC)
        for ct in ["dialogue", "dialogue", "emotion", "self_reflection"]:
            c.records.append(LLMCallRecord(
                timestamp=now, provider="t", model="m",
                prompt_chars=10, response_chars=10,
                prompt_tokens=None, completion_tokens=None,
                latency_ms=100, temperature=0.7, success=True, call_type=ct,
            ))
        by_type = c.calls_by_type()
        assert by_type["dialogue"] == 2
        assert by_type["emotion"] == 1
        assert by_type["self_reflection"] == 1

    def test_tokens_by_type(self) -> None:
        c = MetricsCollector()
        now = datetime.now(tz=UTC)
        c.records.append(LLMCallRecord(
            timestamp=now, provider="t", model="m",
            prompt_chars=400, response_chars=200,
            prompt_tokens=None, completion_tokens=None,
            latency_ms=100, temperature=0.7, success=True, call_type="dialogue",
        ))
        by_type = c.tokens_by_type()
        # 400/4 + 200/4 = 100 + 50 = 150
        assert by_type["dialogue"] == 150

    def test_latency_by_type(self) -> None:
        c = MetricsCollector()
        now = datetime.now(tz=UTC)
        for ms in [100.0, 200.0]:
            c.records.append(LLMCallRecord(
                timestamp=now, provider="t", model="m",
                prompt_chars=10, response_chars=10,
                prompt_tokens=None, completion_tokens=None,
                latency_ms=ms, temperature=0.7, success=True, call_type="emotion",
            ))
        by_type = c.latency_by_type()
        assert by_type["emotion"] == pytest.approx(300.0)


# ---------------------------------------------------------------------------
# InstrumentedProvider
# ---------------------------------------------------------------------------


class TestInstrumentedProvider:
    @pytest.mark.asyncio
    async def test_generate_records_success(self) -> None:
        collector = MetricsCollector()
        wrapped = collector.wrap(_mock_provider("Test output"))

        result = await wrapped.generate("Hello")
        assert result == "Test output"
        assert collector.total_calls == 1
        assert collector.successful_calls == 1
        assert collector.failed_calls == 0

        rec = collector.records[0]
        assert rec.success is True
        assert rec.provider == "test"
        assert rec.prompt_chars == 5
        assert rec.response_chars == 11
        assert rec.latency_ms >= 0

    @pytest.mark.asyncio
    async def test_generate_records_failure(self) -> None:
        provider = _mock_provider()
        provider.generate = AsyncMock(side_effect=RuntimeError("boom"))
        collector = MetricsCollector()
        wrapped = collector.wrap(provider)

        with pytest.raises(RuntimeError, match="boom"):
            await wrapped.generate("Hello")

        assert collector.total_calls == 1
        assert collector.failed_calls == 1
        rec = collector.records[0]
        assert rec.success is False
        assert rec.error == "RuntimeError"

    @pytest.mark.asyncio
    async def test_multiple_calls_aggregate(self) -> None:
        collector = MetricsCollector()
        wrapped = collector.wrap(_mock_provider("ok"))

        for _ in range(5):
            await wrapped.generate("test")

        assert collector.total_calls == 5
        assert collector.avg_latency_ms >= 0
        assert collector.p95_latency_ms >= 0

    @pytest.mark.asyncio
    async def test_generate_with_format_records_metrics(self) -> None:
        """generate_with_format must also record a call record."""
        provider = _mock_provider('{"key": "value"}')
        collector = MetricsCollector()
        wrapped = collector.wrap(provider)

        result = await wrapped.generate_with_format("prompt", "json")
        assert result == '{"key": "value"}'
        assert collector.total_calls == 1
        rec = collector.records[0]
        assert rec.success is True
        assert rec.prompt_chars == 6  # len("prompt")
        assert rec.response_chars == 16

    @pytest.mark.asyncio
    async def test_generate_with_format_records_failure(self) -> None:
        provider = _mock_provider()
        provider.generate_with_format = AsyncMock(side_effect=ValueError("bad"))
        collector = MetricsCollector()
        wrapped = collector.wrap(provider)

        with pytest.raises(ValueError, match="bad"):
            await wrapped.generate_with_format("prompt", "json")

        assert collector.total_calls == 1
        assert collector.failed_calls == 1
        assert collector.records[0].error == "ValueError"

    @pytest.mark.asyncio
    async def test_close_delegates(self) -> None:
        provider = _mock_provider()
        collector = MetricsCollector()
        wrapped = collector.wrap(provider)

        await wrapped.close()
        provider.close.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_getattr_delegates(self) -> None:
        provider = _mock_provider()
        provider.provider_name = "delegated"
        collector = MetricsCollector()
        wrapped = collector.wrap(provider)

        assert wrapped.provider_name == "delegated"

    @pytest.mark.asyncio
    async def test_temperature_and_model_recorded(self) -> None:
        collector = MetricsCollector()
        wrapped = collector.wrap(_mock_provider("ok"))

        await wrapped.generate("p", model="custom-model", temperature=0.3)
        rec = collector.records[0]
        assert rec.model == "custom-model"
        assert rec.temperature == 0.3


# ---------------------------------------------------------------------------
# Call-type context variable
# ---------------------------------------------------------------------------


class TestLLMContext:
    @pytest.mark.asyncio
    async def test_call_type_recorded_from_context(self) -> None:
        collector = MetricsCollector()
        wrapped = collector.wrap(_mock_provider("ok"))

        with llm_context("dialogue"):
            await wrapped.generate("hello")

        assert collector.records[0].call_type == "dialogue"

    @pytest.mark.asyncio
    async def test_call_type_defaults_to_unknown(self) -> None:
        collector = MetricsCollector()
        wrapped = collector.wrap(_mock_provider("ok"))

        await wrapped.generate("hello")
        assert collector.records[0].call_type == "unknown"

    @pytest.mark.asyncio
    async def test_nested_contexts(self) -> None:
        collector = MetricsCollector()
        wrapped = collector.wrap(_mock_provider("ok"))

        with llm_context("outer"):
            await wrapped.generate("a")
            with llm_context("inner"):
                await wrapped.generate("b")
            await wrapped.generate("c")

        types = [r.call_type for r in collector.records]
        assert types == ["outer", "inner", "outer"]


# ---------------------------------------------------------------------------
# LLMCallRecord
# ---------------------------------------------------------------------------


class TestLLMCallRecord:
    def test_construction(self) -> None:
        rec = LLMCallRecord(
            timestamp=datetime.now(tz=UTC),
            provider="openai",
            model="gpt-4",
            prompt_chars=100,
            response_chars=50,
            prompt_tokens=25,
            completion_tokens=15,
            latency_ms=350.5,
            temperature=0.7,
            success=True,
        )
        assert rec.provider == "openai"
        assert rec.prompt_tokens == 25
        assert rec.error is None
        assert rec.call_type == "unknown"

    def test_estimated_tokens_uses_actual_when_available(self) -> None:
        rec = LLMCallRecord(
            timestamp=datetime.now(tz=UTC),
            provider="openai", model="gpt-4",
            prompt_chars=100, response_chars=50,
            prompt_tokens=25, completion_tokens=15,
            latency_ms=100, temperature=0.7, success=True,
        )
        assert rec.estimated_prompt_tokens == 25
        assert rec.estimated_completion_tokens == 15
        assert rec.estimated_total_tokens == 40

    def test_estimated_tokens_falls_back_to_chars(self) -> None:
        rec = LLMCallRecord(
            timestamp=datetime.now(tz=UTC),
            provider="openai", model="gpt-4",
            prompt_chars=400, response_chars=200,
            prompt_tokens=None, completion_tokens=None,
            latency_ms=100, temperature=0.7, success=True,
        )
        assert rec.estimated_prompt_tokens == 100  # 400 / 4
        assert rec.estimated_completion_tokens == 50  # 200 / 4
        assert rec.estimated_total_tokens == 150


# ---------------------------------------------------------------------------
# JSONL persistence
# ---------------------------------------------------------------------------


class TestMetricsPersistence:
    def test_persist_and_reload(self, tmp_path) -> None:
        path = tmp_path / "metrics.jsonl"
        c = MetricsCollector(store_path=path)
        rec = LLMCallRecord(
            timestamp=datetime.now(tz=UTC),
            provider="openai", model="gpt-4",
            prompt_chars=100, response_chars=50,
            prompt_tokens=25, completion_tokens=15,
            latency_ms=350.5, temperature=0.7,
            success=True, call_type="dialogue",
        )
        c.append(rec)
        assert path.exists()

        # Reload into a new collector
        c2 = MetricsCollector(store_path=path)
        assert c2.total_calls == 1
        loaded = c2.records[0]
        assert loaded.provider == "openai"
        assert loaded.call_type == "dialogue"
        assert loaded.latency_ms == pytest.approx(350.5)
        assert loaded.prompt_tokens == 25

    @pytest.mark.asyncio
    async def test_generate_persists_automatically(self, tmp_path) -> None:
        path = tmp_path / "metrics.jsonl"
        collector = MetricsCollector(store_path=path)
        wrapped = collector.wrap(_mock_provider("ok"))

        with llm_context("emotion"):
            await wrapped.generate("hello")

        # Verify file was written
        lines = path.read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == 1

        # Verify reload
        c2 = MetricsCollector(store_path=path)
        assert c2.total_calls == 1
        assert c2.records[0].call_type == "emotion"

    def test_clear_empties_file(self, tmp_path) -> None:
        path = tmp_path / "metrics.jsonl"
        c = MetricsCollector(store_path=path)
        rec = LLMCallRecord(
            timestamp=datetime.now(tz=UTC),
            provider="t", model="m",
            prompt_chars=10, response_chars=10,
            prompt_tokens=None, completion_tokens=None,
            latency_ms=100, temperature=0.7,
            success=True,
        )
        c.append(rec)
        assert c.total_calls == 1

        c.clear()
        assert c.total_calls == 0
        assert path.read_text(encoding="utf-8") == ""

        # Reload confirms empty
        c2 = MetricsCollector(store_path=path)
        assert c2.total_calls == 0

    def test_no_path_means_in_memory_only(self) -> None:
        c = MetricsCollector()
        rec = LLMCallRecord(
            timestamp=datetime.now(tz=UTC),
            provider="t", model="m",
            prompt_chars=10, response_chars=10,
            prompt_tokens=None, completion_tokens=None,
            latency_ms=100, temperature=0.7,
            success=True,
        )
        c.append(rec)
        assert c.total_calls == 1

    def test_missing_file_loads_empty(self, tmp_path) -> None:
        path = tmp_path / "nonexistent.jsonl"
        c = MetricsCollector(store_path=path)
        assert c.total_calls == 0

    def test_corrupt_lines_skipped(self, tmp_path) -> None:
        path = tmp_path / "metrics.jsonl"
        # Write one good line and one bad line
        good = LLMCallRecord(
            timestamp=datetime.now(tz=UTC),
            provider="t", model="m",
            prompt_chars=10, response_chars=10,
            prompt_tokens=None, completion_tokens=None,
            latency_ms=100, temperature=0.7,
            success=True, call_type="dialogue",
        )
        import json
        path.write_text(
            json.dumps(good.to_dict()) + "\n"
            + "NOT VALID JSON\n"
            + json.dumps(good.to_dict()) + "\n",
            encoding="utf-8",
        )
        c = MetricsCollector(store_path=path)
        assert c.total_calls == 2  # bad line skipped

    def test_roundtrip_preserves_all_fields(self, tmp_path) -> None:
        path = tmp_path / "metrics.jsonl"
        original = LLMCallRecord(
            timestamp=datetime(2026, 3, 12, 10, 30, 0, tzinfo=UTC),
            provider="google", model="gemini-flash",
            prompt_chars=1234, response_chars=567,
            prompt_tokens=None, completion_tokens=None,
            latency_ms=1500.75, temperature=0.3,
            success=False, call_type="milestone_review",
            error="LLMTimeoutError",
        )
        c = MetricsCollector(store_path=path)
        c.append(original)

        c2 = MetricsCollector(store_path=path)
        loaded = c2.records[0]
        assert loaded.timestamp == original.timestamp
        assert loaded.provider == original.provider
        assert loaded.model == original.model
        assert loaded.prompt_chars == original.prompt_chars
        assert loaded.response_chars == original.response_chars
        assert loaded.prompt_tokens is None
        assert loaded.completion_tokens is None
        assert loaded.latency_ms == pytest.approx(original.latency_ms)
        assert loaded.temperature == pytest.approx(0.3)
        assert loaded.success is False
        assert loaded.call_type == "milestone_review"
        assert loaded.error == "LLMTimeoutError"
