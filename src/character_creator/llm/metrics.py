"""LLM call metrics collection with JSONL persistence.

Provides :class:`MetricsCollector` — a transparent wrapper around any
:class:`LLMProvider` that records per-call data (latency, token counts,
success/failure) for downstream dashboards and debugging.

Records are persisted to a JSONL file (one JSON object per line) so that
metrics survive across Streamlit reruns and app restarts.  The file is
append-only during a session; historic data is loaded on construction.
"""

from __future__ import annotations

import contextvars
import json
import logging
import time

from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path

    from character_creator.llm.providers import LLMProvider

logger = logging.getLogger(__name__)

# Average chars-per-token for English text (OpenAI tokeniser heuristic)
_CHARS_PER_TOKEN = 4.0

# ---------------------------------------------------------------------------
# Call-type context variable — set by callers to label each LLM call
# ---------------------------------------------------------------------------

_call_type_var: contextvars.ContextVar[str] = contextvars.ContextVar(
    "llm_call_type", default="unknown",
)


@contextmanager
def llm_context(label: str) -> Iterator[None]:
    """Set the call-type label for all LLM calls within this block.

    Usage::

        with llm_context("dialogue"):
            response = await provider.generate(prompt)
    """
    token = _call_type_var.set(label)
    try:
        yield
    finally:
        _call_type_var.reset(token)


@dataclass
class LLMCallRecord:
    """A single LLM API call record."""

    timestamp: datetime
    provider: str
    model: str
    prompt_chars: int
    response_chars: int
    prompt_tokens: int | None
    completion_tokens: int | None
    latency_ms: float
    temperature: float | None
    success: bool
    call_type: str = "unknown"
    error: str | None = None

    @property
    def estimated_prompt_tokens(self) -> int:
        """Return actual prompt tokens if available, else estimate from chars."""
        if self.prompt_tokens is not None:
            return self.prompt_tokens
        return max(1, int(self.prompt_chars / _CHARS_PER_TOKEN))

    @property
    def estimated_completion_tokens(self) -> int:
        """Return actual completion tokens if available, else estimate from chars."""
        if self.completion_tokens is not None:
            return self.completion_tokens
        return max(0, int(self.response_chars / _CHARS_PER_TOKEN))

    @property
    def estimated_total_tokens(self) -> int:
        """Total estimated tokens (prompt + completion)."""
        return self.estimated_prompt_tokens + self.estimated_completion_tokens

    # -- Serialisation -----------------------------------------------------

    def to_dict(self) -> dict:
        """Serialise to a JSON-safe dict."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "provider": self.provider,
            "model": self.model,
            "prompt_chars": self.prompt_chars,
            "response_chars": self.response_chars,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "latency_ms": self.latency_ms,
            "temperature": self.temperature,
            "success": self.success,
            "call_type": self.call_type,
            "error": self.error,
        }

    @classmethod
    def from_dict(cls, data: dict) -> LLMCallRecord:
        """Reconstruct from a dict (inverse of :meth:`to_dict`)."""
        return cls(
            timestamp=datetime.fromisoformat(data["timestamp"]),
            provider=data["provider"],
            model=data["model"],
            prompt_chars=data["prompt_chars"],
            response_chars=data["response_chars"],
            prompt_tokens=data.get("prompt_tokens"),
            completion_tokens=data.get("completion_tokens"),
            latency_ms=data["latency_ms"],
            temperature=data.get("temperature"),
            success=data["success"],
            call_type=data.get("call_type", "unknown"),
            error=data.get("error"),
        )


# ---------------------------------------------------------------------------
# JSONL persistence helpers
# ---------------------------------------------------------------------------


def _load_records(path: Path) -> list[LLMCallRecord]:
    """Load all records from a JSONL file."""
    if not path.exists():
        return []
    records: list[LLMCallRecord] = []
    for line_no, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        stripped = raw_line.strip()
        if not stripped:
            continue
        try:
            records.append(LLMCallRecord.from_dict(json.loads(stripped)))
        except (json.JSONDecodeError, KeyError, TypeError):
            logger.warning("Skipping malformed metrics line %d in %s", line_no, path)
    return records


def _append_record(path: Path, record: LLMCallRecord) -> None:
    """Append a single record as one JSONL line."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record.to_dict()) + "\n")


@dataclass
class MetricsCollector:
    """Collects per-call metrics from an LLM provider.

    When *store_path* is provided, records are persisted to a JSONL file
    and historic data is loaded on construction.  Without a path the
    collector behaves purely in-memory (useful for tests).

    Usage::

        collector = MetricsCollector(store_path=Path("local/metrics.jsonl"))
        wrapped = collector.wrap(provider)
    """

    records: list[LLMCallRecord] = field(default_factory=list)
    store_path: Path | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        """Load any previously persisted records."""
        if self.store_path is not None and not self.records:
            self.records = _load_records(self.store_path)

    def _persist(self, record: LLMCallRecord) -> None:
        """Append *record* to the JSONL store (if configured)."""
        if self.store_path is not None:
            _append_record(self.store_path, record)

    def append(self, record: LLMCallRecord) -> None:
        """Add a record and persist it."""
        self.records.append(record)
        self._persist(record)

    def wrap(self, provider: LLMProvider) -> InstrumentedProvider:
        """Return a wrapper that delegates to *provider* while recording metrics."""
        return InstrumentedProvider(provider, self)

    def clear(self) -> None:
        """Remove all records and truncate the backing file."""
        self.records.clear()
        if self.store_path is not None and self.store_path.exists():
            self.store_path.write_text("", encoding="utf-8")

    # -- Aggregation helpers -----------------------------------------------

    @property
    def total_calls(self) -> int:
        """Total number of LLM calls recorded."""
        return len(self.records)

    @property
    def successful_calls(self) -> int:
        """Number of calls that completed without error."""
        return sum(1 for r in self.records if r.success)

    @property
    def failed_calls(self) -> int:
        """Number of calls that raised an exception."""
        return sum(1 for r in self.records if not r.success)

    @property
    def total_prompt_tokens(self) -> int:
        """Sum of prompt tokens across all calls."""
        return sum(r.prompt_tokens or 0 for r in self.records)

    @property
    def total_completion_tokens(self) -> int:
        """Sum of completion tokens across all calls."""
        return sum(r.completion_tokens or 0 for r in self.records)

    @property
    def total_tokens(self) -> int:
        """Sum of all tokens (prompt + completion)."""
        return self.total_prompt_tokens + self.total_completion_tokens

    @property
    def total_estimated_tokens(self) -> int:
        """Sum of estimated tokens (uses char-based fallback when actual is unavailable)."""
        return sum(r.estimated_total_tokens for r in self.records)

    @property
    def avg_latency_ms(self) -> float:
        """Mean latency in milliseconds."""
        if not self.records:
            return 0.0
        return sum(r.latency_ms for r in self.records) / len(self.records)

    @property
    def p95_latency_ms(self) -> float:
        """95th percentile latency in milliseconds."""
        if not self.records:
            return 0.0
        latencies = sorted(r.latency_ms for r in self.records)
        idx = int(len(latencies) * 0.95)
        return latencies[min(idx, len(latencies) - 1)]

    def calls_by_type(self) -> dict[str, int]:
        """Return call counts grouped by call_type."""
        groups: dict[str, int] = {}
        for r in self.records:
            groups[r.call_type] = groups.get(r.call_type, 0) + 1
        return dict(sorted(groups.items(), key=lambda x: -x[1]))

    def tokens_by_type(self) -> dict[str, int]:
        """Return estimated token totals grouped by call_type."""
        groups: dict[str, int] = {}
        for r in self.records:
            groups[r.call_type] = groups.get(r.call_type, 0) + r.estimated_total_tokens
        return dict(sorted(groups.items(), key=lambda x: -x[1]))

    def latency_by_type(self) -> dict[str, float]:
        """Return total latency (ms) grouped by call_type."""
        groups: dict[str, float] = {}
        for r in self.records:
            groups[r.call_type] = groups.get(r.call_type, 0.0) + r.latency_ms
        return dict(sorted(groups.items(), key=lambda x: -x[1]))


class InstrumentedProvider:
    """Transparent proxy that records metrics for every LLM call.

    Records metrics for both ``generate()`` and ``generate_with_format()``.
    Reads the current :data:`_call_type_var` context variable to tag each
    call with its subsystem origin (dialogue, emotion, memory, etc.).
    """

    def __init__(self, provider: LLMProvider, collector: MetricsCollector) -> None:
        self._provider = provider
        self._collector = collector

    def _record(
        self,
        *,
        prompt: str,
        model: str,
        temperature: float | None,
        result: str | None,
        elapsed_ms: float,
        error: Exception | None,
    ) -> None:
        """Append a call record to the collector (and persist)."""
        self._collector.append(
            LLMCallRecord(
                timestamp=datetime.now(tz=UTC),
                provider=self._provider.provider_name,
                model=model,
                prompt_chars=len(prompt),
                response_chars=len(result) if result else 0,
                prompt_tokens=None,
                completion_tokens=None,
                latency_ms=elapsed_ms,
                temperature=temperature,
                success=error is None,
                call_type=_call_type_var.get(),
                error=type(error).__name__ if error else None,
            )
        )

    async def generate(
        self,
        prompt: str,
        model: str | None = None,
        temperature: float | None = None,
    ) -> str:
        """Delegate to the wrapped provider and record metrics."""
        resolved_model = model or self._provider.config.large_model
        resolved_temp = temperature if temperature is not None else self._provider.config.temperature

        t0 = time.perf_counter()
        try:
            result = await self._provider.generate(prompt, model, temperature)
        except Exception as exc:
            self._record(
                prompt=prompt, model=resolved_model, temperature=resolved_temp,
                result=None, elapsed_ms=(time.perf_counter() - t0) * 1000, error=exc,
            )
            raise
        else:
            self._record(
                prompt=prompt, model=resolved_model, temperature=resolved_temp,
                result=result, elapsed_ms=(time.perf_counter() - t0) * 1000, error=None,
            )
            return result

    async def generate_with_format(
        self,
        prompt: str,
        response_format: str,
        model: str | None = None,
        temperature: float | None = None,
    ) -> str:
        """Delegate format-aware generation and record metrics."""
        resolved_model = model or self._provider.config.large_model
        resolved_temp = temperature if temperature is not None else self._provider.config.temperature

        t0 = time.perf_counter()
        try:
            result = await self._provider.generate_with_format(
                prompt, response_format, model, temperature,
            )
        except Exception as exc:
            self._record(
                prompt=prompt, model=resolved_model, temperature=resolved_temp,
                result=None, elapsed_ms=(time.perf_counter() - t0) * 1000, error=exc,
            )
            raise
        else:
            self._record(
                prompt=prompt, model=resolved_model, temperature=resolved_temp,
                result=result, elapsed_ms=(time.perf_counter() - t0) * 1000, error=None,
            )
            return result

    async def close(self) -> None:
        """Close the underlying provider's HTTP client."""
        await self._provider.close()

    def __getattr__(self, name: str) -> object:
        """Delegate anything not explicitly defined to the wrapped provider."""
        return getattr(self._provider, name)
