"""Streamlit dashboard page — LLM metrics and session diagnostics.

Renders real-time charts and summary cards for token usage, call
latencies, error rates, and per-subsystem activity from the current
session's :class:`MetricsCollector`.
"""

from __future__ import annotations

import csv
import io

from datetime import UTC, datetime

import pandas as pd
import streamlit as st

from character_creator.llm.metrics import MetricsCollector

# Human-readable labels for call_type tags
_TYPE_LABELS: dict[str, str] = {
    "dialogue": "Dialogue",
    "internal_monologue": "Monologue",
    "emotion": "Emotion",
    "memory_condensation": "Memory (condense)",
    "memory_promotion": "Memory (promote)",
    "experience_classification": "Experience class.",
    "self_reflection": "Self-reflection",
    "behaviour_extraction": "Behaviour extract.",
    "dissonance_detection": "Dissonance detect.",
    "milestone_review": "Milestone review",
    "unknown": "Other",
}


def _label(call_type: str) -> str:
    return _TYPE_LABELS.get(call_type, call_type)


def _get_collector() -> MetricsCollector:
    """Return the session-scoped MetricsCollector (create if missing)."""
    if "metrics_collector" not in st.session_state:
        st.session_state.metrics_collector = MetricsCollector()
    return st.session_state.metrics_collector


# ---------------------------------------------------------------------------
# Public page entry point
# ---------------------------------------------------------------------------


def page_dashboard() -> None:
    """Render the metrics dashboard."""
    st.markdown(
        '<div class="stage-header">'
        "<h1>📊 Dashboard</h1>"
        "<p>LLM performance, token budget, and session diagnostics</p>"
        "</div>",
        unsafe_allow_html=True,
    )

    collector = _get_collector()

    if not collector.records:
        st.info(
            "No LLM calls recorded yet. Start a scene to see metrics here.",
            icon="📭",
        )
        return

    df = _build_dataframe(collector)

    # ------------------------------------------------------------------ KPIs
    _render_kpi_row(collector)
    st.divider()

    # --------------------------------------------------------- Subsystem split
    col_left, col_right = st.columns(2)
    with col_left:
        _render_calls_by_type(collector)
    with col_right:
        _render_tokens_by_type(collector)
    st.divider()

    # ------------------------------------------------- Latency & token charts
    col_left2, col_right2 = st.columns(2)
    with col_left2:
        _render_latency_chart(df)
    with col_right2:
        _render_token_chart(df)
    st.divider()

    # ---------------------------------------- Timeline & error / cost section
    col_left3, col_right3 = st.columns(2)
    with col_left3:
        _render_call_timeline(df)
    with col_right3:
        _render_error_breakdown(collector)
    st.divider()

    # ------------------------------------------------------------------ Table
    _render_call_log(df)


# ---------------------------------------------------------------------------
# DataFrame builder
# ---------------------------------------------------------------------------


def _build_dataframe(collector: MetricsCollector) -> pd.DataFrame:
    """Convert records to a DataFrame used by all charts."""
    rows = []
    for i, r in enumerate(collector.records):
        rows.append({
            "Call": i + 1,
            "Time": r.timestamp.strftime("%H:%M:%S"),
            "Type": _label(r.call_type),
            "Provider": r.provider,
            "Model": r.model[:25] if r.model else "",
            "Prompt (ch)": r.prompt_chars,
            "Response (ch)": r.response_chars,
            "Est. Tokens": r.estimated_total_tokens,
            "Latency (ms)": round(r.latency_ms, 1),
            "Temp": round(r.temperature, 2) if r.temperature is not None else None,
            "Status": "✓" if r.success else f"✗ {r.error or ''}",
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# KPI cards
# ---------------------------------------------------------------------------


def _render_kpi_row(collector: MetricsCollector) -> None:
    """Show headline numbers across the top of the page."""
    cols = st.columns(6)
    with cols[0]:
        st.metric("Total Calls", collector.total_calls)
    with cols[1]:
        st.metric("Successful", collector.successful_calls)
    with cols[2]:
        rate = (
            f"{collector.successful_calls / collector.total_calls * 100:.0f}%"
            if collector.total_calls
            else "—"
        )
        st.metric("Success Rate", rate)
    with cols[3]:
        st.metric("Avg Latency", f"{collector.avg_latency_ms:,.0f} ms")
    with cols[4]:
        st.metric("P95 Latency", f"{collector.p95_latency_ms:,.0f} ms")
    with cols[5]:
        tokens = collector.total_estimated_tokens
        st.metric("Est. Tokens", f"~{tokens:,}")


# ---------------------------------------------------------------------------
# Subsystem breakdown
# ---------------------------------------------------------------------------


def _render_calls_by_type(collector: MetricsCollector) -> None:
    """Horizontal bar chart of call counts per subsystem."""
    st.subheader("Calls by Subsystem")
    raw = collector.calls_by_type()
    if not raw:
        return
    chart_df = pd.DataFrame({
        "Subsystem": [_label(k) for k in raw],
        "Calls": list(raw.values()),
    })
    st.bar_chart(chart_df, x="Subsystem", y="Calls", color="#a78bfa", horizontal=True)


def _render_tokens_by_type(collector: MetricsCollector) -> None:
    """Horizontal bar chart of estimated tokens per subsystem."""
    st.subheader("Tokens by Subsystem (est.)")
    raw = collector.tokens_by_type()
    if not raw:
        return
    chart_df = pd.DataFrame({
        "Subsystem": [_label(k) for k in raw],
        "Tokens": list(raw.values()),
    })
    st.bar_chart(chart_df, x="Subsystem", y="Tokens", color="#38bdf8", horizontal=True)


# ---------------------------------------------------------------------------
# Per-call charts
# ---------------------------------------------------------------------------


def _render_latency_chart(df: pd.DataFrame) -> None:
    """Bar chart of latency per call, coloured by subsystem type."""
    st.subheader("Latency per Call (ms)")
    st.bar_chart(df, x="Call", y="Latency (ms)", color="Type")


def _render_token_chart(df: pd.DataFrame) -> None:
    """Stacked bar of prompt + response characters per call."""
    st.subheader("Characters Sent / Received")
    chart_df = df[["Call", "Prompt (ch)", "Response (ch)"]].set_index("Call")
    st.bar_chart(chart_df, color=["#38bdf8", "#34d399"])


def _render_call_timeline(df: pd.DataFrame) -> None:
    """Area chart of cumulative latency over calls."""
    st.subheader("Cumulative Latency")
    cumulative = df["Latency (ms)"].cumsum() / 1000
    timeline_df = pd.DataFrame({"Call": df["Call"], "Cumulative (s)": cumulative})
    st.area_chart(timeline_df, x="Call", y="Cumulative (s)", color="#f472b6")


def _render_error_breakdown(collector: MetricsCollector) -> None:
    """Success message or error-type bar chart."""
    st.subheader("Call Outcomes")
    success_count = collector.successful_calls
    fail_count = collector.failed_calls
    if fail_count == 0:
        st.success(f"All {success_count} calls succeeded", icon="✅")
    else:
        error_types: dict[str, int] = {}
        for r in collector.records:
            if not r.success and r.error:
                error_types[r.error] = error_types.get(r.error, 0) + 1
        err_df = pd.DataFrame({
            "Error": list(error_types.keys()),
            "Count": list(error_types.values()),
        })
        st.bar_chart(err_df, x="Error", y="Count", color="#f87171")
        st.warning(
            f"{fail_count} of {collector.total_calls} calls failed",
            icon="⚠️",
        )


# ---------------------------------------------------------------------------
# Call log table
# ---------------------------------------------------------------------------


def _render_call_log(df: pd.DataFrame) -> None:
    """Full scrollable table of every LLM call plus CSV export."""
    st.subheader("Call Log")
    st.dataframe(df, use_container_width=True, hide_index=True)

    # Export button
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=list(df.columns))
    writer.writeheader()
    for row in df.to_dict("records"):
        writer.writerow(row)
    st.download_button(
        "⬇ Export CSV",
        buf.getvalue(),
        file_name=f"llm_metrics_{datetime.now(tz=UTC).strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv",
    )
