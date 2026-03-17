"""Streamlit UI -- Character Creator.

A theatrical, immersive interface for creating characters and watching them
come alive through LLM-powered dialogue.  Supports both **step-wise**
(default) and **continuous** dialogue modes, and lets the user jump into
the conversation with their own profile.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import math
import time

from typing import TYPE_CHECKING, Any

import streamlit as st

from character_creator.utils.path import ensure_src_in_path

# Ensure imports work regardless of working directory
ensure_src_in_path()

from character_creator.api.models import (  # noqa: E402
    BackgroundRequest,
    CharacterCreateRequest,
    PersonalityRequest,
    PersonalityTraitsRequest,
    ValuesRequest,
)
from character_creator.api.routes.characters import (  # noqa: E402
    character_repository,
    initialize_default_characters,
)
from character_creator.core.character import Character  # noqa: E402
from character_creator.core.constants import EMOTIONAL_STATES  # noqa: E402
from character_creator.core.dialogue import DialogueContext, DialogueSystem  # noqa: E402
from character_creator.core.interaction import (  # noqa: E402
    InteractionRecord,
    SQLiteInteractionRepository,
)
from character_creator.core.memory import Background  # noqa: E402
from character_creator.core.personality import Personality  # noqa: E402
from character_creator.llm.metrics import MetricsCollector  # noqa: E402
from character_creator.llm.providers import get_llm_provider  # noqa: E402
from character_creator.ui.dashboard import page_dashboard  # noqa: E402
from character_creator.utils.config import settings  # noqa: E402

if TYPE_CHECKING:
    from character_creator.core.trait_evolution import MilestoneReview

# ---------------------------------------------------------------------------
# Globals / singletons
# ---------------------------------------------------------------------------

logger = logging.getLogger(__name__)

MAX_EXCHANGES = 30

_interaction_repo = SQLiteInteractionRepository(
    settings.interactions_db_path,
)

_USER_PROFILE_PATH = settings.user_profile_path

# ---------------------------------------------------------------------------
# User profile persistence
# ---------------------------------------------------------------------------

_DEFAULT_USER_PROFILE: dict[str, Any] = {
    "name": "You",
    "description": (
        "A curious and engaged conversationalist who enjoys exploring "
        "ideas and connecting with interesting characters."
    ),
    "personality": {
        "traits": {
            "assertiveness": 0.6,
            "warmth": 0.7,
            "openness": 0.8,
            "conscientiousness": 0.6,
            "emotional_stability": 0.7,
            "humor_inclination": 0.6,
            "formality": 0.4,
        },
        "values": {
            "priority_keywords": ["curiosity", "honesty", "growth"],
            "beliefs": ["Everyone has something worth saying"],
            "dislikes": ["pretentiousness"],
            "strengths": ["active listening", "empathy"],
            "weaknesses": ["occasional impatience"],
        },
        "speech_patterns": ["Conversational and direct"],
        "quirks": [],
    },
    "background": {
        "age": 30,
        "origin": "",
        "occupation": "",
        "motivations": ["Understand different perspectives"],
        "fears": [],
        "desires": ["Meaningful conversation"],
    },
}


def _load_user_profile() -> dict[str, Any]:
    """Load user profile from disk, falling back to defaults."""
    if _USER_PROFILE_PATH.exists():
        try:
            return json.loads(_USER_PROFILE_PATH.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            pass
    return dict(_DEFAULT_USER_PROFILE)


def _save_user_profile(data: dict[str, Any]) -> None:
    """Persist user profile to disk."""
    _USER_PROFILE_PATH.write_text(
        json.dumps(data, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def _build_user_character(data: dict[str, Any]) -> Character:
    """Construct a Character object from the user profile dict."""
    return Character(
        name=data["name"],
        description=data["description"],
        personality=Personality.from_dict(data.get("personality", {})),
        background=Background.from_dict(data.get("background", {})),
    )


# ---------------------------------------------------------------------------
# Custom CSS
# ---------------------------------------------------------------------------

_CUSTOM_CSS = """
<style>
/* -- chat bubbles ------------------------------------------------------ */
.chat-bubble {
    padding: 0.9rem 1.15rem;
    border-radius: 1rem;
    margin-bottom: 0.35rem;
    line-height: 1.55;
    font-size: 0.97rem;
}
.chat-speaker {
    font-weight: 700;
    margin-bottom: 0.2rem;
    font-size: 0.92rem;
}
.chat-thought {
    font-style: italic;
    opacity: 0.72;
    font-size: 0.86rem;
    margin-top: 0.35rem;
    padding-left: 0.5rem;
    border-left: 2px solid rgba(167,139,250,0.4);
}
.chat-mood {
    font-size: 0.78rem;
    opacity: 0.55;
    margin-top: 0.15rem;
}

/* -- character cards --------------------------------------------------- */
.char-card {
    background: linear-gradient(135deg, #2a2640 0%, #1e1b2e 100%);
    border: 1px solid rgba(167,139,250,0.25);
    border-radius: 1rem;
    padding: 1.3rem;
    margin-bottom: 0.7rem;
    transition: border-color 0.2s;
}
.char-card:hover { border-color: rgba(167,139,250,0.6); }
.char-card h4 { margin: 0 0 0.3rem 0; color: #a78bfa; }
.char-card p  { margin: 0; font-size: 0.9rem; opacity: 0.8; }

/* -- stat pill --------------------------------------------------------- */
.stat-pill {
    display: inline-block;
    background: rgba(167,139,250,0.15);
    border: 1px solid rgba(167,139,250,0.3);
    border-radius: 2rem;
    padding: 0.3rem 0.8rem;
    font-size: 0.82rem;
    margin: 0.15rem 0.15rem;
}

/* -- stage header ------------------------------------------------------ */
.stage-header {
    text-align: center;
    padding: 1.5rem 0 0.5rem 0;
}
.stage-header h1 {
    font-size: 2.4rem;
    background: linear-gradient(135deg, #a78bfa, #f472b6);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0.2rem;
}
.stage-header p {
    opacity: 0.6;
    font-size: 1.05rem;
}

/* -- trait bar --------------------------------------------------------- */
.trait-row {
    display: flex;
    align-items: center;
    margin-bottom: 0.35rem;
    font-size: 0.88rem;
}
.trait-label {
    width: 130px;
    flex-shrink: 0;
    opacity: 0.7;
}
.trait-bar-bg {
    flex: 1;
    height: 8px;
    background: rgba(255,255,255,0.08);
    border-radius: 4px;
    overflow: hidden;
}
.trait-bar-fill {
    height: 100%;
    border-radius: 4px;
    transition: width 0.4s ease;
}

/* -- user badge ------------------------------------------------------- */
.user-badge {
    display: inline-block;
    background: linear-gradient(135deg, #f472b6, #a78bfa);
    color: #fff;
    border-radius: 0.4rem;
    padding: 0.1rem 0.45rem;
    font-size: 0.72rem;
    margin-left: 0.4rem;
    vertical-align: middle;
}

/* -- evolution panel -------------------------------------------------- */
.evo-header {
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 0.07em;
    color: #a78bfa;
    opacity: 0.85;
    margin: 0.7rem 0 0.3rem 0;
    text-transform: uppercase;
}
.evo-narrative {
    font-style: italic;
    font-size: 0.83rem;
    opacity: 0.75;
    line-height: 1.55;
    margin-bottom: 0.5rem;
    padding: 0.35rem 0.6rem;
    background: rgba(167,139,250,0.07);
    border-left: 2px solid rgba(167,139,250,0.45);
    border-radius: 0 4px 4px 0;
}
.evo-trait-row {
    display: flex;
    align-items: center;
    margin-bottom: 0.2rem;
    font-size: 0.82rem;
}
.evo-trait-label {
    width: 115px;
    flex-shrink: 0;
    opacity: 0.7;
}
.evo-bar-wrap {
    flex: 1;
    height: 6px;
    background: rgba(255,255,255,0.06);
    border-radius: 3px;
    position: relative;
}
.evo-bar-new {
    position: absolute;
    top: 0; left: 0;
    height: 100%;
    border-radius: 3px;
}
.evo-bar-marker {
    position: absolute;
    top: -1px;
    width: 2px;
    height: calc(100% + 2px);
    border-radius: 1px;
    background: rgba(255,255,255,0.55);
    z-index: 1;
}
.evo-delta {
    width: 52px;
    text-align: right;
    font-size: 0.77rem;
    font-family: monospace;
    font-weight: 600;
}
.evo-delta-pos { color: #34d399; }
.evo-delta-neg { color: #f87171; }
.evo-justification {
    font-size: 0.76rem;
    opacity: 0.6;
    font-style: italic;
    padding: 0 0 0.35rem 0.5rem;
    line-height: 1.4;
}
.evo-micro-label {
    font-size: 0.7rem;
    opacity: 0.5;
    margin: 0.15rem 0 0.1rem 0;
    letter-spacing: 0.03em;
}
</style>
"""

_PALETTE = [
    "#a78bfa", "#f472b6", "#38bdf8", "#34d399",
    "#facc15", "#fb923c", "#f87171", "#c084fc",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _color_for(name: str) -> str:
    """Deterministic colour based on character name."""
    return _PALETTE[hash(name) % len(_PALETTE)]


def _trait_bar(label: str, value: float, color: str = "#a78bfa") -> str:
    """Return HTML for a slim trait progress bar."""
    pct = int(value * 100)
    return (
        '<div class="trait-row">'
        f'<span class="trait-label">{label}</span>'
        '<div class="trait-bar-bg">'
        f'<div class="trait-bar-fill" style="width:{pct}%;background:{color};"></div>'
        "</div>"
        f'<span style="width:40px;text-align:right;opacity:0.6;font-size:0.82rem">'
        f"{value:.2f}</span>"
        "</div>"
    )


def _render_evolution_panel(
    char: Character,
    initial_traits: dict[str, float],
    color: str,
    review: MilestoneReview | None = None,
) -> None:
    """Render a rich visual evolution summary inside a character's expander.

    Shows a narrative summary (from milestone review when available), then
    per-trait delta bars with before/after values and, where the milestone
    review produced justifications, a short italic quote explaining each shift.

    Args:
        char: The character whose current traits are shown.
        initial_traits: Trait values at scene start (for diff computation).
        color: Accent colour for this character.
        review: Optional ``MilestoneReview`` from ``_finish_scene``.
    """
    cur_traits = char.personality.traits.to_dict()
    changed = {
        k: (initial_traits.get(k, 0.0), cur_traits[k])
        for k in cur_traits
        if abs(cur_traits[k] - initial_traits.get(k, 0.0)) > 0.005
    }

    if not changed and (review is None or not review.narrative_summary):
        return

    # Build justification map from review shifts
    justifications: dict[str, str] = {}
    if review:
        for shift in review.shifts:
            if shift.justification:
                justifications[shift.trait_name] = shift.justification

    st.markdown(
        '<div class="evo-header">🌱 Character Arc</div>',
        unsafe_allow_html=True,
    )

    # Narrative summary
    if review and review.narrative_summary:
        st.markdown(
            f'<div class="evo-narrative">{review.narrative_summary}</div>',
            unsafe_allow_html=True,
        )

    if changed:
        st.markdown(
            '<div class="evo-micro-label">Trait Shifts</div>',
            unsafe_allow_html=True,
        )
        for trait_name, (old_val, new_val) in sorted(
            changed.items(), key=lambda x: abs(x[1][1] - x[1][0]), reverse=True,
        ):
            delta = new_val - old_val
            delta_cls = "evo-delta-pos" if delta > 0 else "evo-delta-neg"
            delta_arrow = "▲" if delta > 0 else "▼"
            pct_old = int(old_val * 100)
            pct_new = int(new_val * 100)
            label = trait_name.replace("_", " ").title()
            justif = justifications.get(trait_name, "")

            # Filled bar at new value; vertical tick at old value (always visible)
            row_html = (
                f'<div class="evo-trait-row">'
                f'<span class="evo-trait-label">{label}</span>'
                f'<div class="evo-bar-wrap">'
                f'<div class="evo-bar-new" style="width:{pct_new}%;background:{color};opacity:0.75;"></div>'
                f'<div class="evo-bar-marker" style="left:{pct_old}%;"></div>'
                f'</div>'
                f'<span class="evo-delta {delta_cls}">'
                f'{delta_arrow}&thinsp;{abs(delta):.2f}</span>'
                f'</div>'
            )
            if justif:
                row_html += (
                    f'<div class="evo-justification">"{justif}"</div>'
                )
            st.markdown(row_html, unsafe_allow_html=True)


def _radar_svg(
    traits: dict[str, float], size: int = 200, color: str = "#a78bfa",
) -> str:
    """Generate an inline SVG radar / spider chart for personality traits."""
    labels = list(traits.keys())
    values = list(traits.values())
    n = len(labels)
    if n < 3:
        return ""

    cx, cy, r = size // 2, size // 2, size // 2 - 30
    angle_step = 2 * math.pi / n

    # grid rings
    grid = ""
    for ring in (0.25, 0.5, 0.75, 1.0):
        pts = " ".join(
            f"{cx + r * ring * math.sin(i * angle_step):.1f},"
            f"{cy - r * ring * math.cos(i * angle_step):.1f}"
            for i in range(n)
        )
        grid += (
            f'<polygon points="{pts}" fill="none" '
            f'stroke="rgba(255,255,255,0.08)" />'
        )

    # axis lines
    axes = ""
    for i in range(n):
        x = cx + r * math.sin(i * angle_step)
        y = cy - r * math.cos(i * angle_step)
        axes += (
            f'<line x1="{cx}" y1="{cy}" x2="{x:.1f}" y2="{y:.1f}" '
            f'stroke="rgba(255,255,255,0.06)" />'
        )

    # data polygon
    data_pts = " ".join(
        f"{cx + r * values[i] * math.sin(i * angle_step):.1f},"
        f"{cy - r * values[i] * math.cos(i * angle_step):.1f}"
        for i in range(n)
    )
    data_poly = (
        f'<polygon points="{data_pts}" fill="{color}" fill-opacity="0.25" '
        f'stroke="{color}" stroke-width="2" />'
    )

    # dots + labels
    dots = ""
    for i in range(n):
        x = cx + r * values[i] * math.sin(i * angle_step)
        y = cy - r * values[i] * math.cos(i * angle_step)
        dots += f'<circle cx="{x:.1f}" cy="{y:.1f}" r="3" fill="{color}" />'

        lx = cx + (r + 18) * math.sin(i * angle_step)
        ly = cy - (r + 18) * math.cos(i * angle_step)
        anchor = "middle"
        if math.sin(i * angle_step) > 0.3:
            anchor = "start"
        elif math.sin(i * angle_step) < -0.3:
            anchor = "end"
        short = labels[i].replace("_", " ").title()[:12]
        dots += (
            f'<text x="{lx:.1f}" y="{ly:.1f}" text-anchor="{anchor}" '
            f'fill="rgba(255,255,255,0.55)" font-size="10">{short}</text>'
        )

    return (
        f'<svg viewBox="0 0 {size} {size}" width="{size}" height="{size}" '
        f'xmlns="http://www.w3.org/2000/svg">'
        f"{grid}{axes}{data_poly}{dots}</svg>"
    )


def _emotion_icon(emotion: str) -> str:
    """Map an emotional context string to a small emoji."""
    em = emotion.lower()
    icons: dict[str, str] = {
        "enthusiastic": "✨", "curious": "🔍", "thoughtful": "💭",
        "apologetic": "🙏", "resistant": "🛡️", "neutral": "○",
        "happy": "😊", "sad": "😢", "angry": "😠", "excited": "🎉",
    }
    return icons.get(em, "○")


def _render_chat_bubble(
    speaker_name: str,
    dialogue: str,
    pre_thought: str,
    internal: str,
    emotional: str,
    color: str,
    *,
    is_user: bool = False,
) -> str:
    """Build the HTML string for a single chat bubble."""
    icon = _emotion_icon(emotional)
    badge = '<span class="user-badge">YOU</span>' if is_user else ""
    html = (
        f'<div class="chat-bubble" style="border-left:3px solid {color};">'
        f'<div class="chat-speaker" style="color:{color}">'
        f"{speaker_name}{badge}</div>"
    )
    if pre_thought:
        html += f'<div class="chat-thought">💭 {pre_thought}</div>'
    html += dialogue
    if internal:
        html += f'<div class="chat-thought">🧠 {internal}</div>'
    html += f'<div class="chat-mood">{icon} {emotional}</div></div>'
    return html


def _parse_csv(txt: str) -> list[str]:
    """Split comma-separated text into a trimmed list."""
    return [s.strip() for s in txt.split(",") if s.strip()]


# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------

# Keys used by the step-wise scene state machine:
#   scene_active         - bool, True while a scene is running
#   scene_context        - DialogueContext
#   scene_record         - InteractionRecord
#   scene_provider       - LLMProvider
#   scene_loop           - asyncio event loop
#   scene_system         - DialogueSystem
#   scene_exchanges      - list of rendered exchange dicts for display
#   scene_waiting        - True when paused between steps
#   scene_mode           - "stepwise" | "continuous"
#   scene_max            - int, exchange cap configured at start
#   scene_start_time     - float
#   scene_initial_configs - dict[name, config_snapshot] frozen at scene start
#   scene_counter        - int, monotonically-increasing scene id for widget keys


# ---------------------------------------------------------------------------
# Provider model presets -- used by the welcome gate & settings page
# ---------------------------------------------------------------------------

_PROVIDER_PRESETS: dict[str, dict[str, str]] = {
    "google": {
        "label": "Google (Gemini)",
        "default_model": "gemini-2.0-flash",
        "key_hint": "Starts with AIza…",
        "signup_url": "https://aistudio.google.com/apikey",
    },
    "openai": {
        "label": "OpenAI",
        "default_model": "gpt-4o-mini",
        "key_hint": "Starts with sk-…",
        "signup_url": "https://platform.openai.com/api-keys",
    },
    "anthropic": {
        "label": "Anthropic (Claude)",
        "default_model": "claude-sonnet-4-20250514",
        "key_hint": "Starts with sk-ant-…",
        "signup_url": "https://console.anthropic.com/settings/keys",
    },
}

_PROVIDER_IDS = list(_PROVIDER_PRESETS.keys())


def _active_provider() -> str:
    """Return the LLM provider id currently in effect (session > settings)."""
    return str(
        st.session_state.get("llm_provider", settings.llm_provider)
    ).lower()


def _active_model() -> str:
    """Return the LLM model currently in effect (session > settings)."""
    return str(
        st.session_state.get("llm_model", settings.default_model)
    )


def _active_api_key() -> str:
    """Return the API key for the active provider.

    Prefers the user-supplied key in session state; falls back to the
    env-file key only when running locally **and** a key is configured.
    This ensures a public deployment never leaks server-side secrets.
    """
    # Session-state key always wins
    session_key = st.session_state.get("llm_api_key", "")
    if session_key:
        return str(session_key)
    # Fallback to env -- safe only in local dev
    if settings.is_local:
        provider = _active_provider()
        env_map: dict[str, str | None] = {
            "anthropic": settings.anthropic_api_key,
            "openai": settings.openai_api_key,
            "google": settings.google_api_key,
        }
        return env_map.get(provider) or ""
    return ""


def _init_state() -> None:
    """Ensure all session-state keys exist."""
    if "db_init" not in st.session_state:
        initialize_default_characters()
        st.session_state.db_init = True
    st.session_state.setdefault("scene_active", False)
    st.session_state.setdefault("scene_exchanges", [])
    st.session_state.setdefault("scene_waiting", False)
    st.session_state.setdefault("scene_mode", "stepwise")
    st.session_state.setdefault("scene_milestone_reviews", {})
    st.session_state.setdefault("onboarded", False)
    # LLM config -- session-state overrides for runtime safety
    st.session_state.setdefault("llm_provider", settings.llm_provider)
    st.session_state.setdefault("llm_model", settings.default_model)
    st.session_state.setdefault("llm_api_key", "")  # never persisted
    # metrics collector for the dashboard (file-backed)
    if "metrics_collector" not in st.session_state:
        st.session_state.metrics_collector = MetricsCollector(
            store_path=settings.metrics_path,
        )
    # user profile
    if "user_profile" not in st.session_state:
        st.session_state.user_profile = _load_user_profile()
    # mark onboarded if user has previously set a real name & has an API key
    if (
        not st.session_state.onboarded
        and st.session_state.user_profile.get("name", "You") != "You"
        and _active_api_key()
    ):
        st.session_state.onboarded = True


# ---------------------------------------------------------------------------
# LLM helpers
# ---------------------------------------------------------------------------


def _get_provider() -> Any:
    """Create and return an LLM provider using the active runtime config.

    Prefers session-state values (set by the user at onboarding / settings)
    over file-based settings, ensuring a public deployment never reads
    server-side keys.
    """
    provider_id = _active_provider()
    api_key = _active_api_key()
    if not api_key:
        msg = (
            f"No API key configured for provider '{provider_id}'. "
            "Please set one in Settings."
        )
        raise ValueError(msg)
    return get_llm_provider(provider_id, api_key=api_key)

def _generate_exchange(
    loop: asyncio.AbstractEventLoop,
    system: DialogueSystem,
    context: DialogueContext,
    speaker: Character,
) -> tuple[str, str, str, str]:
    """Run one LLM exchange and return (pre_thought, dialogue, internal, emotional)."""
    pre_thought, dialogue, internal = loop.run_until_complete(
        system.generate_response(context, speaker),
    )
    emotional = loop.run_until_complete(
        system.infer_emotional_context(speaker, dialogue),
    )
    return pre_thought, dialogue, internal, emotional


def _run_post_exchange_hooks(
    loop: asyncio.AbstractEventLoop,
    system: DialogueSystem,
    speaker: Character,
    dialogue: str,
    emotional: str,
    exchange_index: int,
) -> None:
    """Run evolution subsystems after each exchange.

    Mirrors the hooks in ``DialogueSystem.generate_sequential_dialogue``:
    memory condensation, experience-based trait shifts, self-reflection,
    and dissonance detection.
    """
    # Memory condensation
    if speaker.memory_store.working_is_full:
        loop.run_until_complete(
            system.memory_condenser.trigger_condensation(
                speaker.memory_store, speaker,
            ),
        )

    # Experience-based trait micro-shifts
    loop.run_until_complete(
        system.trait_shift_engine.process_exchange(
            speaker, dialogue, emotional, exchange_index,
        ),
    )

    # Periodic self-reflection
    loop.run_until_complete(
        system.self_reflection_engine.maybe_reflect(speaker, exchange_index),
    )

    # Periodic dissonance detection
    loop.run_until_complete(
        system.dissonance_detector.maybe_detect(speaker, exchange_index),
    )


def _add_exchange_to_state(  # noqa: PLR0913
    context: DialogueContext,
    record: InteractionRecord,
    speaker: Character,
    dialogue: str,
    pre_thought: str,
    internal: str,
    emotional: str,
    *,
    is_user: bool = False,
) -> dict[str, Any]:
    """Record an exchange in context, record, and build display dict."""
    context.add_exchange(
        speaker=speaker,
        text=dialogue,
        emotional_context=emotional,
        pre_exchange_thought=pre_thought,
        internal_thought=internal,
    )
    record.add_exchange(
        speaker=speaker.name,
        text=dialogue,
        emotional_context=emotional,
        pre_exchange_thought=pre_thought,
        internal_thought=internal,
    )

    # Track config changes for audit trail
    initial_configs = st.session_state.get("scene_initial_configs", {})
    initial = initial_configs.get(speaker.name, {})
    if initial and _is_character_modified(speaker, initial):
        diff = _get_character_diff(speaker, initial)
        config_changes: list[dict[str, Any]] = record.metadata.get(
            "config_changes", [],
        )
        config_changes.append({
            "exchange_idx": record.exchange_count,
            "character": speaker.name,
            "timestamp": time.time(),
            "changes": diff,
            "config_snapshot": {
                "personality": speaker.personality.to_dict(),
                "background": speaker.background.to_dict(),
                "description": speaker.description,
                "emotional_state": str(speaker.current_emotional_state),
            },
        })
        record.metadata["config_changes"] = config_changes

    return {
        "speaker": speaker.name,
        "text": dialogue,
        "pre_thought": pre_thought,
        "internal": internal,
        "emotional": emotional,
        "is_user": is_user,
    }


def _finish_scene() -> None:
    """Cleanly tear down the active scene and persist the record."""
    loop = st.session_state.get("scene_loop")
    provider = st.session_state.get("scene_provider")
    record: InteractionRecord | None = st.session_state.get("scene_record")
    system: DialogueSystem | None = st.session_state.get("scene_system")
    ctx: DialogueContext | None = st.session_state.get("scene_context")
    trait_snapshots: dict[str, dict[str, float]] = st.session_state.get(
        "scene_trait_snapshots", {},
    )

    # Run milestone reviews before closing the event loop and capture results
    if system and ctx and loop and not loop.is_closed() and ctx.exchanges:
        try:
            reviews = loop.run_until_complete(
                system.run_milestone_reviews(ctx, trait_snapshots),
            )
            st.session_state.scene_milestone_reviews = {
                r.character_name: r for r in reviews
            }
        except Exception as exc:  # noqa: BLE001
            logger.warning("Milestone reviews failed: %s", exc)
            st.session_state.scene_milestone_reviews = {}

    # Persist evolved characters back to the database
    if ctx:
        for character in ctx.characters:
            character_repository.update(character)

    if provider and loop and not loop.is_closed():
        with contextlib.suppress(Exception):
            loop.run_until_complete(provider.close())
    if loop and not loop.is_closed():
        loop.close()
    if record:
        # Store final character configs for complete audit trail
        ctx: DialogueContext | None = st.session_state.get("scene_context")
        if ctx:
            record.metadata["final_configs"] = {
                c.name: {
                    "personality": c.personality.to_dict(),
                    "background": c.background.to_dict(),
                    "description": c.description,
                    "emotional_state": str(c.current_emotional_state),
                }
                for c in ctx.characters
            }
        record.finish()
        _interaction_repo.save(record)

    st.session_state.scene_active = False
    st.session_state.scene_waiting = False


# ---------------------------------------------------------------------------
# Pages
# ---------------------------------------------------------------------------


def page_stage() -> None:
    """The Stage -- hero landing page."""
    st.markdown(
        '<div class="stage-header">'
        "<h1>Character Creator</h1>"
        "<p>Craft personalities. Set the scene. Watch them come alive.</p>"
        "</div>",
        unsafe_allow_html=True,
    )

    all_chars = character_repository.list_all()
    recent = _interaction_repo.list_all()[:5]

    col_l, col_r = st.columns([3, 2], gap="large")

    with col_l:
        st.markdown("### 🎬 Quick Start")
        st.markdown(
            "Select characters, describe a scene, and watch AI-powered dialogue "
            "unfold in real time -- personality-driven, emotionally aware, and "
            "completely unique every time.  You can even jump into the "
            "conversation yourself!",
        )

        st.markdown("##### Your Cast")
        if not all_chars:
            st.info(
                "No characters yet -- head to the **Workshop** to create some.",
            )
        else:
            cols = st.columns(min(len(all_chars), 3))
            for i, char in enumerate(all_chars[:6]):
                c = _color_for(char.name)
                with cols[i % len(cols)]:
                    desc_preview = (
                        char.description[:90] + "…"
                        if len(char.description) > 90
                        else char.description
                    )
                    st.markdown(
                        '<div class="char-card">'
                        f"<h4>{char.name}</h4>"
                        f"<p>{desc_preview}</p>"
                        f'<span class="stat-pill" style="border-color:{c}">'
                        f"Assert {char.personality.traits.assertiveness:.1f}"
                        "</span>"
                        f'<span class="stat-pill" style="border-color:{c}">'
                        f"Warmth {char.personality.traits.warmth:.1f}"
                        "</span>"
                        "</div>",
                        unsafe_allow_html=True,
                    )

    with col_r:
        st.markdown("### 📜 Recent Sessions")
        if not recent:
            st.caption(
                "No dialogue sessions yet. "
                "Head to the **Scene** page to start one!",
            )
        for rec in recent:
            dur = f"{rec.duration_seconds:.1f}s" if rec.duration_seconds else "-"
            st.markdown(
                '<div class="char-card">'
                f'<h4>{rec.topic or "Untitled"}</h4>'
                f"<p>{'  ·  '.join(rec.characters)} -- "
                f"{rec.exchange_count} exchanges · {dur}</p>"
                "</div>",
                unsafe_allow_html=True,
            )

    st.divider()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Characters", len(all_chars))
    c2.metric("Sessions", _interaction_repo.count())
    total_ex = sum(r.exchange_count for r in _interaction_repo.list_all())
    c3.metric("Total Exchanges", total_ex)
    c4.metric("Provider", _active_provider().title())


# ---------------------------------------------------------------------------
# Cast (character browser)
# ---------------------------------------------------------------------------


def _render_personality_tab(char: Character, color: str) -> None:
    """Render the Personality tab content for a character."""
    traits_dict = char.personality.traits.to_dict()
    pc, vc = st.columns(2)
    with pc:
        st.markdown("##### Trait Profile")
        for trait_name, val in traits_dict.items():
            st.markdown(
                _trait_bar(trait_name.replace("_", " ").title(), val, color),
                unsafe_allow_html=True,
            )
    with vc:
        vals = char.personality.values
        st.markdown("##### Core Values")
        if vals.priority_keywords:
            st.markdown(
                " · ".join(f"**{k}**" for k in vals.priority_keywords),
            )
        if vals.beliefs:
            for b in vals.beliefs:
                st.markdown(f"- {b}")
        if vals.strengths:
            st.markdown("**Strengths:** " + ", ".join(vals.strengths))
        if vals.weaknesses:
            st.markdown("**Weaknesses:** " + ", ".join(vals.weaknesses))

    if char.personality.speech_patterns or char.personality.quirks:
        st.markdown("##### Voice & Mannerisms")
        sp_col, q_col = st.columns(2)
        with sp_col:
            for pat in char.personality.speech_patterns:
                st.markdown(f"🗣️ {pat}")
        with q_col:
            for q in char.personality.quirks:
                st.markdown(f"✦ {q}")


def _render_background_tab(char: Character) -> None:
    """Render the Background tab content for a character."""
    bg = char.background
    bl, br = st.columns(2)
    with bl:
        if bg.motivations:
            st.markdown("##### Motivations")
            for m in bg.motivations:
                st.markdown(f"🎯 {m}")
        if bg.fears:
            st.markdown("##### Fears")
            for f_ in bg.fears:
                st.markdown(f"😨 {f_}")
    with br:
        if bg.desires:
            st.markdown("##### Desires")
            for d in bg.desires:
                st.markdown(f"💫 {d}")
        if bg.relationships:
            st.markdown("##### Relationships")
            for rel_name, rel_desc in bg.relationships.items():
                st.markdown(f"**{rel_name}:** {rel_desc}")


def page_cast() -> None:
    """Browse and inspect characters -- radar, values, background."""
    st.markdown("## 🎭 The Cast")
    all_chars = character_repository.list_all()

    if not all_chars:
        st.info("No characters yet. Create one in the **Workshop**.")
        return

    names = [c.name for c in all_chars]
    selected_name = st.selectbox(
        "Choose a character", names, label_visibility="collapsed",
    )
    char = character_repository.read(selected_name)
    if not char:
        return

    color = _color_for(char.name)

    h_left, h_right = st.columns([2, 1])
    with h_left:
        st.markdown(f"### {char.name}")
        st.caption(char.description)
        st.markdown(
            f'<span class="stat-pill">Age {char.background.age}</span>'
            f'<span class="stat-pill">{char.background.occupation or "--"}</span>'
            f'<span class="stat-pill">{char.background.origin or "--"}</span>',
            unsafe_allow_html=True,
        )
    with h_right:
        traits_dict = char.personality.traits.to_dict()
        st.markdown(
            _radar_svg(traits_dict, size=210, color=color),
            unsafe_allow_html=True,
        )

    tab_personality, tab_background, tab_inner = st.tabs(
        ["Personality", "Background", "Inner World"],
    )
    with tab_personality:
        _render_personality_tab(char, color)
    with tab_background:
        _render_background_tab(char)
    with tab_inner:
        st.markdown("##### Current Emotional State")
        st.markdown(f"*{char.current_emotional_state.base.title()}*")
        st.markdown("##### Self-Perception")
        st.text(char.get_character_self_perception())

    st.divider()
    if st.button("🗑️ Delete this character", type="secondary"):
        character_repository.delete(selected_name)
        st.rerun()


# ---------------------------------------------------------------------------
# Scene  (step-wise + continuous dialogue)
# ---------------------------------------------------------------------------


def _is_character_modified(
    char: Character, initial: dict[str, Any],
) -> bool:
    """Check if user-configurable fields differ from the initial snapshot."""
    if not initial:
        return False
    if char.personality.to_dict() != initial.get("personality", {}):
        return True
    if char.background.to_dict() != initial.get("background", {}):
        return True
    if char.description != initial.get("description", ""):
        return True
    return str(char.current_emotional_state) != initial.get("emotional_state", "neutral")


def _get_character_diff(
    char: Character, initial: dict[str, Any],
) -> list[str]:
    """Return a human-readable list of changed fields."""
    changes: list[str] = []
    if not initial:
        return changes
    # Traits
    cur_traits = char.personality.traits.to_dict()
    ini_traits = initial.get("personality", {}).get("traits", {})
    changes.extend(
        f"{k}: {ini_traits.get(k, 0):.2f}→{cur_traits[k]:.2f}"
        for k in cur_traits
        if abs(cur_traits.get(k, 0) - ini_traits.get(k, 0)) > 0.01
    )
    # Values
    cur_vals = char.personality.values.to_dict()
    ini_vals = initial.get("personality", {}).get("values", {})
    changes.extend(
        f"values.{k}"
        for k in cur_vals
        if cur_vals.get(k) != ini_vals.get(k)
    )
    # Background
    cur_bg = char.background.to_dict()
    ini_bg = initial.get("background", {})
    changes.extend(
        f"background.{k}"
        for k in ("age", "origin", "occupation", "motivations", "fears", "desires")
        if cur_bg.get(k) != ini_bg.get(k)
    )
    if char.description != initial.get("description", ""):
        changes.append("description")
    if str(char.current_emotional_state) != initial.get("emotional_state", "neutral"):
        changes.append(f"emotional_state->{char.current_emotional_state}")
    return changes


_EMOTIONAL_OPTIONS = list(EMOTIONAL_STATES)


def _render_setup_cast_preview(
    selected_names: list[str], all_chars: list[Character],
) -> None:
    """Editable cast sidebar on the Scene setup page.

    Every parameter that influences dialogue behaviour is configurable here --
    trait sliders, emotional state, values, and background fields.  Changes
    are written back to the repository so they persist into the scene and are
    flagged with a *Modified* badge plus a save-as option.
    """
    selected = [c for c in all_chars if c.name in selected_names]
    if not selected:
        st.caption("Select characters to preview their profiles.")
        return

    # Snapshot originals on first render so we can detect changes
    if "setup_initial_configs" not in st.session_state:
        st.session_state.setup_initial_configs = {}
    initial_configs: dict[str, dict[str, Any]] = st.session_state.setup_initial_configs
    for char in selected:
        if char.name not in initial_configs:
            initial_configs[char.name] = {
                "personality": char.personality.to_dict(),
                "background": char.background.to_dict(),
                "description": char.description,
                "emotional_state": str(char.current_emotional_state),
            }

    st.markdown(
        '<div style="font-size:0.85rem;opacity:0.6;margin-bottom:0.5rem">'
        '🎭 <b>SELECTED CAST</b> '
        '<span style="font-size:0.7rem;opacity:0.7">(✏️ editable)</span>'
        "</div>",
        unsafe_allow_html=True,
    )

    for char in selected:
        color = _color_for(char.name)
        kp = f"setup_{char.name.replace(' ', '_')}"
        initial = initial_configs.get(char.name, {})
        is_modified = _is_character_modified(char, initial)
        badge = " ⚠️" if is_modified else ""

        with st.expander(f"{char.name}{badge}", expanded=False):
            # ── Description ───────────────────────────────────────
            st.caption(char.description)

            # ── Personality trait sliders ──────────────────────────
            st.markdown(
                '<div style="font-size:0.78rem;opacity:0.6;margin-top:0.3rem">'
                "<b>Personality Traits</b></div>",
                unsafe_allow_html=True,
            )
            traits_dict = char.personality.traits.to_dict()
            for t_name, t_val in traits_dict.items():
                new_val = st.slider(
                    t_name.replace("_", " ").title(),
                    min_value=0.0,
                    max_value=1.0,
                    value=t_val,
                    step=0.05,
                    key=f"{kp}_{t_name}",
                )
                setattr(char.personality.traits, t_name, new_val)

            # Dynamic radar chart
            st.markdown(
                _radar_svg(char.personality.traits.to_dict(), size=140, color=color),
                unsafe_allow_html=True,
            )

            # ── Emotional state ───────────────────────────────────
            emo_idx = (
                _EMOTIONAL_OPTIONS.index(char.current_emotional_state.base)
                if char.current_emotional_state.base in _EMOTIONAL_OPTIONS
                else 0
            )
            new_emo = st.selectbox(
                "Emotional State",
                _EMOTIONAL_OPTIONS,
                index=emo_idx,
                key=f"{kp}_emotional",
            )
            char.update_emotional_state(new_emo)

            # ── Values ────────────────────────────────────────────
            st.markdown(
                '<div style="font-size:0.78rem;opacity:0.6;margin-top:0.5rem">'
                "<b>Values</b></div>",
                unsafe_allow_html=True,
            )
            vals = char.personality.values

            new_keywords = st.text_input(
                "Core Values",
                value=", ".join(vals.priority_keywords),
                key=f"{kp}_values",
            )
            vals.priority_keywords = _parse_csv(new_keywords)

            new_strengths = st.text_input(
                "Strengths",
                value=", ".join(vals.strengths),
                key=f"{kp}_strengths",
            )
            vals.strengths = _parse_csv(new_strengths)

            new_weaknesses = st.text_input(
                "Weaknesses",
                value=", ".join(vals.weaknesses),
                key=f"{kp}_weaknesses",
            )
            vals.weaknesses = _parse_csv(new_weaknesses)

            # ── Background ────────────────────────────────────────
            st.markdown(
                '<div style="font-size:0.78rem;opacity:0.6;margin-top:0.5rem">'
                "<b>Background</b></div>",
                unsafe_allow_html=True,
            )
            bg = char.background

            new_age = st.number_input(
                "Age", value=bg.age, min_value=1, max_value=200,
                key=f"{kp}_age",
            )
            bg.age = new_age

            new_occ = st.text_input(
                "Occupation", value=bg.occupation,
                key=f"{kp}_occupation",
            )
            bg.occupation = new_occ

            new_motivations = st.text_input(
                "Motivations",
                value=", ".join(bg.motivations),
                key=f"{kp}_motivations",
            )
            bg.motivations = _parse_csv(new_motivations)

            # ── Modified indicator & save ─────────────────────────
            if is_modified:
                diff = _get_character_diff(char, initial)
                st.markdown(
                    '<div style="margin:0.5rem 0">'
                    '<span style="background:#ff6b35;color:white;padding:2px 8px;'
                    'border-radius:10px;font-size:0.75rem">⚠️ Modified</span>'
                    "</div>",
                    unsafe_allow_html=True,
                )
                if diff:
                    st.caption("Changed: " + ", ".join(diff[:5]))
                save_name = st.text_input(
                    "Save as",
                    value=f"{char.name} (edited)",
                    key=f"{kp}_save_name",
                )
                if st.button(
                    "💾 Save to Repository",
                    key=f"{kp}_save",
                ):
                    existing = character_repository.read(save_name)
                    if existing:
                        st.warning(
                            f"'{save_name}' already exists -- "
                            "choose a different name.",
                        )
                    else:
                        new_char = Character(
                            name=save_name,
                            description=char.description,
                            personality=Personality.from_dict(
                                char.personality.to_dict(),
                            ),
                            background=Background.from_dict(
                                char.background.to_dict(),
                            ),
                        )
                        character_repository.create(new_char)
                        st.success(f"Saved '{save_name}'!")

            # Persist edits back to repository so _start_scene picks them up
            character_repository.update(char)


def _render_cast_panel(characters: list[Character]) -> None:  # noqa: PLR0912
    """Render an interactive cast panel with editable character profiles.

    Trait sliders, values inputs, and background fields are live-editable.
    Changes are synced back to the Character objects used by the dialogue
    engine, and flagged with a *Modified* badge + save-to-repository option.
    Characters with AI-driven trait evolutions show a rich evolution panel
    with narrative summary and per-trait delta bars.
    """
    scene_num = st.session_state.get("scene_counter", 0)
    initial_configs = st.session_state.get("scene_initial_configs", {})
    milestone_reviews = st.session_state.get("scene_milestone_reviews", {})

    st.markdown(
        '<div style="font-size:0.85rem;opacity:0.6;margin-bottom:0.5rem">'
        '🎭 <b>CAST</b> '
        '<span style="font-size:0.7rem;opacity:0.7">(✏️ editable)</span>'
        "</div>",
        unsafe_allow_html=True,
    )

    for char in characters:
        color = _color_for(char.name)
        kp = f"s{scene_num}_{char.name.replace(' ', '_')}"
        initial = initial_configs.get(char.name, {})
        is_modified = _is_character_modified(char, initial)

        # Detect AI-driven trait evolutions (compare to scene-start snapshot)
        initial_traits = initial.get("personality", {}).get("traits", {})
        cur_traits = char.personality.traits.to_dict()
        has_evolution = initial_traits and any(
            abs(cur_traits.get(k, 0.0) - initial_traits.get(k, 0.0)) > 0.005
            for k in cur_traits
        )
        review = milestone_reviews.get(char.name)
        has_arc = has_evolution or (review is not None and bool(review.narrative_summary))

        if has_arc:
            badge = " 🌱"
        elif is_modified:
            badge = " ⚠️"
        else:
            badge = ""

        # Stable expander key (does not include badge so state survives badge changes)
        expander_key = f"{kp}_expander"
        # Auto-expand once when evolutions are first detected; preserve user state
        # thereafter.  st.session_state[expander_key] is managed by Streamlit after
        # the first render.
        if expander_key not in st.session_state:
            # First render in this scene — open if there are already evolutions
            st.session_state[expander_key] = bool(has_arc)
        elif has_arc and not st.session_state.get(f"{expander_key}_arc_shown"):
            # Evolutions just appeared — auto-expand once, then leave user in control
            st.session_state[expander_key] = True
        if has_arc:
            st.session_state[f"{expander_key}_arc_shown"] = True

        with st.expander(f"{char.name}{badge}", key=expander_key):
            # ── Evolution panel (shown first when evolutions are present) ──
            if has_arc:
                _render_evolution_panel(char, initial_traits, color, review)
                st.divider()

            # ── Description ───────────────────────────────────────────
            st.caption(char.description)

            # ── Personality trait sliders ──────────────────────────────
            st.markdown(
                '<div style="font-size:0.78rem;opacity:0.6;margin-top:0.3rem">'
                "<b>Personality Traits</b></div>",
                unsafe_allow_html=True,
            )
            traits_dict = char.personality.traits.to_dict()
            for t_name, t_val in traits_dict.items():
                new_val = st.slider(
                    t_name.replace("_", " ").title(),
                    min_value=0.0,
                    max_value=1.0,
                    value=t_val,
                    step=0.05,
                    key=f"{kp}_{t_name}",
                )
                setattr(char.personality.traits, t_name, new_val)

            # Dynamic radar chart (reflects current slider values)
            st.markdown(
                _radar_svg(char.personality.traits.to_dict(), size=140, color=color),
                unsafe_allow_html=True,
            )

            # ── Emotional state ───────────────────────────────────────
            emo_idx = (
                _EMOTIONAL_OPTIONS.index(char.current_emotional_state.base)
                if char.current_emotional_state.base in _EMOTIONAL_OPTIONS
                else 0
            )
            new_emo = st.selectbox(
                "Emotional State",
                _EMOTIONAL_OPTIONS,
                index=emo_idx,
                key=f"{kp}_emotional",
            )
            char.update_emotional_state(new_emo)

            # ── Values ────────────────────────────────────────────────
            st.markdown(
                '<div style="font-size:0.78rem;opacity:0.6;margin-top:0.5rem">'
                "<b>Values</b></div>",
                unsafe_allow_html=True,
            )
            vals = char.personality.values

            new_keywords = st.text_input(
                "Core Values",
                value=", ".join(vals.priority_keywords),
                key=f"{kp}_values",
            )
            vals.priority_keywords = _parse_csv(new_keywords)

            new_strengths = st.text_input(
                "Strengths",
                value=", ".join(vals.strengths),
                key=f"{kp}_strengths",
            )
            vals.strengths = _parse_csv(new_strengths)

            new_weaknesses = st.text_input(
                "Weaknesses",
                value=", ".join(vals.weaknesses),
                key=f"{kp}_weaknesses",
            )
            vals.weaknesses = _parse_csv(new_weaknesses)

            # ── Background ────────────────────────────────────────────
            st.markdown(
                '<div style="font-size:0.78rem;opacity:0.6;margin-top:0.5rem">'
                "<b>Background</b></div>",
                unsafe_allow_html=True,
            )
            bg = char.background

            new_age = st.number_input(
                "Age", value=bg.age, min_value=1, max_value=200,
                key=f"{kp}_age",
            )
            bg.age = new_age

            new_occ = st.text_input(
                "Occupation", value=bg.occupation,
                key=f"{kp}_occupation",
            )
            bg.occupation = new_occ

            new_motivations = st.text_input(
                "Motivations",
                value=", ".join(bg.motivations),
                key=f"{kp}_motivations",
            )
            bg.motivations = _parse_csv(new_motivations)

            # ── Modified indicator & save ─────────────────────────────
            if is_modified:
                diff = _get_character_diff(char, initial)
                st.markdown(
                    '<div style="margin:0.5rem 0">'
                    '<span style="background:#ff6b35;color:white;padding:2px 8px;'
                    'border-radius:10px;font-size:0.75rem">⚠️ Modified</span>'
                    "</div>",
                    unsafe_allow_html=True,
                )
                # Only show non-trait manual changes in the caption (trait
                # evolutions are already visualised in the evolution panel above)
                # Trait diff entries have the format "trait_name: old→new"
                manual_changes = [
                    d for d in diff
                    if not any(d.startswith(f"{t}:") for t in cur_traits)
                ]
                if manual_changes:
                    st.caption("Changed: " + ", ".join(manual_changes[:5]))
                save_name = st.text_input(
                    "Save as",
                    value=f"{char.name} (edited)",
                    key=f"{kp}_save_name",
                )
                if st.button(
                    "💾 Save to Repository",
                    key=f"{kp}_save",
                ):
                    existing = character_repository.read(save_name)
                    if existing:
                        st.warning(
                            f"'{save_name}' already exists -- "
                            "choose a different name.",
                        )
                    else:
                        new_char = Character(
                            name=save_name,
                            description=char.description,
                            personality=Personality.from_dict(
                                char.personality.to_dict(),
                            ),
                            background=Background.from_dict(
                                char.background.to_dict(),
                            ),
                        )
                        character_repository.create(new_char)
                        st.success(f"Saved '{save_name}'!")


def _render_scene_exchanges() -> None:
    """Render all exchanges accumulated so far."""
    for ex in st.session_state.scene_exchanges:
        c = _color_for(ex["speaker"])
        st.markdown(
            _render_chat_bubble(
                ex["speaker"], ex["text"], ex.get("pre_thought", ""), ex["internal"],
                ex["emotional"], c, is_user=ex.get("is_user", False),
            ),
            unsafe_allow_html=True,
        )


def _start_scene(
    selected_names: list[str],
    scene_desc: str,
    topic: str,
    num_exchanges: int,
    mode: str,
) -> None:
    """Initialise a new scene in session state."""
    characters = [character_repository.read(n) for n in selected_names]
    characters = [c for c in characters if c]

    provider = _get_provider()
    loop = asyncio.new_event_loop()

    context = DialogueContext(
        characters=characters,
        scene_description=scene_desc,
        topic=topic,
    )

    # Snapshot initial character configs for change detection & audit
    initial_configs = {
        c.name: {
            "personality": c.personality.to_dict(),
            "background": c.background.to_dict(),
            "description": c.description,
            "emotional_state": str(c.current_emotional_state),
        }
        for c in characters
    }

    # Snapshot trait values for milestone reviews at scene end
    trait_snapshots = {
        c.name: c.personality.traits.to_dict() for c in characters
    }

    record = InteractionRecord(
        scene_description=scene_desc,
        topic=topic,
        characters=[c.name for c in characters],
        provider=_active_provider(),
        model=_active_model(),
        max_exchanges=num_exchanges,
        metadata={
            "initial_configs": initial_configs,
            "config_changes": [],
        },
    )

    scene_num = st.session_state.get("scene_counter", 0) + 1

    st.session_state.update({
        "scene_active": True,
        "scene_context": context,
        "scene_record": record,
        "scene_provider": provider,
        "scene_loop": loop,
        "scene_system": DialogueSystem(
            st.session_state.metrics_collector.wrap(provider),
        ),
        "scene_exchanges": [],
        "scene_waiting": mode == "stepwise",
        "scene_mode": mode,
        "scene_max": num_exchanges,
        "scene_start_time": time.time(),
        "scene_initial_configs": initial_configs,
        "scene_trait_snapshots": trait_snapshots,
        "scene_counter": scene_num,
        "scene_milestone_reviews": {},
    })


def _run_one_step(speaker: Character | None = None) -> None:
    """Execute a single AI exchange, optionally forcing a speaker."""
    ctx: DialogueContext = st.session_state.scene_context
    record: InteractionRecord = st.session_state.scene_record
    system: DialogueSystem = st.session_state.scene_system
    loop: asyncio.AbstractEventLoop = st.session_state.scene_loop

    if speaker is None:
        speaker = system.generate_next_speaker(ctx)

    exchange_index = len(ctx.exchanges)

    pre_thought, dialogue, internal, emotional = _generate_exchange(
        loop, system, ctx, speaker,
    )
    ex = _add_exchange_to_state(
        ctx, record, speaker, dialogue, pre_thought, internal, emotional,
    )
    st.session_state.scene_exchanges.append(ex)

    _run_post_exchange_hooks(
        loop, system, speaker, dialogue, emotional, exchange_index,
    )


def _run_user_step(user_char: Character, user_text: str) -> None:
    """Insert the user's own dialogue into the conversation."""
    ctx: DialogueContext = st.session_state.scene_context
    record: InteractionRecord = st.session_state.scene_record

    emotional = DialogueSystem._infer_emotional_context_heuristic(user_char, user_text)
    ex = _add_exchange_to_state(
        ctx, record, user_char, user_text, "", "", emotional,
        is_user=True,
    )
    st.session_state.scene_exchanges.append(ex)


def page_director() -> None:
    """Set the scene & run dialogue -- step-wise or continuous."""
    st.markdown("## 🎬 Scene")
    st.caption(
        "Set the scene, choose your cast, and watch the dialogue unfold -- "
        "one step at a time or in a continuous flow.",
    )

    # ---- if a scene is already active, show it --------------------------
    if st.session_state.scene_active:
        _director_active_scene()
        return

    # ---- scene setup -----------------------------------------------------
    all_chars = character_repository.list_all()
    if len(all_chars) < 2:
        st.warning(
            "You need at least **2 characters** to run a scene. "
            "Head to the **Workshop**.",
        )
        return

    names = [c.name for c in all_chars]

    # Two-column layout: scene setup on the left (70%), cast sidebar on the right (30%)
    setup_col, cast_preview_col = st.columns([7, 3], gap="large")

    with setup_col:
        with st.container(border=True):
            selected_names = st.multiselect(
                "Cast",
                names,
                default=names[:3] if len(names) >= 3 else names[:2],
                help="Pick 2-5 characters for the scene.",
            )
            scene_desc = st.text_area(
                "Scene",
                value=(
                    "A cozy, dimly-lit café on a rainy evening. Mismatched "
                    "chairs surround small wooden tables, the aroma of fresh "
                    "coffee fills the air, and soft jazz plays in the background."
                ),
                height=90,
            )
            topic = st.text_input(
                "Topic",
                value="What does it mean to live a meaningful life?",
            )
            mode = st.radio(
                "Mode",
                ["Step-wise (default)", "Continuous"],
                horizontal=True,
                help=(
                    "**Step-wise**: pause after each exchange -- you can pick "
                    "the next speaker or speak yourself.  "
                    "**Continuous**: generate all exchanges end-to-end."
                ),
            )
            num_exchanges = st.slider(
                "Max exchanges", 1, MAX_EXCHANGES, 20,
                help=f"Exchange cap (max {MAX_EXCHANGES}).",
            )
            st.markdown(
                f'<span class="stat-pill">Provider: {_active_provider()}</span>'
                f'<span class="stat-pill">Model: {_active_model()}</span>',
                unsafe_allow_html=True,
            )

        can_start = (
            len(selected_names) >= 2
            and bool(scene_desc.strip())
            and bool(topic.strip())
        )

        if st.button(
            "🎬  Action!",
            type="primary",
            use_container_width=True,
            disabled=not can_start,
        ):
            resolved_mode = (
                "continuous" if mode.startswith("Continuous") else "stepwise"
            )
            try:
                _start_scene(
                    selected_names, scene_desc, topic,
                    num_exchanges, resolved_mode,
                )
            except Exception as exc:  # noqa: BLE001
                st.error(f"Could not start scene: {exc}")
                st.stop()

            if resolved_mode == "continuous":
                _director_run_continuous()
            else:
                st.rerun()

        # ---- recent sessions ---------------------------------------------
        _render_recent_sessions()

    with cast_preview_col:
        _render_setup_cast_preview(selected_names, all_chars)


def _director_active_scene() -> None:
    """Render the in-progress step-wise scene UI."""
    ctx: DialogueContext = st.session_state.scene_context
    record: InteractionRecord = st.session_state.scene_record
    exchanges_so_far = len(st.session_state.scene_exchanges)
    max_ex = st.session_state.scene_max

    # header info (full width)
    st.markdown(
        f'<span class="stat-pill">Scene: {ctx.scene_description[:60]}…</span>'
        f'<span class="stat-pill">Topic: {ctx.topic[:40]}</span>'
        f'<span class="stat-pill">'
        f"{exchanges_so_far}/{max_ex} exchanges</span>",
        unsafe_allow_html=True,
    )
    st.progress(min(exchanges_so_far / max(max_ex, 1), 1.0))

    # Two-column layout: dialogue on the left (70%), cast panel on the right (30%)
    scene_col, cast_col = st.columns([7, 3], gap="large")

    with cast_col:
        _render_cast_panel(ctx.characters)

    with scene_col:
        # display conversation so far
        _render_scene_exchanges()

        # check if done
        if exchanges_so_far >= max_ex:
            _finish_scene()
            elapsed = time.time() - st.session_state.get("scene_start_time", 0)
            st.success(
                f"✨ Scene complete -- {record.exchange_count} exchanges "
                f"in {elapsed:.1f}s",
            )
            if st.button("🔄 New Scene"):
                st.rerun()
            _render_recent_sessions()
            return

        st.divider()

        # --- step controls ---
        system: DialogueSystem = st.session_state.scene_system
        suggested = system.generate_next_speaker(ctx)
        char_names = [c.name for c in ctx.characters]

        st.markdown(
            f"**Next up:** {suggested.name} "
            f"*(suggested by personality engine)*",
        )

        ctrl_col1, ctrl_col2, ctrl_col3 = st.columns([2, 2, 1])

        with ctrl_col1:
            override = st.selectbox(
                "Or pick a speaker",
                ["(suggested) " + suggested.name, *char_names],
                key="step_speaker_override",
            )

        with ctrl_col3:
            end_btn = st.button(
                "⏹️ End Scene",
                use_container_width=True,
            )

        with ctrl_col2:
            play_btn = st.button(
                "▶️  Next Exchange",
                type="primary",
                use_container_width=True,
            )

        # user's own input
        st.markdown("---")
        user_profile = st.session_state.user_profile
        user_char = _build_user_character(user_profile)
        st.markdown(
            f"**Or speak as {user_profile['name']}** "
            f'<span class="user-badge">YOU</span>',
            unsafe_allow_html=True,
        )
        user_input = st.text_input(
            "Your reply",
            placeholder="Type your own dialogue here and press Enter…",
            key="step_user_input",
            label_visibility="collapsed",
        )
        send_btn = st.button("💬 Send as yourself", disabled=not user_input)

    # ---- handle actions (outside columns for proper rerun) ---------------
    if end_btn:
        _finish_scene()
        st.rerun()
        return

    if send_btn and user_input:
        # Ensure user character is in the context so others can address them
        if user_char.name not in [c.name for c in ctx.characters]:
            ctx.characters.append(user_char)
        _run_user_step(user_char, user_input)
        st.rerun()
        return

    if play_btn:
        # determine speaker
        speaker: Character | None = None
        if override and not override.startswith("(suggested)"):
            speaker = next(
                (c for c in ctx.characters if c.name == override), None,
            )
        try:
            with st.spinner(
                f"✨ {(speaker or suggested).name} is thinking…",
            ):
                _run_one_step(speaker)
        except Exception as exc:  # noqa: BLE001
            st.error(f"LLM error: {exc}")
        st.rerun()
        return


def _director_run_continuous() -> None:
    """Run all exchanges end-to-end (continuous mode)."""
    ctx: DialogueContext = st.session_state.scene_context
    record: InteractionRecord = st.session_state.scene_record
    system: DialogueSystem = st.session_state.scene_system
    loop: asyncio.AbstractEventLoop = st.session_state.scene_loop
    num = st.session_state.scene_max

    st.divider()

    # Two-column layout: dialogue stream on the left (70%), cast panel on the right (30%)
    scene_col, cast_col = st.columns([7, 3], gap="large")

    with cast_col:
        _render_cast_panel(ctx.characters)

    with scene_col:
        chat_container = st.container()
        status = st.empty()
        bar = st.progress(0)

        for i in range(num):
            bar.progress(i / num)
            speaker = system.generate_next_speaker(ctx)
            status.markdown(
                f"<div style='opacity:0.6;font-style:italic'>"
                f"[{i + 1}/{num}] {speaker.name} is thinking…</div>",
                unsafe_allow_html=True,
            )
            try:
                pre_thought, dialogue, internal, emotional = _generate_exchange(
                    loop, system, ctx, speaker,
                )
            except Exception as exc:  # noqa: BLE001
                status.error(
                    f"LLM error after {record.exchange_count} exchanges: {exc}",
                )
                break

            ex = _add_exchange_to_state(
                ctx, record, speaker, dialogue, pre_thought, internal, emotional,
            )
            st.session_state.scene_exchanges.append(ex)

            _run_post_exchange_hooks(
                loop, system, speaker, dialogue, emotional, i,
            )

            with chat_container:
                c = _color_for(speaker.name)
                st.markdown(
                    _render_chat_bubble(
                        speaker.name, dialogue, pre_thought, internal, emotional, c,
                    ),
                    unsafe_allow_html=True,
                )

        bar.progress(1.0)
        elapsed = time.time() - st.session_state.get("scene_start_time", 0)
        _finish_scene()
        status.success(
            f"✨ Scene complete -- {record.exchange_count} exchanges "
            f"in {elapsed:.1f}s",
        )


def _render_recent_sessions() -> None:
    """Show the last few persisted sessions in expanders."""
    st.divider()
    st.markdown("##### Recent Sessions")
    recents = _interaction_repo.list_all()[:5]
    if not recents:
        st.caption(
            "No sessions yet. Hit **Action!** above to create the first one.",
        )
    for rec in recents:
        dur = f"{rec.duration_seconds:.1f}s" if rec.duration_seconds else "-"
        label = (
            f"**{rec.topic or 'Untitled'}** -- "
            f"{'  ·  '.join(rec.characters)} -- "
            f"{rec.exchange_count} exchanges · {dur}"
        )
        with st.expander(label):
            for ex in rec.exchanges:
                c = _color_for(ex.get("speaker", ""))
                st.markdown(
                    _render_chat_bubble(
                        ex.get("speaker", "?"),
                        ex.get("text", ""),
                        ex.get("pre_exchange_thought", ""),
                        ex.get("internal_thought", ""),
                        ex.get("emotional_context", "neutral"),
                        c,
                    ),
                    unsafe_allow_html=True,
                )


# ---------------------------------------------------------------------------
# Workshop (character creation)
# ---------------------------------------------------------------------------


def page_workshop() -> None:
    """Create a new character from scratch."""
    st.markdown("## 🛠️ Workshop")
    st.caption("Forge a new character from scratch.")

    with st.container(border=True):
        wl, wr = st.columns(2)
        with wl:
            char_name = st.text_input(
                "Name", placeholder="e.g. Mira, Jax, Ophelia",
            )
            description = st.text_area(
                "Description",
                placeholder=(
                    "A sharp-eyed barista with paint-stained sleeves "
                    "and an ironic smile…"
                ),
                height=100,
            )
        with wr:
            age = st.number_input("Age", 1, 150, 30)
            origin = st.text_input(
                "Origin", placeholder="Brooklyn, New York",
            )
            occupation = st.text_input(
                "Occupation", placeholder="Freelance illustrator",
            )

    st.markdown("#### Personality")
    tc1, tc2, tc3, tc4 = st.columns(4)
    with tc1:
        assertiveness = st.slider(
            "Assertiveness", 0.0, 1.0, 0.5, key="ws_assert",
        )
        warmth = st.slider("Warmth", 0.0, 1.0, 0.5, key="ws_warmth")
    with tc2:
        openness = st.slider("Openness", 0.0, 1.0, 0.5, key="ws_open")
        conscientiousness = st.slider(
            "Conscientiousness", 0.0, 1.0, 0.5, key="ws_consc",
        )
    with tc3:
        emotional_stability = st.slider(
            "Emotional Stability", 0.0, 1.0, 0.5, key="ws_emot",
        )
        humor = st.slider("Humor", 0.0, 1.0, 0.5, key="ws_humor")
    with tc4:
        formality = st.slider(
            "Formality", 0.0, 1.0, 0.5, key="ws_formal",
        )
        preview_traits = {
            "assertiveness": assertiveness,
            "warmth": warmth,
            "openness": openness,
            "conscientiousness": conscientiousness,
            "emotional_stability": emotional_stability,
            "humor_inclination": humor,
            "formality": formality,
        }
        st.markdown(
            _radar_svg(preview_traits, size=150, color="#a78bfa"),
            unsafe_allow_html=True,
        )

    st.markdown("#### Values & Voice")
    vl, vr = st.columns(2)
    with vl:
        values_text = st.text_area(
            "Core Values (comma-separated)",
            placeholder="courage, honesty, curiosity", height=70,
        )
        beliefs_text = st.text_area(
            "Beliefs (comma-separated)",
            placeholder="Everyone deserves a second chance", height=70,
        )
        strengths_text = st.text_area(
            "Strengths", placeholder="empathy, quick thinking", height=60,
        )
        weaknesses_text = st.text_area(
            "Weaknesses", placeholder="stubbornness, overthinking", height=60,
        )
    with vr:
        dislikes_text = st.text_area(
            "Dislikes", placeholder="dishonesty, small talk", height=70,
        )
        speech_text = st.text_area(
            "Speech Patterns",
            placeholder="Uses dry humour, pauses mid-sentence", height=70,
        )
        quirks_text = st.text_area(
            "Quirks", placeholder="Taps pen when nervous", height=60,
        )
        motivations_text = st.text_area(
            "Motivations", placeholder="Create something lasting", height=60,
        )

    fears_text = st.text_input(
        "Fears", placeholder="mediocrity, abandonment",
    )
    desires_text = st.text_input(
        "Desires", placeholder="recognition, deep connection",
    )

    bc1, bc2 = st.columns(2)
    with bc1:
        create_btn = st.button(
            "✨ Create Character", type="primary", use_container_width=True,
        )
    with bc2:
        preview_btn = st.button("👁️ Preview", use_container_width=True)

    if create_btn or preview_btn:
        if not char_name or not description or len(description) < 10:
            st.error(
                "Please fill in at least a name and description (10+ chars).",
            )
            st.stop()

        char = _build_character_from_form(
            char_name, description, age, origin, occupation,
            assertiveness, warmth, openness, conscientiousness,
            emotional_stability, humor, formality,
            values_text, beliefs_text, dislikes_text,
            strengths_text, weaknesses_text, speech_text,
            quirks_text, motivations_text, fears_text, desires_text,
        )

        if preview_btn:
            with st.expander("📋 Character Profile", expanded=True):
                st.text(char.get_character_profile())
        else:
            try:
                character_repository.create(char)
                st.success(f"✨ **{char_name}** has joined the cast!")
                st.balloons()
                time.sleep(1)
                st.rerun()
            except ValueError as e:
                st.error(f"Error: {e}")


def _build_character_from_form(  # noqa: PLR0913
    name: str, description: str,
    age: int, origin: str, occupation: str,
    assertiveness: float, warmth: float, openness: float,
    conscientiousness: float, emotional_stability: float,
    humor: float, formality: float,
    values_text: str, beliefs_text: str, dislikes_text: str,
    strengths_text: str, weaknesses_text: str, speech_text: str,
    quirks_text: str, motivations_text: str,
    fears_text: str, desires_text: str,
) -> Character:
    """Construct a Character from workshop / profile form fields."""
    traits_req = PersonalityTraitsRequest(
        assertiveness=assertiveness, warmth=warmth, openness=openness,
        conscientiousness=conscientiousness,
        emotional_stability=emotional_stability,
        humor_inclination=humor, formality=formality,
    )
    values_req = ValuesRequest(
        priority_keywords=_parse_csv(values_text),
        beliefs=_parse_csv(beliefs_text),
        dislikes=_parse_csv(dislikes_text),
        strengths=_parse_csv(strengths_text),
        weaknesses=_parse_csv(weaknesses_text),
    )
    personality_req = PersonalityRequest(
        traits=traits_req, values=values_req,
        speech_patterns=_parse_csv(speech_text),
        quirks=_parse_csv(quirks_text),
    )
    background_req = BackgroundRequest(
        age=age, origin=origin, occupation=occupation,
        motivations=_parse_csv(motivations_text),
        fears=_parse_csv(fears_text),
        desires=_parse_csv(desires_text),
    )
    request = CharacterCreateRequest(
        name=name, description=description,
        personality=personality_req, background=background_req,
    )
    data = request.model_dump()
    return Character(
        name=data["name"],
        description=data["description"],
        personality=Personality.from_dict(data["personality"]),
        background=Background.from_dict(data["background"]),
    )


# ---------------------------------------------------------------------------
# My Profile (user character editor)
# ---------------------------------------------------------------------------


def page_my_profile() -> None:
    """Edit the user's own character profile used for joining scenes."""
    st.markdown("## 👤 My Profile")
    st.caption(
        "This is **your** character -- the personality other characters "
        "see when you join the conversation.",
    )

    # Explanatory hints (collapsible so they don't overwhelm repeat visitors)
    with st.expander("💡 How does my profile affect conversations?", expanded=False):
        st.markdown(
            "Your profile shapes how AI characters perceive and respond to you "
            "when you join a scene.\n\n"
            "- **Assertiveness & Warmth** control whether characters treat you "
            "as a leader or a peer, and how openly they share.\n"
            "- **Openness** makes characters more willing to explore abstract or "
            "unconventional topics with you.\n"
            "- **Humor & Formality** set the *tone* -- a high-humor, low-formality "
            "profile encourages banter; the reverse keeps things serious.\n"
            "- **Values & Beliefs** give characters something to agree or "
            "disagree with, driving richer conflict and connection.\n"
            "- **Background details** (occupation, motivations, fears) provide "
            "context the AI uses to craft more personal, grounded responses.\n\n"
            "*Tip: you don't need to fill everything in -- even a name and a few "
            "trait sliders are enough to make your presence felt.*",
        )

    prof = st.session_state.user_profile
    p_traits = prof.get("personality", {}).get("traits", {})
    p_vals = prof.get("personality", {}).get("values", {})
    p_bg = prof.get("background", {})

    with st.container(border=True):
        pl, pr = st.columns(2)
        with pl:
            name = st.text_input(
                "Display Name", value=prof.get("name", "You"),
                key="up_name",
            )
            description = st.text_area(
                "Description",
                value=prof.get("description", ""),
                height=100, key="up_desc",
            )
        with pr:
            age = st.number_input(
                "Age", 1, 150,
                value=p_bg.get("age", 30), key="up_age",
            )
            origin = st.text_input(
                "Origin", value=p_bg.get("origin", ""), key="up_origin",
            )
            occupation = st.text_input(
                "Occupation", value=p_bg.get("occupation", ""),
                key="up_occ",
            )

    st.markdown("#### Personality")
    tc1, tc2, tc3, tc4 = st.columns(4)
    with tc1:
        assertiveness = st.slider(
            "Assertiveness", 0.0, 1.0,
            value=p_traits.get("assertiveness", 0.6), key="up_assert",
        )
        warmth = st.slider(
            "Warmth", 0.0, 1.0,
            value=p_traits.get("warmth", 0.7), key="up_warmth",
        )
    with tc2:
        openness = st.slider(
            "Openness", 0.0, 1.0,
            value=p_traits.get("openness", 0.8), key="up_open",
        )
        conscientiousness = st.slider(
            "Conscientiousness", 0.0, 1.0,
            value=p_traits.get("conscientiousness", 0.6), key="up_consc",
        )
    with tc3:
        emotional_stability = st.slider(
            "Emotional Stability", 0.0, 1.0,
            value=p_traits.get("emotional_stability", 0.7), key="up_emot",
        )
        humor = st.slider(
            "Humor", 0.0, 1.0,
            value=p_traits.get("humor_inclination", 0.6), key="up_humor",
        )
    with tc4:
        formality = st.slider(
            "Formality", 0.0, 1.0,
            value=p_traits.get("formality", 0.4), key="up_formal",
        )
        preview_traits = {
            "assertiveness": assertiveness, "warmth": warmth,
            "openness": openness,
            "conscientiousness": conscientiousness,
            "emotional_stability": emotional_stability,
            "humor_inclination": humor, "formality": formality,
        }
        st.markdown(
            _radar_svg(preview_traits, size=150, color="#f472b6"),
            unsafe_allow_html=True,
        )

    st.markdown("#### Values & Voice")
    vl, vr = st.columns(2)
    with vl:
        values_text = st.text_area(
            "Core Values",
            value=", ".join(p_vals.get("priority_keywords", [])),
            height=60, key="up_vals",
        )
        beliefs_text = st.text_area(
            "Beliefs",
            value=", ".join(p_vals.get("beliefs", [])),
            height=60, key="up_beliefs",
        )
        strengths_text = st.text_area(
            "Strengths",
            value=", ".join(p_vals.get("strengths", [])),
            height=50, key="up_str",
        )
        weaknesses_text = st.text_area(
            "Weaknesses",
            value=", ".join(p_vals.get("weaknesses", [])),
            height=50, key="up_weak",
        )
    with vr:
        dislikes_text = st.text_area(
            "Dislikes",
            value=", ".join(p_vals.get("dislikes", [])),
            height=60, key="up_dis",
        )
        speech_text = st.text_area(
            "Speech Patterns",
            value=", ".join(
                prof.get("personality", {}).get("speech_patterns", []),
            ),
            height=60, key="up_speech",
        )
        motivations_text = st.text_area(
            "Motivations",
            value=", ".join(p_bg.get("motivations", [])),
            height=50, key="up_motiv",
        )
        fears_text = st.text_input(
            "Fears",
            value=", ".join(p_bg.get("fears", [])),
            key="up_fears",
        )

    desires_text = st.text_input(
        "Desires",
        value=", ".join(p_bg.get("desires", [])),
        key="up_desires",
    )

    c_save, c_reset = st.columns(2)
    with c_save:
        if st.button(
            "💾 Save Profile", type="primary", use_container_width=True,
        ):
            new_profile: dict[str, Any] = {
                "name": name,
                "description": description,
                "personality": {
                    "traits": {
                        "assertiveness": assertiveness,
                        "warmth": warmth,
                        "openness": openness,
                        "conscientiousness": conscientiousness,
                        "emotional_stability": emotional_stability,
                        "humor_inclination": humor,
                        "formality": formality,
                    },
                    "values": {
                        "priority_keywords": _parse_csv(values_text),
                        "beliefs": _parse_csv(beliefs_text),
                        "dislikes": _parse_csv(dislikes_text),
                        "strengths": _parse_csv(strengths_text),
                        "weaknesses": _parse_csv(weaknesses_text),
                    },
                    "speech_patterns": _parse_csv(speech_text),
                    "quirks": [],
                },
                "background": {
                    "age": age,
                    "origin": origin,
                    "occupation": occupation,
                    "motivations": _parse_csv(motivations_text),
                    "fears": _parse_csv(fears_text),
                    "desires": _parse_csv(desires_text),
                },
            }
            st.session_state.user_profile = new_profile
            _save_user_profile(new_profile)
            st.success("Profile saved!")

    with c_reset:
        if st.button("🔄 Reset to Defaults", use_container_width=True):
            st.session_state.user_profile = dict(_DEFAULT_USER_PROFILE)
            _save_user_profile(_DEFAULT_USER_PROFILE)
            st.rerun()

    # preview
    st.divider()
    st.markdown("##### Live Preview")
    char = _build_user_character(st.session_state.user_profile)
    st.text(char.get_character_profile())


# ---------------------------------------------------------------------------
# Archive
# ---------------------------------------------------------------------------


def page_archive() -> None:
    """Browse past dialogue sessions."""
    st.markdown("## 📜 Archive")
    st.caption("Browse past dialogue sessions.")

    records = _interaction_repo.list_all()
    if not records:
        st.info(
            "No sessions recorded yet. "
            "Run a scene from the **Scene** page.",
        )
        return

    st.metric("Total Sessions", len(records))

    for rec in records:
        dur = (
            f"{rec.duration_seconds:.1f}s"
            if rec.duration_seconds
            else "in-progress"
        )
        header = (
            f"**{rec.topic or 'Untitled'}** -- "
            f"{'  ·  '.join(rec.characters)} -- "
            f"{rec.exchange_count} exchanges · {dur}"
        )
        with st.expander(header):
            ic1, ic2, ic3 = st.columns(3)
            ic1.caption(f"Provider: {rec.provider}")
            ic2.caption(f"Model: {rec.model}")
            ic3.caption(f"ID: {rec.interaction_id[:12]}…")

            st.markdown(f"*{rec.scene_description}*")
            st.divider()

            for ex in rec.exchanges:
                c = _color_for(ex.get("speaker", ""))
                st.markdown(
                    _render_chat_bubble(
                        ex.get("speaker", "?"),
                        ex.get("text", ""),
                        ex.get("pre_exchange_thought", ""),
                        ex.get("internal_thought", ""),
                        ex.get("emotional_context", "neutral"),
                        c,
                    ),
                    unsafe_allow_html=True,
                )

            if st.button(
                "🗑️ Delete", key=f"del_{rec.interaction_id}",
            ):
                _interaction_repo.delete(rec.interaction_id)
                st.rerun()


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------


def page_settings() -> None:
    """Live LLM configuration and system stats."""
    st.markdown("## ⚙️ Settings")

    # ── LLM configuration (live-editable) ─────────────────────────────────
    st.markdown("##### LLM Provider")
    st.caption(
        "Changes take effect immediately. Your API key is held only in "
        "your browser session and is **never** persisted to disk.",
    )

    sl, sr = st.columns(2)

    with sl:
        provider_labels = [p["label"] for p in _PROVIDER_PRESETS.values()]
        cur_provider = _active_provider()
        cur_idx = (
            _PROVIDER_IDS.index(cur_provider)
            if cur_provider in _PROVIDER_IDS
            else 0
        )
        new_label = st.radio(
            "Provider",
            provider_labels,
            index=cur_idx,
            horizontal=True,
            key="settings_provider",
        )
        new_provider_id = _PROVIDER_IDS[provider_labels.index(new_label)]
        preset = _PROVIDER_PRESETS[new_provider_id]

        new_model = st.text_input(
            "Model",
            value=_active_model(),
            key="settings_model",
            help="The model identifier sent to the provider.",
        )

    with sr:
        new_key = st.text_input(
            "API Key",
            type="password",
            value=_active_api_key(),
            placeholder=preset["key_hint"],
            key="settings_api_key",
            help=(
                f"[Get a key from {preset['label']}]"
                f"({preset['signup_url']})"
            ),
        )

        st.markdown(
            f"**Temperature:** `{settings.dialogue_temperature}`  \n"
            f"**Fast model:** `{settings.fast_model}`",
        )

    if st.button("💾  Apply changes", type="primary"):
        st.session_state.llm_provider = new_provider_id
        st.session_state.llm_model = (
            new_model.strip() or preset["default_model"]
        )
        st.session_state.llm_api_key = new_key.strip()
        st.success(
            f"Switched to **{preset['label']}** / "
            f"`{st.session_state.llm_model}`.",
        )

    if not _active_api_key():
        st.warning(
            "⚠️ No API key set -- scenes won't work until you provide one.",
        )

    st.divider()

    # ── System stats ──────────────────────────────────────────────────────
    st.markdown("##### System Stats")
    s1, s2, s3 = st.columns(3)
    all_chars = character_repository.list_all()
    s1.metric("Characters", len(all_chars))
    s2.metric("Stored Sessions", _interaction_repo.count())
    total_ex = sum(r.exchange_count for r in _interaction_repo.list_all())
    s3.metric("Total Exchanges", total_ex)

    st.divider()

    # ── Capabilities ──────────────────────────────────────────────────────
    st.markdown("##### Capabilities")
    feat_l, feat_r = st.columns(2)
    with feat_l:
        st.markdown(
            "- ✅ Live LLM dialogue generation\n"
            "- ✅ Step-wise & continuous dialogue modes\n"
            "- ✅ User can join conversations in-character\n"
            "- ✅ Personality-driven speaker selection\n"
            "- ✅ Internal monologue & emotional tracking\n"
            "- ✅ Persistent character database (SQLite)",
        )
    with feat_r:
        st.markdown(
            "- ✅ Multi-provider support (OpenAI / Anthropic / Google)\n"
            "- ✅ 7-dimensional personality trait system\n"
            "- ✅ Rich background & memory model\n"
            f"- ✅ Configurable exchange cap (1-{MAX_EXCHANGES})\n"
            "- ✅ Editable user profile for self-insertion\n"
            "- ✅ Interaction archiving & replay",
        )

    st.divider()

    # ── Security note ─────────────────────────────────────────────────────
    st.markdown("##### 🔒 Security")
    st.caption(
        "• API keys are stored **in-memory only** (Streamlit session state) "
        "and are never written to disk, logs, or the database.  \n"
        "• Keys are sent exclusively to the selected LLM provider over HTTPS.  \n"
        "• When you close your browser tab the key is gone.",
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Welcome gate
# ---------------------------------------------------------------------------


def _page_welcome() -> None:
    """Onboarding -- collect user name + LLM provider / API key."""
    st.markdown(
        '<div style="text-align:center;padding:2.5rem 0 1rem 0;">'
        '<span style="font-size:3.5rem">🎭</span>'
        "</div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div style="text-align:center">'
        '<h1 style="'
        "font-size:2.6rem;"
        "background:linear-gradient(135deg,#a78bfa,#f472b6);"
        "-webkit-background-clip:text;"
        '-webkit-text-fill-color:transparent;">'
        "Welcome to Character Creator</h1>"
        '<p style="opacity:0.65;font-size:1.1rem;max-width:520px;'
        'margin:0 auto">'
        "Craft unique personalities, set the scene, and watch them "
        "come alive through AI-powered dialogue -- or jump in yourself."
        "</p></div>",
        unsafe_allow_html=True,
    )

    st.markdown("")

    # ---- centred form column --------------------------------------------
    _, col, _ = st.columns([1, 2, 1])
    with col:
        # ── About you ────────────────────────────────────────────────
        st.markdown(
            '<p style="text-align:center;opacity:0.8;font-size:1.05rem">'
            "First, what should we call you?</p>",
            unsafe_allow_html=True,
        )
        entered_name = st.text_input(
            "Your name",
            placeholder="Enter your preferred name…",
            label_visibility="collapsed",
            key="welcome_name_input",
        )

        st.markdown("")
        st.divider()

        # ── LLM provider config ──────────────────────────────────────
        st.markdown(
            '<p style="text-align:center;opacity:0.8;font-size:1.05rem">'
            "Now, connect an AI provider</p>",
            unsafe_allow_html=True,
        )
        st.caption(
            "Character Creator uses a large-language model to power "
            "dialogue. Choose your provider and paste an API key below. "
            "Your key is stored **only in your browser session** -- it is "
            "never saved to disk or sent anywhere except your chosen "
            "provider.",
        )

        # Provider selector -- tabs for visual clarity
        provider_labels = [p["label"] for p in _PROVIDER_PRESETS.values()]
        current_provider = st.session_state.get(
            "llm_provider", settings.llm_provider,
        )
        default_idx = (
            _PROVIDER_IDS.index(current_provider)
            if current_provider in _PROVIDER_IDS
            else 0
        )
        selected_label = st.radio(
            "Provider",
            provider_labels,
            index=default_idx,
            horizontal=True,
            key="welcome_provider_radio",
        )
        # Resolve back to provider id
        provider_id = _PROVIDER_IDS[provider_labels.index(selected_label)]
        preset = _PROVIDER_PRESETS[provider_id]

        # API key -- use password type so it's masked
        api_key = st.text_input(
            "API Key",
            type="password",
            placeholder=preset["key_hint"],
            key="welcome_api_key",
            help=(
                f"[Get a key from {preset['label']}]"
                f"({preset['signup_url']})"
            ),
        )

        # Model -- pre-filled with a sensible default, user can change
        model = st.text_input(
            "Model",
            value=preset["default_model"],
            key="welcome_model",
            help="The model identifier your provider should use.",
        )

        st.markdown("")

        # ── Go button ────────────────────────────────────────────────
        has_name = bool(entered_name and entered_name.strip())
        has_key = bool(api_key and api_key.strip())
        can_go = has_name and has_key

        if not has_key and has_name:
            st.info(
                "🔑 An API key is required to power the dialogue engine. "
                f"[Get one from {preset['label']}]({preset['signup_url']})",
                icon="i",
            )

        if st.button(
            "✨  Let's go",
            type="primary",
            use_container_width=True,
            disabled=not can_go,
        ):
            # Persist name
            clean_name = entered_name.strip()
            prof = st.session_state.user_profile
            prof["name"] = clean_name
            st.session_state.user_profile = prof
            _save_user_profile(prof)

            # Store LLM config in session (never on disk)
            st.session_state.llm_provider = provider_id
            st.session_state.llm_model = model.strip() or preset["default_model"]
            st.session_state.llm_api_key = api_key.strip()

            st.session_state.onboarded = True
            st.rerun()

    st.markdown("")
    _, foot_col, _ = st.columns([1, 2, 1])
    with foot_col:
        st.caption(
            "You can change your name, profile, and LLM settings at any "
            "time from **My Profile** and **Settings**.",
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    """Application entry point."""
    st.set_page_config(
        page_title="Character Creator",
        page_icon="🎭",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.markdown(_CUSTOM_CSS, unsafe_allow_html=True)
    _init_state()

    # ---- onboarding gate -------------------------------------------------
    if not st.session_state.onboarded:
        _page_welcome()
        return

    # ---- main app --------------------------------------------------------
    with st.sidebar:
        st.markdown(
            '<div style="text-align:center;padding:0.5rem 0 1rem 0;">'
            '<span style="font-size:2rem">🎭</span><br/>'
            '<b style="font-size:1.1rem;letter-spacing:1px">'
            "CHARACTER CREATOR</b>"
            "</div>",
            unsafe_allow_html=True,
        )
        page = st.radio(
            "Navigate",
            [
                "My Profile",
                "Cast",
                "Scene",
                "Workshop",
                "Archive",
                "Dashboard",
                "Settings",
            ],
            label_visibility="collapsed",
        )

        st.divider()
        prof_name = st.session_state.user_profile.get("name", "You")
        st.caption(
            f"Signed in as: **{prof_name}**  \n"
            f"Provider: **{_active_provider()}**  \n"
            f"Model: **{_active_model()}**",
        )
        st.caption("v0.3.0")

    pages = {
        "My Profile": page_my_profile,
        "Cast": page_cast,
        "Scene": page_director,
        "Workshop": page_workshop,
        "Archive": page_archive,
        "Dashboard": page_dashboard,
        "Settings": page_settings,
    }
    pages[page]()


if __name__ == "__main__":
    main()
