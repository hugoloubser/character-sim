"""Live dialogue demo with real LLM calls.

Demonstrates:
    1. Connecting to a configured LLM provider (Anthropic by default).
    2. Using ``create_default_characters()`` to load rich character profiles.
    3. Running a multi-character dialogue scene through ``DialogueSystem``.
    4. Persisting the interaction via ``InteractionRepository``.
    5. Logging everything (console + file) via the logging infrastructure.

Usage::

    # Default — 5 exchanges, Anthropic provider
    python -m character_creator.demos.live_dialogue

    # Custom exchange count (max 10)
    python -m character_creator.demos.live_dialogue --exchanges 8

    # Different provider
    python -m character_creator.demos.live_dialogue --provider openai --exchanges 3

"""

from __future__ import annotations

import argparse
import asyncio
import sys
import textwrap

from pathlib import Path

from character_creator.core.database import create_default_characters
from character_creator.core.dialogue import DialogueContext, DialogueSystem
from character_creator.core.interaction import (
    InMemoryInteractionRepository,
    InteractionRecord,
    SQLiteInteractionRepository,
)
from character_creator.llm.providers import LLMError, get_llm_provider
from character_creator.utils.config import settings
from character_creator.utils.logging import LogAccessLayer, setup_logging

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MAX_EXCHANGES = 10
DEFAULT_EXCHANGES = 5
DEFAULT_PROVIDER = "anthropic"

DEFAULT_SCENE = (
    "A cozy, dimly-lit café on a rainy evening. Mismatched chairs "
    "surround small wooden tables, the aroma of fresh coffee fills the air, "
    "and soft jazz plays in the background."
)
DEFAULT_TOPIC = "What does it mean to live a meaningful life?"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Run a live LLM-powered multi-character dialogue.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            examples:
              %(prog)s                          # 5 exchanges, anthropic
              %(prog)s --exchanges 8            # 8 exchanges
              %(prog)s --provider openai -n 3   # OpenAI, 3 exchanges
        """),
    )
    p.add_argument(
        "-n",
        "--exchanges",
        type=int,
        default=DEFAULT_EXCHANGES,
        help=f"Number of dialogue exchanges (1-{MAX_EXCHANGES}, default: {DEFAULT_EXCHANGES}).",
    )
    p.add_argument(
        "--provider",
        type=str,
        default=settings.llm_provider or DEFAULT_PROVIDER,
        help=f"LLM provider name (default: {settings.llm_provider or DEFAULT_PROVIDER}).",
    )
    p.add_argument(
        "--scene",
        type=str,
        default=DEFAULT_SCENE,
        help="Scene description for the dialogue.",
    )
    p.add_argument(
        "--topic",
        type=str,
        default=DEFAULT_TOPIC,
        help="Conversation topic.",
    )
    p.add_argument(
        "--db",
        type=str,
        default=None,
        help="Path to SQLite database for persisting the interaction (omit for in-memory).",
    )
    p.add_argument(
        "--characters",
        nargs="+",
        default=None,
        help="Names of characters to include (default: Alice, Zoey, Kai).",
    )
    return p


# ---------------------------------------------------------------------------
# Core demo logic
# ---------------------------------------------------------------------------


async def run_dialogue(
    *,
    provider_name: str,
    num_exchanges: int,
    scene: str,
    topic: str,
    character_names: list[str] | None,
    db_path: str | None,
) -> InteractionRecord:
    """Execute a live dialogue and persist the result.

    Args:
        provider_name: LLM provider to use.
        num_exchanges: How many exchanges to generate (capped at MAX_EXCHANGES).
        scene: Scene description.
        topic: Conversation topic.
        character_names: Which default characters to include (None ⇒ default trio).
        db_path: If given, persist to SQLite; otherwise use in-memory repo.

    Returns:
        The completed ``InteractionRecord``.

    Raises:
        SystemExit: On unrecoverable LLM errors (with helpful message).

    """
    # -- Clamp exchanges ---------------------------------------------------
    num_exchanges = max(1, min(num_exchanges, MAX_EXCHANGES))

    # -- Select characters -------------------------------------------------
    all_chars = create_default_characters()
    char_map = {c.name.lower(): c for c in all_chars}

    if character_names:
        selected = []
        for name in character_names:
            key = name.lower()
            if key not in char_map:
                available = ", ".join(c.name for c in all_chars)
                print(f"⚠  Unknown character '{name}'. Available: {available}")
                sys.exit(1)
            selected.append(char_map[key])
    else:
        selected = [char_map["alice"], char_map["zoey"], char_map["kai"]]

    # -- Provider ----------------------------------------------------------
    # Resolve the API key from settings so .env is honoured even when the
    # corresponding OS environment variable is not set.
    api_key_map = {
        "anthropic": settings.anthropic_api_key,
        "openai": settings.openai_api_key,
        "google": settings.google_api_key,
    }
    api_key = api_key_map.get(provider_name.lower())

    print(f"\n🔌  Initialising {provider_name} provider …")
    try:
        provider = get_llm_provider(provider_name, api_key=api_key)
    except (ValueError, LLMError) as exc:
        print(f"❌  Could not create LLM provider: {exc}")
        sys.exit(1)

    # -- Repository --------------------------------------------------------
    if db_path:
        repo = SQLiteInteractionRepository(Path(db_path))
        print(f"💾  Persisting interactions to {db_path}")
    else:
        repo = InMemoryInteractionRepository()

    # -- Record ------------------------------------------------------------
    record = InteractionRecord(
        scene_description=scene,
        topic=topic,
        characters=[c.name for c in selected],
        provider=provider_name,
        model=settings.default_model,
        max_exchanges=num_exchanges,
    )

    # -- Dialogue ----------------------------------------------------------
    dialogue_system = DialogueSystem(provider)
    context = DialogueContext(
        characters=selected,
        scene_description=scene,
        topic=topic,
    )

    char_list = ", ".join(c.name for c in selected)
    print(f"🎭  Characters: {char_list}")
    print(f"🎬  Scene: {scene}")
    print(f"💬  Topic: {topic}")
    print(f"🔄  Exchanges: {num_exchanges}")
    print("-" * 72)

    try:
        for i in range(num_exchanges):
            speaker = dialogue_system.generate_next_speaker(context)
            print(f"\n[{i + 1}/{num_exchanges}] {speaker.name} is thinking …")

            pre_thought, dialogue, internal = await dialogue_system.generate_response(context, speaker)
            emotional = await dialogue_system.infer_emotional_context(speaker, dialogue)

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

            # Pretty-print the exchange
            print(f"  💭 ({speaker.name} thinks: {pre_thought})")
            print(f"  💬 {speaker.name}: {dialogue}")
            print(f"  🧠 ({speaker.name}'s thought: {internal})")
            print(f"  😶 Mood: {emotional}")

    except LLMError as exc:
        print(f"\n❌  LLM error after {record.exchange_count} exchange(s): {exc}")
        print("     (Check your API key and billing status.)")
    except Exception as exc:  # noqa: BLE001
        print(f"\n❌  Unexpected error: {exc}")
    finally:
        record.finish()
        await provider.close()

    # -- Persist -----------------------------------------------------------
    repo.save(record)
    print("\n" + "=" * 72)
    print(f"✅  Interaction complete — {record.exchange_count} exchange(s) in "
          f"{record.duration_seconds:.1f}s")
    print(f"    ID: {record.interaction_id}")

    return record


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------


def main() -> None:
    """Parse CLI args, configure logging, and run the demo."""
    parser = _build_parser()
    args = parser.parse_args()

    # Set up logging (logs/ directory in local data dir)
    log_dir = settings.log_dir
    log_file = setup_logging(log_dir=log_dir, prefix="live_dialogue", level=settings.log_level)
    print(f"📝  Log file: {log_file}")

    asyncio.run(
        run_dialogue(
            provider_name=args.provider,
            num_exchanges=args.exchanges,
            scene=args.scene,
            topic=args.topic,
            character_names=args.characters,
            db_path=args.db,
        )
    )

    # Show a summary from the log access layer
    logs = LogAccessLayer(log_dir)
    latest = logs.latest(prefix="live_dialogue")
    if latest:
        print(f"\n📂  Latest log: {latest}")
        print(f"    Lines: {len(logs.tail(latest, n=9999))}")


if __name__ == "__main__":
    main()
