"""Streaming TTS port + adapter factory (Ports & Adapters).

Mirror of ``stt.py`` for the speaking side: the voice turn loop consumes a
provider-agnostic per-turn TTS object; this module defines that contract and
the factory that builds the per-call client for the configured provider.

The client is per-call (ONE persistent provider connection — connect cost is
paid at call setup so turn 1 is fast); the turn object is per-utterance.
The turn contract the loop relies on:

  open()             -> start a synthesis context for this turn
  send_text(text)    -> stream text fragments as the LLM produces them;
                        the literal sentinel "<flush>" forces synthesis of
                        buffered text (first-sentence latency, long deltas)
  send_filler(text)  -> speak canned filler (tool-call acknowledgment)
  end()              -> no more text; resolves when the final audio arrived
  abort()            -> barge-in: kill synthesis mid-stream, drop the context

Audio is delivered via the ``on_audio`` callback passed to ``new_turn`` as
s16le mono PCM at the transport's playback rate, incrementally while
synthesis runs — and synthesis MUST be interruptible mid-stream (``abort``),
because barge-in is the plugin's whole point. A request/response HTTP TTS
cannot satisfy this contract without faking it; prefer providers with a
streaming websocket.

Default provider is Cartesia. Adding a provider = one client module
implementing TTS/TTSTurn + one branch in ``make_tts`` + an
``extra.tts_provider`` value.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Awaitable, Callable, Dict, Optional, Protocol, runtime_checkable

logger = logging.getLogger(__name__)


@runtime_checkable
class TTSTurn(Protocol):
    """One utterance's synthesis context. See the module docstring for the
    call sequence the turn loop drives."""

    async def open(self) -> None: ...

    async def send_text(self, text: str) -> None: ...

    async def send_filler(self, text: str) -> None: ...

    async def end(self) -> None: ...

    async def abort(self) -> None: ...


@runtime_checkable
class TTS(Protocol):
    """The per-call TTS client port. ``provider``/``model``/``voice`` are
    public for logs and telemetry."""

    provider: str
    model: str
    voice: str

    async def connect(self) -> None: ...

    def new_turn(
        self, on_audio: Callable[[bytes], Awaitable[None]]
    ) -> TTSTurn: ...

    async def close(self) -> None: ...


DEFAULT_TTS_PROVIDER = "cartesia"


def make_tts(provider: Optional[str], *, extra: Dict[str, Any]) -> TTS:
    """Build the per-call TTS client for ``provider``.

    Resolves the provider's API key from the environment, its tuning knobs
    from ``extra``, and raises a clear error for an unknown provider. The
    client constructors raise on missing key/voice, so a misconfigured
    provider fails fast (and the adapter's partial-start cleanup unwinds).
    """
    provider = (provider or DEFAULT_TTS_PROVIDER).strip().lower()

    if provider == "cartesia":
        from . import cartesia_tts

        api_key = os.getenv("CARTESIA_API_KEY", "").strip()
        if not api_key:
            raise RuntimeError("CARTESIA_API_KEY is not set")
        voice_id = (
            extra.get("cartesia_voice_id")
            or os.getenv("CARTESIA_VOICE_ID", "").strip()
        )
        return cartesia_tts.CartesiaTTSClient(
            api_key,
            voice_id,
            model=extra.get("cartesia_model") or cartesia_tts.DEFAULT_MODEL,
            speed=_opt_float_extra(extra, "tts_speed"),
            volume=_opt_float_extra(extra, "tts_volume"),
            emotion=(extra.get("tts_emotion") or None),
        )

    raise RuntimeError(
        f"unknown tts_provider={provider!r}; "
        "supported: 'cartesia' (default)")


def _opt_float_extra(extra: Dict[str, Any], key: str) -> Optional[float]:
    """Parse an optional float from extra, returning None (not a default)
    when unset — so a provider generation_config field is only sent when the
    user actually set it. Bad values warn and are ignored."""
    raw = extra.get(key)
    if raw is None or (isinstance(raw, str) and not raw.strip()):
        return None
    try:
        return float(raw)
    except (TypeError, ValueError):
        logger.warning("voice: invalid %s=%r; ignoring", key, raw)
        return None
