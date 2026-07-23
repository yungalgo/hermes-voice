"""Voice platform plugin — realtime Daily (WebRTC) voice calls.

Standalone hermes-agent platform plugin (installed via the
``hermes_agent.plugins`` pip entry point, or dir-dropped into
``~/.hermes/plugins/``). Turn orchestration lives in turn_loop.py; this
module owns plugin registration, requirement checks, and the adapter
lifecycle.

Standalone mode: this agent holds DAILY_API_KEY, creates its own private
Daily room + a single-use meeting token at connect time, joins immediately,
and logs the room URL for the owner to share. caller == owner.

Opinionated stack: Deepgram Flux (streaming STT + model-integrated turn
detection), Cartesia (streaming TTS, default — the per-turn TTS is built
behind a factory so another provider can be slotted in), and the plugin's
voice_model slot (``platforms.voice.extra.voice_model``) for generation.
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from typing import Any, Dict, Optional, Tuple

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import (
    BasePlatformAdapter,
    SendResult,
)

from . import daily_transport, turn_loop
from . import stt as stt_mod
from . import tts as tts_mod

logger = logging.getLogger(__name__)

DAILY_API = "https://api.daily.co/v1"
STANDALONE_ROOM_TTL_S = 3600

# Call watchdog: an abandoned call — tab closed, or the room expired and
# ejected the agent — would otherwise leave the billable STT stream running.
# The watchdog tears the call down when humans are gone, the call ended
# remotely, or a hard age cap is hit.
DEFAULT_IDLE_TEARDOWN_S = 60.0   # extra.idle_teardown_s
DEFAULT_MAX_CALL_S = 1800.0      # extra.max_call_s
WATCHDOG_POLL_S = 0.5


# All sibling modules are stdlib-only at module level (heavy SDKs are
# deferred inside functions), so plain relative imports replace the old
# in-tree dual-import shim that hermes' flat test loader required.


def _daily_available() -> bool:
    try:
        import daily  # noqa: F401
        return True
    except ImportError:
        return False


def _websockets_available() -> bool:
    try:
        import websockets  # noqa: F401
        return True
    except ImportError:
        return False


def check_requirements() -> bool:
    """Deps importable + the unconditional call keys present (cheap env read,
    no config load): Daily (transport) and Cartesia (TTS) are needed for every
    call. The STT key is provider-dependent (Deepgram for the default Flux,
    Cartesia for Ink-2), so it is checked in validate_config / at connect,
    where the platform's extra config is available."""
    if not _daily_available() or not _websockets_available():
        return False
    return bool(
        os.getenv("DAILY_API_KEY", "").strip()
        and os.getenv("CARTESIA_API_KEY", "").strip()
    )


def _resolve_float_extra(extra: Dict[str, Any], key: str, default: float) -> float:
    raw = extra.get(key)
    if raw is None or (isinstance(raw, str) and not raw.strip()):
        return default
    try:
        return float(raw)
    except (TypeError, ValueError):
        logger.warning("voice: invalid %s=%r; using default %s",
                       key, raw, default)
        return default


def validate_config(config) -> bool:
    """Config-aware readiness: transport key + the configured STT provider's
    key. Cartesia's key is already required by check_requirements (TTS), so
    only the default Flux path adds a key requirement here."""
    if not os.getenv("DAILY_API_KEY", "").strip():
        return False
    extra = getattr(config, "extra", None) or {}
    stt_provider = (extra.get("stt_provider") or "deepgram_flux").strip().lower()
    if stt_provider == "deepgram_flux":
        return bool(os.getenv("DEEPGRAM_API_KEY", "").strip())
    return True


def is_connected(config) -> bool:
    return check_requirements() and validate_config(config)


def register(ctx) -> None:
    """Plugin entry point — called by the Hermes plugin system at startup."""
    ctx.register_platform(
        name="voice",
        label="Voice",
        adapter_factory=lambda cfg: VoiceAdapter(cfg),
        check_fn=check_requirements,
        validate_config=validate_config,
        is_connected=is_connected,
        required_env=["DAILY_API_KEY", "DEEPGRAM_API_KEY", "CARTESIA_API_KEY"],
        install_hint="pip install daily-python==0.29.1 websockets  # into hermes' venv (plugin code arrives via git, not PyPI)",
        pii_safe=True,
        emoji="📞",
        allow_update_command=False,
        platform_hint=(
            "You are on a live voice call. Speak naturally and BRIEFLY — "
            "1-3 short sentences per reply unless asked for detail. Never use "
            "markdown, bullet lists, code blocks, or URLs; everything you "
            "write is read aloud. If a tool call will take a while, say so "
            "in a few words first."
        ),
    )


class VoiceAdapter(BasePlatformAdapter):
    """Daily-room voice call adapter (standalone mode).

    This agent holds DAILY_API_KEY, creates its own private room + meeting
    token via the Daily REST API at connect time, joins immediately, and logs
    the room URL for the owner to share.

    One active call at a time (daily-python virtual devices are process-level
    singletons — see daily_transport.py).
    """

    def __init__(self, config: PlatformConfig):
        platform = Platform("voice")
        super().__init__(config=config, platform=platform)
        self._call_lock = asyncio.Lock()
        self._active_call: Optional[Dict[str, Any]] = None

    async def connect(self, *, is_reconnect: bool = False) -> bool:
        # is_reconnect (base.py contract): we hold no server-side message
        # queue — a dropped call is simply a new room on reconnect.
        # Standalone: create our own private room + tokens, join immediately.
        # The room is private, so the shareable URL must carry an owner token —
        # the bare room URL alone cannot join a private room.
        room_url, agent_token, share_url = await self._create_standalone_room()
        await self._start_call(room_url, agent_token)
        self._mark_connected()
        # The join URL is the room's join permission (private room + owner
        # token) — and it is a JWT, which hermes' RedactingFormatter mangles
        # in every log handler (console and file). So the log line only
        # ANNOUNCES the call; the actual URL is handed over redaction-free:
        #   1. print() to stdout (bypasses logging formatters), and
        #   2. written 0600 to <hermes home>/voice-call-url.txt.
        url_file = None
        try:
            from hermes_constants import get_hermes_home

            url_file = get_hermes_home() / "voice-call-url.txt"
            url_file.write_text(share_url + "\n", encoding="utf-8")
            os.chmod(url_file, 0o600)
        except OSError:
            logger.warning("voice: could not write the join URL file",
                           exc_info=True)
            url_file = None
        print(
            "voice: standalone call ready — open this URL to talk to your "
            f"agent (keep the token private):\n    {share_url}",
            flush=True,
        )
        logger.warning(
            "voice: standalone call ready — join URL printed to stdout%s "
            "(log lines redact the token, so don't copy the URL from a log)",
            f" and written to {url_file}" if url_file else "")
        return True

    async def _start_call(self, room_url: str, token: str) -> None:
        async with self._call_lock:
            if self._active_call is not None:
                await self._end_call_locked("replaced-by-new-call")
            loop = asyncio.get_running_loop()
            extra = self.config.extra or {}

            # STT: streaming ASR + model-integrated turn detection
            # (start/eager-end/resumed/end-of-turn events the turn loop reacts
            # to). Deepgram Flux is the default; the make_stt factory is the
            # seam so another provider (e.g. Cartesia Ink-2) can be A/B'd
            # behind the same port. Inbound caller audio is at the Daily
            # SPEAKER_RATE; each adapter handles its own rate (Flux resamples
            # 24k->16k internally; Ink-2 accepts the native rate). The factory
            # resolves the provider's API key and raises if it's missing.
            stt_provider = (extra.get("stt_provider")
                            or "deepgram_flux").strip().lower()
            stt = stt_mod.make_stt(
                stt_provider,
                input_rate=daily_transport.SPEAKER_RATE,
                extra=extra,
            )
            transport = None
            tts_client = None
            try:
                await stt.start()
                logger.info("voice: STT provider=%s", stt.provider)

                async def on_audio_in(pcm: bytes) -> None:
                    await stt.send_audio(pcm)

                transport = daily_transport.DailyTransport(loop, on_audio_in)
                await transport.join(room_url, token)

                # TTS: built through the tts.py port + make_tts factory
                # (Cartesia is the bundled adapter; extra.tts_provider is the
                # slot). ONE persistent provider connection for the whole call
                # (connect is too costly to pay per turn), opened here so
                # turn 1 is fast; the loop gets a per-turn object via
                # tts_factory below.
                tts_client = tts_mod.make_tts(
                    extra.get("tts_provider"), extra=extra)
                await tts_client.connect()
                logger.info("voice: TTS provider=%s model=%s voice=%s",
                            tts_client.provider, tts_client.model,
                            tts_client.voice)
            except BaseException:
                # Partial-start cleanup: a failure here (bad TTS config, room
                # join error, ...) must not leak a live STT socket or a joined
                # transport. daily-python's virtual devices are process-level
                # singletons — a leaked reader thread makes every subsequent
                # retry fail with PyO3 "Already borrowed".
                if transport is not None:
                    try:
                        transport.begin_teardown()
                        await transport.leave()
                    except Exception:
                        logger.warning(
                            "voice: transport cleanup after failed start "
                            "also failed", exc_info=True)
                if tts_client is not None:
                    try:
                        await tts_client.close()
                    except Exception:
                        pass
                try:
                    await stt.stop()
                except Exception:
                    pass
                raise

            async def tts_factory(on_audio):
                turn = tts_client.new_turn(on_audio)
                await turn.open()
                return turn

            vloop = turn_loop.VoiceTurnLoop(
                stt, tts_factory, transport, extra=extra)
            task = asyncio.create_task(vloop.run())
            self._active_call = {
                "stt": stt, "transport": transport, "loop": vloop,
                "task": task, "tts_client": tts_client,
                "started_at": time.monotonic(), "room_url": room_url}
            self._active_call["watchdog"] = asyncio.create_task(
                self._call_watchdog(transport))
            logger.info("voice: call started in %s", room_url)

    async def _call_watchdog(self, transport) -> None:
        """Tear the call down when it is no longer worth paying for:
          - no human (remote) participant for extra.idle_teardown_s
            (tab closed / never joined),
          - the call ended remotely (room expired, agent ejected, fatal
            client error),
          - call age exceeds extra.max_call_s (hard cost cap).
        Polls every WATCHDOG_POLL_S. asr_seconds_est in the teardown summary
        makes the cost of every call auditable."""
        extra = self.config.extra or {}
        idle_teardown_s = _resolve_float_extra(
            extra, "idle_teardown_s", DEFAULT_IDLE_TEARDOWN_S)
        max_call_s = _resolve_float_extra(
            extra, "max_call_s", DEFAULT_MAX_CALL_S)
        idle_since: Optional[float] = None
        while True:
            await asyncio.sleep(WATCHDOG_POLL_S)
            call = self._active_call
            if call is None or call.get("transport") is not transport:
                return
            now = time.monotonic()
            age_s = now - call["started_at"]
            reason = None
            abnormal = transport.abnormal_end
            if abnormal is not None:
                reason = "remote-end"
                logger.warning(
                    "voice: WATCHDOG teardown reason=%s detail=%r age_s=%.0f "
                    "— call ended remotely (ejection/expiry/error)",
                    reason, abnormal, age_s)
            elif age_s >= max_call_s:
                reason = "max-call-duration"
                logger.warning(
                    "voice: WATCHDOG teardown reason=%s age_s=%.0f "
                    "max_call_s=%.0f — hard cost cap hit",
                    reason, age_s, max_call_s)
            elif transport.remote_participant_count == 0:
                if idle_since is None:
                    idle_since = now
                elif now - idle_since >= idle_teardown_s:
                    reason = "no-human-participants"
                    logger.warning(
                        "voice: WATCHDOG teardown reason=%s idle_s=%.0f "
                        "idle_teardown_s=%.0f age_s=%.0f — caller gone "
                        "(tab closed / never joined)",
                        reason, now - idle_since, idle_teardown_s, age_s)
            else:
                idle_since = None
            if reason is not None:
                await self._end_call(reason)
                return

    async def _end_call(self, reason: str) -> None:
        async with self._call_lock:
            await self._end_call_locked(reason)

    async def _end_call_locked(self, reason: str) -> None:
        call = self._active_call
        if call is None:
            return
        self._active_call = None
        # FIRST: kill the billable keep-alive feed. Every await below can take
        # real time, and the keep-alive must not pump paid STT audio while the
        # call winds down.
        call["transport"].begin_teardown()
        watchdog = call.get("watchdog")
        if watchdog is not None and watchdog is not asyncio.current_task():
            watchdog.cancel()
            try:
                await watchdog
            except (asyncio.CancelledError, Exception):
                pass
        await call["loop"].stop()
        call["task"].cancel()
        try:
            await call["task"]
        except (asyncio.CancelledError, Exception):
            pass
        await call["stt"].stop()
        tts_client = call.get("tts_client")
        if tts_client is not None:
            await tts_client.close()
        await call["transport"].leave()
        summary = {
            "event": "voice_call_teardown",
            "reason": reason,
            "room_url": call.get("room_url"),
            "call_s": round(time.monotonic() - call["started_at"], 1),
            "asr_seconds_est": round(call["stt"].asr_seconds_est, 1),
            "stt_provider": call["stt"].provider,
        }
        # Durable + WARNING-level emit: this is the per-call cost record; it
        # must survive log levels/rotation on the agent volume.
        turn_loop.emit_telemetry(summary)
        logger.info("voice: call ended reason=%s", reason)

    async def _create_standalone_room(self) -> Tuple[str, str, str]:
        """Create a private room + two short-lived meeting tokens: one the agent
        joins with, and one embedded in the shareable URL so the owner can join
        the private room (the bare URL cannot). Returns
        (room_url, agent_token, share_url)."""
        import httpx

        daily_key = os.environ["DAILY_API_KEY"]
        headers = {"Authorization": f"Bearer {daily_key}"}
        exp = int(time.time()) + STANDALONE_ROOM_TTL_S
        async with httpx.AsyncClient(timeout=15.0) as client:
            room_resp = await client.post(
                f"{DAILY_API}/rooms", headers=headers,
                json={"privacy": "private", "properties": {"exp": exp}},
            )
            room_resp.raise_for_status()
            room = room_resp.json()

            async def _mint_token(is_owner: bool) -> str:
                resp = await client.post(
                    f"{DAILY_API}/meeting-tokens", headers=headers,
                    json={"properties": {"room_name": room["name"],
                                         "is_owner": is_owner, "exp": exp}},
                )
                resp.raise_for_status()
                return resp.json()["token"]

            agent_token = await _mint_token(False)
            human_token = await _mint_token(True)
        share_url = f"{room['url']}?t={human_token}"
        return room["url"], agent_token, share_url

    async def disconnect(self) -> None:
        await self._end_call("adapter-disconnect")

    async def send(self, chat_id, content, reply_to=None, metadata=None) -> SendResult:
        """Out-of-band sends (cron etc.): speak if a call is live.

        The gateway never routes normal chat through this adapter (the turn
        loop owns the conversation), but the abstract method must exist
        (base.py:2257). Honest failure beats a hidden queue.
        """
        call = self._active_call
        if call is None:
            return SendResult(success=False, error="no active voice call")
        tts = call["loop"]._tts
        if tts is None:
            return SendResult(
                success=False,
                error="agent not mid-utterance; queueing not supported in v0",
            )
        await tts.send_text(content)
        return SendResult(success=True, message_id="voice")

    async def get_chat_info(self, chat_id: str) -> Dict[str, Any]:
        return {"name": "Voice Call", "type": "dm"}
