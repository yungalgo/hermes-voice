"""Fixture-driven unit tests for the voice platform plugin. No live network —
Daily / Deepgram / Cartesia are all faked. Covers plugin registration +
requirements, the Cartesia extra-config parsers, the Flux event -> turn /
barge-in logic in the turn loop (the in-process barge-in is the plugin's
whole point), the 24k->16k resampler, and the plugin-scoped voice_model
slot resolution.

Requires a hermes-agent checkout on sys.path (tests/conftest.py wires it —
see HERMES_AGENT_SRC).
"""

from __future__ import annotations
import asyncio

from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import PlatformConfig

from hermes_voice import adapter as _adapter
from hermes_voice import control
from hermes_voice import cartesia_ink_stt as ink
from hermes_voice import cartesia_tts as cartesia
from hermes_voice import daily_transport
from hermes_voice import deepgram_flux_stt as flux
from hermes_voice import stt as stt_port
from hermes_voice import tts as tts_port
from hermes_voice import turn_loop
from hermes_voice import voice_model


# --------------------------------------------------------------------------- #
# Fakes
# --------------------------------------------------------------------------- #

class _FakeFluxSTT:
    """Yields a fixed list of normalized Flux events, then ends (so the
    turn loop's `async for` over events() returns)."""

    provider = "deepgram_flux"   # the STT port surface the turn loop tags

    def __init__(self, events):
        self._events = events

    async def events(self):
        for ev in self._events:
            yield ev


class _FakeTransport:
    def __init__(self, playing: bool = False):
        self._playing = playing
        self.cleared = 0
        self.first_write_t = None

    def is_playing(self) -> bool:
        return self._playing

    def clear_output(self) -> None:
        self.cleared += 1

    def reset_write_mark(self) -> None:
        pass

    async def send_audio(self, pcm: bytes) -> None:
        pass


async def _noop_tts_factory(on_audio):  # never called: _start_turn is patched
    raise AssertionError("tts_factory should not run in _consume_flux tests")


def _make_loop(events, *, playing=False, extra=None):
    return turn_loop.VoiceTurnLoop(
        _FakeFluxSTT(events), _noop_tts_factory, _FakeTransport(playing=playing),
        extra=extra or {})


# --------------------------------------------------------------------------- #
# Plugin registration + requirements
# --------------------------------------------------------------------------- #

def test_register_registers_voice_platform():
    ctx = MagicMock()
    _adapter.register(ctx)
    ctx.register_platform.assert_called_once()
    kwargs = ctx.register_platform.call_args.kwargs
    assert kwargs["name"] == "voice"
    assert kwargs["label"] == "Voice"
    assert kwargs["required_env"] == ["CARTESIA_API_KEY"]
    assert kwargs["pii_safe"] is True
    assert kwargs["allow_update_command"] is False
    for fn in ("check_fn", "validate_config", "is_connected", "adapter_factory"):
        assert callable(kwargs[fn])


def test_adapter_factory_returns_voice_adapter():
    # Production ordering: plugin discovery registers the platform with the
    # REAL platform_registry before the gateway ever constructs an adapter —
    # that registration is what lets Platform._missing_ mint the "voice"
    # pseudo-member for an out-of-tree plugin. Mirror it here.
    from gateway.platform_registry import PlatformEntry, platform_registry

    ctx = MagicMock()
    _adapter.register(ctx)
    kwargs = ctx.register_platform.call_args.kwargs
    if not platform_registry.is_registered("voice"):
        platform_registry.register(PlatformEntry(
            name=kwargs["name"], label=kwargs["label"],
            adapter_factory=kwargs["adapter_factory"],
            check_fn=kwargs["check_fn"]))

    adapter = kwargs["adapter_factory"](PlatformConfig(enabled=True, extra={}))
    assert isinstance(adapter, _adapter.VoiceAdapter)


def test_check_requirements_needs_shared_transport_and_tts(monkeypatch):
    # Mode-specific signaling, Daily REST, STT, and voice selection are
    # config-aware and deliberately not gated by this config-free hook.
    monkeypatch.setattr(_adapter, "_daily_available", lambda: True)
    monkeypatch.setattr(_adapter, "_websockets_available", lambda: True)
    monkeypatch.delenv("DAILY_API_KEY", raising=False)
    monkeypatch.delenv("DEEPGRAM_API_KEY", raising=False)
    monkeypatch.delenv("SECOND_BRAIN_SIGNALING_KEY", raising=False)
    monkeypatch.delenv("CARTESIA_API_KEY", raising=False)
    assert _adapter.check_requirements() is False
    monkeypatch.setenv("CARTESIA_API_KEY", "ck")
    assert _adapter.check_requirements() is True


def test_check_requirements_false_without_deps(monkeypatch):
    monkeypatch.setattr(_adapter, "_daily_available", lambda: False)
    monkeypatch.setattr(_adapter, "_websockets_available", lambda: True)
    monkeypatch.setenv("CARTESIA_API_KEY", "ck")
    assert _adapter.check_requirements() is False


def test_validate_config_is_mode_and_stt_provider_aware(monkeypatch):
    for name in (
        "DAILY_API_KEY", "DEEPGRAM_API_KEY", "CARTESIA_API_KEY",
        "CARTESIA_VOICE_ID", "SECOND_BRAIN_SIGNALING_URL",
        "SECOND_BRAIN_SIGNALING_KEY",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("CARTESIA_API_KEY", "ck")
    monkeypatch.setenv("CARTESIA_VOICE_ID", "voice")

    standalone = PlatformConfig(enabled=True, extra={})
    orchestrated = PlatformConfig(enabled=True, extra={"mode": "orchestrated"})
    ink = PlatformConfig(enabled=True, extra={"stt_provider": "cartesia_ink"})
    assert _adapter.validate_config(standalone) is False
    monkeypatch.setenv("DAILY_API_KEY", "dk")
    assert _adapter.validate_config(standalone) is False
    monkeypatch.setenv("DEEPGRAM_API_KEY", "dg")
    assert _adapter.validate_config(standalone) is True
    monkeypatch.delenv("DEEPGRAM_API_KEY")
    assert _adapter.validate_config(ink) is True

    # Orchestrated mode neither reads nor requires DAILY_API_KEY.
    monkeypatch.delenv("DAILY_API_KEY")
    monkeypatch.setenv("DEEPGRAM_API_KEY", "dg")
    assert _adapter.validate_config(orchestrated) is False
    monkeypatch.setenv("SECOND_BRAIN_SIGNALING_URL", "https://signal.test/events")
    monkeypatch.setenv("SECOND_BRAIN_SIGNALING_KEY", "signal-key")
    assert _adapter.validate_config(orchestrated) is True
    assert _adapter.validate_config(
        PlatformConfig(enabled=True, extra={"mode": "invalid"})
    ) is False

def test_parse_command_rejects_malformed_and_unknown_keys():
    assert control.parse_command(
        '{"type":"leave_room","callId":"call-1"}'
    ) == {"type": "leave_room", "callId": "call-1"}
    bad_payloads = (
        "[]",
        '{"type":"unknown","callId":"call-1"}',
        '{"type":"leave_room","callId":"call-1","extra":"x"}',
        '{"type":"leave_room","callId":1}',
        '{"type":"join_room","callId":"c","sessionId":"s",'
        '"roomUrl":"u"}',
    )
    for payload in bad_payloads:
        with pytest.raises(ValueError):
            control.parse_command(payload)


class _FakeControlResponse:
    def __init__(self, status_code, lines=(), *, hold=False):
        self.status_code = status_code
        self.request = object()
        self._lines = lines
        self._hold = hold
        self.holding = asyncio.Event()
        self._release = asyncio.Event()
        self.exited = False

    async def aiter_lines(self):
        for line in self._lines:
            yield line
        if self._hold:
            self.holding.set()
            await self._release.wait()


class _FakeControlStream:
    def __init__(self, response):
        self._response = response

    async def __aenter__(self):
        return self._response

    async def __aexit__(self, exc_type, exc, traceback):
        self._response.exited = True


class _FakeControlHttpClient:
    def __init__(self, responses):
        self._responses = list(responses)
        self.stream_calls = []
        self.post_calls = []
        self.options = None
        self.closed = False

    def stream(self, method, url):
        self.stream_calls.append((method, url))
        return _FakeControlStream(self._responses.pop(0))

    async def post(self, url, *, json):
        self.post_calls.append((url, json))
        response = MagicMock()
        response.raise_for_status.return_value = None
        return response

    async def aclose(self):
        self.closed = True


def _install_control_http(monkeypatch, responses):
    client = _FakeControlHttpClient(responses)

    def factory(**options):
        client.options = options
        return client

    monkeypatch.setattr(control.httpx, "AsyncClient", factory)
    return client


@pytest.mark.asyncio
async def test_control_start_blocks_until_first_http_200(monkeypatch):
    responses = [
        _FakeControlResponse(503),
        _FakeControlResponse(200, hold=True),
    ]
    http_client = _install_control_http(monkeypatch, responses)
    sleep_started = asyncio.Event()
    resume_reconnect = asyncio.Event()

    async def gated_sleep(delay):
        sleep_started.set()
        await resume_reconnect.wait()

    monkeypatch.setattr(control.asyncio, "sleep", gated_sleep)
    client = control.VoiceControlClient(
        "https://signal.test/events", "secret", AsyncMock())
    start_task = asyncio.create_task(client.start())
    await sleep_started.wait()
    assert start_task.done() is False

    resume_reconnect.set()
    await start_task

    assert http_client.stream_calls == [
        ("GET", "https://signal.test/events"),
        ("GET", "https://signal.test/events"),
    ]
    assert http_client.options["headers"] == {"Authorization": "Bearer secret"}
    await client.close()
    assert responses[1].exited is True


@pytest.mark.asyncio
async def test_control_reconnect_backoff_resets_after_http_200(monkeypatch):
    responses = [
        _FakeControlResponse(503),
        _FakeControlResponse(503),
        _FakeControlResponse(200),
        _FakeControlResponse(200, hold=True),
    ]
    _install_control_http(monkeypatch, responses)
    real_sleep = asyncio.sleep
    delays = []

    async def record_sleep(delay):
        delays.append(delay)
        await real_sleep(0)

    monkeypatch.setattr(control.asyncio, "sleep", record_sleep)
    monkeypatch.setattr(
        control.random, "uniform", lambda low, high: (low + high) / 2)
    client = control.VoiceControlClient(
        "https://signal.test/events", "secret", AsyncMock())

    await client.start()
    await responses[3].holding.wait()

    assert delays == [0.5, 1.0, 0.5]
    await client.close()


@pytest.mark.asyncio
async def test_control_close_interrupts_open_stream(monkeypatch):
    response = _FakeControlResponse(200, hold=True)
    http_client = _install_control_http(monkeypatch, [response])
    client = control.VoiceControlClient(
        "https://signal.test/events", "secret", AsyncMock())
    await client.start()
    await response.holding.wait()

    await client.close()

    assert response.exited is True
    assert http_client.closed is True


@pytest.mark.asyncio
async def test_control_close_interrupts_backoff_sleep(monkeypatch):
    _install_control_http(monkeypatch, [_FakeControlResponse(503)])
    sleep_started = asyncio.Event()
    sleep_cancelled = asyncio.Event()

    async def blocked_sleep(delay):
        sleep_started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            sleep_cancelled.set()
            raise

    monkeypatch.setattr(control.asyncio, "sleep", blocked_sleep)
    client = control.VoiceControlClient(
        "https://signal.test/events", "secret", AsyncMock())
    start_task = asyncio.create_task(client.start())
    await sleep_started.wait()

    await client.close()

    assert sleep_cancelled.is_set() is True
    with pytest.raises(RuntimeError, match="closed before connecting"):
        await start_task


@pytest.mark.asyncio
async def test_control_post_event_is_strict_and_uses_same_url(monkeypatch):
    http_client = _install_control_http(monkeypatch, [])
    client = control.VoiceControlClient(
        "https://signal.test/events", "secret", AsyncMock())

    with pytest.raises(ValueError):
        await client.post_event({
            "type": "ended", "callId": "call-1", "reason": "left",
            "extra": "rejected",
        })
    with pytest.raises(ValueError):
        await client.post_event({
            "type": "transcript", "callId": "call-1", "sessionId": "s",
            "role": "user", "content": "hello", "final": False,
        })
    event = {"type": "ended", "callId": "call-1", "reason": "left"}
    await client.post_event(event)

    assert http_client.post_calls == [("https://signal.test/events", event)]
    await client.close()


@pytest.mark.asyncio
async def test_standalone_connect_creates_and_joins_room(monkeypatch, tmp_path):
    adapter = _adapter.VoiceAdapter(PlatformConfig(enabled=True, extra={}))
    adapter._create_standalone_room = AsyncMock(
        return_value=("https://room.test/r", "agent-token", "https://share.test/r")
    )
    adapter._start_call = AsyncMock()
    adapter._mark_connected = MagicMock()
    import hermes_constants
    monkeypatch.setattr(hermes_constants, "get_hermes_home", lambda: tmp_path)

    assert await adapter.connect() is True
    adapter._create_standalone_room.assert_awaited_once_with()
    adapter._start_call.assert_awaited_once_with(
        "https://room.test/r", "agent-token")


@pytest.mark.asyncio
async def test_orchestrated_connect_uses_signaling_without_daily(monkeypatch):
    events = []

    class FakeControl:
        def __init__(self, url, bearer, on_command):
            events.append(("init", url, bearer, on_command))

        async def start(self):
            events.append(("start",))

        async def close(self):
            events.append(("close",))

    monkeypatch.delenv("DAILY_API_KEY", raising=False)
    monkeypatch.setenv("SECOND_BRAIN_SIGNALING_URL", "https://signal.test/events")
    monkeypatch.setenv("SECOND_BRAIN_SIGNALING_KEY", "signal-key")
    monkeypatch.setattr(_adapter.control, "VoiceControlClient", FakeControl)
    adapter = _adapter.VoiceAdapter(PlatformConfig(
        enabled=True, extra={"mode": " Orchestrated "}))
    adapter._create_standalone_room = AsyncMock()
    adapter._mark_connected = MagicMock()

    assert await adapter.connect() is True
    adapter._create_standalone_room.assert_not_awaited()
    assert events[0][:3] == (
        "init", "https://signal.test/events", "signal-key")
    assert events[1] == ("start",)


@pytest.mark.asyncio
async def test_join_room_passes_exact_control_fields():
    adapter = _adapter.VoiceAdapter(PlatformConfig(
        enabled=True, extra={"mode": "orchestrated"}))
    adapter._start_call = AsyncMock()
    command = {
        "type": "join_room",
        "callId": "call-1",
        "sessionId": "session-1",
        "roomUrl": "https://room.test/r",
        "token": "agent-token",
    }

    await adapter._handle_control_command(command)

    adapter._start_call.assert_awaited_once_with(
        "https://room.test/r",
        "agent-token",
        call_id="call-1",
        session_id="session-1",
    )


@pytest.mark.asyncio
async def test_join_failure_posts_error_then_ended():
    adapter = _adapter.VoiceAdapter(PlatformConfig(
        enabled=True, extra={"mode": "orchestrated"}))
    adapter._start_call = AsyncMock(side_effect=RuntimeError("join failed"))
    control_client = MagicMock()
    control_client.post_event = AsyncMock()
    adapter._control = control_client

    await adapter._handle_control_command({
        "type": "join_room",
        "callId": "call-1",
        "sessionId": "session-1",
        "roomUrl": "https://room.test/r",
        "token": "agent-token",
    })

    assert [call.args[0] for call in control_client.post_event.await_args_list] == [
        {"type": "error", "callId": "call-1", "code": "agent_join_failed"},
        {"type": "ended", "callId": "call-1", "reason": "agent_join_failed"},
    ]


@pytest.mark.asyncio
async def test_second_join_finishes_old_teardown_before_new_start(monkeypatch):
    order = []

    class FakeSTT:
        provider = "fake-stt"
        asr_seconds_est = 0.0

        def __init__(self, label):
            self.label = label

        async def start(self):
            order.append((self.label, "start"))

        async def stop(self):
            order.append((self.label, "stop"))

        async def send_audio(self, pcm):
            pass

    class FakeTransport:
        abnormal_end = None
        remote_participant_count = 1

        def __init__(self, loop=None, on_audio_in=None, *, label="new"):
            self.label = label

        async def join(self, room_url, token):
            order.append((self.label, "join", room_url, token))

        def begin_teardown(self):
            order.append((self.label, "begin_teardown"))

        async def leave(self):
            order.append((self.label, "leave"))

    class FakeTTS:
        provider = "fake-tts"
        model = "fake-model"
        voice = "fake-voice"

        def __init__(self, label):
            self.label = label

        async def connect(self):
            order.append((self.label, "connect"))

        async def close(self):
            order.append((self.label, "close"))

        def new_turn(self, on_audio):
            raise AssertionError("no turn should start in this lifecycle test")

    class FakeVoiceLoop:
        def __init__(self, stt, tts_factory, transport, *, extra):
            self.label = transport.label
            self._run_forever = asyncio.Event()

        async def run(self):
            await self._run_forever.wait()

        async def stop(self):
            order.append((self.label, "loop_stop"))

    def make_stt(provider, *, input_rate, extra):
        order.append(("new", "create_stt"))
        return FakeSTT("new")

    def make_tts(provider, *, extra):
        order.append(("new", "create_tts"))
        return FakeTTS("new")

    async def post_event(event):
        order.append(("event", event))

    monkeypatch.setattr(_adapter.stt_mod, "make_stt", make_stt)
    monkeypatch.setattr(_adapter.daily_transport, "DailyTransport", FakeTransport)
    monkeypatch.setattr(_adapter.tts_mod, "make_tts", make_tts)
    monkeypatch.setattr(_adapter.turn_loop, "VoiceTurnLoop", FakeVoiceLoop)
    monkeypatch.setattr(
        _adapter.turn_loop, "emit_telemetry",
        lambda summary: order.append(("telemetry", summary["reason"])),
    )
    adapter = _adapter.VoiceAdapter(PlatformConfig(
        enabled=True, extra={"mode": "orchestrated"}))
    control_client = MagicMock()
    control_client.post_event = AsyncMock(side_effect=post_event)
    adapter._control = control_client
    old_task = asyncio.create_task(asyncio.Event().wait())
    adapter._active_call = {
        "stt": FakeSTT("old"),
        "transport": FakeTransport(label="old"),
        "loop": FakeVoiceLoop(
            None, None, FakeTransport(label="old"), extra={}),
        "task": old_task,
        "tts_client": FakeTTS("old"),
        "started_at": 0.0,
        "room_url": "https://room.test/old",
        "call_id": "old-call",
        "session_id": "old-session",
    }

    await adapter._handle_control_command({
        "type": "join_room",
        "callId": "new-call",
        "sessionId": "new-session",
        "roomUrl": "https://room.test/new",
        "token": "new-token",
    })

    old_leave = order.index(("old", "leave"))
    ended = order.index(("event", {
        "type": "ended",
        "callId": "old-call",
        "reason": "replaced-by-new-call",
    }))
    new_create = order.index(("new", "create_stt"))
    assert old_leave < ended < new_create
    assert adapter._active_call["call_id"] == "new-call"
    assert adapter._active_call["session_id"] == "new-session"

    adapter._control = None
    await adapter._end_call("test-cleanup")


@pytest.mark.asyncio
async def test_stale_leave_room_is_noop():
    adapter = _adapter.VoiceAdapter(PlatformConfig(
        enabled=True, extra={"mode": "orchestrated"}))
    adapter._active_call = {"call_id": "current-call"}
    adapter._end_call_locked = AsyncMock()

    await adapter._handle_control_command({
        "type": "leave_room", "callId": "stale-call"})

    adapter._end_call_locked.assert_not_awaited()


@pytest.mark.asyncio
async def test_disconnect_ends_media_before_control():
    order = []
    adapter = _adapter.VoiceAdapter(PlatformConfig(
        enabled=True, extra={"mode": "orchestrated"}))

    async def end_call(reason):
        order.append(("media", reason))

    control_client = MagicMock()
    control_client.close = AsyncMock(side_effect=lambda: order.append(("control",)))
    adapter._end_call = end_call
    adapter._control = control_client

    await adapter.disconnect()

    assert order == [("media", "adapter-disconnect"), ("control",)]

# --------------------------------------------------------------------------- #
# Extra-config parsers
# --------------------------------------------------------------------------- #

def test_resolve_float_extra_default_and_parse():
    assert _adapter._resolve_float_extra({}, "eot_threshold", 0.7) == 0.7
    assert _adapter._resolve_float_extra({"eot_threshold": "0.5"}, "eot_threshold", 0.7) == 0.5
    # garbage falls back to the default, never raises
    assert _adapter._resolve_float_extra({"eot_threshold": "nope"}, "eot_threshold", 0.7) == 0.7


def test_opt_float_extra_returns_none_when_unset():
    assert tts_port._opt_float_extra({}, "tts_speed") is None
    assert tts_port._opt_float_extra({"tts_speed": ""}, "tts_speed") is None
    assert tts_port._opt_float_extra({"tts_speed": "1.2"}, "tts_speed") == 1.2


# --------------------------------------------------------------------------- #
# tts port — make_tts factory (provider slot)
# --------------------------------------------------------------------------- #

def test_make_tts_unknown_provider_raises_clear():
    with pytest.raises(RuntimeError, match="unknown tts_provider"):
        tts_port.make_tts("elevenlabs", extra={})


def test_make_tts_missing_key_raises_clear(monkeypatch):
    monkeypatch.delenv("CARTESIA_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="CARTESIA_API_KEY is not set"):
        tts_port.make_tts(None, extra={})


def test_make_tts_missing_voice_raises_clear(monkeypatch):
    monkeypatch.setenv("CARTESIA_API_KEY", "ck")
    monkeypatch.delenv("CARTESIA_VOICE_ID", raising=False)
    with pytest.raises(ValueError, match="voice_id is required"):
        tts_port.make_tts("cartesia", extra={})


def test_make_tts_builds_cartesia_with_port_attrs(monkeypatch):
    monkeypatch.setenv("CARTESIA_API_KEY", "ck")
    client = tts_port.make_tts(
        None,  # None/empty -> DEFAULT_TTS_PROVIDER
        extra={"cartesia_voice_id": "v-1", "cartesia_model": "sonic-x",
               "tts_speed": "1.2"},
    )
    # Public port attributes the adapter logs/telemetry rely on.
    assert client.provider == "cartesia"
    assert client.voice == "v-1"
    assert client.model == "sonic-x"
    assert client._gen_cfg["speed"] == 1.2


# --------------------------------------------------------------------------- #
# Flux event -> turn / barge-in logic (the moat)
# --------------------------------------------------------------------------- #

@pytest.mark.asyncio
async def test_eager_then_end_starts_a_single_turn(monkeypatch):
    loop = _make_loop([
        {"event": "eager_end_of_turn", "transcript": "hello there"},
        {"event": "end_of_turn", "transcript": "hello there"},
    ])
    started = []
    monkeypatch.setattr(loop, "_start_turn",
                        lambda text, **k: started.append(text))
    await loop._consume_flux()
    # Eager starts the turn; the matching end_of_turn lets it stand (no restart).
    assert started == ["hello there"]


@pytest.mark.asyncio
async def test_end_of_turn_without_eager_starts_turn(monkeypatch):
    loop = _make_loop([
        {"event": "end_of_turn", "transcript": "what time is it"},
    ])
    started = []
    monkeypatch.setattr(loop, "_start_turn",
                        lambda text, **k: started.append(text))
    await loop._consume_flux()
    assert started == ["what time is it"]


@pytest.mark.asyncio
async def test_start_of_turn_barges_in_while_audio_playing(monkeypatch):
    # State is LISTENING but the transport is still playing the agent's tail —
    # barge-in must fire on that tail, which is the bug the is_playing() check
    # fixes.
    loop = _make_loop(
        [{"event": "start_of_turn", "transcript": "wait"}], playing=True)
    barged = []

    async def _fake_barge():
        barged.append(True)

    monkeypatch.setattr(loop, "_barge_in", _fake_barge)
    await loop._consume_flux()
    assert barged == [True]


@pytest.mark.asyncio
async def test_start_of_turn_idle_does_not_barge(monkeypatch):
    loop = _make_loop(
        [{"event": "start_of_turn", "transcript": "hi"}], playing=False)
    barged = []

    async def _fake_barge():
        barged.append(True)

    monkeypatch.setattr(loop, "_barge_in", _fake_barge)
    await loop._consume_flux()
    assert barged == []   # nothing playing -> pre-warm, not barge


@pytest.mark.asyncio
async def test_start_of_turn_respects_allow_interruptions_false(monkeypatch):
    loop = _make_loop(
        [{"event": "start_of_turn", "transcript": "wait"}],
        playing=True, extra={"allow_interruptions": False})
    barged = []

    async def _fake_barge():
        barged.append(True)

    monkeypatch.setattr(loop, "_barge_in", _fake_barge)
    await loop._consume_flux()
    assert barged == []


@pytest.mark.asyncio
async def test_turn_resumed_cancels_the_speculative_turn(monkeypatch):
    loop = _make_loop([
        {"event": "eager_end_of_turn", "transcript": "hello"},
        {"event": "turn_resumed", "transcript": ""},
    ])
    monkeypatch.setattr(loop, "_start_turn", lambda text, **k: None)
    barged = []

    async def _fake_barge():
        barged.append(True)

    monkeypatch.setattr(loop, "_barge_in", _fake_barge)
    await loop._consume_flux()
    assert barged == [True]   # the user kept talking -> kill the eager turn


def test_resolve_bool_extra():
    assert turn_loop._resolve_bool_extra({}, "k", True) is True
    assert turn_loop._resolve_bool_extra({"k": "false"}, "k", True) is False
    assert turn_loop._resolve_bool_extra({"k": "on"}, "k", False) is True
    assert turn_loop._resolve_bool_extra({"k": False}, "k", True) is False


# --------------------------------------------------------------------------- #
# Flux 24k -> 16k resampler (pure function)
# --------------------------------------------------------------------------- #

def test_resample_to_16k_reduces_sample_count():
    pcm = bytes(2400 * 2)   # 2400 s16le samples @ 24k (silence)
    out = flux._resample_to_16k(pcm, 24000)
    # 24k -> 16k is a 2/3 ratio: 2400 -> 1600 samples (3200 bytes).
    assert len(out) == 1600 * 2


def test_resample_to_16k_interpolates_a_ramp():
    import array
    src = array.array("h", [0, 300, 600, 900])   # 4 samples
    out = array.array("h")
    out.frombytes(flux._resample_to_16k(src.tobytes(), 24000))
    # 4 -> round(4*2/3)=3 samples, endpoints preserved, middle interpolated.
    assert len(out) == 3
    assert out[0] == 0 and out[-1] == 900
    assert 0 < out[1] < 900


def test_resample_to_16k_passthrough_at_16k():
    pcm = b"\x01\x00\x02\x00\x03\x00"
    assert flux._resample_to_16k(pcm, 16000) == pcm


def test_resample_to_16k_empty():
    assert flux._resample_to_16k(b"", 24000) == b""


# --------------------------------------------------------------------------- #
# Flux wire-message normalization (the event contract the turn loop consumes)
# --------------------------------------------------------------------------- #

def test_flux_normalize_start_of_turn():
    out = flux._normalize_flux_message(
        {"type": "TurnInfo", "event": "StartOfTurn", "transcript": "hi",
         "end_of_turn_confidence": 0.2})
    assert out["event"] == "start_of_turn"
    assert out["transcript"] == "hi"
    assert out["confidence"] == 0.2


def test_flux_normalize_eager_and_end_events():
    eager = flux._normalize_flux_message(
        {"type": "TurnInfo", "event": "EagerEndOfTurn"})
    end = flux._normalize_flux_message(
        {"type": "TurnInfo", "event": "EndOfTurn"})
    resumed = flux._normalize_flux_message(
        {"type": "TurnInfo", "event": "TurnResumed"})
    assert eager["event"] == "eager_end_of_turn"
    assert end["event"] == "end_of_turn"
    assert resumed["event"] == "turn_resumed"


def test_flux_normalize_missing_transcript_becomes_empty():
    out = flux._normalize_flux_message(
        {"type": "TurnInfo", "event": "StartOfTurn", "transcript": None})
    assert out["transcript"] == ""


def test_flux_normalize_drops_unknown_and_non_turninfo():
    assert flux._normalize_flux_message(
        {"type": "TurnInfo", "event": "Bogus"}) is None
    assert flux._normalize_flux_message({"type": "Metadata"}) is None
    assert flux._normalize_flux_message({"type": "Connected"}) is None


# --------------------------------------------------------------------------- #
# Daily transport queue logic (the barge-in audio path — no SDK needed)
# --------------------------------------------------------------------------- #

async def _noop_audio_in(pcm: bytes) -> None:
    pass


@pytest.mark.asyncio
async def test_transport_send_audio_splits_large_pcm():
    import asyncio
    t = daily_transport.DailyTransport(asyncio.get_running_loop(), _noop_audio_in)
    big = bytes(daily_transport.OUT_CHUNK_BYTES * 3 + 100)
    await t.send_audio(big)
    sizes = []
    while not t._out_q.empty():
        sizes.append(len(t._out_q.get_nowait()))
    assert len(sizes) == 4  # 3 full chunks + a 100-byte remainder
    assert all(s <= daily_transport.OUT_CHUNK_BYTES for s in sizes)
    assert sum(sizes) == len(big)


@pytest.mark.asyncio
async def test_transport_send_audio_small_stays_one_chunk():
    import asyncio
    t = daily_transport.DailyTransport(asyncio.get_running_loop(), _noop_audio_in)
    await t.send_audio(b"\x00" * 200)
    assert t._out_q.qsize() == 1


@pytest.mark.asyncio
async def test_transport_is_playing_and_clear_output():
    import asyncio
    t = daily_transport.DailyTransport(asyncio.get_running_loop(), _noop_audio_in)
    assert t.is_playing() is False
    await t.send_audio(b"\x00" * 200)
    assert t.is_playing() is True
    t.clear_output()                      # barge-in drops queued audio
    assert t.is_playing() is False


# --------------------------------------------------------------------------- #
# Cartesia request builder (pure) + the <flush> mapping
# --------------------------------------------------------------------------- #

def test_cartesia_request_basic_payload():
    c = cartesia.CartesiaTTSClient("k", "voice-1")
    req = c._request("ctx-1", "hello there", True)
    assert req["transcript"] == "hello there"
    assert req["voice"] == {"mode": "id", "id": "voice-1"}
    assert req["output_format"]["encoding"] == "pcm_s16le"
    assert req["output_format"]["sample_rate"] == cartesia.OUTPUT_SAMPLE_RATE
    assert req["context_id"] == "ctx-1"
    assert req["continue"] is True
    assert "flush" not in req
    assert "generation_config" not in req


def test_cartesia_request_flush_maps_to_flag():
    c = cartesia.CartesiaTTSClient("k", "voice-1")
    req = c._request("ctx-1", "", True, flush=True)
    assert req["flush"] is True
    assert req["transcript"] == ""


def test_cartesia_request_carries_generation_config():
    c = cartesia.CartesiaTTSClient("k", "voice-1", speed=1.2, emotion="calm")
    req = c._request("ctx-1", "hi", False)
    assert req["generation_config"] == {"speed": 1.2, "emotion": "calm"}
    assert req["continue"] is False


def test_cartesia_requires_a_voice_id():
    with pytest.raises(ValueError):
        cartesia.CartesiaTTSClient("k", "")


# --------------------------------------------------------------------------- #
# Cartesia Ink-2 wire-message normalization (the A/B adapter's event contract)
# --------------------------------------------------------------------------- #

def test_ink_normalize_maps_every_turn_event():
    # The whole point of the seam: Ink-2's turn.* events normalize onto the
    # SAME EV_* contract Flux produces, so the turn loop drives both identically.
    cases = {
        "turn.start": stt_port.EV_START,
        "turn.update": stt_port.EV_UPDATE,
        "turn.eager_end": stt_port.EV_EAGER_EOT,
        "turn.resume": stt_port.EV_RESUMED,
        "turn.end": stt_port.EV_END,
    }
    for raw_type, expected in cases.items():
        out = ink._normalize_ink_message({"type": raw_type})
        assert out is not None
        assert out["event"] == expected


def test_ink_normalize_resume_is_turn_resumed():
    # Explicit: the resume event must map to turn_resumed so the loop cancels
    # the speculative (eager) turn — the trickiest mapping to get right.
    out = ink._normalize_ink_message({"type": "turn.resume"})
    assert out["event"] == "turn_resumed"


def test_ink_normalize_transcript_field():
    # Verified live: the transcript lives in "transcript".
    assert ink._normalize_ink_message(
        {"type": "turn.end", "transcript": "hi there"})["transcript"] == "hi there"


def test_ink_normalize_defaults_missing_fields():
    out = ink._normalize_ink_message({"type": "turn.start"})
    assert out["transcript"] == ""
    assert out["confidence"] is None
    assert out["words"] == []
    assert out["turn_index"] is None


def test_ink_normalize_carries_turn_id():
    # Verified live: id field is "turn_id" (string); Ink-2 sends no confidence
    # or per-word list, so those stay None/[] for shape-parity with Flux.
    out = ink._normalize_ink_message({
        "type": "turn.update", "transcript": "partial", "turn_id": "3"})
    assert out["transcript"] == "partial"
    assert out["turn_index"] == "3"
    assert out["confidence"] is None
    assert out["words"] == []


def test_ink_normalize_drops_unknown_types():
    assert ink._normalize_ink_message({"type": "metadata"}) is None
    assert ink._normalize_ink_message({"type": "error", "error": "x"}) is None
    assert ink._normalize_ink_message({}) is None


def test_ink_requires_api_key():
    with pytest.raises(ValueError):
        ink.CartesiaInkSTT("")


# --------------------------------------------------------------------------- #
# STT port surface + factory (both providers behind one seam)
# --------------------------------------------------------------------------- #

def test_both_providers_expose_provider_tag():
    dg = flux.DeepgramFluxSTT("dg-key")
    ck = ink.CartesiaInkSTT("ck-key")
    assert dg.provider == "deepgram_flux"
    assert ck.provider == "cartesia_ink"


def test_both_providers_satisfy_the_port_surface():
    dg = flux.DeepgramFluxSTT("dg-key")
    ck = ink.CartesiaInkSTT("ck-key")
    for obj in (dg, ck):
        for attr in ("start", "send_audio", "events", "configure", "stop"):
            assert callable(getattr(obj, attr))
        assert isinstance(obj.asr_seconds_est, float)
        assert isinstance(obj.provider, str)
        # runtime_checkable Protocol: both adapters are STT instances.
        assert isinstance(obj, stt_port.STT)


def test_ev_constants_reexported_from_flux():
    # Existing `from deepgram_flux_stt import EV_*` imports must keep working
    # even though the canonical home moved to stt.py.
    assert flux.EV_START == stt_port.EV_START == "start_of_turn"
    assert flux.EV_RESUMED == stt_port.EV_RESUMED == "turn_resumed"
    assert flux.EV_END == stt_port.EV_END == "end_of_turn"


def test_make_stt_defaults_to_flux(monkeypatch):
    monkeypatch.setenv("DEEPGRAM_API_KEY", "dg")
    stt = stt_port.make_stt("deepgram_flux", input_rate=24000, extra={})
    assert stt.provider == "deepgram_flux"


def test_make_stt_builds_cartesia_ink(monkeypatch):
    monkeypatch.setenv("CARTESIA_API_KEY", "ck")
    stt = stt_port.make_stt("cartesia_ink", input_rate=24000, extra={})
    assert stt.provider == "cartesia_ink"


def test_make_stt_passes_flux_thresholds(monkeypatch):
    monkeypatch.setenv("DEEPGRAM_API_KEY", "dg")
    stt = stt_port.make_stt(
        "deepgram_flux", input_rate=24000,
        extra={"eot_threshold": "0.8", "eager_eot_threshold": "0.4"})
    assert stt._eot_threshold == 0.8
    assert stt._eager_eot_threshold == 0.4


def test_make_stt_unknown_provider_raises(monkeypatch):
    monkeypatch.setenv("DEEPGRAM_API_KEY", "dg")
    with pytest.raises(RuntimeError, match="unknown stt_provider"):
        stt_port.make_stt("whisper-x", input_rate=24000, extra={})


def test_make_stt_missing_key_raises_clear(monkeypatch):
    monkeypatch.delenv("CARTESIA_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="CARTESIA_API_KEY is not set"):
        stt_port.make_stt("cartesia_ink", input_rate=24000, extra={})


@pytest.mark.asyncio
async def test_ink_configure_is_noop():
    ck = ink.CartesiaInkSTT("ck-key")
    # No socket open; must not raise, and must do nothing harmful.
    await ck.configure(eot_threshold=0.9)
    await ck.configure()


# --------------------------------------------------------------------------- #
# voice_model — plugin-scoped slot resolution (rehomed from gateway/run.py)
# --------------------------------------------------------------------------- #

@pytest.fixture()
def _no_voice_env(monkeypatch):
    for var in ("VOICE_MODEL_PROVIDER", "VOICE_MODEL_NAME",
                "VOICE_MODEL_BASE_URL", "VOICE_MODEL_KEY_ENV"):
        monkeypatch.delenv(var, raising=False)


def test_read_voice_model_cfg_absent_means_main_model(_no_voice_env):
    assert voice_model.read_voice_model_cfg(None) is None
    assert voice_model.read_voice_model_cfg({}) is None
    # non-dict / missing model / auto provider all mean "ride the main model"
    assert voice_model.read_voice_model_cfg({"voice_model": "groq"}) is None
    assert voice_model.read_voice_model_cfg(
        {"voice_model": {"provider": "groq"}}) is None
    assert voice_model.read_voice_model_cfg(
        {"voice_model": {"provider": "auto", "model": "m"}}) is None
    assert voice_model.read_voice_model_cfg(
        {"voice_model": {"provider": "", "model": "m"}}) is None


def test_read_voice_model_cfg_from_extra(_no_voice_env):
    vm = {"provider": "groq", "model": "llama-4-scout"}
    assert voice_model.read_voice_model_cfg({"voice_model": vm}) == vm


def test_read_voice_model_cfg_env_wins_over_extra(_no_voice_env, monkeypatch):
    monkeypatch.setenv("VOICE_MODEL_PROVIDER", "cerebras")
    monkeypatch.setenv("VOICE_MODEL_NAME", "env-model")
    cfg = voice_model.read_voice_model_cfg(
        {"voice_model": {"provider": "groq", "model": "yaml-model"}})
    assert cfg == {"provider": "cerebras", "model": "env-model"}
    # env provider "auto" disables the slot even when extra is set
    monkeypatch.setenv("VOICE_MODEL_PROVIDER", "auto")
    assert voice_model.read_voice_model_cfg(
        {"voice_model": {"provider": "groq", "model": "yaml-model"}}) is None


def test_resolve_voice_runtime_kwargs_merges_runtime_and_model(
        _no_voice_env, monkeypatch):
    import hermes_cli.runtime_provider as rp

    seen = {}

    def fake_resolve(*, requested, explicit_base_url, explicit_api_key,
                     target_model):
        seen.update(requested=requested, api_key=explicit_api_key)
        return {"api_key": "rk", "base_url": "https://x", "provider": requested,
                "api_mode": "chat", "command": None, "args": None,
                "credential_pool": None}

    monkeypatch.setattr(rp, "resolve_runtime_provider", fake_resolve)
    monkeypatch.setenv("MY_KEY_ENV", "indirect-key")
    out = voice_model.resolve_voice_runtime_kwargs(
        {"voice_model": {"provider": "groq", "model": "llama-4-scout",
                         "key_env": "MY_KEY_ENV"}})
    assert out is not None
    assert out["model"] == "llama-4-scout"
    assert out["provider"] == "groq"
    assert out["args"] == []
    # key_env indirection resolved before the provider lookup
    assert seen == {"requested": "groq", "api_key": "indirect-key"}


def test_resolve_voice_runtime_kwargs_failure_falls_back(
        _no_voice_env, monkeypatch):
    import hermes_cli.runtime_provider as rp

    def boom(**_kwargs):
        raise RuntimeError("no such provider")

    monkeypatch.setattr(rp, "resolve_runtime_provider", boom)
    # A bad override must never raise into the call path — main model rides.
    assert voice_model.resolve_voice_runtime_kwargs(
        {"voice_model": {"provider": "nope", "model": "m"}}) is None


def test_voice_model_name_for_telemetry(_no_voice_env):
    assert voice_model.voice_model_name({}) is None
    assert voice_model.voice_model_name(
        {"voice_model": {"provider": "groq", "model": "llama-4-scout"}}
    ) == "llama-4-scout"
