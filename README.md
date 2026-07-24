# hermes-voice

Realtime voice platform plugin for [hermes-agent](https://github.com/NousResearch/hermes-agent):
call your agent over WebRTC — real-time, full-duplex, with in-process barge-in.

The plugin joins a [Daily](https://daily.co) (WebRTC) room as a participant,
listens with [Deepgram Flux](https://developers.deepgram.com/docs/flux)
(streaming speech-to-text with model-integrated turn detection), thinks with
your agent, and speaks back with [Cartesia](https://cartesia.ai) (streaming
text-to-speech) — all in real time. Unlike an external voice orchestrator that
wraps the model as a black box, the turn loop runs **inside** the agent
session: a barge-in cancels the LLM's actual generation (not just the audio),
long tool calls get a spoken acknowledgment instead of silence, and voice
shares the same agent, tools, and memory as every other channel.

Originally proposed in-tree as
[NousResearch/hermes-agent#51827](https://github.com/NousResearch/hermes-agent/pull/51827);
repackaged as a standalone plugin per the project's standing policy that
third-party service integrations ship as standalone plugin repos.

## Install

```bash
hermes plugins install yungalgo/hermes-voice
```

(git-clones into `~/.hermes/plugins/` and prompts to enable). Or install as
a pip package straight from the repo — this registers the
`hermes_agent.plugins` entry point:

```bash
pip install git+https://github.com/yungalgo/hermes-voice
```

Either way, also make sure the plugin's own dependencies are importable in
hermes' environment (the git route does not install them):

```bash
pip install daily-python==0.29.1 'websockets>=14,<16' 'httpx>=0.27,<1'
```

Non-bundled plugins are **opt-in**: enable it once with

```bash
hermes plugins enable voice-platform
```

(or add `voice-platform` to `plugins.enabled` in `config.yaml`).

Every mode needs `CARTESIA_API_KEY` and a Cartesia voice selected through
`CARTESIA_VOICE_ID` or `extra.cartesia_voice_id`. The selected STT provider
also determines its credential: the default `deepgram_flux` needs
`DEEPGRAM_API_KEY`; `cartesia_ink` reuses `CARTESIA_API_KEY`.

Standalone mode additionally needs `DAILY_API_KEY`. Orchestrated mode does
not read that key; it instead requires `SECOND_BRAIN_SIGNALING_URL` and the
dedicated bearer secret `SECOND_BRAIN_SIGNALING_KEY`.

## A faster model for voice (optional)

Real-time voice is latency-critical, so you can point voice turns at a fast,
non-reasoning model while your main model stays whatever you normally use.
The slot is plugin-scoped config:

```yaml
gateway:
  platforms:
    voice:
      extra:
        voice_model:
          provider: groq
          model: meta-llama/llama-4-scout-17b-16e-instruct
```

or, env-only (env wins over YAML): `VOICE_MODEL_PROVIDER` + `VOICE_MODEL_NAME`
(plus optional `VOICE_MODEL_BASE_URL` / `VOICE_MODEL_KEY_ENV`).

Leave it out (or set `provider: auto`) and voice rides your main model. A
mis-configured slot logs a warning and falls back to the main model — it never
fails a live call.

## Configuration

Standalone is the default. Set `extra.mode` to `orchestrated` only when an
external controller owns room allocation. Other tuning is shared by both modes:

```yaml
gateway:
  platforms:
    voice:
      extra:
        mode: standalone             # or orchestrated
        stt_provider: deepgram_flux  # or cartesia_ink (rides the Cartesia key)
        tts_provider: cartesia       # the bundled TTS adapter
        cartesia_voice_id: ""        # Cartesia voice id (or set CARTESIA_VOICE_ID)
        cartesia_model: sonic-3.5
        eot_threshold: 0.7           # Flux end-of-turn confidence (higher = waits longer)
        eager_eot_threshold: 0.5     # Flux eager/speculative end-of-turn
        allow_interruptions: true    # barge-in when the caller starts speaking
        tts_speed: ""                # optional Cartesia speed (e.g. 1.0)
        greeting_text: "Hi, how can I help?" # optional literal text spoken when the call opens
        idle_teardown_s: 60          # hang up after the caller is gone this long
        max_call_s: 1800             # hard call-duration cap
```

`greeting_text` is literal spoken text sent directly to TTS; it is not an LLM
instruction or prompt.

### Provider slots

Every leg of the loop sits behind an explicit slot; today's catalog is
deliberately small:

| Leg | Port / slot | Bundled adapters |
| --- | --- | --- |
| LLM | hermes provider registry + `voice_model` | everything hermes supports — nothing to add here |
| STT | `stt.py` (`extra.stt_provider`) | `deepgram_flux` (default), `cartesia_ink` |
| TTS | `tts.py` (`extra.tts_provider`) | `cartesia` |

Adding a provider is one adapter module implementing the port's Protocol plus
a branch in its `make_stt`/`make_tts` factory — the turn loop only ever sees
normalized turn events (STT) and the per-turn synthesis contract (TTS). The
contracts are documented in each port module; the non-negotiable capabilities
are model-integrated turn events (or a VAD shim) for STT, and mid-stream
abort (barge-in) for TTS.

## Making a call

With the keys set and the plugin enabled, start the gateway (`hermes gateway`).
The agent creates its own private Daily room, joins it, and hands you the join
URL on startup — **printed to stdout** and written to
`<hermes home>/voice-call-url.txt`:

```
voice: standalone call ready — open this URL to talk to your agent (keep the token private):
    https://<you>.daily.co/<room>?t=<token>
```

(Log lines redact token-shaped strings, so always take the URL from stdout or
the file, never from a log file.)

Open that URL in a browser, allow the microphone, and talk. The room is
private, so the URL carries a short-lived owner **token** — that token is the
join permission; share the URL only with people you want to reach your agent.
The room is short-lived (~1 hour), and the agent ends the call automatically
when the caller leaves, the room expires, or a maximum-duration cap is hit.

### Orchestrated calls

With `extra.mode: orchestrated`, startup opens an authenticated outbound SSE
stream at `SECOND_BRAIN_SIGNALING_URL`; it does not create or join a room.
The controller sends exact `join_room` commands containing `callId`,
`sessionId`, `roomUrl`, and the agent token. `leave_room` names the active
`callId`. Commands with missing, extra, or incorrectly typed fields are
rejected. The same URL receives strict JSON event callbacks over authenticated
POST requests. Only one call is active; a new join fully ends the old call.

## Notes

- One active call per agent process (the WebRTC virtual audio devices are
  process-level singletons).
- Per-call cost telemetry (STT seconds, timing legs) is written to
  `voice-telemetry.jsonl` under the hermes home (override the path with
  `VOICE_TELEMETRY_SINK`).
- The plugin runs in-process inside hermes-agent and imports a few of its
  internal helpers (`gateway.run`, `hermes_cli.tools_config`,
  `hermes_cli.runtime_provider`); pin your hermes-agent version and run this
  repo's test suite against it before upgrading either side.

## Development

```bash
git clone https://github.com/yungalgo/hermes-voice && cd hermes-voice
# tests import gateway.* / hermes_cli.* from a hermes-agent checkout:
export HERMES_AGENT_SRC=/path/to/hermes-agent
uv run --with-editable . --project "$HERMES_AGENT_SRC" pytest tests
```
