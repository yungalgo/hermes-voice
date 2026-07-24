"""Authenticated outbound control channel for orchestrated voice calls."""

from __future__ import annotations

import asyncio
import inspect
import json
import logging
import random
from collections.abc import Awaitable, Callable
from typing import Any

import httpx

logger = logging.getLogger(__name__)

Command = dict[str, str]
CommandHandler = Callable[[Command], Awaitable[None]]
_RECONNECT_DELAYS = (0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 30.0)
_COMMAND_KEYS = {
    "join_room": frozenset(("type", "callId", "sessionId", "roomUrl", "token")),
    "leave_room": frozenset(("type", "callId")),
}
_EVENT_KEYS = {
    "status": frozenset(("type", "callId", "status")),
    "transcript": frozenset(
        ("type", "callId", "sessionId", "role", "content", "final")
    ),
    "ended": frozenset(("type", "callId", "reason")),
    "error": frozenset(("type", "callId", "code")),
}


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"duplicate JSON key: {key}")
        value[key] = item
    return value


def parse_command(data: str) -> Command:
    """Parse one SSE data payload into an exact supported command shape."""
    try:
        command = json.loads(data, object_pairs_hook=_reject_duplicate_keys)
    except (json.JSONDecodeError, ValueError) as exc:
        raise ValueError("command is not valid JSON") from exc
    if type(command) is not dict:
        raise ValueError("command must be a JSON object")
    command_type = command.get("type")
    if type(command_type) is not str or command_type not in _COMMAND_KEYS:
        raise ValueError("unknown command type")
    if set(command) != _COMMAND_KEYS[command_type]:
        raise ValueError(f"invalid keys for {command_type}")
    if any(type(value) is not str for value in command.values()):
        raise ValueError("all command fields must be strings")
    return command


def _validate_event(event: dict[str, Any]) -> None:
    if type(event) is not dict:
        raise ValueError("event must be a dict")
    event_type = event.get("type")
    if type(event_type) is not str or event_type not in _EVENT_KEYS:
        raise ValueError("unknown event type")
    if set(event) != _EVENT_KEYS[event_type]:
        raise ValueError(f"invalid keys for {event_type}")
    if event_type == "transcript":
        string_keys = ("type", "callId", "sessionId", "role", "content")
        if any(type(event[key]) is not str for key in string_keys):
            raise ValueError("transcript string fields must be strings")
        if event["role"] not in ("user", "assistant") or event["final"] is not True:
            raise ValueError("invalid transcript role or final flag")
        return
    if any(type(value) is not str for value in event.values()):
        raise ValueError("all event fields must be strings")
    if event_type == "status" and event["status"] not in (
        "listening",
        "thinking",
        "speaking",
    ):
        raise ValueError("invalid voice status")


class VoiceControlClient:
    """Maintain one authenticated SSE stream and POST events to its URL."""

    def __init__(
        self,
        event_url: str,
        bearer: str,
        on_command: CommandHandler,
    ) -> None:
        event_url = event_url.strip()
        bearer = bearer.strip()
        if not event_url or not bearer:
            raise ValueError("event_url and bearer are required")
        self._event_url = event_url
        self._on_command = on_command
        self._client = httpx.AsyncClient(
            headers={"Authorization": f"Bearer {bearer}"},
            timeout=httpx.Timeout(15.0, read=None),
        )
        self._first_connection = asyncio.Event()
        self._task: asyncio.Task[None] | None = None
        self._closed = False

    async def start(self) -> None:
        """Start reconnecting and wait until the first HTTP 200 SSE stream."""
        if self._closed:
            raise RuntimeError("voice control client is closed")
        if self._task is None:
            self._task = asyncio.create_task(self._run(), name="voice-control-sse")
        try:
            await self._first_connection.wait()
            if self._closed:
                raise RuntimeError("voice control client closed before connecting")
        except asyncio.CancelledError:
            await self.close()
            raise

    async def close(self) -> None:
        """Cancel an active stream or backoff sleep, then close the client."""
        if self._closed:
            return
        self._closed = True
        task = self._task
        if task is not None:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
        self._first_connection.set()
        await self._client.aclose()

    async def post_event(self, event: dict[str, Any]) -> None:
        """Validate and POST one callback to the same signaling URL."""
        if self._closed:
            raise RuntimeError("voice control client is closed")
        _validate_event(event)
        response = await self._client.post(self._event_url, json=event)
        response.raise_for_status()

    async def _run(self) -> None:
        delay_index = 0
        while True:
            try:
                async with self._client.stream("GET", self._event_url) as response:
                    if response.status_code != 200:
                        raise httpx.HTTPStatusError(
                            f"SSE endpoint returned HTTP {response.status_code}",
                            request=response.request,
                            response=response,
                        )
                    self._first_connection.set()
                    delay_index = 0
                    await self._consume(response)
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.warning("voice: control stream disconnected; reconnecting", exc_info=True)

            base_delay = _RECONNECT_DELAYS[
                min(delay_index, len(_RECONNECT_DELAYS) - 1)
            ]
            delay_index += 1
            await asyncio.sleep(random.uniform(base_delay * 0.8, base_delay * 1.2))

    async def _consume(self, response: httpx.Response) -> None:
        data_lines: list[str] = []
        async for line in response.aiter_lines():
            if line == "":
                await self._dispatch(data_lines)
                data_lines.clear()
            elif line.startswith("data:"):
                value = line[5:]
                data_lines.append(value[1:] if value.startswith(" ") else value)
        await self._dispatch(data_lines)

    async def _dispatch(self, data_lines: list[str]) -> None:
        if not data_lines:
            return
        try:
            command = parse_command("\n".join(data_lines))
        except ValueError as exc:
            logger.warning("voice: rejected malformed control command: %s", exc)
            return
        result = self._on_command(command)
        if not inspect.isawaitable(result):
            raise TypeError("on_command must return an awaitable")
        await result
