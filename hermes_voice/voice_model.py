"""Dedicated voice-model slot — plugin-scoped config resolution.

History: upstream PR #51827 proposed ``voice_model:`` as a top-level
config.yaml sibling of ``model:``/``fallback_model:``, which required core
edits (``gateway/run.py`` resolution helpers + ``hermes_cli/config.py``
root-key validation). Upstream policy forbids plugins touching core, so the
slot is plugin-scoped here, with identical semantics:

Config (preferred; written by an operator or control plane)::

    gateway:
      platforms:
        voice:
          extra:
            voice_model:
              provider: groq
              model: meta-llama/llama-4-scout-17b-16e-instruct
              # optional: api_key / key_env (or api_key_env) / base_url

Env fallback (hermes convention: env wins over YAML)::

    VOICE_MODEL_PROVIDER / VOICE_MODEL_NAME
    [VOICE_MODEL_BASE_URL / VOICE_MODEL_KEY_ENV]

Semantics (unchanged from the PR): an absent slot, an empty ``model``, or
``provider`` empty/``auto`` all mean "voice turns ride the main model" and
resolve to ``None``. A resolution failure logs a warning and falls back to
the main model — a bad override never fails a live call.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


def read_voice_model_cfg(extra: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """The effective voice-model mapping, or None when voice rides the main
    model. Env vars take precedence over ``extra.voice_model`` (hermes'
    env > YAML convention)."""
    env_provider = os.getenv("VOICE_MODEL_PROVIDER", "").strip()
    env_model = os.getenv("VOICE_MODEL_NAME", "").strip()
    if env_provider and env_model:
        if env_provider.lower() == "auto":
            return None
        cfg: Dict[str, Any] = {"provider": env_provider, "model": env_model}
        base_url = os.getenv("VOICE_MODEL_BASE_URL", "").strip()
        if base_url:
            cfg["base_url"] = base_url
        key_env = os.getenv("VOICE_MODEL_KEY_ENV", "").strip()
        if key_env:
            cfg["key_env"] = key_env
        return cfg

    vm = (extra or {}).get("voice_model")
    if not isinstance(vm, dict) or not vm.get("model"):
        return None
    if str(vm.get("provider") or "").strip().lower() in ("", "auto"):
        return None
    return vm


def voice_model_name(extra: Optional[Dict[str, Any]]) -> Optional[str]:
    """The configured voice model's name (telemetry/logging), or None when
    voice turns run on the main model."""
    vm = read_voice_model_cfg(extra)
    return (vm or {}).get("model") or None


def resolve_voice_runtime_kwargs(
    extra: Optional[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    """Resolve the voice-model slot into AIAgent runtime kwargs.

    Voice is a full agentic turn (tools, memory, the iteration loop)
    delivered in a latency-critical spoken modality, so it gets its own
    dedicated model slot rather than sharing the main model's latency
    profile. Returns endpoint/key/provider/api_mode kwargs plus ``model``
    when the slot is configured, or None when it is unset / ``auto``
    (caller uses the main model).

    Mirrors ``gateway.run._try_resolve_fallback_provider`` structurally.
    """
    from hermes_cli.runtime_provider import resolve_runtime_provider

    vm = read_voice_model_cfg(extra)
    if vm is None:
        return None
    try:
        explicit_api_key = vm.get("api_key")
        if not explicit_api_key:
            key_env = str(vm.get("key_env") or vm.get("api_key_env") or "").strip()
            if key_env:
                explicit_api_key = os.getenv(key_env, "").strip() or None
        runtime = resolve_runtime_provider(
            requested=vm.get("provider"),
            explicit_base_url=vm.get("base_url"),
            explicit_api_key=explicit_api_key,
            target_model=vm.get("model"),
        )
    except Exception as exc:
        logger.warning(
            "voice_model resolution failed (%s); falling back to the main model",
            exc,
        )
        return None
    logger.info(
        "Voice model resolved: %s model=%s",
        vm.get("provider") or runtime.get("provider"),
        vm.get("model"),
    )
    return {
        "api_key": runtime.get("api_key"),
        "base_url": runtime.get("base_url"),
        "provider": runtime.get("provider"),
        "api_mode": runtime.get("api_mode"),
        "command": runtime.get("command"),
        "args": list(runtime.get("args") or []),
        "credential_pool": runtime.get("credential_pool"),
        "model": vm.get("model"),
    }
