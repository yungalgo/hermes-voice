"""hermes-voice — realtime voice platform plugin for hermes-agent.

Loaded by the hermes plugin system either as a pip entry point
(group ``hermes_agent.plugins``, name ``voice-platform``) or as a
directory plugin under ``~/.hermes/plugins/``. Either way the plugin
system calls :func:`register`.
"""

from .adapter import register

__all__ = ["register"]
