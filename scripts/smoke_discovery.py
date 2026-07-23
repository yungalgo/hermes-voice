"""Discovery smoke: verify hermes finds + loads the plugin and registers the
'voice' platform. Run inside a hermes-agent environment with the plugin
installed (pip route) or dir-dropped (set HERMES_HOME accordingly):

    HERMES_HOME=/tmp/hvp-smoke-home uv run --with <wheel> python scripts/smoke_discovery.py

Exits non-zero on any failed expectation. Requires no provider keys —
registration must succeed without them; check_fn gates connection, not
discovery.
"""

import os
import sys

for k in ("DAILY_API_KEY", "DEEPGRAM_API_KEY", "CARTESIA_API_KEY"):
    os.environ.pop(k, None)

lookup_key = sys.argv[1] if len(sys.argv) > 1 else "voice-platform"

from hermes_cli.plugins import discover_plugins, get_plugin_manager  # noqa: E402

discover_plugins(force=True)
mgr = get_plugin_manager()
lp = mgr._plugins.get(lookup_key)
assert lp is not None, f"plugin {lookup_key!r} not discovered; found: {sorted(mgr._plugins)}"
print(f"manifest: key={lookup_key} source={lp.manifest.source} "
      f"enabled={lp.enabled} error={lp.error}")
assert lp.enabled, f"plugin not enabled: {lp.error}"

from gateway.platform_registry import platform_registry  # noqa: E402

assert platform_registry.is_registered("voice"), "platform 'voice' not registered"
entry = platform_registry.get("voice")
print(f"platform: label={entry.label} plugin={entry.plugin_name} source={entry.source}")
assert entry.source == "plugin"
print(f"check_fn (no keys): {entry.check_fn()}")
assert entry.check_fn() is False, "check_fn must be False without the three keys"

from gateway.config import Platform  # noqa: E402

assert Platform("voice").value == "voice"
print("Platform('voice') pseudo-member OK")
print("SMOKE PASS")
