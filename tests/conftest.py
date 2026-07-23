"""Test wiring: the plugin runs in-process inside hermes-agent, so its unit
tests need a hermes-agent checkout importable (gateway.*, hermes_cli.*).

Set HERMES_AGENT_SRC to the checkout root; defaults to a (gitignored)
`hermes-agent/` clone at this repo's root. Fails fast listing everything
tried — no silent skip.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

_here = Path(__file__).resolve()
_candidates = [
    Path(p)
    for p in (
        os.environ.get("HERMES_AGENT_SRC"),
        _here.parents[1] / "hermes-agent",  # clone at repo root
    )
    if p
]

for _src in _candidates:
    if (_src / "gateway" / "config.py").is_file():
        sys.path.insert(0, str(_src))
        break
else:
    raise RuntimeError(
        "hermes-agent checkout not found; set HERMES_AGENT_SRC to a "
        "hermes-agent repo root (the plugin's tests import gateway.* and "
        "hermes_cli.* from it). Tried: "
        + ", ".join(str(p) for p in _candidates)
    )
