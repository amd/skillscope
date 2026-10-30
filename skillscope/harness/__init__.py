# Copyright Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: MIT

"""The agent harnesses skillscope can drive.

A harness is *what* runs a case; the engine is *where*. See `base.Harness` for
the contract and `docs/harnesses.md` for how to add one.
"""

from __future__ import annotations

from .base import Harness, RunSpec
from .claude_code import ClaudeCode
from .copilot import Copilot

DEFAULT = "claude-code"

_REGISTRY: dict[str, Harness] = {}


def register(harness: Harness) -> Harness:
    """Make `harness` selectable by its name. Registering a name twice is a bug."""
    if harness.name in _REGISTRY:
        raise ValueError(f"harness {harness.name!r} is already registered")
    _REGISTRY[harness.name] = harness
    return harness


def names() -> list[str]:
    return sorted(_REGISTRY)


def get(name: str | None = None) -> Harness:
    """The harness called `name`, or the default one when none is named."""
    key = name or DEFAULT
    try:
        return _REGISTRY[key]
    except KeyError:
        raise SystemExit(
            f"error: unknown harness {key!r}. Known harnesses: {', '.join(names())}."
        ) from None


register(ClaudeCode())
register(Copilot())

__all__ = ["DEFAULT", "Harness", "RunSpec", "get", "names", "register"]
