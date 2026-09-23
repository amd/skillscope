# Copyright Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: MIT

"""The eval engines built on ``inspect_ai``.

The legacy engine drives the `claude` CLI directly; these hand the work to
`inspect_ai` -- ``claude-code`` runs the CLI inside the sandbox through
`inspect_swe`, ``claude-cli`` runs it on the host. All of them produce the same
outcome objects, so everything downstream -- `summarize`, `render_markdown`,
the report writers -- is shared.

`inspect_ai` is an optional dependency, so nothing here is imported at module
scope by the rest of the package. Call `require()` before touching a submodule
to turn a missing wheel into an actionable message rather than a traceback.
"""

from __future__ import annotations

def install_hint(engine: str = "inspect") -> str:
    """Why this run cannot start, naming the engine that was actually asked for.

    Every engine but `legacy` runs on inspect_ai, so any of them can raise
    this. Naming `inspect` regardless sent a CI job looking for a flag it had
    not passed.
    """
    return (
        f"error: --engine {engine} needs the inspect extra. Install it with:\n"
        "    pip install 'skillscope[inspect]'"
    )


# Kept for callers that predate the engine argument.
INSTALL_HINT = install_hint()


def require(engine: str = "inspect") -> None:
    """Raise SystemExit with an install hint when `inspect_ai` is missing."""
    try:
        import inspect_ai  # noqa: F401
    except ModuleNotFoundError as exc:  # pragma: no cover -- environment shape
        raise SystemExit(install_hint(engine)) from exc


def available() -> bool:
    """Whether the inspect extra is installed (for diagnostics, not control flow)."""
    try:
        import inspect_ai  # noqa: F401
    except ModuleNotFoundError:
        return False
    return True
