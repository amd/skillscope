# Copyright Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: MIT

"""The eval engines built on ``inspect_ai``.

The legacy engine drives the `claude` CLI directly; these hand the work to
`inspect_ai` -- ``claude-code`` runs the CLI inside the sandbox through
`inspect_swe`, ``claude-code-no-sandbox`` runs it on the host. All of them produce the same
outcome objects, so everything downstream -- `summarize`, `render_markdown`,
the report writers -- is shared.

`inspect_ai` is the `inspect` extra, so nothing here is imported at module
scope by the rest of the package and a light install can still run the
commands that need no agent. Call `require(engine)` before touching a
submodule, to turn a missing wheel into an actionable message rather than a
traceback.
"""

from __future__ import annotations

from .. import install

def install_hint(engine: str) -> str:
    """Why this run cannot start, naming the engine that was actually asked for.

    Every engine runs on inspect_ai, which is the `inspect` extra. The
    commands that need no agent import none of it, so the default install
    leaves it out, and this is what a graded command says when nobody asked
    for it. Through the action nobody sees this: `bootstrap/launch.py` asks
    for the extra by command. It is for a hand-rolled install.

    `--force`, because the tool is already installed and `uv tool install`
    would otherwise leave it alone and report success.

    The engine is named rather than defaulted: a default once sent a CI job
    looking for a flag it had not passed, and the value it defaulted to is
    one argparse now rejects.
    """
    return (
        f"error: --engine {engine} needs the eval engine, which is the "
        "`inspect` extra and is not in the default install. Add it with:\n"
        f"    {install.install_command('inspect', force=True)}"
    )


def require(engine: str) -> None:
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
