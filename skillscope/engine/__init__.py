# Copyright Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: MIT

"""The inspect-backed eval engine (``--engine inspect``).

The legacy engine drives the `claude` CLI directly; this one hands the work to
`inspect_ai`. Both produce the same outcome objects, so everything downstream --
`summarize`, `render_markdown`, the report writers -- is shared.

`inspect_ai` is an optional dependency, so nothing here is imported at module
scope by the rest of the package. Call `require()` before touching a submodule
to turn a missing wheel into an actionable message rather than a traceback.
"""

from __future__ import annotations

INSTALL_HINT = (
    "error: --engine inspect needs the inspect extra. Install it with:\n"
    "    pip install 'skillscope[inspect]'"
)


def require() -> None:
    """Raise SystemExit with an install hint when `inspect_ai` is missing."""
    try:
        import inspect_ai  # noqa: F401
    except ModuleNotFoundError as exc:  # pragma: no cover -- environment shape
        raise SystemExit(INSTALL_HINT) from exc


def available() -> bool:
    """Whether the inspect extra is installed (for diagnostics, not control flow)."""
    try:
        import inspect_ai  # noqa: F401
    except ModuleNotFoundError:
        return False
    return True
