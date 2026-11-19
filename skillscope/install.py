# Copyright Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: MIT

"""The one install line this tool tells anyone to run.

`skillscope` on PyPI is an unrelated project, so a bare-name install gets
neither this tool nor any of its extras. Every hint has to name the
repository, and `uv tool install` rather than `pip install`, because an
extra has to be asked for against the URL rather than beside it.

A function rather than a string per call site. The hints and the docs drifted
apart once already: `docs/usage.md` named the repository while three hints in
the code still named the bare package, and the only way a reader found out
which of the two was right was to run one and watch it install somebody
else's project.
"""

from __future__ import annotations

REPOSITORY = "git+https://github.com/amd/skillscope"


def install_command(extra: str | None = None, *, force: bool = False) -> str:
    """The command that installs this tool, asking for `extra` where named.

    `force` for a hint that has to replace an install already on the machine:
    `uv tool install` leaves an installed tool alone otherwise, so an upgrade
    hint without it reads as a command that did nothing.
    """
    package = f"skillscope[{extra}]" if extra else "skillscope"
    flag = "--force " if force else ""
    return f"uv tool install {flag}'{package} @ {REPOSITORY}'"
