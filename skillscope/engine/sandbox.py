# Copyright Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: MIT

"""Which sandbox a skill's cases run in.

Docker on Linux, `local` on Windows. inspect's sandbox layer -- and every tool
built on it -- assumes a POSIX guest, so there is no Windows container option
here; the Windows legs trade isolation for running on the platform they are
meant to test. DevLab's ephemeral, off-network runners are what covers that gap.

A skill declares its needs in `evals/machine.yml`, which already exists to say
what class of machine a skill wants. An optional `sandbox:` key names a compose
file relative to the skill directory, so a skill that must reach the network to
pull a model, or needs `/dev/dri`, says so instead of every skill paying for it.
"""

from __future__ import annotations

import os
import sys

from .. import datasets

# Escape hatch for local development and wiring runs: `SKILLSCOPE_SANDBOX=local`
# skips the container entirely. Not for CI -- a graded run that quietly dropped
# its sandbox would report the same numbers with none of the isolation.
SANDBOX_ENV = "SKILLSCOPE_SANDBOX"

# Hardware-free skills get no network. Skills that need egress ship their own
# compose file and opt out of this default.
DEFAULT_COMPOSE = "compose.yaml"


def is_windows() -> bool:
    return sys.platform.startswith("win")


def for_skill(skill: str):
    """The `sandbox` spec for a skill's task, or None to use inspect's default.

    Returns a `(type, config)` tuple when a compose file is declared, a bare
    type name otherwise -- both are accepted as `Task(sandbox=...)`.
    """
    override = os.environ.get(SANDBOX_ENV, "").strip()
    if override:
        return override

    if is_windows():
        return "local"

    compose = _declared_compose(skill)
    if compose is not None:
        return ("docker", str(compose))
    return "docker"


def _declared_compose(skill: str):
    """Path to the compose file a skill's `machine.yml` names, if any."""
    name = (datasets._read_machine(skill) or {}).get("sandbox")
    if not name:
        return None

    path = datasets.skill_path(skill) / name
    if not path.is_file():
        raise SystemExit(
            f"error: {skill}: evals/machine.yml names sandbox '{name}', "
            f"but {path} does not exist."
        )
    return path
