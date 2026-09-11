# Copyright Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: MIT

"""Which sandbox a skill's cases run in.

Two decisions, kept apart because they are made by different people.

**Which provider** is a property of the machine: Docker by default, `podman` on
a host that has that instead, `local` where there is no container at all.
`SKILLSCOPE_SANDBOX` selects it, because whoever runs the job knows what the
runner has and a skill does not. Any provider inspect can resolve works --
`podman` comes from `inspect-podman`, which registers itself through an
`inspect_ai` entry point, so installing it is the whole setup.

**What the sandbox has to provide** is a property of the skill, declared in
`evals/machine.yml` with a `sandbox:` key naming a compose file. A skill that
must reach the network to pull a model, or that needs a device bound in, says
so there instead of every skill paying for what one of them needs.

Windows is the exception to both: inspect's sandbox layer, and every tool built
on it, assumes a POSIX guest, so the Windows legs run `local` and trade
isolation for running on the platform they are meant to test. Ephemeral,
off-network runners are what covers that gap.
"""

from __future__ import annotations

import os
import sys

from .. import datasets

# Which provider to use. Set it to what the runner actually has: `podman` on a
# host without Docker, `local` to skip the container entirely. `local` is for
# working locally, not for CI -- a graded run that quietly dropped its sandbox
# would report the same numbers with none of the isolation.
SANDBOX_ENV = "SKILLSCOPE_SANDBOX"

DEFAULT_PROVIDER = "docker"

# Providers that take no configuration, so a skill's compose file cannot apply.
UNCONFIGURED = {"local"}

# `local` runs in the same filesystem as the harness: the sandbox API works, but
# nothing is isolated. Named so a report can say which it was.
NOT_ISOLATED = {"local"}


def describe() -> dict:
    """What the report should say about isolation.

    A report that shows the same numbers whether or not a case was contained
    invites the reader to assume it was. Both engines say it outright instead,
    so "these ran isolated and those did not" is answerable from the artifact
    rather than from whoever remembers how the job was configured.
    """
    name = provider()
    return {"sandbox": name, "sandbox_isolated": name not in NOT_ISOLATED}


def is_windows() -> bool:
    return sys.platform.startswith("win")


def provider() -> str:
    """The sandbox provider for this run."""
    override = os.environ.get(SANDBOX_ENV, "").strip()
    if override:
        return override
    if is_windows():
        return "local"
    return DEFAULT_PROVIDER


def for_skill(skill: str):
    """The `sandbox` spec for a skill's task.

    Returns a `(provider, config)` tuple when the skill declares a compose file
    and the provider can take one, a bare provider name otherwise -- both are
    accepted as `Task(sandbox=...)`.
    """
    name = provider()
    if name in UNCONFIGURED:
        return name

    # The provider is the machine's choice and the compose file is the skill's,
    # so selecting a provider must not silently discard what the skill asked
    # for: a skill that needs network egress would otherwise run without it and
    # fail for a reason nothing in the report explains.
    compose = _declared_compose(skill)
    return (name, str(compose)) if compose is not None else name


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
