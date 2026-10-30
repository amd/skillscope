# Copyright Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: MIT

"""Provide what a skill's behavioral cases need, where its agent runs.

A skill declares its prerequisites once, in ``evals/machine.yml`` (parsed by
`datasets.machine_prerequisites`), and every engine applies that one
declaration. The alternative it replaces -- a Python ``setup_session`` hook --
could only prepare the host, so a skill that also ran under ``claude-code``
had to describe its environment a second time in a compose file, and the two
descriptions drifted.

Each prerequisite is a ``check`` and an ``install`` command plus the ``env``
the agent needs once it is installed. The rule is the same on every engine,
and lives in `provide` so it cannot differ between them:

* ``check`` exits 0: the prerequisite is already there. Nothing is installed
  and its ``env`` is not applied -- the environment that provides it already
  says where it is. An author writes ``check`` so that is true only when the
  agent would work without the ``env``.
* Otherwise ``install`` runs, then ``check`` again *with* the ``env`` applied,
  because an installed tool is usually only findable through it. Still failing
  is the prerequisite's failure, reported with the step's own output.

Steps see two inputs, and ``env`` values may name only these:

* ``SKILL_SOURCE_DIR`` -- the skill's source tree, as `sources.resolve` finds
  it. Host only, and unset there too when no tree can be found: a container
  has neither the tree nor, by default, a network to fetch it with. A step
  that needs it sees it unset; an ``env`` value naming it fails before
  anything is installed, with the reason it is missing.
* ``SKILL_CACHE_DIR`` -- scratch space for what ``install`` builds. Per skill
  on the host, removed after its cases; a fixed path in each case's container,
  so an image that bakes the same ``install`` in at that path makes ``check``
  pass and skips the per-case cost. That path is provisional until a skill's
  image is built against it.

Where each engine runs it:

* ``claude-code-no-sandbox``: once per skill, before its task is built, in
  `on_host`. The resulting ``env`` reaches the agent through the solver, not
  through ``os.environ`` -- nothing a skill installs is visible to the next.
* ``claude-code``: in each case's container, before the agent, in
  `claude_code_agent`, because inspect starts one container per case.
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import platform
import shutil
import signal
import subprocess
import tempfile
from collections.abc import Awaitable, Callable, Iterator
from dataclasses import dataclass
from pathlib import Path

from .. import datasets, deadline, sources
from . import no_sandbox
from ..datasets import Prerequisite

# Where a container keeps the inputs. Fixed rather than random so that an image
# can bake an `install` in at the same `SKILL_CACHE_DIR` and have `check` find
# it. Provisional until a skill's image is built against it: documented as such
# in docs/authoring-evals.md, and nothing pins the path yet.
CONTAINER_ROOT = "/opt/skillscope"
CONTAINER_CACHE = f"{CONTAINER_ROOT}/cache"
# Where the files a step names are copied, and removed again before the agent
# starts: evals/ also holds the answer key, and only what a step names leaves.
CONTAINER_STEPS = f"{CONTAINER_ROOT}/steps"

# How much of a failing step's output the report carries. The end of a build
# log is where its error is; the start is usually configuration chatter.
OUTPUT_TAIL = 2000

Runner = Callable[[list[str], dict[str, str]], Awaitable[tuple[int, str]]]


class PrerequisiteError(Exception):
    """A prerequisite could not be provided. Fails the cases that needed it."""

    def __init__(self, message: str) -> None:
        super().__init__(message)
        # inspect reports a sample's error by `.message` where there is one and
        # by `repr()` where there is not, which would put escaped newlines and
        # the class name in the report instead of the step's output.
        self.message = message


def host_platform() -> str:
    """This host as one of `datasets.COMMAND_PLATFORMS`."""
    name = platform.system().lower()
    return {"darwin": "macos"}.get(name, name)


def expand(
    env: dict[str, str], inputs: dict[str, str], unavailable: dict[str, str]
) -> dict[str, str]:
    """`env` with ``${SKILL_SOURCE_DIR}`` and ``${SKILL_CACHE_DIR}`` filled in.

    The declaration was checked to name nothing else, so an input missing from
    `inputs` is one this engine could not supply, and `unavailable` says why.
    """
    expanded: dict[str, str] = {}
    for name, value in env.items():
        for key in datasets.PREREQUISITE_INPUTS:
            token = "${" + key + "}"
            if token in value:
                if key not in inputs:
                    raise PrerequisiteError(
                        f"`{name}` refers to {token}, which is not available "
                        f"here: {unavailable.get(key, 'nothing provides it')}"
                    )
                value = value.replace(token, inputs[key])
        expanded[name] = value
    return expanded


def _tail(output: str) -> str:
    output = output.strip()
    return output if len(output) <= OUTPUT_TAIL else "..." + output[-OUTPUT_TAIL:]


@dataclass(frozen=True)
class Provision:
    """The env the agent needs, and which prerequisites produced it.

    `satisfied` passed `check`, so nothing was installed and none of their
    `env` was applied. `installed` are the ones this run provided.
    """

    env: dict[str, str]
    satisfied: tuple[str, ...]
    installed: tuple[str, ...]


def _provision_line(skill: str, provision: Provision) -> str:
    """Name each branch `provide` took. The count alone does not."""
    parts: list[str] = []
    if provision.satisfied:
        parts.append(f"{', '.join(provision.satisfied)} satisfied by the environment")
    if provision.installed:
        parts.append(f"{', '.join(provision.installed)} installed")
    return f"[prerequisites] {skill}: {'; '.join(parts)}"


async def provide(
    declared: list[Prerequisite],
    inputs: dict[str, str],
    where: str,
    run: Runner,
    unavailable: dict[str, str] | None = None,
) -> Provision:
    """Apply `declared` in order.

    The result's `env` is what the agent needs. `satisfied` and `installed`
    name which prerequisites were already present and which this run provided,
    because a count alone is the same either way.

    `where` is the platform the steps run on, and `run` runs one argv with
    extra environment, returning its exit code and combined output. Each
    prerequisite's steps see the env of the ones before it, so a later one can
    build on an earlier one. `unavailable` explains any input missing from
    `inputs`, for the error raised if an `env` value names it.
    """
    applied: dict[str, str] = {}
    satisfied: list[str] = []
    installed: list[str] = []
    for item in declared:
        check = item.command("check", where)
        install = item.command("install", where)
        if check is None or install is None:
            raise PrerequisiteError(
                f"prerequisite `{item.name}` declares no command for {where}."
            )
        step_env = {**inputs, **applied}
        code, _ = await run(check, step_env)
        if code == 0:
            satisfied.append(item.name)
            continue
        # Before the install rather than after it: an `env` this engine cannot
        # fill should not cost a full build to discover.
        env = expand(item.env, inputs, unavailable or {})
        code, output = await run(install, step_env)
        if code != 0:
            # The step's own output first: a report keeps only the start of
            # this. Then, for an input the step mentions and this engine could
            # not supply, why it was unset -- not for every failed install.
            said = " ".join(install) + output
            missing = "".join(
                f"\n({name} was not available: {why})"
                for name, why in (unavailable or {}).items()
                if name in said
            )
            raise PrerequisiteError(
                f"prerequisite `{item.name}`: install exited {code}: "
                f"{_tail(output)}{missing}"
            )
        code, output = await run(check, {**step_env, **env})
        if code != 0:
            raise PrerequisiteError(
                f"prerequisite `{item.name}`: still not present after install "
                f"(check exited {code}).\n{_tail(output)}"
            )
        applied.update(env)
        installed.append(item.name)
    return Provision(applied, tuple(satisfied), tuple(installed))


# What the `--timeout` watchdog's hard exit has to clean up, because `os._exit`
# reaches neither the `finally` below nor the kill in `_host_runner`. Same
# pattern as `hooks.py`.
_created_caches: set[str] = set()
# Host steps still running, by pid. Each leads a session or process group of
# its own (see `_host_runner`), so the pid names everything it started too.
_live_steps: set[int] = set()


def _discard_all_caches() -> None:
    for path in list(_created_caches):
        shutil.rmtree(path, ignore_errors=True)
    _created_caches.clear()


def _kill_live_steps() -> None:
    """Kill every running host step and what it started. For the watchdog.

    Synchronous, because the watchdog fires from a thread with no event loop.
    """
    for pid in list(_live_steps):
        try:
            if os.name == "nt":
                subprocess.run(
                    ["taskkill", "/F", "/T", "/PID", str(pid)],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    timeout=5,
                )
            else:
                os.killpg(pid, signal.SIGKILL)
        except (OSError, subprocess.SubprocessError):
            pass
        _live_steps.discard(pid)


def _on_expire() -> None:
    # Steps first: a build still running would go on writing into the cache
    # being removed.
    _kill_live_steps()
    _discard_all_caches()


deadline.on_expire(_on_expire)


def _host_runner(cwd: Path) -> Runner:
    async def run(argv: list[str], extra: dict[str, str]) -> tuple[int, str]:
        # Asynchronous rather than `subprocess.run`, and stopped on the way
        # out, because a build is the slowest thing a run does outside inspect:
        # the deadline's graceful stop interrupts the main thread, and a
        # blocking call would sit out the whole build before noticing -- long
        # enough for the hard kill to take the report with it.
        #
        # In a session of its own, so the stop reaches what the step started
        # too: an install is usually a script running a build, and the build
        # would otherwise go on writing into a cache that is being removed.
        spawn: dict = {}
        if os.name == "nt":
            spawn["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
        else:
            spawn["start_new_session"] = True
        env = {
            name: value
            for name, value in os.environ.items()
            # Only what this run resolved: a value left in the runner's own
            # environment would stand in for an input this run could not find.
            if name not in datasets.PREREQUISITE_INPUTS
        }
        try:
            proc = await asyncio.create_subprocess_exec(
                *argv,
                cwd=cwd,
                env={**env, **extra},
                stdin=asyncio.subprocess.DEVNULL,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
                **spawn,
            )
        except (OSError, ValueError) as exc:
            return 127, f"could not start {argv[0]!r}: {exc}"
        _live_steps.add(proc.pid)
        try:
            output, _ = await proc.communicate()
        except BaseException:
            # Shielded: this runs while the step is being cancelled, and the
            # kill has to finish rather than be cancelled in turn.
            await asyncio.shield(no_sandbox._terminate(proc))
            raise
        finally:
            _live_steps.discard(proc.pid)
        return proc.returncode, output.decode(errors="replace")

    return run


@contextlib.contextmanager
def on_host(skill: str) -> Iterator[dict[str, str]]:
    """Provide `skill`'s prerequisites on this host for the length of the block.

    Yields the env its agent needs -- empty for a skill that declares none,
    without creating anything. Raises `PrerequisiteError` before yielding if
    one cannot be provided.
    ``os.environ`` is never changed: the env goes to the agent through its
    solver, so nothing provided here leaks into the next skill.
    """
    declared = datasets.machine_prerequisites(skill)
    if not declared:
        yield {}
        return

    cache = Path(tempfile.mkdtemp(prefix=f"skillscope-prerequisites-{skill}-"))
    _created_caches.add(str(cache))
    try:
        inputs = {"SKILL_CACHE_DIR": str(cache)}
        unavailable: dict[str, str] = {}
        # Resolved here but not required: a skill whose steps never use its
        # source tree must not fail because there is none -- not inside a
        # checkout, or a vendored skill whose fetch failed. A step that needs
        # it sees it unset, and an `env` value naming it fails with the
        # resolver's own reason.
        try:
            inputs["SKILL_SOURCE_DIR"] = str(sources.resolve(skill, cache).path)
        except (SystemExit, Exception) as exc:
            unavailable["SKILL_SOURCE_DIR"] = str(exc).removeprefix("error: ")
        print(f"[prerequisites] {skill}: {len(declared)} to provide", flush=True)
        provision = asyncio.run(
            provide(
                declared,
                inputs,
                host_platform(),
                _host_runner(datasets.machine_path(skill).parent),
                unavailable,
            )
        )
        print(_provision_line(skill, provision), flush=True)
        yield provision.env
    finally:
        shutil.rmtree(cache, ignore_errors=True)
        _created_caches.discard(str(cache))


def _named_files(skill: str, declared: list[Prerequisite]) -> dict[str, Path]:
    """The files beside machine.yml that a container step names, by argv item.

    Only these are copied in. The rest of evals/ includes the answer key, and a
    step has no business carrying it into the agent's room -- so naming a
    dataset file is refused rather than copied.
    """
    evals_dir = datasets.machine_path(skill).parent.resolve()
    keys = {
        (datasets.skill_path(skill) / rel).resolve()
        for rel in (datasets.DATASET_RELPATH, datasets.EXTENDED_DATASET_RELPATH)
    }
    named: dict[str, Path] = {}
    for item in declared:
        for step in ("check", "install"):
            for arg in item.command(step, "linux") or []:
                # Every argv item is probed, including the body of an inline
                # `sh -c` script, which can be no valid path at all.
                try:
                    candidate = (evals_dir / arg).resolve()
                    if not candidate.is_file() or evals_dir not in candidate.parents:
                        continue
                except (OSError, ValueError):
                    continue
                if candidate in keys:
                    raise PrerequisiteError(
                        f"prerequisite `{item.name}` names {arg}, which is this "
                        "skill's answer key and is never copied into the agent's "
                        "container."
                    )
                named[candidate.relative_to(evals_dir).as_posix()] = candidate
    return named


def claude_code_agent(skill: str, make_agent: Callable[[dict[str, str]], object]):
    """Solver: provide `skill`'s prerequisites in this case's container, then run.

    `make_agent` builds the agent from the env the prerequisites produced. It is
    built here, per case, rather than once with the task, because that env is
    only known once `check` has run in this case's container: a prerequisite
    the image already provides gets none of its declared env, same as on the
    host.

    `None` for a skill that declares none, so its task is built exactly as it
    was before prerequisites existed.

    Assumes a Linux container: the fixed paths and `mkdir`/`rm` are the
    guest's. `verify.require` refuses the unisolated `local` provider and
    Windows before any task is built, which is what makes that hold.
    """
    declared = datasets.machine_prerequisites(skill)
    if not declared:
        return None
    named = _named_files(skill, declared)

    from inspect_ai.agent import as_solver
    from inspect_ai.solver import solver
    from inspect_ai.util import OutputLimitExceededError, sandbox

    @solver
    def _with_prerequisites():
        async def solve(state, generate):
            box = sandbox()

            async def run(argv: list[str], extra: dict[str, str]) -> tuple[int, str]:
                # inspect raises for output too large to return or not text;
                # as a failed step, `provide` reports it under the
                # prerequisite's name rather than as a bare sandbox error.
                try:
                    result = await box.exec(argv, cwd=CONTAINER_STEPS, env=extra)
                except (OutputLimitExceededError, OSError, UnicodeDecodeError) as exc:
                    return 126, f"{type(exc).__name__}: {exc}"
                return result.returncode, result.stdout + result.stderr

            made = await box.exec(["mkdir", "-p", CONTAINER_STEPS, CONTAINER_CACHE])
            if not made.success:
                raise PrerequisiteError(
                    f"could not create {CONTAINER_ROOT} in the container: "
                    f"{_tail(made.stderr)}"
                )
            try:
                # Inside the `try`, so a copy that fails partway still has the
                # files already copied removed before anything else runs.
                for relative, path in named.items():
                    await box.write_file(
                        f"{CONTAINER_STEPS}/{relative}", path.read_bytes()
                    )
                print(
                    f"[prerequisites] {skill}: {len(declared)} to provide",
                    flush=True,
                )
                provision = await provide(
                    declared,
                    {"SKILL_CACHE_DIR": CONTAINER_CACHE},
                    "linux",
                    run,
                    {
                        "SKILL_SOURCE_DIR": "a container has no copy of the "
                        "skill's source tree. Build what needs it into the "
                        "image instead."
                    },
                )
                print(_provision_line(skill, provision), flush=True)
            except BaseException:
                # Best effort, and never in place of the error being raised.
                with contextlib.suppress(Exception):
                    await box.exec(["rm", "-rf", CONTAINER_STEPS])
                raise
            removed = await box.exec(["rm", "-rf", CONTAINER_STEPS])
            if not removed.success:
                # The agent must not start with copied evals/ files in reach.
                raise PrerequisiteError(
                    f"could not remove {CONTAINER_STEPS} before the agent "
                    f"starts: {_tail(removed.stderr)}"
                )
            return await as_solver(make_agent(provision.env))(state, generate)

        return solve

    return _with_prerequisites()
