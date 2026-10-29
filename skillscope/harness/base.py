# Copyright Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: MIT

"""What a harness is, as far as the engines are concerned.

A *harness* is the agent that is handed a prompt and a workspace and does the
work: Claude Code, GitHub Copilot CLI, Codex, and so on. It is a different
question from where it runs (`--engine`: on the host or in a container), and
this module is the contract for answering it.

The engines drive a harness as a subprocess, so the contract is small: how to
find it, where it looks for skills, how to start one case, what environment to
start it in, and how to read what it prints. Everything else -- staging, the
stop rule, usage, grading -- stays in the engine, once, for every harness.

Adding a harness is one subclass of `Harness`, registered in
`skillscope/harness/__init__.py`, plus recorded output for the contract test in
`tests/test_harness.py`. `docs/harnesses.md` has the checklist.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable, Mapping

from .. import deadline

# What the preflight asks, and how long a CLI gets to answer. A reply of any
# kind proves the credential, the model name and the network; the words are
# only there so the reply is short.
PREFLIGHT_PROMPT = "Reply with the single word: ok"
PREFLIGHT_TIMEOUT_S = 60.0

# `--version` output by executable, so a report names the build once however
# many cases ask.
_VERSIONS: dict[str, str | None] = {}


def _kill_tree(proc: subprocess.Popen) -> None:
    """Kill a process and everything it started.

    `subprocess.run(timeout=)` kills only the process it launched. A CLI
    installed with npm is a shim that starts the real program, so killing the
    shim leaves the program running with the output pipe open, and the wait that
    follows the kill lasts until the program ends on its own.
    """
    try:
        if os.name == "nt":
            subprocess.run(
                ["taskkill", "/F", "/T", "/PID", str(proc.pid)],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
        else:
            import signal

            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
    except (OSError, ProcessLookupError):
        pass
    try:
        proc.kill()
    except (OSError, ProcessLookupError):
        pass


def _run_bounded(
    cmd: list[str], *, stdin: str, cwd: str, env: dict[str, str], timeout: float
) -> subprocess.CompletedProcess:
    """`subprocess.run` that really stops at `timeout`, children included."""
    spawn: dict = {}
    if os.name == "nt":
        spawn["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
    else:
        spawn["start_new_session"] = True
    proc = subprocess.Popen(
        cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        cwd=cwd, env=env, text=True, encoding="utf-8", **spawn,
    )
    try:
        out, err = proc.communicate(stdin, timeout=timeout)
    except subprocess.TimeoutExpired:
        _kill_tree(proc)
        proc.communicate()
        raise
    return subprocess.CompletedProcess(cmd, proc.returncode, out, err)


@dataclass(frozen=True)
class RunSpec:
    """Everything one case hands to a harness.

    A value object rather than keyword arguments so a harness that needs one
    more input in future (a trust flag, a home directory) reads it from here
    instead of every caller growing a parameter.
    """

    workspace: str
    prompt: str
    model: str | None = None
    effort: str | None = None
    # A throwaway config directory, so the runner's own settings and user-level
    # skills stay out of the session. `None` when the caller does not isolate.
    # A path wherever the agent runs, like `workspace`: on the host a host path,
    # in a container a path inside it.
    config_dir: str | None = None
    # The harness's own spend or session controls, already built by the caller.
    extra_flags: tuple[str, ...] = ()


class Harness(ABC):
    """One agent harness. Subclass, fill in the class attributes, register."""

    # The name `--harness` and the reports use.
    name: str
    # The executable looked up on PATH.
    binary: str
    # Where, relative to the workspace, the harness discovers skills.
    skills_dir: str
    # Whether activating a skill is a tool call the stream names. A harness with
    # no such call (it activates a skill by reading its SKILL.md) sets this
    # `False`, and activation is read from the file access instead.
    has_skill_tool: bool = True
    # Whether the prompt is written to the process's stdin. A harness that takes
    # it as an argument builds that into `command` and sets this `False`; its
    # stdin is closed either way, because some CLIs wait on it.
    prompt_on_stdin: bool = True
    # The model used when the caller names none. Not a shared default because
    # models are named differently, and cost very differently, per harness.
    default_model: str
    # The environment variables a credential can come from. A routing run can
    # only keep the runner's own configuration out of the session when the
    # harness authenticates from one of them; a login held in its own config
    # directory would be hidden along with everything else.
    credentials: tuple[str, ...] = ()
    # Whether a container leg exists for this harness (`--engine sandbox`).
    supports_sandbox: bool = False

    def resolve_model(self, model: str | None) -> str:
        """The model name this harness's CLI takes, for what the caller asked.

        `None` means the caller named none. A harness with its own names maps
        the generic aliases (`opus`, `sonnet`, `haiku`) onto them here.
        """
        return model or self.default_model

    def pin_model(self, model: str) -> str:
        """The model a CI run is held to.

        Automated runs are pinned so results stay comparable from one run to the
        next. A harness that pins says so here; the default does not.
        """
        return model

    def can_isolate_room(self, environ: Mapping[str, str] | None = None) -> bool:
        """Whether the runner's own config can be kept out of a routing session."""
        environ = os.environ if environ is None else environ
        return any(environ.get(name, "").strip() for name in self.credentials)

    def cost_flags(self, max_budget_usd: float | None) -> list[str]:
        """The CLI's own spend controls for a case, or none when it has none."""
        return []

    def version(self) -> str | None:
        """What `<binary> --version` prints, for the report. `None` if it cannot."""
        executable = shutil.which(self.binary)
        if executable is None:
            return None
        if executable not in _VERSIONS:
            try:
                proc = subprocess.run(
                    [executable, "--version"], capture_output=True, text=True,
                    encoding="utf-8", timeout=30,
                )
                lines = [line.strip() for line in (proc.stdout or "").splitlines()]
                _VERSIONS[executable] = next((line for line in lines if line), None)
            except (OSError, subprocess.SubprocessError):
                _VERSIONS[executable] = None
        return _VERSIONS[executable]

    def preflight(
        self, model: str | None, timeout: float = PREFLIGHT_TIMEOUT_S
    ) -> tuple[bool, str]:
        """Prove, before any case runs, that the CLI can reach its model.

        One trivial prompt through the harness's own command, so what is checked
        is what a case will use: the executable, the credential, the model name
        and the network. Returns `(ok, detail)`, with a short reason on failure.
        """
        executable = shutil.which(self.binary)
        if executable is None:
            return False, f"'{self.binary}' CLI not found on PATH"

        bound = deadline.active()
        if bound is not None:
            leftover = bound.remaining()
            if leftover <= 0:
                return False, bound.message()
            timeout = min(timeout, leftover)

        with tempfile.TemporaryDirectory(prefix="skillscope-preflight-") as workspace, \
                tempfile.TemporaryDirectory(prefix="skillscope-preflight-home-") as home:
            spec = RunSpec(
                workspace=workspace, prompt=PREFLIGHT_PROMPT, model=model,
                config_dir=home,
            )
            try:
                proc = _run_bounded(
                    self.host_command(spec),
                    stdin=PREFLIGHT_PROMPT if self.prompt_on_stdin else "",
                    cwd=workspace, env=self.host_env(spec), timeout=timeout,
                )
            except subprocess.TimeoutExpired:
                return False, (
                    f"{self.binary} preflight timed out after {timeout:g}s "
                    "(is the network reachable?)"
                )
        if proc.returncode != 0:
            detail = (proc.stderr or proc.stdout or f"exit code {proc.returncode}").strip()
            return False, detail[:500]
        return True, "ok"

    def require(self) -> None:
        """Refuse to start on the host when the harness is not installed there.

        This asks the runner's `PATH`, which is the right question for a run
        started on the host. A leg that starts the agent in a container asks
        about the image instead.
        """
        if not shutil.which(self.binary):
            raise SystemExit(f"error: '{self.binary}' CLI not found on PATH")

    @abstractmethod
    def command(self, spec: RunSpec) -> list[str]:
        """The argv that runs one case, headless, streaming JSON events.

        The executable is named, not located: `self.binary`, never an absolute
        path on this machine. Whoever starts the process resolves it, so the same
        command can be started on the host or inside a container.
        """

    @abstractmethod
    def env(self, spec: RunSpec) -> dict[str, str]:
        """What the harness sets in the process's environment, and nothing else.

        Its own settings, and whatever points it at `spec.config_dir` so the
        runner's own settings and skills stay out of the session. Not the runner's
        environment: `PATH`, `HOME` and every key on the machine are not the
        harness's to carry, and a container must not inherit them. Whoever starts
        the process puts this on top of the environment it has.
        """

    def host_command(self, spec: RunSpec) -> list[str]:
        """`command(spec)` with its executable resolved on this machine.

        What a run started on the host executes. The lookup is `shutil.which`
        rather than leaving it to the operating system, because on Windows a CLI
        installed with npm is a `.cmd` shim that only the former finds.
        """
        argv = self.command(spec)
        return [shutil.which(argv[0]) or argv[0], *argv[1:]]

    def host_env(self, spec: RunSpec) -> dict[str, str]:
        """The environment a run started on the host gets: this machine's, with
        what the harness sets on top."""
        return {**os.environ, **self.env(spec)}

    def normalize(self, event: dict) -> list[dict]:
        """Translate one event the harness printed into canonical events.

        The canonical form is the one the engines already read: Claude Code's
        stream-json, reduced to `assistant` messages carrying `tool_use`
        blocks, `user` messages carrying `tool_result` blocks, the
        `system`/`init` event naming the skills the session registered, and a
        final `result` event. A harness whose stream already has that shape
        keeps this default; any other maps its events onto it, and returns an
        empty list for the ones that carry nothing the engines read.

        Stateless, which is enough when each event can be translated on its
        own. A harness that cannot -- one whose closing event does not repeat
        the answer, so it has to be remembered from an earlier one -- overrides
        `normalizer` instead.
        """
        return [event]

    def normalizer(self) -> Callable[[dict], list[dict]]:
        """A translator for the events of one run, free to keep state.

        The engine asks for a fresh one per case, so nothing a harness
        remembers can leak from one case into the next. A harness instance is
        shared by every case in a run, and cases run concurrently; the state
        therefore lives in the returned callable, never on `self`.

        The default is the stateless `normalize`.
        """
        return self.normalize
