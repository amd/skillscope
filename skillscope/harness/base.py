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

import shutil
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Callable


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
    config_dir: Path | None = None
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

    def require(self) -> None:
        """Refuse to start when the harness is not installed."""
        if not shutil.which(self.binary):
            raise SystemExit(f"error: '{self.binary}' CLI not found on PATH")

    @abstractmethod
    def command(self, spec: RunSpec) -> list[str]:
        """The argv that runs one case, headless, streaming JSON events."""

    @abstractmethod
    def env(self, spec: RunSpec) -> dict[str, str]:
        """The environment that process gets, config isolation included."""

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
