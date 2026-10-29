# Copyright Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: MIT

"""Claude Code as a harness.

Not new behaviour: this is the command line and environment the host engine
has always used for the `claude` CLI, moved behind the `Harness` contract so
the engine no longer assumes it. Claude Code's stream-json is the canonical
event form, so `normalize` is the default identity.
"""

from __future__ import annotations

import shutil

from .. import agent
from .base import Harness, RunSpec


class ClaudeCode(Harness):
    name = "claude-code"
    binary = "claude"
    skills_dir = ".claude/skills"

    def command(self, spec: RunSpec) -> list[str]:
        cmd = [
            shutil.which(self.binary), "-p",
            "--output-format", "stream-json", "--verbose",
            "--dangerously-skip-permissions",
            "--add-dir", spec.workspace,
        ]
        if spec.model:
            cmd += ["--model", spec.model]
        if spec.effort:
            cmd += ["--effort", spec.effort]
        cmd += list(spec.extra_flags)
        return cmd

    def env(self, spec: RunSpec) -> dict[str, str]:
        env = agent.claude_env()
        if spec.config_dir is not None:
            env["CLAUDE_CONFIG_DIR"] = str(spec.config_dir)
        return env
