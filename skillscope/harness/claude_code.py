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

from typing import Mapping

from .. import agent
from .. import routing
from .base import Harness, RunSpec


class ClaudeCode(Harness):
    name = "claude-code"
    binary = "claude"
    skills_dir = ".claude/skills"
    default_model = "opus"
    credentials = routing.ENV_CREDENTIALS
    supports_sandbox = True

    def pin_model(self, model: str) -> str:
        # CI forces `opus`; the rule and its exemptions live in `agent`.
        return agent.enforce_model_policy(model) or model

    def can_isolate_room(self, environ: Mapping[str, str] | None = None) -> bool:
        if environ is None:
            return routing.can_isolate_config()
        return super().can_isolate_room(environ)

    def cost_flags(self, max_budget_usd: float | None) -> list[str]:
        # Imported here: the engines import the harness package, not the reverse.
        from ..engine import routing as engine_routing

        return engine_routing.host_cost_flags(max_budget_usd)

    def command(self, spec: RunSpec) -> list[str]:
        cmd = [
            self.binary, "-p",
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
        env = agent.claude_settings()
        if spec.config_dir is not None:
            env["CLAUDE_CONFIG_DIR"] = spec.config_dir
        return env
