# Copyright Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: MIT

"""GitHub Copilot CLI as a harness.

`copilot --output-format json` prints one JSON event per line, none of them in
Claude Code's shape, so this translates. What the recorded output shows (CLI
1.0.105, `tests/fixtures/harness/copilot/`):

* A tool call is `tool.execution_start` (`toolName`, `arguments`,
  `toolCallId`) and its result is `tool.execution_complete`. Activating a skill
  is a call to the tool named `skill` with `arguments.skill` set, which is the
  call the engines already read. A call the permission rules deny is still
  recorded, with `success: false`.
* The closing `result` event carries an exit code and premium-request usage and
  **no answer**. The answer is the last assistant message, printed earlier, so
  the translator remembers it. This is why a harness can keep state per run.
* Token counts arrive once, in `session.usage_checkpoint`, per model.
* There is no event naming the skills the session loaded. `copilot skill list
  --json` does, without a model call, and is how the room can be checked.
* Streaming deltas, reasoning and bookkeeping events carry nothing the engines
  read and are dropped.

Other facts worth knowing before changing this:

* It authenticates from `COPILOT_GITHUB_TOKEN`, `GH_TOKEN` or `GITHUB_TOKEN` in
  the environment, which is why an empty `COPILOT_HOME` is enough to isolate
  the session from the runner's own configuration and user-level skills.
* Copilot ships built-in skills, so they are in every session's room beside the
  ones staged here.
* It discovers skills in `.claude/skills`, the directory Claude Code uses.
* A run is billed in premium requests rather than dollars, so a spend cap in
  dollars cannot bind.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from typing import Callable, Mapping

from .. import agent, deadline
from .base import Harness, RunSpec, _run_bounded

# The model a CI run is held to. One premium request per prompt where the largest
# model costs fifteen, which matters for a run that makes dozens of them; and one
# fixed model, so scores stay comparable from one run to the next.
CI_MODEL = "claude-sonnet-5.5"

# How long `skill list` may take. It makes no model call, so a minute is a hang.
ROOM_TIMEOUT_S = 60.0

# For a run that is only asking a question, the one tool name it may use, and no
# tool is called that. `--available-tools` is an allowlist: the model is offered
# only what it names, so naming nothing real leaves it nothing. That is what it
# replaced the denylist for. `--excluded-tools` has to spell the CLI's own tool
# names, and a list written from memory had `shell` where the CLI's tool is
# `powershell` (it also runs `read_powershell`, `view`, `create`, `edit`, ...), so
# the shell tool stayed on offer. Checked with CLI 1.0.105: with this flag the
# model reports no tool it can see and none is called; an empty value is ignored.
# Nothing is pre-approved either, so a tool that did get through would be denied.
JUDGE_TOOLS = "skillscope-no-tools"

# The generic aliases, in this CLI's names.
ALIASES = {
    "opus": "claude-opus-5.5",
    "sonnet": "claude-sonnet-5.5",
    "haiku": "claude-haiku-5.5",
}


def _tool_use(call_id: str, name: str, arguments: dict) -> dict:
    return {
        "type": "assistant",
        "message": {
            "content": [
                {"type": "tool_use", "id": call_id, "name": name, "input": arguments}
            ]
        },
    }


def _tool_result(call_id: str, content: str) -> dict:
    return {
        "type": "user",
        "message": {
            "content": [
                {"type": "tool_result", "tool_use_id": call_id, "content": content}
            ]
        },
    }


def _count(value: object) -> int:
    """A token count, or 0 for anything that is not a plain number."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return 0
    return int(value)


def _field(container: object, key: str) -> object:
    """`container[key]` when the container is an object, else None."""
    return container.get(key) if isinstance(container, dict) else None


def _tokens(data: dict) -> tuple[int, int]:
    """Input and output tokens, summed over every model the run used."""
    metrics = _field(_field(data, "accountingSnapshot"), "modelMetrics")
    total_in = total_out = 0
    for per_model in metrics.values() if isinstance(metrics, dict) else ():
        usage = per_model.get("usage") if isinstance(per_model, dict) else None
        if isinstance(usage, dict):
            total_in += _count(usage.get("inputTokens"))
            total_out += _count(usage.get("outputTokens"))
    return total_in, total_out


class Copilot(Harness):
    name = "copilot"
    binary = "copilot"
    skills_dir = ".claude/skills"
    default_model = CI_MODEL
    credentials = ("COPILOT_GITHUB_TOKEN", "GH_TOKEN", "GITHUB_TOKEN")

    def resolve_model(self, model: str | None) -> str:
        wanted = model or self.default_model
        return ALIASES.get(wanted.lower(), wanted)

    def pin_model(self, model: str) -> str:
        if not agent.is_automated_env() or model == CI_MODEL:
            return model
        if model.lower().startswith(agent.NO_PROVIDER_PREFIXES):
            return model
        agent._safe_print(
            f"[skillscope] automated run: coercing model '{model}' -> "
            f"'{CI_MODEL}' to pin the CI model."
        )
        return CI_MODEL

    def command(self, spec: RunSpec) -> list[str]:
        # The prompt goes in on stdin, which this CLI reads when no `-p` is
        # given: no quoting to get wrong, and no command-line length limit.
        #
        # `--allow-all` is Claude Code's `--dangerously-skip-permissions`: a
        # non-interactive run cannot answer a prompt, so anything not
        # pre-approved is denied and the agent could not do the work. Like the
        # host engine's Claude Code, it runs unsandboxed. The built-in MCP
        # servers are off so a case is graded on the skill, not on a tool the
        # skill's author never had in mind.
        cmd = [self.binary, "--output-format", "json"]
        if not spec.tool_free:
            cmd.append("--allow-all")
        cmd += ["--no-ask-user", "--disable-builtin-mcps"]
        if spec.tool_free:
            cmd.append(f"--available-tools={JUDGE_TOOLS}")
        if spec.model:
            cmd += ["--model", spec.model]
        if spec.effort:
            cmd += ["--reasoning-effort", spec.effort]
        if spec.tool_free and spec.extra_flags:
            # Appended last, a flag here could grant back what the allowlist took
            # away. Refused like Claude Code's tool-free command: a question that
            # needs another flag is one this harness does not answer.
            raise NotImplementedError(
                "a tool-free copilot command takes no extra flags: they could "
                "re-grant what the allowlist removed"
            )
        cmd += list(spec.extra_flags)
        return cmd

    def env(self, spec: RunSpec) -> dict[str, str]:
        # A self-update in the middle of a graded run changes what is being
        # graded, and the version is part of what a report is read against. The
        # runner's own setting still wins, as it always did. The credential is
        # the runner's to supply: this is only what the harness itself sets.
        env = {"COPILOT_AUTO_UPDATE": os.environ.get("COPILOT_AUTO_UPDATE", "false")}
        if spec.config_dir is not None:
            env["COPILOT_HOME"] = spec.config_dir
        return env

    def room(self, workspace: str, env: Mapping[str, str]) -> list[str] | None:
        # `skill list --json` reports what a session started here would load,
        # without a model call. Built-in skills ship with the CLI and are in
        # every session, so they are not part of what was staged.
        timeout = ROOM_TIMEOUT_S
        bound = deadline.active()
        if bound is not None:
            # Spent like the preflight and the judge: a check that cannot finish
            # inside the command's budget is not made, and the room is reported
            # as unverified.
            if bound.remaining() <= 0:
                return None
            timeout = min(timeout, bound.remaining())
        executable = shutil.which(self.binary)
        if executable is None:
            return None
        try:
            # Bounded like the preflight and the judge: the CLI is a shim when it
            # was installed with npm, and a plain `subprocess.run` timeout kills
            # the shim and leaves the program running with the pipe open.
            proc = _run_bounded(
                [executable, "skill", "list", "--json"],
                stdin="", cwd=workspace, env=dict(env), timeout=timeout,
            )
            listed = json.loads(proc.stdout or "")
        except (OSError, subprocess.SubprocessError, ValueError, TypeError):
            return None
        if proc.returncode != 0 or not isinstance(listed, list):
            return None
        return [
            str(entry["name"])
            for entry in listed
            if isinstance(entry, dict)
            and isinstance(entry.get("name"), str)
            and entry["name"]
            and entry.get("source") != "builtin"
            # A skill the CLI reports as switched off is not in the room.
            and entry.get("enabled") is not False
        ]

    def normalizer(self) -> Callable[[dict], list[dict]]:
        # What the closing event needs and does not carry, remembered from the
        # events before it. Per run, because the next case must not see it.
        answer = {"text": ""}
        tokens = {"input": 0, "output": 0}
        started: set[str] = set()

        def normalize(event: dict) -> list[dict]:
            # Any JSON value is a legal line of output, not only an object.
            if not isinstance(event, dict):
                return []
            kind = event.get("type")
            data = event.get("data")
            data = data if isinstance(data, dict) else {}

            if kind == "tool.execution_start":
                call_id, name = data.get("toolCallId"), data.get("toolName")
                if not (isinstance(call_id, str) and call_id and isinstance(name, str) and name):
                    return []
                started.add(call_id)
                arguments = data.get("arguments")
                return [_tool_use(call_id, name, arguments if isinstance(arguments, dict) else {})]

            if kind == "tool.execution_complete":
                call_id = data.get("toolCallId")
                if call_id not in started:
                    # A result no call can claim would shift every later one
                    # onto the wrong call, which the engine pairs by position.
                    return []
                if data.get("success"):
                    content = _field(data.get("result"), "content")
                else:
                    content = _field(data.get("error"), "message")
                return [
                    _tool_result(
                        call_id,
                        content if isinstance(content, str) else "",
                    )
                ]

            if kind == "assistant.message":
                # A sub-agent's message is its own, not the run's answer.
                from_sub_agent = event.get("agentId") or data.get("agentId")
                content = data.get("content")
                if isinstance(content, str) and content.strip() and not from_sub_agent:
                    answer["text"] = content
                return []

            if kind == "session.usage_checkpoint":
                tokens["input"], tokens["output"] = _tokens(data)
                return []

            if kind == "result":
                failed = bool(event.get("exitCode"))
                canonical: list[dict] = []
                if tokens["input"] or tokens["output"]:
                    # The usage recorder reads tokens off assistant events and
                    # this CLI reports them once, at the end.
                    canonical.append(
                        {
                            "type": "assistant",
                            "message": {
                                "content": [],
                                "usage": {
                                    "input_tokens": tokens["input"],
                                    "output_tokens": tokens["output"],
                                },
                            },
                        }
                    )
                canonical.append(
                    {
                        "type": "result",
                        "subtype": "error" if failed else "success",
                        "is_error": failed,
                        "result": answer["text"],
                    }
                )
                return canonical

            return []

        return normalize
