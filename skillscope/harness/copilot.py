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

import os
from typing import Callable

from .base import Harness, RunSpec


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


def _tokens(data: dict) -> tuple[int, int]:
    """Input and output tokens, summed over every model the run used."""
    metrics = ((data.get("accountingSnapshot") or {}).get("modelMetrics")) or {}
    total_in = total_out = 0
    for per_model in metrics.values():
        usage = per_model.get("usage") if isinstance(per_model, dict) else None
        if isinstance(usage, dict):
            total_in += int(usage.get("inputTokens") or 0)
            total_out += int(usage.get("outputTokens") or 0)
    return total_in, total_out


class Copilot(Harness):
    name = "copilot"
    binary = "copilot"
    skills_dir = ".claude/skills"

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
        cmd = [
            self.binary,
            "--output-format", "json",
            "--allow-all",
            "--no-ask-user",
            "--disable-builtin-mcps",
        ]
        if spec.model:
            cmd += ["--model", spec.model]
        if spec.effort:
            cmd += ["--reasoning-effort", spec.effort]
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

    def normalizer(self) -> Callable[[dict], list[dict]]:
        # What the closing event needs and does not carry, remembered from the
        # events before it. Per run, because the next case must not see it.
        answer = {"text": ""}
        tokens = {"input": 0, "output": 0}

        def normalize(event: dict) -> list[dict]:
            kind = event.get("type")
            data = event.get("data")
            data = data if isinstance(data, dict) else {}

            if kind == "tool.execution_start":
                arguments = data.get("arguments")
                return [
                    _tool_use(
                        str(data.get("toolCallId", "")),
                        str(data.get("toolName", "")),
                        arguments if isinstance(arguments, dict) else {},
                    )
                ]

            if kind == "tool.execution_complete":
                if data.get("success"):
                    content = (data.get("result") or {}).get("content")
                else:
                    content = (data.get("error") or {}).get("message")
                return [
                    _tool_result(
                        str(data.get("toolCallId", "")),
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
