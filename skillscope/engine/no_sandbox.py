# Copyright Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: MIT

"""Real Claude Code, driven as a subprocess, inside inspect's framework.

`inspect_swe` runs the CLI *inside* the sandbox and reaches the model through a
bridge whose proxy is a Linux binary -- which is why it cannot run on Windows at
all. This runs the CLI on the host instead, the way the legacy engine does, and
maps what it did into inspect's messages so the scorers, the judge and the
`.eval` transcript all work unchanged.

The point is fidelity. A skill is written for this harness, so the real harness
runs everywhere and only the isolation differs: `inspect_swe` in a container on
Linux, this on any platform -- and on Windows this is the only option, because
inspect's sandbox layer assumes a POSIX guest whichever agent drives it.

Unsandboxed by construction: the CLI runs on the host, in the sample's own
working directory. That is what the legacy engine already does, so it is not a
regression -- but the report says `sandbox_isolated: false` rather than leaving
a reader to assume otherwise.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from .. import agent as legacy_agent

# Tool calls and results are reconstructed from the stream, so they need ids
# that are merely unique within a sample rather than meaningful.
_CALL_PREFIX = "cli"


def require_local() -> None:
    """This driver runs on the host, so the sandbox has to be the host."""
    from . import sandbox as sandbox_spec

    provider = sandbox_spec.provider()
    if provider not in sandbox_spec.NOT_ISOLATED:
        raise SystemExit(
            f"error: --engine claude-code-no-sandbox runs the CLI on the host, so it needs "
            f"SKILLSCOPE_SANDBOX=local, not {provider!r}. For a sandboxed run of "
            "the real harness on Linux, use --engine claude-code."
        )
    if not shutil.which("claude"):
        raise SystemExit("error: 'claude' CLI not found on PATH")


async def _workspace() -> str:
    """The directory this sample's files live in."""
    from inspect_ai.util import sandbox
    from inspect_ai.util._sandbox.local import LocalSandboxEnvironment

    return sandbox().as_type(LocalSandboxEnvironment).directory.name


def events_to_messages(events: list[dict], prompt: str) -> tuple[list, str]:
    """Turn the CLI's stream into inspect messages, and the final answer.

    The scorers read tool calls to see what the agent did and assistant text to
    see what it told the user, so both have to survive the crossing. Reusing
    the legacy walk keeps one parser for one stream format.
    """
    from inspect_ai.model import ChatMessageAssistant, ChatMessageTool
    from inspect_ai.tool import ToolCall

    tool_uses: list[tuple[str, str]] = []
    tool_results: list[str] = []
    for event in events:
        legacy_agent._walk(event, tool_uses, tool_results)

    messages: list = []
    for index, (name, arguments) in enumerate(tool_uses):
        call_id = f"{_CALL_PREFIX}-{index}"
        try:
            parsed = json.loads(arguments)
        except json.JSONDecodeError:
            parsed = {"raw": arguments}
        messages.append(
            ChatMessageAssistant(
                content="",
                tool_calls=[
                    ToolCall(id=call_id, function=name, arguments=parsed)
                ],
            )
        )
        if index < len(tool_results):
            messages.append(
                ChatMessageTool(
                    content=tool_results[index],
                    tool_call_id=call_id,
                    function=name,
                )
            )

    final = ""
    for event in events:
        if event.get("type") == "result" and isinstance(event.get("result"), str):
            final = event["result"]
    if final:
        messages.append(ChatMessageAssistant(content=final))
    return messages, final


def install_skill(skill_dir: Path, workspace: str) -> None:
    """Put the skill where the real harness looks for it.

    `.claude/skills/<name>` inside a directory the CLI is given with
    `--add-dir`, which is what the legacy engine has always done and what
    `inspect_swe` does via its own `skills=` argument. The react agent reaches
    the same place differently, through inspect's `skill()` tool -- so a solver
    that replaces the react agent has to do this itself or the agent runs with
    no skill at all, answering from the prompt and scoring like it.
    """
    dest = Path(workspace) / ".claude" / "skills" / skill_dir.name
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(skill_dir, dest, dirs_exist_ok=True)


def claude_code_no_sandbox(model: str | None, effort: str | None, skill_dir: Path):
    """Solver: install the skill, run the real CLI once, record what it did."""
    from inspect_ai.model import ModelOutput
    from inspect_ai.solver import solver

    @solver
    def _claude_code_no_sandbox():
        async def solve(state, generate):
            from inspect_ai.util import subprocess as sandbox_subprocess

            workspace = await _workspace()
            install_skill(skill_dir, workspace)
            prompt = state.input_text

            cmd = [
                shutil.which("claude"), "-p",
                "--output-format", "stream-json", "--verbose",
                "--dangerously-skip-permissions",
                "--add-dir", workspace,
            ]
            if model:
                cmd += ["--model", model]
            if effort:
                cmd += ["--effort", effort]

            result = await sandbox_subprocess(
                cmd, input=prompt, cwd=workspace, env=legacy_agent.claude_env(),
            )

            events: list[dict] = []
            for line in (result.stdout or "").splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    events.append(json.loads(line))
                except json.JSONDecodeError:
                    continue

            if not events:
                raise RuntimeError(
                    "claude produced no parseable stream-json output "
                    f"(exit {result.returncode}). stderr: {(result.stderr or '')[:300]}"
                )

            for event in events:
                legacy_agent.usage.record_stream_event(event)

            messages, final = events_to_messages(events, prompt)
            state.messages.extend(messages)
            state.output = ModelOutput.from_content(
                model=model or "claude", content=final or "(no final message)"
            )
            return state

        return solve

    return _claude_code_no_sandbox()
