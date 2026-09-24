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

import asyncio
import json
import os
import shutil
import signal
import subprocess
from pathlib import Path
from typing import Callable

from .. import agent as legacy_agent

# Tool calls and results are reconstructed from the stream, so they need ids
# that are merely unique within a sample rather than meaningful.
_CALL_PREFIX = "cli"

# Where an early stop records why it happened, for a reader downstream. The
# sample carries no inspect limit when this driver stops the run itself -- the
# bound was enforced here, not by inspect -- so without this the report would
# describe a budgeted stop as a run that ended on its own.
STOP_REASON_KEY = "skillscope_host_stop_reason"

# What `stop_when` returns when the stream reached the CLI's own ending. Named
# rather than spelled inline because the mapper reads it back.
STOP_RESULT = "result"


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


def run_completed(events: list[dict]) -> tuple[bool, str]:
    """Whether the CLI reached the end of the run, and what it said if so.

    Pure, and separated from message construction on purpose: it is the
    distinction the routing mapper depends on -- "the agent reached for no
    skill" against "the agent never ran" -- and the unit suite runs without the
    inspect extra, so a rule only reachable through inspect's message objects
    would have no coverage where it matters.

    A `result` event is the evidence. The CLI emits one when it finishes,
    carrying the closing answer or nothing at all; a stream without one is a
    run that did not get there.
    """
    completed = False
    final = ""
    for event in events:
        if event.get("type") != "result":
            continue
        completed = True
        if isinstance(event.get("result"), str):
            final = event["result"]
    return completed, final


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

    completed, final = run_completed(events)

    # Recorded even when empty, which is the point. A run that finished
    # without a tool call and without a closing sentence is a routing result:
    # the agent reached for no skill. Appending nothing made it
    # indistinguishable from a CLI that never ran, and the routing mapper --
    # which has to tell those apart, and cannot do it from an empty message
    # list -- graded two such cases as infrastructure failures where the
    # legacy engine graded them `correct_trigger` and `true_negative`.
    #
    # The placeholder matches what `state.output` has always used for the same
    # case; only the message list was inconsistent with it.
    if completed or final:
        messages.append(ChatMessageAssistant(content=final or "(no final message)"))
    return messages, final


async def _terminate(proc) -> None:
    """Kill the CLI and everything it started.

    The CLI spawns helpers, and killing only the process we launched leaves
    them running against the same budget the stop was meant to protect. The
    legacy engine learned this and kills the whole group; this is that, in
    asyncio's vocabulary. Spawned into its own session/group precisely so this
    one signal can reach all of it.
    """
    if proc.returncode is not None:
        return
    try:
        if os.name == "nt":
            # No process groups to signal; the tree walk is taskkill's job.
            killer = await asyncio.create_subprocess_exec(
                "taskkill", "/F", "/T", "/PID", str(proc.pid),
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
            await killer.wait()
        else:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
    except (OSError, ProcessLookupError):
        pass
    try:
        proc.kill()
    except (OSError, ProcessLookupError):
        pass
    try:
        await asyncio.wait_for(proc.wait(), timeout=15)
    except (asyncio.TimeoutError, ProcessLookupError):
        pass


def _subprocess_concurrency():
    """inspect's own subprocess limiter, so this driver stays inside its budget.

    Reached by name: `concurrency()` is keyed, so asking for "subprocesses"
    joins the same semaphore `inspect_ai.util.subprocess` uses rather than
    opening a second, unaccounted pool beside it. The size only matters to
    whoever creates it first, and inspect normally has by the time a solver
    runs. Defended anyway -- the limit is private, and a driver that refused to
    run because an internal name moved would be worse than one that ran
    unlimited.
    """
    import contextlib

    from inspect_ai.util import concurrency

    try:
        from inspect_ai.util._subprocess import max_subprocesses_context_var

        limit = max_subprocesses_context_var.get()
    except Exception:  # pragma: no cover -- upstream internals moved
        return contextlib.nullcontext()
    return concurrency("subprocesses", limit, resizable=True)


async def _stream_until(
    cmd: list[str],
    prompt: str,
    workspace: str,
    env: dict,
    stop_when: Callable[[dict], str | None],
) -> tuple[list[dict], str | None, int | None, str]:
    """Run the CLI, reading its stream, and stop the moment `stop_when` says to.

    The reason this exists rather than `inspect_ai.util.subprocess`: that call
    returns once the process has exited, so nothing it gives back can arrive in
    time to end the run early. For a behavioral case that is exactly right --
    the skill has to finish for the scorers to have anything to read. For a
    routing case it means the agent answers the question on its first tool call
    and then does the entire job anyway, unwatched and paid for. Measured over
    one 67-case room, 152 of this leg's 220 tool calls happened after the
    decision it was being asked for.

    So: read the stream line by line, hand each event to the caller's rule, and
    kill the process group the first time it says stop. That is what the legacy
    engine does, and this leg replaces the legacy engine.
    """
    spawn: dict = {}
    if os.name == "nt":
        spawn["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
    else:
        spawn["start_new_session"] = True

    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        cwd=workspace,
        env=env,
        **spawn,
    )

    errors: list[str] = []

    async def _drain_stderr() -> None:
        # Drained concurrently, not at the end: a full stderr pipe blocks the
        # CLI, and a CLI blocked on a pipe nobody is reading never reaches the
        # decision this function is waiting for.
        try:
            while True:
                line = await proc.stderr.readline()
                if not line:
                    return
                errors.append(line.decode("utf-8", "replace"))
        except (asyncio.CancelledError, ValueError):
            return

    stderr_task = asyncio.create_task(_drain_stderr())

    events: list[dict] = []
    stop_reason: str | None = None
    try:
        proc.stdin.write(prompt.encode("utf-8"))
        await proc.stdin.drain()
        proc.stdin.close()
    except (BrokenPipeError, ConnectionResetError, OSError):
        # The CLI rejected the prompt or died early. Whatever it managed to
        # say is read below and reported the same way as any other short run.
        pass

    try:
        while True:
            raw = await proc.stdout.readline()
            if not raw:
                break
            line = raw.decode("utf-8", "replace").strip()
            if not line:
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            events.append(event)
            stop_reason = stop_when(event)
            if stop_reason is not None:
                break
    finally:
        await _terminate(proc)
        stderr_task.cancel()
        try:
            await stderr_task
        except (asyncio.CancelledError, Exception):
            pass

    return events, stop_reason, proc.returncode, "".join(errors)


def install_skill(skill_dir: Path, workspace: str) -> None:
    """Put the skill where the real harness looks for it.

    `.claude/skills/<name>` inside a directory the CLI is given with
    `--add-dir`, which is what the legacy engine has always done and what
    `inspect_swe` does via its own `skills=` argument. Staging is the driver's
    job, and this driver is the one that has to do it by hand: skip it and the
    agent runs with no skill at all, answering from the prompt and scoring like
    it -- which looks like a bad skill rather than a missing one.
    """
    dest = Path(workspace) / ".claude" / "skills" / skill_dir.name
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(skill_dir, dest, dirs_exist_ok=True)


def claude_code_no_sandbox(
    model: str | None,
    effort: str | None,
    skill_dir: Path,
    config_dir: Path | None = None,
    extra_flags: list[str] | None = None,
    stop_when_factory: Callable[[], Callable[[dict], str | None]] | None = None,
):
    """Solver: install the skill, run the real CLI once, record what it did.

    `extra_flags` are the CLI's own cost controls -- `--max-budget-usd`,
    `--no-session-persistence` -- which the legacy engine passes and this
    driver could not, because it did not build that part of the command line.
    The caller probes for them with `routing.supported_flags` first: an older
    build rejects an unknown flag and every case fails identically, which reads
    as a routing collapse rather than a flag problem.

    `config_dir` redirects the CLI away from the runner's own `~/.claude`, the
    way the legacy engine does. Optional because a behavioral case installs one
    skill and grades what the agent produced, so a stray user-level skill is at
    worst noise. A routing case grades *which* skill fired, and a stray one
    joins the room for every case -- so that caller passes it and refuses to
    run without it.

    `stop_when_factory` builds the rule that decides, per stream event, whether
    the run has answered the question being asked of it. Opt-in because the two
    callers want opposite things from the same CLI: a behavioral case is graded
    on what the skill *produced*, so it has to run to the end, and passing a
    rule here would cut the work being measured. A routing case is graded on
    which skill fired, and everything after that is paid for and unread. Left
    `None`, this runs through `inspect_ai.util.subprocess` exactly as before.

    A factory rather than the rule itself because the solver is built once and
    run for every sample, while a budget counts *per case*. Handed a single
    rule, its tally would carry from one case into the next and the room would
    run out of budget partway through -- which has happened here before, and
    reads as a skill that stopped triggering rather than a counter that never
    reset.
    """
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
            cmd += list(extra_flags or [])

            env = legacy_agent.claude_env()
            if config_dir is not None:
                env["CLAUDE_CONFIG_DIR"] = str(config_dir)

            events: list[dict] = []
            if stop_when_factory is None:
                result = await sandbox_subprocess(
                    cmd, input=prompt, cwd=workspace, env=env,
                )
                returncode, stderr = result.returncode, result.stderr or ""
                for line in (result.stdout or "").splitlines():
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        events.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
            else:
                async with _subprocess_concurrency():
                    events, stop_reason, returncode, stderr = await _stream_until(
                        cmd, prompt, workspace, env, stop_when_factory()
                    )
                # Recorded whatever it was, including `None` for a stream that
                # ended on its own. A stop this driver performed leaves no
                # inspect limit behind, so this is the only trace of it.
                if stop_reason is not None and stop_reason != STOP_RESULT:
                    state.store.set(STOP_REASON_KEY, stop_reason)

            if not events:
                raise RuntimeError(
                    "claude produced no parseable stream-json output "
                    f"(exit {returncode}). stderr: {stderr[:300]}"
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
