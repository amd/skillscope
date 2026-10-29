# Copyright Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: MIT

"""Behavioral evals under `inspect_ai`.

Shared by every engine that runs on the framework: the task, the scorer, the
sandbox and the reporting live here, and the caller supplies the solver that
drives the agent. `claude-code-no-sandbox` passes one; `claude-code` builds its own task in
`verify.py` because `inspect_swe` supplies the whole agent rather than a solver.

`run()` matches `behavior.run()` -- same arguments, same `BehaviorOutcome`
list -- so swapping engines is a one-line substitution in the CLI and every
report path downstream is untouched.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from .. import agent, config, deadline, routing as routing_core
from ..behavior import BehaviorOutcome
from ..datasets import Case
from . import convert, hooks, models, sandbox as sandbox_spec, scorers, stats

# An agent that never decides it is finished must still stop. The legacy engine
# bounded this with `--case-timeout` and a process kill; inspect expresses it
# declaratively, and a message cap catches the loop a wall-clock cap only ends
# after paying for it.
MESSAGE_LIMIT = 120

# A model that reaches no provider never calls the submit tool, so it loops to
# whatever cap it is given -- and every turn is a real sandbox round trip. The
# wiring run proves the machinery in a handful of turns; the rest is the mock
# failing to finish, slowly.
MOCK_MESSAGE_LIMIT = 6


# How much of the command's budget to keep back from inspect's own per-sample
# limit. The `--timeout` deadline ends the process with `os._exit`, which takes
# the report and the transcript with it -- so a run that overruns says only
# that it overran. Handing inspect the whole budget makes the two fire together
# and the hard kill wins the race. Stopping the sample early enough for inspect
# to score what exists and write the log turns a timeout into evidence: eight
# Instinct runs have now overrun and not one of them said where it got to.
TIMEOUT_RESERVE_S = 120


def task_time_limit(bound) -> int | None:
    """The per-sample limit to give inspect, inside the command's own deadline."""
    if bound is None:
        return None
    return max(60, int(bound.remaining() - TIMEOUT_RESERVE_S))


def realtime_logging() -> bool:
    """Whether inspect should keep its live sample buffer for this run.

    The buffer exists so `inspect view` can watch a run in progress, and it
    lives in a sqlite file under the user data directory, named after the task.
    On Windows that directory is the service account's profile, which is long
    enough that a task named after a longer skill crosses MAX_PATH -- sqlite
    then answers "unable to open database file" and the whole task dies. Two
    skills on the same runner passed and one did not, purely on the length of
    its name.

    Nothing watches a CI run live, and the `.eval` log is written either way,
    so the buffer is cost without benefit exactly where it breaks.
    """
    return not sys.platform.startswith("win")


def message_limit_for(model: str) -> int:
    """How many turns this model should be allowed before the case is stopped."""
    if model.lower().startswith(agent.NO_PROVIDER_PREFIXES):
        return MOCK_MESSAGE_LIMIT
    return MESSAGE_LIMIT


def build_task(
    skill: str,
    cases: list[Case],
    model: str,
    ctx: dict | None = None,
    *,
    solver_factory,
    baseline: bool = False,
):
    """One inspect `Task` per skill: its cases, its skill installed, its scorer.

    `solver_factory` is what drives the agent; everything around it -- scoring,
    judging, reporting -- stays the same whichever one is passed. It receives
    the skill's directory because staging is the driver's job: each driver puts
    the skill where the agent it runs will look for it.

    Required, and keyword-only. It was optional while a built-in react agent
    was the default, and a caller that forgot it silently graded a different
    agent than it asked for.

    `baseline` only names the task. Whether the skill is actually staged is
    the solver's business, so the flag and the factory have to agree -- the
    name is what makes a disagreement visible in the log directory rather
    than invisible in the numbers.
    """
    from inspect_ai import Task

    skill_dir = config.active().skill_path(skill)
    samples = [convert.sample_from_case(c, skill_dir, ctx) for c in cases]

    # `evals/hooks.py`, where the skill ships one. `cleanup` rather than a
    # trailing solver: inspect runs it in a `finally` under a shielded cancel
    # scope, so teardown still happens when the agent raises -- which is the
    # property a hook that removes containers exists for.
    hook = hooks.load(skill)

    bound = deadline.active()
    return Task(
        name=f"behavioral-{skill}" + ("-baseline" if baseline else ""),
        metadata={BASELINE_KEY: baseline, SKILL_KEY: skill},
        tags=["skillscope", "baseline" if baseline else "graded"],
        dataset=samples,
        setup=hooks.setup_solver(hook, skill, cases),
        solver=solver_factory(skill_dir),
        cleanup=hooks.cleanup_fn(hook, skill, cases),
        scorer=scorers.expectations(),
        sandbox=sandbox_spec.for_skill(skill),
        message_limit=message_limit_for(model),
        time_limit=task_time_limit(bound),
    )


def _failed(skill: str, cases: list[Case], detail: str) -> list[BehaviorOutcome]:
    """One failed outcome per case, for a skill that could not be run at all."""
    return [
        BehaviorOutcome(
            id=case.id,
            skill=skill,
            prompt=case.prompt,
            passed=False,
            elapsed_s=0.0,
            error=detail,
            degraded=routing_core.is_provider_error(detail),
        )
        for case in cases
    ]


def _inspect_room(routing_engine, sample, skill: str) -> tuple[str | None, str | None]:
    """`(the skill that fired, why the room was not what it claimed)`.

    One walk over the tool calls, answering both questions, because the second
    is only meaningful beside the first. `detect_activation` reports a skill
    nobody staged as `other:<name>`, which is exactly the contamination this
    looks for -- the runner's own `~/.claude/skills` are registered next to
    the staged ones, and a behavioral run passes no `config_dir` to keep them
    out.

    Reading the skill's own source is the second route, and the one isolation
    cannot close: the command runs from the repo root, so a directory the
    staging step skipped is still on disk for an agent to find. A baseline
    that read it was not a baseline.

    Best effort, by the same argument as `_activated`: a shape this cannot
    walk is an unknown room, not a failed case.
    """
    fired: str | None = None
    why: str | None = None
    marks = (f"skills/{skill}", f"/{skill}/", "_bundled_skills")
    try:
        for call in routing_engine._tool_calls(sample):
            hit = routing_core.detect_activation(
                routing_engine._activation_event(call),
                [skill],
                allow_body_path=routing_engine.ALLOW_BODY_PATH,
            )
            if hit and hit.startswith("other:") and why is None:
                why = f"another skill fired: {hit.removeprefix('other:')}"
            elif hit and fired is None:
                fired = hit
            if why is None:
                text = json.dumps(getattr(call, "arguments", None) or {}, ensure_ascii=False)
                if any(mark in text for mark in marks):
                    why = f"read the skill's own files ({skill})"
    except Exception:  # noqa: BLE001 -- see the docstring
        return fired, why
    return fired, why


def _activated(routing_engine, sample, skill: str) -> str | None:
    """The skill the agent reached for in this sample, if any.

    Graded by `routing.detect_activation`, the same rule and the same tool-name
    matching the routing leg uses, so "fired" means one thing across the two
    commands. A baseline sample has no skill installed and so returns None,
    which is the point: it is what makes "the skill fired and the outcome
    differed" sayable at all.

    Best effort. A log shape this cannot walk leaves the field None rather
    than failing a case that was otherwise graded fine -- an unknown activation
    is a weaker report, not a broken run.
    """
    try:
        for call in routing_engine._tool_calls(sample):
            hit = routing_core.detect_activation(
                routing_engine._activation_event(call),
                [skill],
                allow_body_path=routing_engine.ALLOW_BODY_PATH,
            )
            if hit:
                return hit
    except Exception:  # noqa: BLE001 -- see the docstring
        return None
    return None


def _outcomes(log, skill: str, cases: list[Case]) -> list[BehaviorOutcome]:
    """Map one inspect `EvalLog` back onto skillscope's outcome objects.

    A task that produced no samples at all reports one failed outcome per case
    rather than an empty list: an infrastructure failure must not render as
    "every expectation met".

    But a task that produced *some* samples keeps them, whatever its status.
    `log.status` goes to `error` when any sample raised, so failing the batch
    on it threw away every case the run had already graded and paid for --
    one container that would not start, and a skill's whole report read as a
    total failure. `engine/routing.py:_outcomes` has always done it this way,
    for the same reason; the two now agree. Cases with no sample are still
    errored below, so nothing goes quietly missing.
    """
    # Imported here rather than at module scope: `engine/routing.py` imports
    # this module, so a top-level import would close the cycle.
    from . import routing as routing_engine  # noqa: PLC0415

    prompts = {c.id: c.prompt for c in cases}
    outcomes: list[BehaviorOutcome] = []

    if not log.samples:
        detail = getattr(log.error, "message", None) or "the task produced no samples"
        return _failed(skill, cases, f"inspect task failed: {detail}")

    for sample in log.samples:
        case_id = str(sample.id)
        checks: list[dict] = []
        error: str | None = None

        for score in (sample.scores or {}).values():
            checks.extend((score.metadata or {}).get(scorers.CHECKS, []))

        if sample.error is not None:
            error = f"{sample.error.message}"
        elif not checks:
            error = "case has no behavioral assertions to grade"

        activated, contaminated = _inspect_room(routing_engine, sample, skill)

        outcomes.append(
            BehaviorOutcome(
                id=case_id,
                skill=skill,
                prompt=prompts.get(case_id, ""),
                passed=error is None and bool(checks) and all(c["passed"] for c in checks),
                elapsed_s=round(getattr(sample, "total_time", None) or 0.0, 2),
                checks=checks,
                error=error,
                degraded=routing_core.is_provider_error(error),
                activated=activated,
                contaminated=contaminated,
            )
        )

    # Samples inspect never reported back are still cases somebody asked for.
    # Now that a partial log is kept rather than discarded wholesale, the cases
    # missing from it are the ones that need saying -- silence here would read
    # as a smaller, cleaner run rather than an incomplete one.
    failure = getattr(getattr(log, "error", None), "message", None)
    missing = (
        f"inspect task failed before this case ran: {failure}"
        if failure
        else "inspect returned no sample for this case"
    )
    graded = {outcome.id for outcome in outcomes}
    outcomes.extend(
        _failed(skill, [c for c in cases if c.id not in graded], missing)
    )
    return outcomes


# Eval-level metadata naming the `run_eval` call that started an eval, so a
# stopped run reads back its own log and nobody else's.
CALL_TOKEN = "skillscope_call"

# Which half of a staleness comparison a log is, and whose skill it is about.
# The task name says the same thing, but a name is a string somebody has to
# parse; inspect filters on metadata and tags, and so can anything reading a
# log directory afterwards. Written by both engines, so a reader does not have
# to know which one produced the log it is holding.
BASELINE_KEY = "skillscope_baseline"
SKILL_KEY = "skillscope_skill"


def stop_requested() -> bool:
    """Whether the deadline's graceful stage has asked the command to stop."""
    bound = deadline.active()
    return bound is not None and bound.interrupted


def run_eval(inspect_eval, task, *, log_dir: str, **kwargs):
    """`inspect_eval(task, ...)`, keeping the logs of a run the deadline stopped.

    When the graceful stage interrupts a run inside inspect's loop, inspect
    cancels the samples, runs `Task.cleanup`, writes the log as `cancelled` --
    and then returns `[]`. Measured with five of six samples finished: all five
    were in the log on disk, and none came back from `eval()`. Without this, a
    run that overran its wall clock dropped every case it had already graded
    and paid for, and the gate blamed "every case errored".

    The interrupt does not always land there. In `eval()`'s synchronous setup
    it surfaces as `KeyboardInterrupt`, and on Python 3.10 anyio can surface it
    as `CancelledError`; both are taken as the stop they are, and the readback
    runs the same way.

    So the log is read back from `log_dir`, and only this call's. The default
    directory accumulates across runs and can be shared by another command
    running beside this one -- including one running the very same task, since
    routing's is always `routing`. A task name or a time window cannot tell
    those apart, so each call tags its eval with a token of its own and reads
    back only what carries it. Files that predate the call are skipped before
    their headers are read, which is all that `before` is for. An empty
    result is the caller's to report, since only it knows which cases were
    asked for.
    """
    import asyncio
    import uuid

    from inspect_ai.log import read_eval_log

    token = uuid.uuid4().hex
    before = _log_files(log_dir)
    try:
        logs = inspect_eval(
            task, log_dir=log_dir, metadata={CALL_TOKEN: token}, **kwargs
        )
    except (KeyboardInterrupt, asyncio.CancelledError):
        if not stop_requested():
            raise
        logs = []
    if logs or not stop_requested():
        return logs
    ours = [
        uri for uri in sorted(_log_files(log_dir) - before)
        if (read_eval_log(uri, header_only=True).eval.metadata or {}).get(CALL_TOKEN)
        == token
    ]
    return [read_eval_log(uri) for uri in ours]


def _log_files(log_dir: str) -> set[str]:
    """The logs in `log_dir`, as URIs inspect can read back on every platform.

    Not `inspect_ai.log.list_eval_logs`: on Windows the names it returns do
    not survive the trip back into `read_eval_log`. It builds them as
    `file://C:...`, one slash short, so the drive letter parses as a host and
    is dropped -- every readback failed with FileNotFoundError on a path
    starting `\\Users`. `Path.as_uri()` gives `file:///C:/...`. Local only,
    which is every directory skillscope writes logs to.
    """
    root = Path(log_dir)
    if not root.is_dir():
        return set()
    return {
        path.resolve().as_uri()
        for path in root.iterdir()
        if path.suffix in (".eval", ".json")
    }


def run(
    skills: list[str],
    cases: list[Case],
    model: str,
    effort: str,
    *,
    solver_factory,
    log_dir: str | None = None,
    baseline: bool = False,
) -> list[BehaviorOutcome]:
    """Run every behavioral case, grouped by skill. Mirrors `behavior.run`.

    `baseline` is passed through to the task and used in the progress lines.
    It does not itself remove the skill: the caller's `solver_factory` does
    that, and this says so in the output.
    """
    from inspect_ai import eval as inspect_eval

    sandbox_spec.require_provider()

    outcomes: list[BehaviorOutcome] = []
    for skill in skills:
        skill_cases = [c for c in cases if c.skill == skill and c.has_behavior]
        if not skill_cases:
            continue

        if stop_requested():
            # The stop is for the command, not the task it landed in. Starting
            # the next skill would spend the reserve meant for cleanup and meet
            # the hard kill, which loses the report along with everything else.
            outcomes.extend(_failed(skill, skill_cases, deadline.active().message()))
            continue

        label = "baseline" if baseline else "behavioral"
        print(f"[{label}] {skill}: {len(skill_cases)} case(s)", flush=True)
        try:
            logs = run_eval(
                inspect_eval,
                build_task(
                    skill,
                    skill_cases,
                    model,
                    solver_factory=solver_factory,
                    baseline=baseline,
                ),
                model=model,
                model_args=models.model_args(model),
                log_dir=log_dir or str(Path(".skillscope") / "logs"),
                log_realtime=realtime_logging(),
                # A sample that raises is that case's failure, not the batch's.
                # inspect's default ends the task on the first error, so one
                # container that would not start discarded every case already
                # graded -- `_outcomes` then reported the whole skill failed.
                # Errored samples still come back in the log and are still
                # graded as failures; what changes is that their neighbours
                # survive.
                fail_on_error=False,
                # skillscope's own progress lines are the report; inspect's rich
                # display takes over the terminal and produces nothing useful
                # when a CI job pipes stdout to a file.
                display="plain",
            )
        except SystemExit as exc:
            # One skill's broken setup is that skill's failure, not everybody's.
            # A malformed sandbox declaration used to abort the whole command,
            # throwing away results for skills already graded and paid for --
            # the same reason the structural gate reads only the skills a run
            # is about.
            outcomes.extend(_failed(skill, skill_cases, str(exc)))
            continue

        if not logs and stop_requested():
            # Stopped before inspect wrote anything. Failed rather than absent,
            # so the report still counts the cases it was asked for.
            outcomes.extend(_failed(skill, skill_cases, deadline.active().message()))
            continue

        for log in logs:
            stats.record_log(log)
            outcomes.extend(_outcomes(log, skill, skill_cases))

    for outcome in outcomes:
        passed = sum(1 for c in outcome.checks if c["passed"])
        print(
            f"  [{'PASS' if outcome.passed else 'FAIL'}] {outcome.id}: "
            f"{passed}/{len(outcome.checks)} checks in {outcome.elapsed_s}s"
            + (f" -- {outcome.error}" if outcome.error else ""),
            flush=True,
        )
    return outcomes
