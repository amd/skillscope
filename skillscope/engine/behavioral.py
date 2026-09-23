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

import sys
from pathlib import Path

from .. import agent, config, deadline, usage
from ..behavior import BehaviorOutcome
from ..datasets import Case
from . import convert, models, sandbox as sandbox_spec, scorers, stats

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
):
    """One inspect `Task` per skill: its cases, its skill installed, its scorer.

    `solver_factory` is what drives the agent; everything around it -- scoring,
    judging, reporting -- stays the same whichever one is passed. It receives
    the skill's directory because staging is the driver's job: each driver puts
    the skill where the agent it runs will look for it.

    Required, and keyword-only. It was optional while a built-in react agent
    was the default, and a caller that forgot it silently graded a different
    agent than it asked for.
    """
    from inspect_ai import Task

    skill_dir = config.active().skill_path(skill)
    samples = [convert.sample_from_case(c, skill_dir, ctx) for c in cases]

    bound = deadline.active()
    return Task(
        name=f"behavioral-{skill}",
        dataset=samples,
        solver=solver_factory(skill_dir),
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
        )
        for case in cases
    ]


def _outcomes(log, skill: str, cases: list[Case]) -> list[BehaviorOutcome]:
    """Map one inspect `EvalLog` back onto skillscope's outcome objects.

    A task that failed outright reports one failed outcome per case rather than
    an empty list: an infrastructure failure that produced no samples must not
    render as "every expectation met".
    """
    prompts = {c.id: c.prompt for c in cases}
    outcomes: list[BehaviorOutcome] = []

    if log.status == "error" or not log.samples:
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

        outcomes.append(
            BehaviorOutcome(
                id=case_id,
                skill=skill,
                prompt=prompts.get(case_id, ""),
                passed=error is None and bool(checks) and all(c["passed"] for c in checks),
                elapsed_s=round(getattr(sample, "total_time", None) or 0.0, 2),
                checks=checks,
                error=error,
            )
        )
    return outcomes


def run(
    skills: list[str],
    cases: list[Case],
    model: str,
    effort: str,
    *,
    solver_factory,
) -> list[BehaviorOutcome]:
    """Run every behavioral case, grouped by skill. Mirrors `behavior.run`."""
    from inspect_ai import eval as inspect_eval

    sandbox_spec.require_provider()

    outcomes: list[BehaviorOutcome] = []
    for skill in skills:
        skill_cases = [c for c in cases if c.skill == skill and c.has_behavior]
        if not skill_cases:
            continue

        print(f"[behavioral] {skill}: {len(skill_cases)} case(s)", flush=True)
        try:
            logs = inspect_eval(
                build_task(skill, skill_cases, model, solver_factory=solver_factory),
                model=model,
                model_args=models.model_args(model),
                log_dir=str(Path(".skillscope") / "logs"),
                log_realtime=realtime_logging(),
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
