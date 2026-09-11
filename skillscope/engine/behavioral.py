# Copyright Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: MIT

"""Behavioral evals on the inspect engine.

`run()` matches `behavior.run()` -- same arguments, same `BehaviorOutcome`
list -- so swapping engines is a one-line substitution in the CLI and every
report path downstream is untouched.
"""

from __future__ import annotations

from pathlib import Path

from .. import config, deadline, usage
from ..behavior import BehaviorOutcome
from ..datasets import Case
from . import convert, models, sandbox as sandbox_spec, scorers, stats, tools

# An agent that never decides it is finished must still stop. The legacy engine
# bounded this with `--case-timeout` and a process kill; inspect expresses it
# declaratively, and a message cap catches the loop a wall-clock cap only ends
# after paying for it.
MESSAGE_LIMIT = 120


def _tools(skill_dir: Path) -> list:
    """Tools the agent gets for a behavioral run.

    The skill under test, plus the cross-platform set from `engine/tools.py` --
    inspect's own `bash()` and `text_editor()` assume a POSIX guest, which the
    Windows legs do not have.
    """
    from inspect_ai.tool import skill

    return [skill([skill_dir]), *tools.toolset()]


def build_task(skill: str, cases: list[Case], ctx: dict | None = None):
    """One inspect `Task` per skill: its cases, its skill installed, its scorer."""
    from inspect_ai import Task
    from inspect_ai.agent import react

    skill_dir = config.active().skill_path(skill)
    samples = [convert.sample_from_case(c, skill_dir, ctx) for c in cases]

    bound = deadline.active()
    return Task(
        name=f"behavioral-{skill}",
        dataset=samples,
        solver=react(tools=_tools(skill_dir)),
        scorer=scorers.expectations(),
        sandbox=sandbox_spec.for_skill(skill),
        message_limit=MESSAGE_LIMIT,
        time_limit=int(bound.remaining()) if bound is not None else None,
    )


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
        return [
            BehaviorOutcome(
                id=case.id,
                skill=skill,
                prompt=case.prompt,
                passed=False,
                elapsed_s=0.0,
                error=f"inspect task failed: {detail}",
            )
            for case in cases
        ]

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
    skills: list[str], cases: list[Case], model: str, effort: str
) -> list[BehaviorOutcome]:
    """Run every behavioral case, grouped by skill. Mirrors `behavior.run`."""
    from inspect_ai import eval as inspect_eval

    outcomes: list[BehaviorOutcome] = []
    for skill in skills:
        skill_cases = [c for c in cases if c.skill == skill and c.has_behavior]
        if not skill_cases:
            continue

        print(f"[behavioral] {skill}: {len(skill_cases)} case(s)", flush=True)
        logs = inspect_eval(
            build_task(skill, skill_cases),
            model=model,
            model_args=models.model_args(model),
            log_dir=str(Path(".skillscope") / "logs"),
            # skillscope's own progress lines are the report; inspect's rich
            # display takes over the terminal and produces nothing useful when
            # a CI job pipes stdout to a file.
            display="plain",
        )
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
