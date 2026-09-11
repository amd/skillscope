# Copyright Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: MIT

"""The Claude Code verification leg (`--engine claude-code`).

The `inspect` engine grades a skill with a harness-independent agent, which is
a deliberate choice: it tests whether a skill's *instructions* work rather than
how one product reads them. This leg exists to answer the question that choice
raises -- do the results still hold under the real thing?

It runs actual Claude Code inside the sandbox, via `inspect_swe`, and produces
the same outcome objects as the other two engines, so the benchmark tool can
diff its report against theirs with nothing new.

**Reporting only.** It is not a gate. Harness runs are nondeterministic and the
harness is not what we are grading; a divergence here is a question about the
skill, not a build failure.

**Linux only.** `inspect_swe` shells `bash -c` merely to locate the CLI, and the
model proxy it starts in the guest is a Linux binary, so a Windows guest cannot
run this leg at all. That is the whole reason the primary engine does not depend
on it.
"""

from __future__ import annotations

import sys
from pathlib import Path

from .. import config, deadline
from ..behavior import BehaviorOutcome
from ..datasets import Case
from . import behavioral, convert, models, sandbox as sandbox_spec, scorers, stats

INSTALL_HINT = (
    "error: --engine claude-code needs the verify extra. Install it with:\n"
    "    pip install 'skillscope[verify]'"
)


def require() -> None:
    """Fail early and legibly rather than at the first sandbox call."""
    if sys.platform.startswith("win"):
        raise SystemExit(
            "error: --engine claude-code cannot run on Windows. inspect_swe "
            "requires a POSIX guest, both to locate the CLI and to run the "
            "model proxy it installs in the sandbox."
        )
    try:
        import inspect_swe  # noqa: F401
    except ModuleNotFoundError as exc:  # pragma: no cover -- environment shape
        raise SystemExit(INSTALL_HINT) from exc


def build_task(skill: str, cases: list[Case], ctx: dict | None = None):
    """One task per skill, solved by real Claude Code rather than our agent."""
    from inspect_ai import Task
    from inspect_swe import claude_code

    skill_dir = config.active().skill_path(skill)
    samples = [convert.sample_from_case(c, skill_dir, ctx) for c in cases]

    bound = deadline.active()
    return Task(
        name=f"claude-code-{skill}",
        dataset=samples,
        # `skills=` installs into .claude/skills inside the sandbox, which is
        # where the real harness looks -- the point of this leg is that its
        # discovery machinery, not ours, decides what happens.
        solver=claude_code(skills=[skill_dir]),
        scorer=scorers.expectations(),
        sandbox=sandbox_spec.for_skill(skill),
        message_limit=behavioral.MESSAGE_LIMIT,
        time_limit=int(bound.remaining()) if bound is not None else None,
    )


def run(
    skills: list[str], cases: list[Case], model: str, effort: str
) -> list[BehaviorOutcome]:
    """Mirrors `behavior.run`, so the CLI and the benchmark treat it the same."""
    from inspect_ai import eval as inspect_eval

    require()

    outcomes: list[BehaviorOutcome] = []
    for skill in skills:
        skill_cases = [c for c in cases if c.skill == skill and c.has_behavior]
        if not skill_cases:
            continue

        print(f"[claude-code] {skill}: {len(skill_cases)} case(s)", flush=True)
        logs = inspect_eval(
            build_task(skill, skill_cases),
            model=model,
            model_args=models.model_args(),
            log_dir=str(Path(".skillscope") / "logs"),
            display="plain",
        )
        for log in logs:
            stats.record_log(log)
            outcomes.extend(behavioral._outcomes(log, skill, skill_cases))

    for outcome in outcomes:
        passed = sum(1 for c in outcome.checks if c["passed"])
        print(
            f"  [{'PASS' if outcome.passed else 'FAIL'}] {outcome.id}: "
            f"{passed}/{len(outcome.checks)} checks in {outcome.elapsed_s}s"
            + (f" -- {outcome.error}" if outcome.error else ""),
            flush=True,
        )
    return outcomes
