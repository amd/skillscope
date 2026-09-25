# Copyright Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: MIT

"""The Claude Code verification leg (`--engine claude-code`).

The only leg that runs the real agent *and* isolates it. `legacy` and
`claude-code-no-sandbox` both drive the same CLI on the host, so they measure
the machine as it is, with whatever else is installed on it. This one measures
the skill alone -- and when the two disagree, the disagreement is a fact about
one of those environments rather than about the skill.

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
from . import (
    behavioral,
    convert,
    models,
    sandbox as sandbox_spec,
    scorers,
    stats,
    tools,
)

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

    # Not a preference. `inspect_swe` prepares the guest for the CLI by writing
    # $HOME/.claude/settings.json outright, discarding whatever was there. In a
    # container that file belongs to nobody and the write is the setup working
    # as intended. Under the `local` provider $HOME is the developer's own, and
    # the same write destroys their real configuration -- permissions, model,
    # gateway environment -- with no backup and no warning. Observed, not
    # theorised: it cost one settings.json before this guard existed.
    provider = sandbox_spec.provider()
    if provider in sandbox_spec.NOT_ISOLATED:
        raise SystemExit(
            f"error: --engine claude-code needs a real sandbox, but "
            f"{sandbox_spec.SANDBOX_ENV}={provider!r} selects one that shares "
            "the host's filesystem. inspect_swe would overwrite your own "
            "~/.claude/settings.json to set up the agent. Unset "
            f"{sandbox_spec.SANDBOX_ENV} for a container, or use "
            "--engine claude-code-no-sandbox to run the CLI on the host "
            "without it touching your configuration."
        )

    try:
        import inspect_swe  # noqa: F401
    except ModuleNotFoundError as exc:  # pragma: no cover -- environment shape
        raise SystemExit(INSTALL_HINT) from exc

    # Checked here rather than left to the call, because the call is inside a
    # task inspect has already started: the failure arrives as a TypeError
    # underneath a traceback, one leg of a three-leg comparison quietly
    # produces no report, and the run carries on. Observed exactly that way on
    # a runner whose inspect-swe predated the argument and so satisfied the
    # old floor without upgrading.
    import inspect as _inspect

    from inspect_swe import claude_code

    if "effort" not in _inspect.signature(claude_code).parameters:
        raise SystemExit(
            "error: --engine claude-code needs inspect-swe >= 0.2.71 for "
            "claude_code(effort=). Without it this leg runs at the model's "
            "default reasoning effort while the others run at the effort they "
            "were given, so the two are not comparable. Upgrade with:\n"
            "    pip install --upgrade 'inspect-swe>=0.2.71'"
        )


def _ensure_workdir():
    """Create the directory the scorers read, before the agent runs in it.

    The other engines reach it lazily, the first time one of our own tools is
    used. This leg's agent brings its own tools and never calls ours, so
    nothing would create it -- and `cwd` below has to name a directory that
    exists.
    """
    from inspect_ai.solver import solver

    @solver
    def _ensure():
        async def solve(state, generate):
            await tools.workdir()
            return state

        return solve

    return _ensure()


def build_task(
    skill: str,
    cases: list[Case],
    model: str,
    ctx: dict | None = None,
    effort: str | None = None,
):
    """One task per skill, solved by real Claude Code rather than our agent."""
    from inspect_ai import Task
    from inspect_ai.solver import chain
    from inspect_swe import claude_code

    skill_dir = config.active().skill_path(skill)
    samples = [convert.sample_from_case(c, skill_dir, ctx) for c in cases]

    # Where the agent works has to be where the scorers look. A container
    # starts at `/`, and left to itself the agent scattered its output there:
    # every `files_exist` check in the first trial run reported an empty
    # sandbox, which reads as "the agent did nothing" rather than "the agent
    # worked somewhere else". `inspect_swe` takes a working directory rather
    # than instructions, so this is set rather than asked for.
    bound = deadline.active()
    return Task(
        name=f"claude-code-{skill}",
        dataset=samples,
        # `skills=` installs into .claude/skills inside the sandbox, which is
        # where the real harness looks -- the point of this leg is that its
        # discovery machinery, not ours, decides what happens.
        # `effort` reaches the agent rather than being dropped here: unset is
        # not neutral, it is the model's own default, and the legs this one is
        # compared against all pass the value they were given.
        solver=chain(
            _ensure_workdir(),
            claude_code(
                skills=[skill_dir], cwd=tools.workdir_path(), effort=effort or None
            ),
        ),
        scorer=scorers.expectations(),
        sandbox=sandbox_spec.for_skill(skill),
        message_limit=behavioral.message_limit_for(model),
        time_limit=behavioral.task_time_limit(bound),
    )


def run(
    skills: list[str], cases: list[Case], model: str, effort: str
) -> list[BehaviorOutcome]:
    """Mirrors `behavior.run`, so the CLI and the benchmark treat it the same."""
    from inspect_ai import eval as inspect_eval

    require()

    sandbox_spec.require_provider()

    outcomes: list[BehaviorOutcome] = []
    for skill in skills:
        skill_cases = [c for c in cases if c.skill == skill and c.has_behavior]
        if not skill_cases:
            continue

        print(f"[claude-code] {skill}: {len(skill_cases)} case(s)", flush=True)
        logs = inspect_eval(
            build_task(skill, skill_cases, model, effort=effort),
            model=model,
            model_args=models.model_args(model),
            log_dir=str(Path(".skillscope") / "logs"),
            log_realtime=behavioral.realtime_logging(),
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
