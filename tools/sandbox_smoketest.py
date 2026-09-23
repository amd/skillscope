#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: MIT

"""Prove the sandbox machinery holds, without grading anything.

Every engine that runs on `inspect_ai` shares a layer beneath the agent: a
sandbox is started, a case's `workspace:` fixtures are staged into it, the
guest is listed, and the listing is matched against what the case asked for.
That layer is where every cross-platform bug so far has been -- a container
that would not start, a Windows guest that could not be listed, a fixture that
landed somewhere the scorer never looked.

It used to be covered by grading a fixture skill on the harness-independent
engine with a mock model. That engine is gone, and the two that replace it both
drive the real `claude` CLI, so neither can run key-free on a fork's pull
request. Rather than lose the coverage, this exercises the same layer directly:
a stub solver that does nothing at all stands in for the agent.

Doing nothing is the point. The case's expectation is a file the *case* seeded,
so it passes only if the fixture was staged, the sandbox was listed, and the
listing was matched -- and it cannot be passed by an agent that got lucky. No
model is ever called, so this costs nothing and is deterministic.

    tools/sandbox_smoketest.py <repo> [--skill demo-skill]

Exits non-zero, loudly, on the first thing that did not hold.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from skillscope import config, datasets  # noqa: E402
from skillscope.engine import behavioral, sandbox as sandbox_spec  # noqa: E402

# Never reached -- the stub solver returns before anything generates -- but
# `eval()` requires a model, and this is the one that cannot bill anybody.
MODEL = "mockllm/model"


def stub_solver(skill_dir: Path):
    """An agent that does nothing, so only the machinery can pass the case."""
    from inspect_ai.solver import solver

    @solver
    def _noop():
        async def solve(state, generate):
            return state

        return solve

    return _noop()


def fail(message: str) -> None:
    print(f"FAIL: {message}", file=sys.stderr)
    raise SystemExit(1)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("repo", help="Repo holding the fixture skill.")
    parser.add_argument("--skill", default="demo-skill")
    args = parser.parse_args()

    config.use(config.build(Path(args.repo).resolve(), skills_dir="*"))

    cases = [
        case
        for case in datasets.load_dataset(args.skill)
        if case.has_behavior
    ]
    if not cases:
        fail(f"{args.skill} has no case asserting anything, so nothing is proven")

    expected = sandbox_spec.provider()
    print(f"[smoketest] {len(cases)} case(s), sandbox provider {expected!r}", flush=True)

    sandbox_spec.require_provider()
    from inspect_ai import eval as inspect_eval

    logs = inspect_eval(
        behavioral.build_task(args.skill, cases, MODEL, solver_factory=stub_solver),
        model=MODEL,
        log_dir=str(Path(".skillscope") / "logs"),
        log_realtime=behavioral.realtime_logging(),
        display="plain",
    )

    outcomes = []
    for log in logs:
        outcomes.extend(behavioral._outcomes(log, args.skill, cases))

    # An infrastructure failure is not a result. This is what catches a
    # sandbox that never started.
    errored = [o for o in outcomes if o.error]
    if errored:
        fail(f"{len(errored)} case(s) errored: {[o.error for o in errored]}")

    checks = [check for outcome in outcomes for check in outcome.checks]
    if not checks:
        fail("nothing was graded, so nothing was proven")

    # A guest that cannot be listed reports the same shape as an agent that
    # produced nothing, so the difference is asserted rather than assumed.
    unlistable = [c for c in checks if "could not list the sandbox" in (c.get("detail") or "")]
    if unlistable:
        fail(f"the sandbox could not be listed: {unlistable}")

    seeded = [c for c in checks if c.get("kind") == "files_exist"]
    if not seeded:
        fail("the seeded-file check did not run, so staging was never exercised")
    unmet = [c for c in seeded if not c.get("passed")]
    if unmet:
        fail(f"a file the case seeded was not found in the sandbox: {unmet}")

    print(
        f"[smoketest] ok -- {len(checks)} check(s) graded, "
        f"{len(seeded)} seeded fixture(s) found, sandbox {expected!r}."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
