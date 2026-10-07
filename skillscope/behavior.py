# Copyright Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: MIT

"""Behavioral evals: once the skill has fired, does it do the job?

One skill is installed, one prompt runs to completion, and what the agent
actually did is graded against the case's ``expected_behavior`` /
``unexpected_behavior`` / ``logs_contain`` / ``files_exist``. Only evaluations
that assert something beyond the routing decision run here; the trigger
decision itself belongs to ``routing``, which installs several skills at once.

The CLI lives in ``skillscope/cli.py``; this module is the engine.
"""

from __future__ import annotations

import importlib.util
import shutil
import tempfile
import time
import traceback
from dataclasses import asdict, dataclass, field
from pathlib import Path
from types import ModuleType

from . import datasets, deadline
from .agent import Check, Ungradable, claude
from .datasets import Case


@dataclass
class BehaviorOutcome:
    """One behavioral case: what was asserted and what happened."""

    id: str
    skill: str
    prompt: str
    passed: bool
    elapsed_s: float
    checks: list[dict] = field(default_factory=list)
    error: str | None = None


# --------------------------------------------------------------------------
# Hooks: the escape hatch for setup a JSON file cannot express.
# --------------------------------------------------------------------------


def load_hooks(skill: str) -> ModuleType | None:
    """Import ``<skill>/evals/hooks.py`` if it exists.

    A hook module holds environment plumbing only -- cloning a repo, tearing
    down a container, running an external scoring script. Prompts and
    expectations stay in the dataset, so the thing being asserted is always
    readable without opening Python.

    Recognized (all optional)::

        setup_session(cache_dir)      -> dict of template vars, once per skill
        setup(workspace, case, ctx)   -> dict of extra template vars, per case
        teardown(workspace, case, ctx)
        check(run, case, ctx)         -> raise AssertionError to fail the case
                                         (`run.expects` raises `Ungradable`
                                         when the judge could not answer)
    """
    path = datasets.hooks_path(skill)
    if not path.is_file():
        return None
    spec = importlib.util.spec_from_file_location(f"evalhooks_{skill.replace('-', '_')}", path)
    if spec is None or spec.loader is None:
        raise SystemExit(f"error: could not import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def expand(text: str, ctx: dict) -> str:
    """Substitute ``{name}`` placeholders from `ctx`.

    Plain replacement rather than ``str.format`` because prompts routinely
    contain literal braces (JSON snippets, regex quantifiers) that would
    otherwise raise or be swallowed.
    """
    for key, value in ctx.items():
        text = text.replace("{" + key + "}", str(value))
    return text


# --------------------------------------------------------------------------
# Running and grading
# --------------------------------------------------------------------------


def run_case(
    case: Case, ctx: dict, hooks: ModuleType | None, model: str, effort: str
) -> BehaviorOutcome:
    """Stage one skill, run the prompt to completion, grade what happened."""
    assert case.skill is not None
    bound = deadline.active()
    if bound is not None and bound.expired():
        print(f"  [FAIL] {case.id}: {bound.message()}", flush=True)
        return BehaviorOutcome(
            id=case.id,
            skill=case.skill,
            prompt=case.prompt,
            passed=False,
            elapsed_s=0.0,
            error=bound.message(),
        )

    seed = (datasets.skill_path(case.skill) / case.workspace) if case.workspace else None
    started = time.perf_counter()
    case_ctx = dict(ctx)
    checks: list[Check] = []
    error: str | None = None

    try:
        with claude(model, skill=case.skill, effort=effort, seed=seed) as session:
            workspace = session.workspace
            assert workspace is not None
            if hooks is not None and hasattr(hooks, "setup"):
                case_ctx.update(hooks.setup(workspace, case, case_ctx) or {})
            try:
                run = session.prompt(expand(case.prompt, case_ctx))
                checks = run.evaluate(
                    logs_contain=[expand(t, case_ctx) for t in case.logs_contain],
                    files_exist=[expand(p, case_ctx) for p in case.files_exist],
                    expected_behavior=case.expected_behavior,
                    unexpected_behavior=case.unexpected_behavior,
                )
                if hooks is not None and hasattr(hooks, "check"):
                    try:
                        hooks.check(run, case, case_ctx)
                        checks.append(Check("hook", "evals/hooks.py check()", True))
                    except AssertionError as exc:
                        checks.append(Check("hook", "evals/hooks.py check()", False, str(exc)[:400]))
                    except Ungradable as exc:
                        # `run.expects` could not get a verdict. Same standing as
                        # an ungraded dataset expectation: not a pass, but not
                        # evidence against the skill either.
                        checks.append(
                            Check(
                                "hook",
                                "evals/hooks.py check()",
                                False,
                                str(exc)[:400],
                                graded=False,
                            )
                        )
            finally:
                if hooks is not None and hasattr(hooks, "teardown"):
                    hooks.teardown(workspace, case, case_ctx)
    except Exception as exc:  # noqa: BLE001 -- an infra failure is a result too
        error = f"{type(exc).__name__}: {exc}"
        traceback.print_exc()

    elapsed = round(time.perf_counter() - started, 2)
    passed = error is None and all(c.passed for c in checks) and bool(checks)
    ungraded = [c for c in checks if not c.graded]
    if error is None and not checks:
        error = "case has no behavioral assertions to grade"
    elif error is None and ungraded:
        # Still not a pass -- nothing confirmed the skill behaved -- but the
        # reason is that the grader could not answer, so it belongs in the
        # errors column where it gets triaged as infrastructure rather than
        # read as the skill having violated the expectation.
        error = (
            f"{len(ungraded)} of {len(checks)} expectation(s) could not be "
            f"graded: {ungraded[0].detail}"
        )
    print(
        f"  [{'PASS' if passed else 'FAIL'}] {case.id}: "
        f"{sum(1 for c in checks if c.passed)}/{len(checks)} checks in {elapsed}s"
        + (f" -- {error}" if error else ""),
        flush=True,
    )
    return BehaviorOutcome(
        id=case.id,
        skill=case.skill,
        prompt=case.prompt,
        passed=passed,
        elapsed_s=elapsed,
        checks=[asdict(c) for c in checks],
        error=error,
    )


def run(skills: list[str], cases: list[Case], model: str, effort: str) -> list[BehaviorOutcome]:
    """Run every behavioral case, grouped by skill so session setup happens once."""
    outcomes: list[BehaviorOutcome] = []
    for skill in skills:
        skill_cases = [c for c in cases if c.skill == skill and c.has_behavior]
        if not skill_cases:
            continue

        hooks = load_hooks(skill)
        ctx: dict = {}
        cache_dir: Path | None = None
        if hooks is not None and hasattr(hooks, "setup_session"):
            cache_dir = Path(tempfile.mkdtemp(prefix=f"evalcache-{skill}-"))
            print(f"[behavioral] {skill}: running evals/hooks.py setup_session()", flush=True)
            ctx.update(hooks.setup_session(cache_dir) or {})
        try:
            print(f"[behavioral] {skill}: {len(skill_cases)} case(s)", flush=True)
            for case in skill_cases:
                outcomes.append(run_case(case, ctx, hooks, model, effort))
        finally:
            if cache_dir is not None:
                shutil.rmtree(cache_dir, ignore_errors=True)
    return outcomes


def graded(check: dict) -> bool:
    """Whether a check reached a verdict at all.

    A missing key means it did, so a check dict assembled by something
    other than `asdict(Check)` still reads correctly. `summarize` is the
    only place that has to tolerate the absence; everything downstream
    reads what it emits, which always carries the counts.
    """
    return check.get("graded", True)


def summarize(outcomes: list[BehaviorOutcome], meta: dict) -> dict:
    per_skill: dict[str, dict] = {}
    for skill in sorted({o.skill for o in outcomes}):
        subset = [o for o in outcomes if o.skill == skill]
        per_skill[skill] = {
            "cases": len(subset),
            "passed": sum(1 for o in subset if o.passed),
            "checks": sum(len(o.checks) for o in subset),
            "checks_passed": sum(1 for o in subset for c in o.checks if c["passed"]),
            "checks_graded": sum(1 for o in subset for c in o.checks if graded(c)),
            "checks_ungraded": sum(1 for o in subset for c in o.checks if not graded(c)),
        }
    return {
        "meta": meta,
        "totals": {
            "cases": len(outcomes),
            "passed": sum(1 for o in outcomes if o.passed),
            "checks": sum(len(o.checks) for o in outcomes),
            "checks_passed": sum(1 for o in outcomes for c in o.checks if c["passed"]),
            "checks_graded": sum(1 for o in outcomes for c in o.checks if graded(c)),
            "checks_ungraded": sum(1 for o in outcomes for c in o.checks if not graded(c)),
            "errors": sum(1 for o in outcomes if o.error),
        },
        "per_skill": per_skill,
        "cases": [asdict(o) for o in outcomes],
    }


def render_markdown(summary: dict) -> str:
    totals = summary["totals"]
    meta = summary["meta"]
    lines = [
        "## Skill behavioral",
        "",
        f"**{totals['passed']}/{totals['cases']} cases passed** "
        f"({totals['checks_passed']}/{totals['checks_graded']} graded expectations) "
        f"on `{meta['model']}` (effort `{meta['effort']}`).",
        "",
        "| Skill | Cases | Passed | Graded | Met | Ungraded |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for skill, stats in summary["per_skill"].items():
        lines.append(
            f"| `{skill}` | {stats['cases']} | {stats['passed']} | "
            f"{stats['checks_graded']} | {stats['checks_passed']} | "
            f"{stats['checks_ungraded']} |"
        )

    def row(case_id: str, kind: str, expectation: str, detail: str) -> str:
        expectation = expectation.replace("|", "\\|")
        detail = (detail or "").replace("|", "\\|").replace("\n", " ")
        return f"| `{case_id}` | {kind} | {expectation[:120]} | {detail[:160]} |"

    header = ["| Case | Kind | Expectation | Detail |", "| --- | --- | --- | --- |"]

    failures = [c for c in summary["cases"] if not c["passed"]]
    lines += ["", "### Unmet expectations", ""]
    unmet = []
    for case in failures:
        if case["error"]:
            unmet.append(row(case["id"], "error", "(run failed)", case["error"]))
        unmet += [
            row(case["id"], check["kind"], check["expectation"], check["detail"])
            for check in case["checks"]
            if not check["passed"] and graded(check)
        ]
    if unmet:
        lines += header + unmet
    elif failures:
        # `run_case` always records an error alongside an ungraded check, so
        # this is unreachable from a real run. Guarded anyway: `render_markdown`
        # is the one function that may be handed a summary built elsewhere, and
        # "every expectation met" would be a false all-clear on a failed case.
        lines.append("None unmet. See the ungraded expectations below.")
    else:
        lines.append("None. Every behavioral case met every expectation.")

    # Kept apart from the table above on purpose. These expectations were never
    # decided, so listing them as unmet would report a grader that could not
    # answer as a skill that misbehaved.
    ungraded = [
        row(case["id"], check["kind"], check["expectation"], check["detail"])
        for case in summary["cases"]
        for check in case["checks"]
        if not graded(check)
    ]
    if ungraded:
        lines += [
            "",
            "### Ungraded expectations",
            "",
            "The judge could not reach a verdict on these, so they say nothing "
            "about the skill either way and are excluded from the rate above.",
            "",
        ] + header + ungraded
    return "\n".join(lines) + "\n"
