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
    # Set when the failure was the model provider's rather than the skill's.
    # A behavioral case that failed on a gateway 504 is a skill nobody
    # measured, and reporting it beside one that genuinely failed its
    # expectations invites exactly the wrong conclusion.
    degraded: bool = False


# --------------------------------------------------------------------------
# Hooks: the escape hatch for setup a JSON file cannot express.
# --------------------------------------------------------------------------




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






def summarize(outcomes: list[BehaviorOutcome], meta: dict) -> dict:
    per_skill: dict[str, dict] = {}
    for skill in sorted({o.skill for o in outcomes}):
        subset = [o for o in outcomes if o.skill == skill]
        per_skill[skill] = {
            "cases": len(subset),
            "passed": sum(1 for o in subset if o.passed),
            "checks": sum(len(o.checks) for o in subset),
            "checks_passed": sum(1 for o in subset for c in o.checks if c["passed"]),
        }
    return {
        "meta": meta,
        "totals": {
            "cases": len(outcomes),
            "passed": sum(1 for o in outcomes if o.passed),
            "checks": sum(len(o.checks) for o in outcomes),
            "checks_passed": sum(1 for o in outcomes for c in o.checks if c["passed"]),
            "errors": sum(1 for o in outcomes if o.error),
            # Of those errors, the ones the provider caused. A behavioral run
            # with these in it has not measured the skills it names.
            "degraded": sum(1 for o in outcomes if o.degraded),
        },
        "per_skill": per_skill,
        "cases": [asdict(o) for o in outcomes],
    }




def _isolation_note(meta: dict) -> str:
    """One line saying whether the agent was contained while it worked.

    Behavioral runs the agent to completion with permissions bypassed, so
    whether it was isolated changes what the numbers cost to obtain. A report
    that omits it reads as though it were, and the answer differs per platform:
    the Windows legs have no sandbox available at all.
    """
    where = meta.get("sandbox")
    if where is None:
        return ""
    if meta.get("sandbox_isolated"):
        return f"Cases ran isolated, in `{where}`."
    return (
        f"**Cases ran unsandboxed** (`{where}`): the agent worked directly in "
        "the harness's own filesystem, with permissions bypassed."
    )


def render_markdown(summary: dict) -> str:
    totals = summary["totals"]
    meta = summary["meta"]
    lines = [
        "## Skill behavioral",
        "",
        f"**{totals['passed']}/{totals['cases']} cases passed** "
        f"({totals['checks_passed']}/{totals['checks']} individual expectations) "
        f"on `{meta['model']}` (effort `{meta['effort']}`).",
        "",
        _isolation_note(meta),
        "",
        "| Skill | Cases | Passed | Expectations | Met |",
        "| --- | --- | --- | --- | --- |",
    ]
    for skill, stats in summary["per_skill"].items():
        lines.append(
            f"| `{skill}` | {stats['cases']} | {stats['passed']} | "
            f"{stats['checks']} | {stats['checks_passed']} |"
        )

    failures = [c for c in summary["cases"] if not c["passed"]]
    lines += ["", "### Unmet expectations", ""]
    if not failures:
        lines.append("None. Every behavioral case met every expectation.")
    else:
        lines += ["| Case | Kind | Expectation | Detail |", "| --- | --- | --- | --- |"]
        for case in failures:
            if case["error"]:
                detail = case["error"].replace("|", "\\|").replace("\n", " ")
                lines.append(f"| `{case['id']}` | error | (run failed) | {detail[:160]} |")
            for check in case["checks"]:
                if check["passed"]:
                    continue
                expectation = check["expectation"].replace("|", "\\|")
                detail = (check["detail"] or "").replace("|", "\\|").replace("\n", " ")
                lines.append(
                    f"| `{case['id']}` | {check['kind']} | {expectation[:120]} | {detail[:160]} |"
                )
    return "\n".join(lines) + "\n"
