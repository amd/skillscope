# Copyright Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: MIT

"""Is the skill earning its place, or would the agent have done as well alone?

A behavioral run says whether a skill's cases pass. It cannot say whether they
pass *because of the skill*, and those are different questions: a case can be
written so that any competent agent satisfies it, at which point a green report
is measuring the model rather than the thing under test.

So the same cases are run twice -- once with the skill installed and once with
nothing installed at all -- and the two are placed side by side. `install_skill`
calls the second situation out as a failure mode, "answering from the prompt and
scoring like it, which looks like a bad skill rather than a missing one". Asked
for deliberately and labelled, that failure is the measurement.

Four things a case can turn out to be:

``earns_it``
    Passes with the skill, fails without it. The case discriminates, which is
    the only outcome that tells you the skill did something.
``no_signal``
    Passes either way. Says nothing about the skill, and usually says something
    about the case: an expectation any agent meets is not evidence.
``harmful``
    Fails with the skill, passes without it. Rare, and worth stopping for: the
    skill is actively steering the agent wrong.
``fails_either_way``
    Fails both times. A broken case or a genuinely hard one; either way the
    comparison has nothing to report until it passes somewhere.

This never gates. Two runs of a nondeterministic agent disagree with themselves
often enough that a single pair is evidence rather than proof -- the same reason
`tools/benchmark_engines.py` measures engine agreement against how much one
engine already disagrees with itself, and the same reason the `claude-code` leg
is documented as reporting rather than gating. A number here is a question to
go and look into, not a build failure.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

from .behavior import BehaviorOutcome

# (key, column heading, what it means) -- one row per verdict, in the order a
# reader should care about them.
VERDICTS: tuple[tuple[str, str], ...] = (
    ("earns_it", "passed with the skill, failed without it"),
    ("harmful", "failed with the skill, passed without it"),
    ("no_signal", "passed either way"),
    ("fails_either_way", "failed either way"),
    ("ungraded", "one side or both could not be graded"),
)


@dataclass
class Comparison:
    """One case, run both ways."""

    id: str
    skill: str
    verdict: str
    passed_with: bool
    passed_without: bool
    error: str | None = None


def classify(with_skill: BehaviorOutcome, without: BehaviorOutcome) -> str:
    """Which of `VERDICTS` this pair is.

    An errored run on either side is `ungraded` rather than a failure: a
    provider that dropped the connection has measured nothing, and counting it
    as "the skill made no difference" would be the most flattering possible
    reading of an error.
    """
    if with_skill.error or without.error:
        return "ungraded"
    if with_skill.passed and not without.passed:
        return "earns_it"
    if without.passed and not with_skill.passed:
        return "harmful"
    if with_skill.passed and without.passed:
        return "no_signal"
    return "fails_either_way"


def pair(
    with_skill: list[BehaviorOutcome], without: list[BehaviorOutcome]
) -> list[Comparison]:
    """Match the two runs up by case id.

    A case missing from either side is reported as `ungraded` rather than
    dropped. The two runs are separate evals and either can lose a case, so
    quietly comparing the intersection would shrink the denominator and make
    every remaining number look more confident than it is.
    """
    without_by_id = {o.id: o for o in without}
    with_by_id = {o.id: o for o in with_skill}
    comparisons: list[Comparison] = []

    for case_id in sorted(set(with_by_id) | set(without_by_id)):
        this = with_by_id.get(case_id)
        that = without_by_id.get(case_id)
        if this is None or that is None:
            missing = "the baseline run" if that is None else "the graded run"
            known = this or that
            comparisons.append(
                Comparison(
                    id=case_id,
                    skill=known.skill,
                    verdict="ungraded",
                    passed_with=bool(this and this.passed),
                    passed_without=bool(that and that.passed),
                    error=f"no result from {missing}",
                )
            )
            continue
        comparisons.append(
            Comparison(
                id=case_id,
                skill=this.skill,
                verdict=classify(this, that),
                passed_with=this.passed,
                passed_without=that.passed,
                error=this.error or that.error,
            )
        )
    return comparisons


def summarize(comparisons: list[Comparison], meta: dict) -> dict:
    counts = {key: 0 for key, _ in VERDICTS}
    for comparison in comparisons:
        counts[comparison.verdict] += 1

    per_skill: dict[str, dict] = {}
    for skill in sorted({c.skill for c in comparisons}):
        subset = [c for c in comparisons if c.skill == skill]
        per_skill[skill] = {
            "cases": len(subset),
            **{key: sum(1 for c in subset if c.verdict == key) for key, _ in VERDICTS},
        }

    graded = [c for c in comparisons if c.verdict != "ungraded"]
    return {
        "meta": meta,
        "totals": {
            "cases": len(comparisons),
            "graded": len(graded),
            **counts,
            # Of the cases that were graded both ways, the share that moved
            # when the skill was taken away. The number the run exists for.
            "discriminating": (
                round(sum(1 for c in graded if c.verdict == "earns_it") / len(graded), 3)
                if graded
                else None
            ),
        },
        "per_skill": per_skill,
        "cases": [asdict(c) for c in comparisons],
    }


def render_markdown(summary: dict) -> str:
    totals = summary["totals"]
    meta = summary["meta"]
    share = totals["discriminating"]
    headline = (
        f"**{totals['earns_it']}/{totals['graded']} graded cases needed the skill**"
        if totals["graded"]
        else "**Nothing was graded both ways.**"
    )

    lines = [
        "## Skill staleness",
        "",
        f"{headline} on `{meta['model']}` (effort `{meta['effort']}`). Every case "
        "ran twice: once with the skill installed, once with nothing installed.",
        "",
        "Reporting only. Agents disagree with themselves between runs, so treat "
        "a surprising verdict as somewhere to look rather than as a result.",
        "",
        "| Verdict | Cases | Means |",
        "| --- | --- | --- |",
    ]
    for key, meaning in VERDICTS:
        lines.append(f"| `{key}` | {totals[key]} | {meaning} |")

    if share is not None:
        lines += ["", f"Discriminating: {share:.0%} of the cases graded both ways."]

    notable = [c for c in summary["cases"] if c["verdict"] in ("no_signal", "harmful")]
    lines += ["", "### Cases worth a look", ""]
    if not notable:
        lines.append(
            "None. Every graded case either needed the skill or failed both ways."
        )
    else:
        lines += ["| Case | Verdict | What to do |", "| --- | --- | --- |"]
        for case in notable:
            advice = (
                "the skill steered the agent wrong here"
                if case["verdict"] == "harmful"
                else "the agent already did this unaided; the case is not "
                "evidence about the skill"
            )
            lines.append(f"| `{case['id']}` | {case['verdict']} | {advice} |")
    return "\n".join(lines) + "\n"
