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

from dataclasses import asdict, dataclass, field, replace

from .behavior import BehaviorOutcome

# (key, column heading, what it means) -- one row per verdict, in the order a
# reader should care about them.
VERDICTS: tuple[tuple[str, str], ...] = (
    ("earns_it", "passed with the skill, failed without it"),
    ("harmful", "failed with the skill, passed without it"),
    ("no_signal", "passed either way"),
    ("fails_either_way", "failed either way"),
    ("guards_only", "asserts nothing the skill could be credited for"),
    ("ungraded", "one side or both could not be graded"),
)

# What a case claims the skill will *do*. Only these decide a verdict.
POSITIVE_KINDS = ("expected_behavior", "logs_contain", "files_exist")

# What a case forbids. Deliberately excluded from the verdict.
#
# A guard rail exists to catch a future edit to the skill that starts doing the
# forbidden thing, so a well-written one holds whether or not the skill is
# installed -- the base model was never going to run `sudo` unprompted either.
# Counting that as "the skill made no difference" would mark the checks most
# worth keeping as dead weight, and push an author to delete them. They are
# reported on their own terms instead: held, or broken.
GUARD_KIND = "unexpected_behavior"


@dataclass
class Comparison:
    """One case, run both ways.

    Both sides' checks are carried rather than reduced to the verdict. A case
    that failed both ways can still have failed them *differently* -- the first
    run of this command had an agent that tried the task and was stopped by the
    machine on one side, and one that never attempted it on the other, which is
    the skill working. The verdict could not see it and neither could the JSON,
    so the only way to find out was to open the raw inspect logs, which is not
    a report.

    No `diverged` flag is computed from them. The details are judge prose, and
    two runs can word the same behaviour differently; a boolean over that would
    claim a difference the harness cannot stand behind. They are shown instead,
    for a reader to judge.
    """

    id: str
    skill: str
    verdict: str
    passed_with: bool
    passed_without: bool
    error: str | None = None
    checks_with: list[dict] = field(default_factory=list)
    checks_without: list[dict] = field(default_factory=list)
    # Whether the agent actually reached for the skill on the installed side.
    # An `earns_it` with this unset is the weak kind: something differed, but
    # not demonstrably because of the thing under test.
    activated_with: str | None = None
    # Why the baseline was not the empty room it claims to be, if it was not.
    # A contaminated baseline cannot support `no_signal`: "the agent did this
    # unaided" is precisely the claim the contamination refutes.
    baseline_contaminated: str | None = None


def positive(checks: list[dict]) -> list[dict]:
    """The checks that claim the skill will do something."""
    return [c for c in checks if c.get("kind") in POSITIVE_KINDS]


def guards(checks: list[dict]) -> list[dict]:
    """The checks that forbid something."""
    return [c for c in checks if c.get("kind") == GUARD_KIND]


def _met(outcome: BehaviorOutcome) -> bool:
    """Whether this side met every positive claim on the case."""
    claims = positive(outcome.checks)
    return bool(claims) and all(c["passed"] for c in claims)


def classify(with_skill: BehaviorOutcome, without: BehaviorOutcome) -> str:
    """Which of `VERDICTS` this pair is.

    Judged on the positive claims alone -- see `GUARD_KIND` for why the
    forbidden ones are kept out of it.

    An errored run on either side is `ungraded` rather than a failure: a
    provider that dropped the connection has measured nothing, and counting it
    as "the skill made no difference" would be the most flattering possible
    reading of an error.
    """
    if with_skill.error or without.error:
        return "ungraded"
    if not positive(with_skill.checks) and not positive(without.checks):
        # Nothing was claimed, so nothing can be credited. Reporting this as
        # `fails_either_way` would read as a broken case rather than as one
        # that only ever pinned down what must not happen.
        return "guards_only"
    here, there = _met(with_skill), _met(without)
    if here and not there:
        return _voided_if_dirty("earns_it", without)
    if there and not here:
        return _voided_if_dirty("harmful", without)
    verdict = (
        "no_signal" if (here and there)
        else "fails_either_way"
    )
    return _voided_if_dirty(verdict, without)


# Verdicts a contaminated baseline cannot support, and why it cannot.
#
# Contamination is not symmetric. `no_signal` and `harmful` both rest on the
# baseline having *managed* something, and a baseline that reached another
# skill or read this skill's own files did not manage it unaided -- which is
# the whole of what those verdicts claim.
#
# `earns_it` and `fails_either_way` rest on the baseline having *failed*. A
# baseline that failed while holding more than it should have fails all the
# harder without it, so the finding survives its own contamination and voiding
# it would throw away a conclusion the run is entitled to.
VOIDED_BY_CONTAMINATION = ("no_signal", "harmful")


def _voided_if_dirty(verdict: str, without: BehaviorOutcome) -> str:
    if without.contaminated and verdict in VOIDED_BY_CONTAMINATION:
        return "ungraded"
    return verdict


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
                    checks_with=list(this.checks) if this else [],
                    checks_without=list(that.checks) if that else [],
                    activated_with=this.activated if this else None,
                    baseline_contaminated=that.contaminated if that else None,
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
                checks_with=list(this.checks),
                checks_without=list(that.checks),
                activated_with=this.activated,
                baseline_contaminated=that.contaminated,
            )
        )
    return comparisons


def outcomes_from_log(log) -> tuple[str, bool, str | None, list[BehaviorOutcome]]:
    """`(skill, was_baseline, run_key, outcomes)` for one inspect log.

    Rebuilt from the samples rather than from a dataset, because an offline
    reader has no `Case` list: the log is all there is. Prompts are left empty
    for the same reason -- the comparison is about verdicts, and inventing a
    prompt field nobody wrote would be worse than an honest blank.
    """
    from .engine import behavioral, scorers
    from .engine import routing as routing_engine

    eval_meta = log.eval.metadata or {}
    skill = eval_meta.get(behavioral.SKILL_KEY) or log.eval.task
    outcomes: list[BehaviorOutcome] = []
    for sample in log.samples or []:
        checks: list[dict] = []
        for score in (sample.scores or {}).values():
            checks.extend((score.metadata or {}).get(scorers.CHECKS, []))
        error = sample.error.message if sample.error is not None else None
        if error is None and not checks:
            # Exactly what the live path sets (`behavioral._outcomes`), and it
            # has to be the same string for the same reason the detector is
            # shared: a sample whose checks could not be read rebuilds as
            # `checks=[]`, which makes `_met()` False, which makes the case
            # `earns_it` off the back of a harness failure. Without this the
            # rebuild is *more permissive than the run it rebuilds*, quietly
            # restoring a verdict the live run had voided as `ungraded`.
            error = "case has no behavioral assertions to grade"
        outcomes.append(
            BehaviorOutcome(
                id=str(sample.id),
                skill=skill,
                prompt="",
                passed=error is None and bool(checks) and all(c["passed"] for c in checks),
                elapsed_s=round(getattr(sample, "total_time", None) or 0.0, 2),
                checks=checks,
                error=error,
                # Both detected here too. Without the first, a rebuilt
                # comparison reports every `earns_it` as never having fired;
                # without the second it is *more permissive than the run it
                # rebuilds*, quietly restoring verdicts the live run voided.
                **dict(
                    zip(
                        ("activated", "contaminated"),
                        behavioral._inspect_room(routing_engine, sample, skill),
                    )
                ),
            )
        )
    return (
        skill,
        bool(eval_meta.get(behavioral.BASELINE_KEY)),
        eval_meta.get(behavioral.STALENESS_RUN_KEY),
        outcomes,
    )


def from_logs(log_dir: str) -> list[Comparison]:
    """Rebuild the comparison from logs already on disk.

    Two runs become a pair because they carry the same skill, the same
    `STALENESS_RUN_KEY`, and opposite `BASELINE_KEY` -- which is why all three
    are written as metadata rather than inferred from a task name.

    Grouping on the run key is what makes this safe to point at
    `.skillscope/logs`, which is the documented thing to do with it. That
    directory is shared by `behavioral`, `verify` and `routing`, is gitignored
    wholesale, and nothing prunes it; and every ordinary `behavioral` run
    stamps `BASELINE_KEY: False`, so a run key is the only thing separating
    this run's graded half from an unrelated run of the same skill last month.
    Logs with no run key are skipped for that reason: they were not written by
    a staleness run and have no half to be.

    A directory holding only one side of a run is reported as a comparison
    against nothing, the same way a case missing from one side is: silently
    dropping it would read as a smaller, cleaner run.

    Logs that are not skillscope's own are passed over. A shared log directory
    is normal, and failing on a neighbour's file would make this unusable in
    exactly the directories it is meant for.
    """
    from inspect_ai.log import list_eval_logs, read_eval_log

    from .engine import behavioral

    sides: dict[tuple[str, str], dict[bool, list[BehaviorOutcome]]] = {}
    for info in list_eval_logs(log_dir):
        log = read_eval_log(info.name)
        if behavioral.BASELINE_KEY not in (log.eval.metadata or {}):
            continue
        skill, baseline, run_key, outcomes = outcomes_from_log(log)
        if not run_key:
            continue
        sides.setdefault((skill, run_key), {}).setdefault(baseline, []).extend(outcomes)

    comparisons: list[Comparison] = []
    for key in sorted(sides):
        half = sides[key]
        comparisons.extend(
            pair(
                _no_duplicates(half.get(False, [])),
                _no_duplicates(half.get(True, [])),
            )
        )
    return comparisons


def _no_duplicates(outcomes: list[BehaviorOutcome]) -> list[BehaviorOutcome]:
    """`outcomes`, with every case id that appears twice on this side errored.

    One run key and one side can still span several logs -- a run grades one
    skill per task, and a run key covers the whole invocation -- and nothing
    stops two of them carrying the same case id. `pair` builds
    `{o.id: o for o in ...}`, so a duplicate silently collapses to whichever
    came last, and `list_eval_logs` returns newest first, which makes the
    survivor the *older* of the two.

    Both copies are errored, not just the later one: which of them is this
    side's answer is exactly what cannot be told, so keeping either would be
    the guess. `classify` turns the error into `ungraded`, consistent with how
    this module reports every other ambiguity it cannot resolve.
    """
    tally: dict[str, int] = {}
    for outcome in outcomes:
        tally[outcome.id] = tally.get(outcome.id, 0) + 1
    return [
        outcome if tally[outcome.id] == 1
        else replace(
            outcome,
            error="this run has more than one log reporting this case, and "
            "which of them is its answer cannot be told apart",
        )
        for outcome in outcomes
    ]


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

    graded = [
        c for c in comparisons if c.verdict not in ("ungraded", "guards_only")
    ]

    # Guard rails, counted separately. A broken one is the serious finding
    # here: a skill that made the agent do the forbidden thing.
    broken = [
        c
        for c in comparisons
        if any(not g["passed"] for g in guards(c.checks_with))
    ]
    guard_checks = sum(len(guards(c.checks_with)) for c in comparisons)
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
            "guard_checks": guard_checks,
            "guards_broken_with_skill": len(broken),
            # `earns_it` cases where the skill was installed but never fired.
            # The outcome differed; the skill is not shown to be why.
            "baseline_contaminated": sum(
                1 for c in comparisons if c.baseline_contaminated
            ),
            "earned_without_firing": sum(
                1
                for c in comparisons
                if c.verdict == "earns_it" and not c.activated_with
            ),
        },
        "per_skill": per_skill,
        "cases": [asdict(c) for c in comparisons],
    }


def _reasons(case: dict) -> list[tuple[str, str, str]]:
    """Each expectation that failed on either side, with both sides' details.

    Matched on the expectation text, which is what both runs were graded
    against. A check present on one side only still appears, so a run that
    produced fewer checks than its partner cannot hide the difference.
    """
    with_by_text = {c["expectation"]: c for c in case["checks_with"]}
    without_by_text = {c["expectation"]: c for c in case["checks_without"]}
    rows: list[tuple[str, str, str]] = []
    for text in sorted(set(with_by_text) | set(without_by_text)):
        here, there = with_by_text.get(text), without_by_text.get(text)
        if (here and here["passed"]) and (there and there["passed"]):
            continue
        rows.append(
            (
                text,
                _detail(here),
                _detail(there),
            )
        )
    return rows


def _detail(check: dict | None) -> str:
    if check is None:
        return "not graded this side"
    if check["passed"]:
        return "passed"
    return (check["detail"] or "failed").replace("|", "\\|").replace("\n", " ")


def render_markdown(summary: dict) -> str:
    totals = summary["totals"]
    meta = summary["meta"]
    share = totals["discriminating"]
    headline = (
        f"**{totals['earns_it']}/{totals['graded']} graded cases needed the skill**"
        if totals["graded"]
        else "**Nothing was graded both ways.**"
    )

    # Said only when it is known. A `--from-logs` rebuild runs no model, and
    # the invocation's own `--model`/`--effort` defaults describe the command
    # that read the logs rather than the runs that wrote them -- rebuilding a
    # `sonnet`/`low` run with a bare `--from-logs` printed "on `opus` (effort
    # `high`)", which is not a hedge but a false statement about a measurement.
    ran_on = (
        f" on `{meta['model']}` (effort `{meta['effort']}`)"
        if meta.get("model") and meta.get("effort")
        else ""
    )
    lines = [
        "## Skill staleness",
        "",
        f"{headline}{ran_on}. Every case "
        "ran twice: once with the skill installed, once with nothing installed.",
        "",
    ]
    if meta.get("from_logs"):
        lines += [
            f"Rebuilt offline from the logs in `{meta['from_logs']}`; nothing "
            "was run. The model and effort are whatever wrote them, which the "
            "logs do not have to agree on, so they are not claimed here.",
            "",
        ]

    # Nothing discriminated. The row `earns_it | 0` is true and says nothing,
    # and this is the one result the command exists to surface, so it is said
    # rather than left to be noticed in a table.
    if totals["graded"] and not totals["earns_it"]:
        lines += [
            "> **No case in this suite shows the skill doing anything.** Every "
            "graded case came out the same with the skill and without it. "
            "Either the skill has stopped earning its place, or its cases "
            "never tested what it knows -- read the skill and ask whether any "
            "case here would need it. Those are different problems with "
            "different fixes.",
            "",
        ]

    lines += [
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

    dirty = [c for c in summary["cases"] if c["baseline_contaminated"]]
    if dirty:
        voided = [c for c in dirty if c["verdict"] == "ungraded"]
        stood = [c for c in dirty if c["verdict"] != "ungraded"]
        reasons = sorted({c["baseline_contaminated"] for c in dirty})
        note = [
            "",
            f"> **{len(dirty)} case(s) had a baseline that was not empty**: "
            + "; ".join(reasons)
            + ".",
        ]
        if voided:
            note.append(
                f"> {len(voided)} of them are reported `ungraded`: a baseline "
                "that reached another skill, or read this skill's own files, "
                "cannot show what an agent manages unaided.",
            )
        if stood:
            note.append(
                f"> {len(stood)} still stand -- "
                + ", ".join(f"`{c['id']}` ({c['verdict']})" for c in stood)
                + " -- because they rest on the baseline *failing*, and it "
                "failed while holding more than it should have.",
            )
        lines += note

    unfired = totals["earned_without_firing"]
    if unfired:
        lines += [
            "",
            f"> **{unfired} of the {totals['earns_it']} `earns_it` case(s) never "
            "activated the skill.** Installing a skill does not make an agent "
            "use it, so for those the outcome differed without the skill being "
            "shown to be the reason. Read them as a lead rather than a result.",
        ]

    if totals["guard_checks"]:
        broken = totals["guards_broken_with_skill"]
        lines += [
            "",
            f"{totals['guard_checks']} guard rail(s) checked and kept out of "
            "the verdict: they hold whether or not the skill is installed, "
            "which is what they are for. "
            + (
                f"**{broken} case(s) broke one with the skill installed** -- "
                "that is the skill doing the forbidden thing."
                if broken
                else "None was broken with the skill installed."
            ),
        ]

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

    # Every claim that came out differently on the two sides -- whatever the
    # verdict. This used to skip `earns_it` cases, on the reasoning that a
    # skill which earned its keep needs no explaining, which had it exactly
    # backwards: those are the only cases where the evidence lives, and the
    # section printed "nothing to show" on a run whose whole point was that
    # three cases moved.
    detailed = [c for c in summary["cases"] if _reasons(c)]
    lines += ["", "### What the skill changed", ""]
    if not detailed:
        lines.append(
            "Nothing: every claim came out the same with the skill and without it."
        )
    for case in detailed:
        lines += [
            f"**`{case['id']}`** ({case['verdict']})",
            "",
            "| Expectation | With the skill | Without it |",
            "| --- | --- | --- |",
        ]
        for text, here, there in _reasons(case):
            lines.append(
                f"| {text[:70].replace('|', chr(92) + '|')} "
                f"| {here[:90]} | {there[:90]} |"
            )
        lines.append("")
    return "\n".join(lines) + "\n"
