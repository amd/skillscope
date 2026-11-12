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

Three things a case can turn out to be, and an error state:

``earns_it``
    Passes with the skill, fails without it. The case discriminates, which is
    the only outcome that tells you the skill did something. Whether the agent
    actually opened the skill is reported beside it, because installing a
    skill does not make an agent use one and a case that moved without it
    firing moved for a reason nobody has identified.
``harmful``
    Fails with the skill, passes without it. Rare, and worth stopping for: the
    skill is actively steering the agent wrong.
``no_signal``
    Came out the same both ways. Two findings sharing a name, and the pass
    counts beside the verdict say which you are holding. Both sides passing is
    a case any agent satisfies, which is not evidence about the skill. Both
    sides failing is a broken case or a genuinely hard one, which is not
    evidence about the skill either, and points somewhere else entirely.
``ungraded``
    One side or both could not be graded: a side that errored throughout, a
    grader that never answered, or a baseline that turned out not to be the
    empty room it claimed to be.

Three rather than more. Earlier versions split out whether the skill fired,
whether a case asserted anything positive at all, and whether a shared failure
was a broken case or a hard one. Each of those is real and each is still
reported -- as counts and notes beside the verdict, where the evidence for
them already was. None of them was a different verdict: each was one of these
verdicts with a caveat attached, and a caveat reads better as a sentence than
as a row in a table nobody can hold seven of.

Each side can be run more than once (``--attempts``). A side counts as having
passed when any of its attempts met every positive claim, which at one attempt
each is exactly the rule above. "Any" rather than "all" because the question a
side answers is what the agent *can* do: an agent that managed the case once
managed it, and a side that failed the other two times has still shown the
task to be within reach of whatever it was holding.

That makes more attempts a higher bar rather than a steadier reading of the
same one. ``earns_it`` at three attempts needs the baseline to have failed
three times out of three, which is a stronger claim than failing once; and a
baseline that passes once in three takes the case to ``no_signal`` where one
attempt each would have called it ``earns_it``. So a verdict is comparable
with another taken at the same attempt count and not with one taken at a
different count, and raising ``--attempts`` is asking a harder question rather
than asking the same one more carefully.

The counts are carried into the report either way, because `3/3` against `0/3`
and `1/1` against `0/1` are the same verdict and not the same evidence, and
`3/3` against `1/3` is a third thing again.

No verdict here gates, though an expired deadline or a refused run still exits
non-zero: those are statements about the run rather than about a skill. A
nondeterministic agent disagrees with itself between runs often enough that a
single pair is evidence rather than proof -- the same reason
`tools/benchmark_engines.py` measures engine agreement against how much one
engine already disagrees with itself, and the same reason the `claude-code` leg
is documented as reporting rather than gating. A number here is a question to
go and look into, not a build failure. Repeats are what let a reader weigh one;
they are not a licence for this command to start failing builds.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

from .behavior import BehaviorOutcome

# (key, what it means) -- one row per verdict, in the order a reader should
# care about them.
VERDICTS: tuple[tuple[str, str], ...] = (
    ("earns_it", "passed with the skill, failed without it"),
    ("harmful", "failed with the skill, passed without it"),
    ("no_signal", "came out the same both ways"),
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
    """One case, run both ways, each way once or several times.

    No `diverged` flag is computed from the two sides' checks. The details are
    judge prose, and two runs can word the same behaviour differently; a
    boolean over that would claim a difference the harness cannot stand
    behind. They are shown instead, for a reader to judge.
    """

    id: str
    skill: str
    verdict: str
    # Attempts that produced a grade, and how many of them met every positive
    # claim. One attempt each is the default; `passes_with` is then 0 or 1 and
    # the verdict rules read exactly as they did before repeats existed.
    attempts_with: int = 0
    attempts_without: int = 0
    passes_with: int = 0
    passes_without: int = 0
    # Attempts that errored instead. Counted apart from the graded ones so a
    # side that half ran is not reported as a side that half failed: the
    # denominator a reader sees is the number of attempts that answered.
    errored_with: int = 0
    errored_without: int = 0
    error: str | None = None
    # The checks of the attempt the verdict rests on -- one that passed, where
    # the side has one, and otherwise the first. Both sides are carried rather
    # than reduced to the verdict. A case that failed both ways can still have
    # failed them *differently* -- the first run of this command had an agent
    # that tried the task and was stopped by the machine on one side, and one
    # that never attempted it on the other, which is the skill working. The
    # verdict could not see it and neither could the JSON, so the only way to
    # find out was to open the raw inspect logs, which is not a report.
    checks_with: list[dict] = field(default_factory=list)
    checks_without: list[dict] = field(default_factory=list)
    # Every guard rail the installed side was held to, one row per guard,
    # failed if *any* attempt broke it.
    #
    # Kept apart from `checks_with` because that is one attempt's evidence --
    # `_decisive` picks it on positive claims and firing, knowing nothing
    # about guards -- and a guard rail is a claim about the whole side. With
    # `--attempts` above one, reading guards off the decisive attempt meant a
    # skill that ran the forbidden thing on an attempt that was not chosen
    # was reported as having broken nothing at all.
    guards_with: list[dict] = field(default_factory=list)
    # The same, for the baseline. Carried because a guard the *baseline*
    # broke and the installed side held is the skill preventing harm, which
    # is a kind of value the pass/fail comparison cannot see at all: both
    # sides can reach the same answer and only one of them wreck something on
    # the way. The first real run of this command produced one -- an unaided
    # agent attempting a sudo write to `/opt/rocm/share` on a case the skill
    # kept read-only -- and the report credited the skill with nothing.
    guards_without: list[dict] = field(default_factory=list)
    # Whether the agent actually reached for the skill on an attempt the
    # installed side passed. See `_fired_on_pass`: installing a skill does not
    # make an agent use one, so an `earns_it` where this is False is a case
    # that moved for a reason nobody has identified. A boolean rather than the
    # skill's name, because the name was only ever this skill's own -- the
    # room check reports either `None` or exactly the skill under test -- so
    # it carried nothing the row did not already say.
    fired_with: bool = False
    # Why the baseline was not the empty room it claims to be, if it was not.
    # Reported whenever it happened, on any attempt.
    baseline_contaminated: str | None = None
    # Whether that happened on a baseline attempt that *passed*. This is the
    # one that voids a verdict, and the distinction only has teeth once a side
    # can be attempted more than once -- a baseline that was dirty on an
    # attempt it failed cannot have reached a conclusion it never reached.
    baseline_dirty_on_pass: bool = False
    # Wall seconds each side spent on this case, over the attempts that
    # answered. `None` where the side has no honest number -- see `_seconds`,
    # where the distinction between "no time" and "no measurement" is the
    # difference between reporting the finding and inverting it.
    seconds_with: float | None = None
    seconds_without: float | None = None


def positive(checks: list[dict]) -> list[dict]:
    """The checks that claim the skill will do something."""
    return [c for c in checks if c.get("kind") in POSITIVE_KINDS]


def guards(checks: list[dict]) -> list[dict]:
    """The checks that forbid something."""
    return [c for c in checks if c.get("kind") == GUARD_KIND]


def _met(outcome: BehaviorOutcome) -> bool:
    """Whether this attempt met every positive claim on the case."""
    claims = positive(outcome.checks)
    return bool(claims) and all(c["passed"] for c in claims)


def answered(attempts: list[BehaviorOutcome]) -> list[BehaviorOutcome]:
    """The attempts that answered, as opposed to the ones that errored.

    An attempt that errored has measured nothing. Keeping it in the
    denominator would report a provider that dropped the connection as a side
    that failed, which is the most flattering possible reading of an error on
    the graded side and the least flattering on the baseline.

    A grader that could not reach a verdict sets `error` too, from
    `engine/behavioral._outcomes`. That is the same category and the one most
    likely to be mistaken for a result: a sandbox that would not list, or a
    judge that answered in prose, used to arrive here as a check that simply
    failed, which manufactured `earns_it` out of a broken baseline and
    `harmful` out of a broken graded run.
    """
    return [a for a in attempts if not a.error]


def _seconds(attempts: list[BehaviorOutcome]) -> float | None:
    """Wall seconds this side spent on the case, over the attempts that answered.

    Summed rather than averaged or taken from the decisive attempt. The figure
    has to reconcile with the per-half total in the table above it, and a mean
    of medians adds up to nothing a reader can check. `_decisive` would be
    worse still: it prefers an attempt that *passed*, which selects on exactly
    the dimension that correlates with speed, so a baseline's one lucky fast
    pass would be reported as its time while three slow failures vanished. At
    one attempt a side -- the default -- the sum is that attempt's time.

    The denominator is already on the row: `_passes` prints `3/3 with, 0/3
    without` in the next column, so a reader seeing `7m12s` beside `3/3` reads
    it as three attempts' worth without being told.

    `None`, not `0.0`, where no attempt produced a number. This is the one
    column where reporting nothing as zero inverts the finding: the case this
    command exists for is a baseline that ran for half an hour and never
    finished, and `0s` beside the graded half's `2m23s` reads as the baseline
    being instant. Zero is discarded for the same reason -- it is what a
    synthesised failure carries for a case inspect never ran, and an agent run
    is never genuinely instantaneous.
    """
    timed = [a.elapsed_s for a in answered(attempts) if a.elapsed_s]
    return round(sum(timed), 1) if timed else None


def _decisive(attempts: list[BehaviorOutcome]) -> BehaviorOutcome | None:
    """The attempt a side's verdict rests on: one that passed, else the first.

    A side is credited with passing when *any* of its attempts did, so the
    passing attempt is the one whose evidence belongs in the report. Where
    nothing passed, every attempt is equally the answer and the first will do.

    Among passing attempts, one that reached for the skill is preferred over
    one that did not. The verdict no longer turns on firing, but the note
    beside it does, and the failure the preference prevents is unchanged:
    `_fired_on_pass` asks whether the skill fired on any attempt that passed,
    and this is what keeps `fired_with` and `checks_with` pointing at the same
    attempt as that answer. A report saying a case earned its place while
    showing the checks of an attempt where nothing fired would be describing a
    different run from the one it graded.

    A side where every attempt errored falls back to the first of those. Its
    verdict is `ungraded` whatever this returns, and an errored attempt can
    still carry the checks it had graded before it stopped -- which is the
    difference between a report that shows how far the side got and one that
    shows a blank.
    """
    scored = answered(attempts)
    for attempt in scored:
        if _met(attempt) and attempt.activated:
            return attempt
    for attempt in scored:
        if _met(attempt):
            return attempt
    if scored:
        return scored[0]
    return attempts[0] if attempts else None


def _worst_guards(attempts: list[BehaviorOutcome]) -> list[dict]:
    """One row per guard rail this side was held to, at its worst showing.

    A guard rail that held twice and broke once broke: it exists to catch the
    skill doing the forbidden thing, and it caught it. So the rows are keyed
    on the expectation and a failure replaces a pass, which also keeps the
    count at one per guard however many attempts ran.

    Errored attempts are left out, as everywhere else: an attempt that
    measured nothing cannot have seen a guard hold either.
    """
    worst: dict[str, dict] = {}
    for attempt in answered(attempts):
        for guard in guards(attempt.checks):
            seen = worst.get(guard["expectation"])
            if seen is None or (seen["passed"] and not guard["passed"]):
                worst[guard["expectation"]] = guard
    return list(worst.values())


def _harm_prevented(case: dict) -> list[dict]:
    """Guards the baseline broke that the installed side held.

    The skill stopped the agent doing something the case forbids. Asked per
    expectation rather than per case, because a side can hold one guard and
    break another, and it is the specific forbidden thing that a reader needs
    named.

    A guard missing from the baseline is not counted. The baseline has to have
    been *held to* the guard and failed it; a case the baseline never reached
    tells us nothing about what it would have done.
    """
    broke = {
        g["expectation"]: g for g in case["guards_without"] if not g["passed"]
    }
    return [
        broke[g["expectation"]]
        for g in case["guards_with"]
        if g["passed"] and g["expectation"] in broke
    ]


def classify(
    with_skill: list[BehaviorOutcome], without: list[BehaviorOutcome]
) -> str:
    """Which of `VERDICTS` this case's two sides are.

    Three verdicts and an error state, because three is what a pair of runs
    can establish: the two sides differed in the skill's favour, they differed
    against it, or they did not differ.

    Everything the longer taxonomy split out of those is still reported, as
    counts and notes beside the verdict rather than as verdicts of their own.
    Whether the skill fired is `fired_with`, and a case that moved without it
    firing is flagged where it is listed. Whether a case asserts anything the
    skill could be credited for is visible in its claims. Whether a shared
    failure is a broken case or a hard one is the pass counts, which are
    printed next to every row. None of those was ever a different *verdict*:
    each was this verdict with a caveat attached, and a caveat reads better as
    a sentence than as a row in a table nobody can hold seven of.

    Both arguments are a side's attempts: one each is the ordinary run, and
    `--attempts` makes them longer. A side passed when any attempt met every
    positive claim -- see `GUARD_KIND` for why the forbidden ones are kept out
    of it, and the module docstring for why "any" rather than "all".

    A side with nothing left to grade is `ungraded` rather than a failure. One
    attempt that errored out of three is not that case: the other two answered.
    """
    here_scored, there_scored = answered(with_skill), answered(without)
    if not here_scored or not there_scored:
        return "ungraded"
    here = any(_met(a) for a in here_scored)
    there = any(_met(a) for a in there_scored)
    if here and not there:
        return _voided_if_dirty("earns_it", without)
    if there and not here:
        return _voided_if_dirty("harmful", without)
    return _voided_if_dirty("no_signal", without)


def _fired_on_pass(attempts: list[BehaviorOutcome]) -> bool:
    """Whether the skill fired on an attempt the installed side passed.

    Installing a skill does not make an agent use one, so a case that moved
    without the skill firing has moved for some reason nobody has identified.
    That is no longer a verdict of its own, but it is still the difference
    between a case that credits the skill and one that merely coincides with
    it, so it is carried on the comparison and flagged in the report.

    Asked over the whole side rather than over one attempt. Reading
    `activated` off the attempt `_decisive` picked made the answer turn on
    which of two passing attempts ran first: the same pair of runs in the
    other order credited the skill, and the order two attempts happened to run
    in is not a property of the skill.
    """
    return any(_met(a) and a.activated for a in answered(attempts))


# Verdicts a contaminated baseline cannot support, and why it cannot.
#
# Contamination is not symmetric. `harmful` rests on the baseline having
# *managed* something, and a baseline that reached another skill or read this
# skill's own files did not manage it unaided, which is the whole of what the
# verdict claims.
#
# `earns_it` rests on the baseline having *failed*. A baseline that failed
# while holding more than it should have fails all the harder without it, so
# the finding survives its own contamination and voiding it would throw away a
# conclusion the run is entitled to.
#
# `no_signal` is now two findings under one name, and the rule covers both
# without a branch. Both sides passing rests on the baseline having passed, so
# contamination voids it. Both sides failing rests on a baseline that passed
# nothing -- and `dirty_on_pass` only fires on an attempt that passed, so it
# is False there and the verdict stands. Merging the old `fails_either_way`
# and `guards_only` in here therefore voids nothing it should not.
VOIDED_BY_CONTAMINATION = ("no_signal", "harmful")


def dirty_on_pass(without: list[BehaviorOutcome]) -> bool:
    """Whether a baseline attempt that *passed* was contaminated.

    Contamination on an attempt the baseline failed voids nothing: the verdicts
    it voids are the ones claiming the baseline managed something, and an
    attempt that did not manage it cannot be the attempt they rest on. With one
    attempt a side this is the same rule as before -- `no_signal` and `harmful`
    both require the baseline to have passed -- and it only parts company with
    it once a side can pass on one attempt and be dirty on another.
    """
    return any(a.contaminated and _met(a) for a in answered(without))


def _voided_if_dirty(verdict: str, without: list[BehaviorOutcome]) -> str:
    if verdict in VOIDED_BY_CONTAMINATION and dirty_on_pass(without):
        return "ungraded"
    return verdict


def _by_case(outcomes: list[BehaviorOutcome]) -> dict[str, list[BehaviorOutcome]]:
    """One side's attempts, grouped by the case each answers.

    A dict of lists rather than `{o.id: o}`: with `--attempts` above one a case
    has several attempts on a side, and keying on the id alone kept whichever
    came last and silently threw the rest away.
    """
    grouped: dict[str, list[BehaviorOutcome]] = {}
    for outcome in outcomes:
        grouped.setdefault(outcome.id, []).append(outcome)
    return grouped


def _why_dirty(attempts: list[BehaviorOutcome]) -> str | None:
    """Why the baseline was not the empty room it claimed, over its attempts.

    A known reason beats `UNKNOWN_ROOM`. They are different findings -- one
    says the baseline demonstrably held something, the other that nobody could
    tell -- and a side with one attempt of each has the answer to the second
    in the first, so reporting "could not be checked" would be throwing it
    away. Errored attempts are read too: a run that stopped partway still saw
    what it saw before it stopped.
    """
    from .engine.behavioral import UNKNOWN_ROOM

    reasons = [a.contaminated for a in attempts if a.contaminated]
    known = [r for r in reasons if r != UNKNOWN_ROOM]
    return known[0] if known else (reasons[0] if reasons else None)


def _counts(attempts: list[BehaviorOutcome]) -> tuple[int, int, int]:
    """`(graded, passed, errored)` for one side of one case."""
    scored = answered(attempts)
    return (
        len(scored),
        sum(1 for a in scored if _met(a)),
        len(attempts) - len(scored),
    )


def pair(
    with_skill: list[BehaviorOutcome], without: list[BehaviorOutcome]
) -> list[Comparison]:
    """Match the two runs up by case id.

    A case missing from either side is reported as `ungraded` rather than
    dropped. The two runs are separate evals and either can lose a case, so
    quietly comparing the intersection would shrink the denominator and make
    every remaining number look more confident than it is.
    """
    with_by_id = _by_case(with_skill)
    without_by_id = _by_case(without)
    comparisons: list[Comparison] = []

    for case_id in sorted(set(with_by_id) | set(without_by_id)):
        these = with_by_id.get(case_id, [])
        those = without_by_id.get(case_id, [])
        here, there = _decisive(these), _decisive(those)
        with_n, with_passes, with_errors = _counts(these)
        without_n, without_passes, without_errors = _counts(those)
        missing = (
            "the baseline run" if not those
            else "the graded run" if not these
            else None
        )
        comparisons.append(
            Comparison(
                id=case_id,
                skill=(these or those)[0].skill,
                # `classify` answers `ungraded` for a side with no attempt
                # that answered, which a side that is absent entirely also
                # is. Left to it rather than special-cased here, so one
                # function decides what `ungraded` means. `missing` still
                # earns its keep below, where it says *which* side was gone.
                verdict=classify(these, those),
                attempts_with=with_n,
                attempts_without=without_n,
                passes_with=with_passes,
                passes_without=without_passes,
                errored_with=with_errors,
                errored_without=without_errors,
                error=(
                    f"no result from {missing}" if missing
                    else next(
                        (a.error for a in these + those if a.error), None
                    )
                ),
                guards_with=_worst_guards(these),
                guards_without=_worst_guards(those),
                checks_with=list(here.checks) if here else [],
                checks_without=list(there.checks) if there else [],
                fired_with=_fired_on_pass(these),
                baseline_contaminated=_why_dirty(those),
                baseline_dirty_on_pass=dirty_on_pass(those),
                seconds_with=_seconds(these),
                seconds_without=_seconds(those),
            )
        )
    return comparisons


def summarize(comparisons: list[Comparison], meta: dict) -> dict:
    from .engine.behavioral import UNKNOWN_ROOM

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

    # Guard rails, counted separately. A broken one is the serious finding
    # here: a skill that made the agent do the forbidden thing.
    broken = [
        c
        for c in comparisons
        if any(not g["passed"] for g in c.guards_with)
    ]
    guard_checks = sum(len(c.guards_with) for c in comparisons)
    # The other direction, and the one nothing reported. A guard the baseline
    # broke and the skill held is the skill earning its place by what it
    # stopped the agent doing, which no verdict here can express: both sides
    # can reach the same answer with only one of them wrecking something on
    # the way.
    prevented = [c for c in comparisons if _harm_prevented(asdict(c))]
    return {
        "meta": meta,
        "totals": {
            "cases": len(comparisons),
            "graded": len(graded),
            **counts,
            # Of the cases graded both ways, the share that moved when the
            # skill was taken away. Every `earns_it`, including the ones where
            # the skill never fired -- those used to be their own verdict and
            # left out of this number, and are now counted in it and flagged
            # beside it. `earns_it_unfired` is how a reader discounts it, and
            # the report says so next to the figure rather than computing a
            # second share, which would be the old split under another name.
            "discriminating": (
                round(sum(1 for c in graded if c.verdict == "earns_it") / len(graded), 3)
                if graded
                else None
            ),
            # `earns_it` cases where the skill never fired on an attempt
            # that passed. The outcome moved; the skill is not shown to be
            # what moved it.
            "earns_it_unfired": sum(
                1 for c in comparisons
                if c.verdict == "earns_it" and not c.fired_with
            ),
            "guard_checks": guard_checks,
            "harm_prevented": len(prevented),
            "guards_broken_with_skill": len(broken),
            # Counted apart, because they are different findings. One says
            # the baseline was demonstrably not empty; the other says nobody
            # could tell. Both void the same verdicts -- an unanswerable
            # question is worth no more than a failed one -- but a report that
            # totalled them together would claim contamination it never found.
            "baseline_contaminated": sum(
                1
                for c in comparisons
                if c.baseline_contaminated
                and c.baseline_contaminated != UNKNOWN_ROOM
            ),
            "baseline_unknown": sum(
                1
                for c in comparisons
                if c.baseline_contaminated == UNKNOWN_ROOM
            ),
            # Attempts that errored, over every case and both sides. A run
            # that lost a third of its attempts to a flaky provider has
            # measured a third less than its case count suggests, and nothing
            # else in these totals says so.
            "errored_attempts": sum(
                c.errored_with + c.errored_without for c in comparisons
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


def _unsettled(case: dict) -> bool:
    """Whether either side passed some of its attempts and failed others.

    It matters most for `no_signal`, whose advice is to stop treating the case
    as evidence about the skill. That is right when both sides managed it
    every time and wrong when a side managed it once in three: a baseline that
    passed once is enough to deny the case `earns_it` -- see the module
    docstring on what "any attempt passed" costs -- and nowhere near enough to
    call the case dead. Telling an author to delete it on that footing is the
    user-facing harm in the rule, and this is what separates the two.

    Always False at one attempt a side, where a side either passed or did not.
    """
    return any(
        attempts and 0 < passes < attempts
        for passes, attempts in (
            (case["passes_with"], case["attempts_with"]),
            (case["passes_without"], case["attempts_without"]),
        )
    )


def _passes(case: dict) -> str:
    """`with / without` as attempts passed out of attempts graded.

    The verdict is what the two sides did; this is how often they did it. At
    one attempt a side it reads `1/1 with, 0/1 without` and carries no more
    than the verdict already said. Above one attempt it is the whole point,
    because the verdict cannot show it: `3/3` against `0/3` is `earns_it` on
    a baseline that never managed it, `3/3` against `1/3` is `no_signal` on a
    baseline that managed it once, and `2/3` against `1/3` is `no_signal` on
    two sides that both disagreed with themselves. Only the first is a result,
    and only these counts say which one a reader is holding.
    """
    lost = case["errored_with"] + case["errored_without"]
    tail = f" ({lost} attempt(s) errored)" if lost else ""
    return (
        f"{case['passes_with']}/{case['attempts_with']} with, "
        f"{case['passes_without']}/{case['attempts_without']} without{tail}"
    )


def _seconds_cell(case: dict, side: str) -> str:
    """One side's time for one case: a figure, or why there is not one.

    Three states and three strings, because the distinction is the whole
    point of the column. A side that errored throughout has no time to show
    and did not take zero seconds trying -- it is the half-hour baseline that
    never concluded, which is the finding rather than a gap in it.
    """
    seconds = case[f"seconds_{side}"]
    if seconds is not None:
        return _duration(seconds)
    if case[f"errored_{side}"]:
        return "did not finish"
    return "not run"


def _advice(case: dict) -> str:
    """What to do about one case, where the verdict alone does not say.

    This is where the collapsed taxonomy is paid for. `no_signal` now covers
    three situations that want three different actions, and the single
    sentence it used to carry -- "the agent already did this unaided" -- is
    false for two of them. A case that failed on both sides was never managed
    by anybody, and telling its author the agent managed it unaided sends
    them to delete a case that is merely broken.

    Driven off the pass counts, which `_passes` is already printing in the
    column alongside, so a reader can check the advice against the evidence
    for it rather than taking it on trust.
    """
    if case["verdict"] == "harmful":
        return "the skill steered the agent wrong here"
    if case["verdict"] == "earns_it":
        return (
            "something moved, but the agent never opened the skill -- read "
            "the run before crediting it"
        )
    if not positive(case["checks_with"] + case["checks_without"]):
        # The old `guards_only` verdict, as advice. Asked before the pass
        # counts, because a case claiming nothing positive cannot pass and
        # would otherwise be told it is broken or hard -- which would send its
        # author to fix a case that is working exactly as written.
        return (
            "this case claims nothing the skill could be credited for; its "
            "guard rails were still checked"
        )
    if not case["passes_with"] and not case["passes_without"]:
        return (
            "neither side managed it; a broken case or a hard one, and not "
            "yet evidence about the skill"
        )
    if _unsettled(case):
        return (
            "the two sides did not settle; run it again before reading "
            "anything into it"
        )
    return "the agent already did this unaided; the case is not evidence about the skill"


def _cell(value) -> str:
    """A figure, or `n/a` where there is none. Never a zero standing in for one.

    Same spelling as `tools/benchmark_engines.py`, which reports the same
    columns for the same reason.
    """
    return "n/a" if value is None else str(value)


def _duration(seconds: float | None) -> str:
    """`2m23s`, because `143.2s` beside `1680.4s` does not compare at a glance."""
    if seconds is None:
        return "n/a"
    minutes, rest = divmod(int(round(seconds)), 60)
    return f"{minutes}m{rest:02d}s" if minutes else f"{rest}s"


def _ratio(halves: list[dict]) -> str | None:
    """How much longer the baseline took, where both halves have a number."""
    graded, baseline = halves[0]["wall_time_s"], halves[1]["wall_time_s"]
    if not graded or not baseline:
        return None
    return f"{baseline / graded:.1f}x"


def _spend_lead(halves: list[dict]) -> list[str]:
    """The sentence this command exists to print, above everything else.

    A skill's worth is not only whether the agent gets there. An agent holding
    the skill goes straight to the work and an agent without it explores, and
    the gap between those is a result in its own right -- one the verdicts
    cannot express, because both halves can pass and the case still be worth
    keeping. The first run of this command had a graded half finish in 2m23s
    against a baseline still going half an hour later, and the report said
    nothing about it.

    Three shapes, because the honest sentence differs. A baseline that was
    stopped is a floor and is said as one. A graded half slower than its
    baseline is a finding and is said plainly rather than suppressed: a skill
    that costs the agent time is something its author wants to know.
    """
    graded, baseline = halves
    if not baseline["finished"]:
        return [
            f"> **The baseline never finished.** The graded half took "
            f"{_duration(graded['wall_time_s'])}; the baseline was still "
            f"going {_duration(baseline['wall_time_s'])} in, when the bound "
            "stopped it. That gap is itself a result: whatever these cases "
            "ask for, an agent without the skill reaches it only by "
            "exploring, if at all.",
            "",
        ]
    times = _ratio(halves)
    if times is None:
        return []
    if baseline["wall_time_s"] >= graded["wall_time_s"]:
        return [
            f"> **The baseline took {times} the graded half's wall time** -- "
            f"{_duration(graded['wall_time_s'])} with the skill, "
            f"{_duration(baseline['wall_time_s'])} without it.",
            "",
        ]
    return [
        "> **The graded half was the slower of the two** -- "
        f"{_duration(graded['wall_time_s'])} with the skill, "
        f"{_duration(baseline['wall_time_s'])} without it. The skill is "
        "costing the agent time; whether it buys anything for that is the "
        "verdict table below.",
        "",
    ]


def _spend_caveats(halves: list[dict], meta: dict) -> list[str]:
    """Which of these columns compare, and which do not.

    The cost one is not decoration. Dollars arrive only through the host
    leg's final event, which a half the bound killed never reaches -- so the
    half that spent *more* is the likelier of the two to report nothing, and
    a reader who takes a blank for a zero reads the result exactly backwards.
    """
    notes = [
        "> Both halves ran the same cases on the same model, one after the "
        "other, under one `--timeout`: the graded half goes first, so an "
        "overrun there eats the baseline's budget. Wall time is the clock, "
        "and includes waiting for retries and rate limits."
    ]
    if any("cost_usd" not in half for half in halves):
        notes.append(
            "> **A missing cost is a missing number, not a zero.** The "
            "inspect-backed engines report one only where the provider "
            "supplies pricing, which a gateway generally does not and which "
            "inspect's own registry does not supply for any model; "
            "`SKILLSCOPE_MODEL_PRICING` is the way in. Worse for this "
            "report: dollars arrive with a run's final event, which a half "
            "the bound stopped never reaches -- so the half that spent more "
            "is the likelier one to show `n/a`. Tokens and model calls are "
            "the like-for-like columns."
        )
    if meta.get("concurrency"):
        notes.append(
            f"> Up to {meta['concurrency']} cases ran at once on each half, "
            "so wall time measures how long the run took rather than how "
            "much agent time it bought. Model calls and tokens are unaffected."
        )
    if any(not half["finished"] for half in halves):
        stopped = ", ".join(h["label"] for h in halves if not h["finished"])
        notes.append(
            f"> The `{stopped}` row is a floor: the bound stopped it, so it "
            "spent at least this and would have spent more."
        )
    return notes


def _spend_table(halves: list[dict]) -> list[str]:
    naming = {"graded": "with the skill", "baseline": "nothing installed"}
    rows = [
        "| Half | Wall time | Model calls | Tokens | Cost |",
        "| --- | --- | --- | --- | --- |",
    ]
    for half in halves:
        floor = "" if half["finished"] else "+"
        rows.append(
            f"| {naming.get(half['label'], half['label'])} "
            f"| {_duration(half['wall_time_s'])}{floor} "
            f"| {_cell(half.get('model_calls'))} "
            f"| {_cell(half.get('total_tokens'))} "
            f"| {_cell(half.get('cost_usd'))} |"
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
        else "**Nothing was graded both ways**"
    )

    # Still conditional rather than assumed: `summarize` is given whatever
    # meta its caller built, and a report that names a model it was not told
    # about states a measurement that was never taken.
    ran_on = (
        f" on `{meta['model']}` (effort `{meta['effort']}`)"
        if meta.get("model") and meta.get("effort")
        else ""
    )
    # How many times each side was attempted, said plainly: `3/3` against
    # `0/3` and `1/1` against `0/1` are the same verdict and not the same
    # evidence, and a reader who does not know which they are holding cannot
    # tell them apart. Read off the cases rather than the invocation's own
    # `--attempts`, so the report describes the run rather than the request.
    per_side = {c["attempts_with"] for c in summary["cases"]} | {
        c["attempts_without"] for c in summary["cases"]
    }
    most = max(per_side - {0}, default=1)
    # Zero is kept here where `most` discards it. A side with no attempts at
    # all is a case that ran on one side only -- `pair`'s "missing from either
    # side", or a deadline that landed mid-run -- and the sentence below used
    # to say every case ran twice straight over the top of it.
    least = min(per_side, default=1)
    sides = (
        "twice: once with the skill installed, once with nothing installed"
        if most <= 1
        else f"up to {most} times on each side, with the skill installed and "
        "with nothing installed"
    )
    ran = (
        f"Some cases ran on one side only; the rest ran {sides}."
        if least == 0
        else f"Every case ran {sides}."
    )
    lines = [
        "## Skill staleness",
        "",
        f"{headline}{ran_on}. {ran}",
        "",
    ]
    if most > 1:
        # Said where it cannot be missed, because the verdict's meaning moves
        # with the attempt count and nothing else in the report says so. "Any
        # attempt passed" makes a higher count a higher bar: see the module
        # docstring.
        lines += [
            "> **Verdicts from this run compare with other runs at "
            f"{most} attempt(s), not with runs at a different count.** A side "
            "counts as having passed when any of its attempts did, so more "
            "attempts is a harder question: `earns_it` needs the baseline to "
            "have failed every single time.",
            "",
        ]
    if meta.get("max_cases"):
        # A capped run is not a statement about the suite. Said in the one
        # place a reader cannot miss it, because a partial pass that reads as
        # a whole one is the failure a cap invites.
        lines += [
            f"> **This run covered only the first {meta['max_cases']} case(s) "
            "of each selected skill** (`--max-cases`). It says nothing about "
            "the cases it did not run.",
            "",
        ]
    # What each half spent. Placed above the verdicts because it is the
    # cheaper question and often the more useful answer: a skill whose cases
    # all pass either way can still be the difference between two minutes and
    # half an hour, and a verdict table alone reports that as "no signal".
    #
    # Absent from a summary built by something that did not run two halves,
    # so `summarize` stays usable for a caller that has only comparisons.
    halves = meta.get("legs") or []
    if len(halves) == 2:
        lines += _spend_lead(halves)

    # Nothing moved at all. The row `earns_it | 0` is true and says nothing,
    # and this is the one result the command exists to surface, so it is said
    # rather than left to be noticed in a table.
    #
    # Split on whether anything passed, because the same verdict tally means
    # two different things. A suite where both sides passed everything has
    # cases that no longer test the skill. A suite where both sides failed
    # everything has cases that do not work, and telling its author the skill
    # has stopped earning its place would point them at the wrong thing
    # entirely.
    moved = totals["earns_it"] + totals["harmful"]
    anything_passed = any(
        c["passes_with"] or c["passes_without"] for c in summary["cases"]
    )
    # A suite that claims nothing positive anywhere is a third case again,
    # and the one the old `guards_only` verdict used to catch here. Its cases
    # cannot pass, so the "nothing passed" wording below would call a working
    # suite broken; and they cannot move, so the "stopped earning its place"
    # wording would blame a skill for not affecting checks that never asked
    # anything of it.
    claimed = any(
        positive(c["checks_with"] + c["checks_without"]) for c in summary["cases"]
    )
    if totals["graded"] and not moved and not claimed:
        lines += [
            "> **Every case here only forbids things.** Their guard rails were "
            "checked, and nothing was claimed that the skill could be "
            "credited with, so this run says nothing about whether the skill "
            "is earning its place. Add a case that asserts what it should "
            "make the agent *do*.",
            "",
        ]
    elif totals["graded"] and not moved and anything_passed:
        lines += [
            "> **No case in this suite shows the skill doing anything.** Every "
            "graded case came out the same with the skill and without it. "
            "Either the skill has stopped earning its place, or its cases "
            "never tested what it knows -- read the skill and ask whether any "
            "case here would need it. Those are different problems with "
            "different fixes.",
            "",
        ]
    elif totals["graded"] and not moved:
        lines += [
            "> **No case in this suite passed on either side.** That is a "
            "statement about the cases, not about the skill: nothing here has "
            "been shown to work well enough to then ask whether the skill is "
            "why. Fix the cases before reading anything into the verdicts.",
            "",
        ]

    if len(halves) == 2:
        lines += ["### What each half cost", "", *_spend_table(halves), ""]
        lines += [*_spend_caveats(halves, meta), ""]

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

    if totals["errored_attempts"]:
        # Said out loud rather than left in the JSON. An attempt that errored
        # is one nobody paid for an answer from, and a reader comparing `2/2`
        # against `3/3` on the next case has no way to know the first was
        # asked the same question three times.
        lines += [
            "",
            f"{totals['errored_attempts']} attempt(s) errored and are out of "
            "the counts below. A case whose every attempt errored on a side "
            "is `ungraded`; one that lost some of them is graded on the rest, "
            "over the smaller denominator the counts show.",
        ]

    from .engine.behavioral import UNKNOWN_ROOM

    unknown = [
        c
        for c in summary["cases"]
        if c["baseline_contaminated"] == UNKNOWN_ROOM
    ]
    if unknown:
        lines += [
            "",
            f"> **{len(unknown)} case(s) could not be checked for a clean "
            "baseline.** The run did not record where the skill was, and it "
            "is not on this machine, so there is nothing to recognise a "
            "tool call against. They are reported `ungraded` where the "
            "verdict rested on the baseline having managed something. "
            "Re-running them records it.",
        ]

    dirty = [
        c
        for c in summary["cases"]
        if c["baseline_contaminated"] and c["baseline_contaminated"] != UNKNOWN_ROOM
    ]
    if dirty:
        voided = [c for c in dirty if c["verdict"] == "ungraded"]
        # Split three ways, each group worded for what it actually rests on.
        #
        # The third group is what `--attempts` opened up. `_why_dirty` reads
        # every attempt and `dirty_on_pass` reads only the ones that passed,
        # so a baseline that was dirty on an attempt it failed and clean on
        # the one it passed keeps its verdict. Where that verdict is one of
        # `VOIDED_BY_CONTAMINATION` it rests on the baseline having *passed*,
        # so the sentence about resting on a failure is false for it.
        #
        # Driven off `VOIDED_BY_CONTAMINATION` rather than a verdict list
        # written out here, so the two cannot drift apart.
        standing = [c for c in dirty if c["verdict"] != "ungraded"]
        stood = [
            c for c in standing
            if c["verdict"] not in VOIDED_BY_CONTAMINATION
        ]
        elsewhere = [
            c for c in standing if c["verdict"] in VOIDED_BY_CONTAMINATION
        ]
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
        if elsewhere:
            note.append(
                f"> {len(elsewhere)} case(s) rest on a clean attempt -- "
                + ", ".join(
                    f"`{c['id']}` ({c['verdict']})" for c in elsewhere
                )
                + " -- the baseline passed on an attempt that was clean and "
                "was dirty on one it failed, so the attempt the verdict rests "
                "on is not the contaminated one. They stand, on weaker "
                "evidence than a run with no contamination in it at all.",
            )
        lines += note

    unfired = totals["earns_it_unfired"]
    if unfired:
        lines += [
            "",
            f"> **{unfired} of the {totals['earns_it']} `earns_it` case(s) "
            "passed without the skill ever firing.** Installing a skill does "
            "not make an agent use it, so for those the outcome differed "
            "without the skill being shown to be the reason. They are counted "
            "in the figure above and flagged here: read the run before "
            "crediting the skill with them.",
        ]

    saved = [
        (c["id"], g) for c in summary["cases"] for g in _harm_prevented(c)
    ]
    if saved:
        # Said before the guard tally below, which only ever reported guards
        # the *skill* broke. A reader who saw "none was broken with the skill
        # installed" learned that the skill did no harm, and never that it
        # stopped any.
        lines += [
            "",
            f"> **The skill stopped the agent breaking {len(saved)} guard "
            "rail(s) that it broke without it**: "
            + "; ".join(
                f"`{case_id}` -- {g['expectation'][:80]}" for case_id, g in saved
            )
            + ". Both sides may still reach the same answer; this is what the "
            "skill prevented on the way, which no verdict in the table above "
            "can express.",
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

    unsettled = [
        c for c in summary["cases"]
        if c["verdict"] == "no_signal" and _unsettled(c)
    ]
    if unsettled:
        lines += [
            "",
            f"> **{len(unsettled)} case(s) are `no_signal` because a side "
            "disagreed with itself**, not because both sides managed it every "
            "time: "
            + ", ".join(f"`{c['id']}`" for c in unsettled)
            + ". One baseline attempt in three is enough to stop a case "
            "being `earns_it` and is not enough to call it dead. Run them "
            "again before acting on them.",
        ]

    # Former `fails_either_way` cases land here now, because they are
    # `no_signal`. That grows the section on a suite with many hard cases,
    # which is the right trade: the advice column tells them apart, and this
    # is "worth a look" rather than "failures".
    notable = [
        c for c in summary["cases"]
        if c["verdict"] in ("no_signal", "harmful")
        or (c["verdict"] == "earns_it" and not c["fired_with"])
    ]
    lines += ["", "### Cases worth a look", ""]
    if not totals["graded"] and not totals["guard_checks"]:
        # Not the same sentence as "none of them was worth a look". A run with
        # nothing graded has no graded cases to make either claim about, and
        # saying every one of them came out well is the same bug commit
        # `542cad9` fixed one aisle over.
        lines.append(
            "No case was graded both ways, so there is nothing to look at here."
        )
    elif not totals["graded"]:
        # A case whose guards were checked is not nothing, even where no
        # verdict was reached. "Nothing to look at here", printed two
        # paragraphs under "that is the skill doing the forbidden thing",
        # is the strongest finding this command can produce being
        # contradicted by the weakest sentence in it. This section is about
        # the verdict table, so it says that instead.
        lines.append(
            "No case reached a verdict. The guard rails above are what this "
            "run found."
        )
    elif not notable:
        lines.append(
            "None. Every graded case needed the skill, and reached for it."
        )
    else:
        lines += [
            "| Case | Verdict | Passed | Time (with / without) | What to do |",
            "| --- | --- | --- | --- | --- |",
        ]
        for case in notable:
            lines.append(
                f"| `{case['id']}` | {case['verdict']} | {_passes(case)} "
                f"| {_seconds_cell(case, 'with')} / {_seconds_cell(case, 'without')} "
                f"| {_advice(case)} |"
            )

    # Every claim that came out differently on the two sides -- whatever the
    # verdict. This used to skip `earns_it` cases, on the reasoning that a
    # skill which earned its keep needs no explaining, which had it exactly
    # backwards: those are the only cases where the evidence lives, and the
    # section printed "nothing to show" on a run whose whole point was that
    # three cases moved.
    detailed = [c for c in summary["cases"] if _reasons(c)]
    lines += ["", "### What the skill changed", ""]
    if not detailed:
        # Two findings that used to share one sentence. "Every claim matched"
        # is a statement about the skill; "no case was graded on both sides"
        # is a statement about the run, and printing the first for the second
        # tells a reader the run concluded something it never reached.
        compared = any(
            c["checks_with"] and c["checks_without"] for c in summary["cases"]
        )
        lines.append(
            "Nothing: every claim came out the same with the skill and "
            "without it."
            if compared
            else "Nothing was compared: no case was graded on both sides."
        )
    for case in detailed:
        lines += [
            f"**`{case['id']}`** ({case['verdict']}) -- {_passes(case)}; "
            f"{_seconds_cell(case, 'with')} with, "
            f"{_seconds_cell(case, 'without')} without",
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
