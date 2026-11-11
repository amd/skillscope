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

Five things a case can turn out to be:

``earns_it``
    Passes with the skill, fails without it, and the agent reached for the
    skill. The case discriminates, which is the only outcome that tells you
    the skill did something.
``unattributed``
    Passes with the skill and fails without it, but the agent never opened the
    skill on an attempt that passed. Something moved; the skill is not shown
    to be what moved it. Installing a skill does not make an agent use one,
    and a difference that arrived without it is not evidence about it.
``no_signal``
    Passes either way. Says nothing about the skill, and usually says something
    about the case: an expectation any agent meets is not evidence.
``harmful``
    Fails with the skill, passes without it. Rare, and worth stopping for: the
    skill is actively steering the agent wrong.
``fails_either_way``
    Fails both times. A broken case or a genuinely hard one; either way the
    comparison has nothing to report until it passes somewhere.

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

from dataclasses import asdict, dataclass, field, replace

from .behavior import BehaviorOutcome

# (key, what it means) -- one row per verdict, in the order a reader should
# care about them.
VERDICTS: tuple[tuple[str, str], ...] = (
    ("earns_it", "passed with the skill, failed without it, and the skill fired"),
    ("harmful", "failed with the skill, passed without it"),
    ("unattributed", "passed only with the skill, but the skill never fired"),
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
    """One case, run both ways, each way once or several times.

    No `diverged` flag is computed from the two sides' checks. The details are
    judge prose, and two runs can word the same behaviour differently; a
    boolean over that would claim a difference the harness cannot stand
    behind. They are shown instead, for a reader to judge.
    """

    id: str
    skill: str
    verdict: str
    # Which staleness run produced this comparison. Empty on a live run,
    # where there is only one and the report is about it. A rebuild sets it,
    # because a log directory holds every run anyone has ever done in it and
    # a report that does not say which one it read is not a measurement of
    # anything in particular. See `from_logs`.
    run: str = ""
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
    # How much of the case each side satisfied, averaged over its graded
    # attempts. The verdict is binary and these are not: a side passes only by
    # meeting every positive claim, so `fails_either_way` covers a baseline
    # that met four claims of five and one that met none, and `earns_it`
    # covers a baseline that nearly managed it and one that never started.
    # See `_claims`, which is where a reader meets them.
    claims: int = 0
    claims_met_with: float = 0.0
    claims_met_without: float = 0.0
    # Whether the agent actually reached for the skill on the installed side,
    # read off the attempt the verdict rests on -- which `_decisive` picks to
    # be a passing attempt that fired, where the side has one. The verdict
    # itself is decided over every passing attempt and not over this one; a
    # case that would otherwise be `earns_it` and fired on no attempt that
    # passed is `unattributed` instead. See `classify`.
    activated_with: str | None = None
    # Why the baseline was not the empty room it claims to be, if it was not.
    # Reported whenever it happened, on any attempt.
    baseline_contaminated: str | None = None
    # Whether that happened on a baseline attempt that *passed*. This is the
    # one that voids a verdict, and the distinction only has teeth once a side
    # can be attempted more than once -- a baseline that was dirty on an
    # attempt it failed cannot have reached a conclusion it never reached.
    baseline_dirty_on_pass: bool = False


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


def _decisive(attempts: list[BehaviorOutcome]) -> BehaviorOutcome | None:
    """The attempt a side's verdict rests on: one that passed, else the first.

    A side is credited with passing when *any* of its attempts did, so the
    passing attempt is the one whose evidence belongs in the report. Where
    nothing passed, every attempt is equally the answer and the first will do.

    Among passing attempts, one that reached for the skill is preferred over
    one that did not. `classify` asks whether the skill fired on any attempt
    that passed, and this is what keeps `activated_with` and `checks_with`
    pointing at the same attempt as that answer: a report that said `earns_it`
    while showing the checks of an attempt where nothing fired would be
    describing a different run from the one it graded.

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


def _claims_met(attempts: list[BehaviorOutcome]) -> float:
    """Mean positive claims met per graded attempt."""
    scored = answered(attempts)
    if not scored:
        return 0.0
    met = sum(sum(1 for c in positive(a.checks) if c["passed"]) for a in scored)
    return round(met / len(scored), 2)


def classify(
    with_skill: list[BehaviorOutcome], without: list[BehaviorOutcome]
) -> str:
    """Which of `VERDICTS` this case's two sides are.

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
    if not any(positive(a.checks) for a in here_scored + there_scored):
        # Nothing was claimed, so nothing can be credited. Reporting this as
        # `fails_either_way` would read as a broken case rather than as one
        # that only ever pinned down what must not happen.
        return "guards_only"
    here = any(_met(a) for a in here_scored)
    there = any(_met(a) for a in there_scored)
    if here and not there:
        # Installed is not used. The outcome differed, and the thing under
        # test is not shown to be why -- so it is reported as its own finding
        # rather than counted among the cases that needed the skill.
        #
        # Asked over the whole side rather than over one attempt. Reading
        # `activated` off the attempt `_decisive` picked made the verdict turn
        # on which of two passing attempts ran first: the same pair of runs in
        # the other order credited the skill, and the order two attempts
        # happened to run in is not a property of the skill. The module
        # docstring's definition is "never opened the skill on an attempt that
        # passed", and this is that sentence.
        fired = any(_met(a) and a.activated for a in here_scored)
        return _voided_if_dirty("earns_it" if fired else "unattributed", without)
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
# `earns_it`, `unattributed` and `fails_either_way` rest on the baseline
# having *failed*. A baseline that failed while holding more than it should
# have fails all the harder without it, so the finding survives its own
# contamination and voiding it would throw away a conclusion the run is
# entitled to.
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
        # The claims a case makes are the case's, not a side's, so either side
        # can name them. The baseline is asked only when the graded attempt
        # produced none -- a side that errored before it graded anything has
        # no checks to count, and a denominator of 0 would print every case it
        # lost as having claimed nothing.
        #
        # Tested on `here.checks` rather than on `here`. A `BehaviorOutcome`
        # is a dataclass and so always truthy, which made `here or there` fall
        # back only where the graded side was *absent* -- never in the case
        # the comment above describes, which is the one it was written for.
        naming = here if (here and here.checks) else (there or here)
        claims = len(positive(naming.checks)) if naming else 0
        comparisons.append(
            Comparison(
                id=case_id,
                skill=(these or those)[0].skill,
                verdict="ungraded" if missing else classify(these, those),
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
                checks_with=list(here.checks) if here else [],
                checks_without=list(there.checks) if there else [],
                claims=claims,
                claims_met_with=_claims_met(these),
                claims_met_without=_claims_met(those),
                activated_with=here.activated if here else None,
                baseline_contaminated=_why_dirty(those),
                baseline_dirty_on_pass=dirty_on_pass(those),
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
    from . import routing as routing_core
    from .engine import behavioral, scorers
    from .engine import routing as routing_engine

    eval_meta = log.eval.metadata or {}
    skill = eval_meta.get(behavioral.SKILL_KEY) or log.eval.task
    # Where the skill was when the agent ran, not where it is now. Absent from
    # logs written before the key existed, which makes the room unknown rather
    # than clean unless this machine happens to have the skill. See
    # `behavioral.SKILL_DIR_KEY`.
    skill_dir = eval_meta.get(behavioral.SKILL_DIR_KEY)
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
                # inspect's own repeat counter. A run that asked for one
                # attempt writes 1 on every sample, so this is the identity
                # on everything but an `--attempts` run.
                attempt=int(getattr(sample, "epoch", 1) or 1),
                # Exactly what the live path sets, for the same reason as the
                # error string above: a rebuild that left this False would
                # report a provider outage as a skill that failed, which is
                # the rebuild being more permissive than the run it rebuilds.
                degraded=routing_core.is_provider_error(error),
                # Both detected here too. Without the first, a rebuilt
                # comparison reports every `earns_it` as never having fired;
                # without the second it is *more permissive than the run it
                # rebuilds*, quietly restoring verdicts the live run voided.
                #
                # `skill_dir` is what keeps the second true off this machine:
                # the detector needs an absolute path to recognise, and left
                # to work one out it works out this reader's rather than the
                # run's. See `behavioral.SKILL_DIR_KEY`.
                **dict(
                    zip(
                        ("activated", "contaminated"),
                        behavioral._inspect_room(
                            routing_engine, sample, skill, skill_dir
                        ),
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


@dataclass
class Rebuild:
    """The comparisons a log directory yielded, and the runs behind them.

    `runs` holds every staleness run the directory had for each skill, newest
    first, whether or not it was used. Nothing prunes `.skillscope/logs`, so a
    directory accumulates runs and "which one is this" stops being a detail:
    the caller needs it to name the run in the report, and to say what a
    `--run-id` could have asked for instead.
    """

    comparisons: list[Comparison]
    runs: dict[str, list[str]] = field(default_factory=dict)


def from_logs(
    log_dir: str, *, unlabelled: bool = False, run_id: str | None = None
) -> Rebuild:
    """Rebuild the comparison from logs already on disk.

    Two runs become a pair because they carry the same skill, the same
    `STALENESS_RUN_KEY`, and opposite `BASELINE_KEY` -- which is why all three
    are written as metadata rather than inferred from a task name.

    Grouping on the run id is what makes this safe to point at
    `.skillscope/logs`, which is the documented thing to do with it. That
    directory is shared by `behavioral`, `verify` and `routing`, is gitignored
    wholesale, and nothing prunes it; and every ordinary `behavioral` run
    stamps `BASELINE_KEY: False`, so a run id is the only thing separating
    this run's graded half from an unrelated run of the same skill last month.

    One run per skill, the newest, unless `run_id` names another. The run id
    stops two runs being *spliced* into one comparison; it does not stop them
    being *pooled*, and pooling is what a directory of accumulated runs
    produces. Every run of a one-case skill used to add another copy of that
    case to the report, so the second staleness run anybody did turned a
    one-case suite into a two-case one and the headline counted it twice --
    under exactly the invocation the docs give for this. The runs passed over
    are returned alongside, because a report that drops evidence silently is
    the other half of the same fault.

    Newest is read off each log's own `created` stamp rather than the file's
    mtime, which says when the directory was last copied about as often as it
    says when the run happened.

    Logs with no run id are skipped unless `unlabelled` asks for them. See
    `LEGACY_RUN` for what that costs and what it is for. They compete for
    "newest" like any other run once asked for, and stay subject to the
    ambiguity guard if they win.

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
    logs_per_side: dict[tuple[str, str], dict[bool, int]] = {}
    written: dict[tuple[str, str], str] = {}
    for info in list_eval_logs(log_dir):
        log = read_eval_log(info.name)
        if behavioral.BASELINE_KEY not in (log.eval.metadata or {}):
            continue
        skill, baseline, run_key, outcomes = outcomes_from_log(log)
        key = (skill, run_key or LEGACY_RUN)
        sides.setdefault(key, {}).setdefault(baseline, []).extend(outcomes)
        tally = logs_per_side.setdefault(key, {})
        tally[baseline] = tally.get(baseline, 0) + 1
        written[key] = max(written.get(key, ""), str(log.eval.created or ""))

    candidates = [
        key
        for key in sorted(sides)
        if key[1] != LEGACY_RUN or (unlabelled and logs_per_side[key].get(True))
    ]
    # Newest first, with a labelled run always ahead of the unlabelled
    # bucket. `LEGACY_RUN` is a bucket and not a run -- it is every log that
    # carried no id -- so letting it win on a timestamp would pick a pile over
    # a measurement. The run id breaks the remaining ties, so two runs written
    # in the same second order the same way on every machine.
    runs: dict[str, list[str]] = {}
    for key in sorted(
        candidates,
        key=lambda k: (k[1] != LEGACY_RUN, written.get(k, ""), k[1]),
        reverse=True,
    ):
        runs.setdefault(key[0], []).append(key[1])
    chosen = [
        key
        for key in candidates
        if key[1] == (runs[key[0]][0] if run_id is None else run_id)
    ]

    comparisons: list[Comparison] = []
    for key in sorted(chosen):
        half = sides[key]
        counts = logs_per_side[key]
        ambiguous = key[1] == LEGACY_RUN and any(n > 1 for n in counts.values())
        comparisons.extend(
            replace(comparison, run=key[1])
            for comparison in pair(
                _no_duplicates(half.get(False, []), ambiguous),
                _no_duplicates(half.get(True, []), ambiguous),
            )
        )
    return Rebuild(comparisons=comparisons, runs=runs)


# The bucket for logs written before a run id was stamped on them.
#
# Skipped by default, and taken only when the caller asks with
# `--unlabelled-logs`. Both halves of that rule are load-bearing.
#
# Taking them by default was wrong, and not only for old directories. A plain
# `skillscope behavioral` run stamps `BASELINE_KEY: False` and never stamps a
# run id, because only `cmd_staleness` mints one -- so an ordinary behavioral
# log is shaped exactly like a pre-key graded half and always will be. Pooling
# them inflated `cases` and `ungraded` with every skill the user had ever run
# `behavioral` against, and could silently supply the "with skill" side of a
# comparison it was never part of. That is the splicing bug this module's run
# id exists to prevent, arriving through the back door.
#
# Refusing them outright was wrong too, and the reason outlives the migration
# that prompted it. A log directory is a record somebody already paid an agent
# run for; a reader that cannot open one because of a metadata key added after
# it was written has made that run worthless to settle a question it holds the
# answer to. The population is small and does not grow -- every run since the
# key arrived labels itself -- but "small" is the argument for a flag rather
# than for a refusal.
#
# So the caller asserts it. Even then a group needs a baseline side, which
# nothing but `cmd_staleness` ever writes, so a directory of ordinary
# `behavioral` logs stays out whatever flags it is given. And where a side
# holds more than one log there is a choice to make and no way to make it, so
# every case in that group is `ungraded` rather than guessed at.
#
# A labelled run never falls in here, so none of this can weaken a run written
# by this version.
LEGACY_RUN = "unlabelled"


def _no_duplicates(
    outcomes: list[BehaviorOutcome], ambiguous: bool = False
) -> list[BehaviorOutcome]:
    """`outcomes`, with every case this side cannot answer for errored.

    `ambiguous` errors all of them: see `LEGACY_RUN`, where unlabelled logs
    leave several candidates for a side and no way to tell which run each
    belongs to. Otherwise only a case this side answers twice *for the same
    attempt* is errored.

    One run key and one side can still span several logs -- a run grades one
    skill per task, and a run key covers the whole invocation -- and nothing
    stops two of them carrying the same case id. `pair` groups a side's
    attempts by case id, so without this two logs answering the same case
    would read as one case attempted twice, and a collision between runs would
    arrive disguised as the evidence `--attempts` exists to produce.

    The attempt number is what tells a repeat from a collision. A side that
    ran three attempts answers each case three times by design, once per
    attempt; two logs answering the same case *and* the same attempt is the
    ambiguity, and no legitimate run produces one.

    Both copies are errored, not just the later one: which of them is this
    side's answer is exactly what cannot be told, so keeping either would be
    the guess. `classify` reads an errored attempt as one that measured
    nothing, consistent with how this module reports every other ambiguity it
    cannot resolve.
    """
    if ambiguous:
        return [
            replace(
                outcome,
                error="these logs carry no run id and there is more than one "
                "of them on a side, so which run this case belongs to cannot "
                "be told apart",
            )
            for outcome in outcomes
        ]
    tally: dict[tuple[str, int], int] = {}
    for outcome in outcomes:
        key = (outcome.id, outcome.attempt)
        tally[key] = tally.get(key, 0) + 1
    return [
        outcome if tally[(outcome.id, outcome.attempt)] == 1
        else replace(
            outcome,
            error="this run has more than one log reporting this case on the "
            "same attempt, and which of them is its answer cannot be told "
            "apart",
        )
        for outcome in outcomes
    ]


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

    graded = [
        c for c in comparisons if c.verdict not in ("ungraded", "guards_only")
    ]

    # Guard rails, counted separately. A broken one is the serious finding
    # here: a skill that made the agent do the forbidden thing.
    broken = [
        c
        for c in comparisons
        if any(not g["passed"] for g in c.guards_with)
    ]
    guard_checks = sum(len(c.guards_with) for c in comparisons)
    return {
        "meta": meta,
        "totals": {
            "cases": len(comparisons),
            "graded": len(graded),
            **counts,
            # Of the cases that were graded both ways, the share that moved
            # when the skill was taken away *and* can be credited to the
            # skill: `earns_it` alone. Cases that moved without the skill
            # firing are `unattributed` and counted as that; the wider "moved"
            # number is computed in `render_markdown`. This is the narrower
            # one, and the one the run exists for.
            "discriminating": (
                round(sum(1 for c in graded if c.verdict == "earns_it") / len(graded), 3)
                if graded
                else None
            ),
            "guard_checks": guard_checks,
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


def _claims(case: dict) -> str:
    """How much of the case each side met, averaged over its graded attempts.

    A side "passes" only by meeting every positive claim, so the verdict is
    deaf to how close the other side came. `fails_either_way` covers a
    baseline that met four of five claims and one that met none; `earns_it`
    covers a baseline that nearly managed it and one that never started. Those
    are different findings about the case, and an author deciding whether a
    case still tests anything needs the one the verdict threw away.

    Above one attempt a side it also shows what the pass counts cannot: a mean
    of 4.5 of 5 is a side that met everything once and nearly everything the
    next time.
    """
    if not case["claims"]:
        return "-"
    return (
        f"{case['claims_met_with']:g}/{case['claims']} with, "
        f"{case['claims_met_without']:g}/{case['claims']} without"
    )


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
    # How many times each side was attempted, said plainly: `3/3` against
    # `0/3` and `1/1` against `0/1` are the same verdict and not the same
    # evidence, and a reader who does not know which they are holding cannot
    # tell them apart. Read off the cases rather than the invocation's own
    # `--attempts`, so a `--from-logs` rebuild describes the runs it read.
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
    if meta.get("from_logs"):
        lines += [
            f"Rebuilt offline from the logs in `{meta['from_logs']}`; nothing "
            "was run. The model and effort are whatever wrote them, which the "
            "logs do not have to agree on, so they are not claimed here.",
            "",
        ]
        # Which run this is a report about. A log directory accumulates every
        # staleness run anybody has done in it, so naming the one read is the
        # difference between a measurement and a pile of them -- see
        # `from_logs` on what reading the pile used to do to the totals.
        used = {c["skill"]: c["run"] for c in summary["cases"] if c["run"]}
        if used:
            lines += [
                "Read from "
                + ", ".join(
                    f"run `{run}` for `{skill}`"
                    for skill, run in sorted(used.items())
                )
                + ".",
                "",
            ]
        # A run id belongs to an invocation and not to a skill: a staleness
        # run mints one id and stamps it on every skill it covers. So "was
        # this run read" is a question about the whole report rather than
        # about one skill's row -- asked per skill, a run read for `beta` was
        # announced as unread by `alpha`, two lines under the sentence naming
        # it as read.
        #
        # Listed per skill even so, because that is the only form in which
        # the ids are actionable: `--run-id` is global, and which skills an
        # id covers is whichever its own invocation was asked for.
        read = set(used.values())
        unread = {
            skill: [run for run in ids if run not in read]
            for skill, ids in (meta.get("runs") or {}).items()
        }
        unread = {skill: ids for skill, ids in unread.items() if ids}
        spare = {run for ids in unread.values() for run in ids}
        if used and spare:
            shown = "; ".join(
                ", ".join(f"`{run}`" for run in ids[:3])
                + (f" and {len(ids) - 3} more" if len(ids) > 3 else "")
                + f" for `{skill}`"
                for skill, ids in sorted(unread.items())
            )
            lines += [
                f"> **{len(spare)} other staleness run(s) in this directory "
                "were not read.** One run per skill is read"
                + ("" if meta.get("run_id") else ", the most recent")
                + ", because reading several reports the same case once per "
                f"run: {shown}. `--run-id` reads one of them instead. An id "
                "covers whichever skills its own run was asked for, which "
                "need not be every skill here.",
                "",
            ]

    # Nothing moved at all. The row `earns_it | 0` is true and says nothing,
    # and this is the one result the command exists to surface, so it is said
    # rather than left to be noticed in a table.
    #
    # `unattributed` and `harmful` are checked too, and not for tidiness: a run
    # whose only case was `unattributed` printed this, and its second sentence
    # -- "every graded case came out the same with the skill and without it" --
    # was false of the only case in it. That case passed with the skill and
    # failed without it; what it lacked was evidence the skill was why. Those
    # two findings have their own notes, which say the true thing.
    moved = totals["earns_it"] + totals["unattributed"] + totals["harmful"]
    if totals["graded"] and not moved:
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
        # Split four ways, not two, and each group worded for what it actually
        # rests on.
        #
        # A `guards_only` case makes no positive claim at all, so it does not
        # rest on the baseline having failed and saying that it does is a
        # claim the run cannot support -- the same reading of its own
        # contamination rule that `classify` gets right by returning before
        # the rule is consulted, and that this paragraph used to get wrong by
        # filtering on contamination alone.
        #
        # The fourth group is what `--attempts` opened up. `_why_dirty` reads
        # every attempt and `dirty_on_pass` reads only the ones that passed,
        # so a baseline that was dirty on an attempt it failed and clean on
        # the one it passed keeps its verdict. Where that verdict is one of
        # `VOIDED_BY_CONTAMINATION` it rests on the baseline having *passed*,
        # so the sentence about resting on a failure is false for it.
        #
        # Driven off `VOIDED_BY_CONTAMINATION` rather than a verdict list
        # written out here, so the two cannot drift apart.
        silent = [c for c in dirty if c["verdict"] == "guards_only"]
        standing = [
            c for c in dirty
            if c["verdict"] not in ("ungraded", "guards_only")
        ]
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
        if silent:
            note.append(
                f"> {len(silent)} are `guards_only` -- "
                + ", ".join(f"`{c['id']}`" for c in silent)
                + " -- and claim nothing either way, so a dirty baseline "
                "neither voids them nor leaves them standing on anything.",
            )
        lines += note

    unfired = totals["unattributed"]
    if unfired:
        lines += [
            "",
            f"> **{unfired} case(s) passed only with the skill installed, but "
            "the skill never fired on the attempt that passed.** Installing a "
            "skill does not make an agent use it, so for those the outcome "
            "differed without the skill being shown to be the reason. They "
            "are `unattributed` rather than `earns_it`: a lead to go and read, "
            "not a case the skill has been credited with.",
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

    notable = [
        c for c in summary["cases"]
        if c["verdict"] in ("no_signal", "harmful", "unattributed")
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
        # A `guards_only` case is not graded and is not nothing: it ran on
        # both sides and its guards were checked. "Nothing to look at here",
        # printed two paragraphs under "that is the skill doing the forbidden
        # thing", is the strongest finding this command can produce being
        # contradicted by the weakest sentence in it. This section is about
        # the verdict table, so it says that instead.
        lines.append(
            "No case reached a verdict. The guard rails above are what this "
            "run found."
        )
    elif not notable:
        lines.append(
            "None. Every graded case either needed the skill or failed both ways."
        )
    else:
        lines += [
            "| Case | Verdict | Passed | Claims met | What to do |",
            "| --- | --- | --- | --- | --- |",
        ]
        for case in notable:
            advice = (
                "the two sides did not settle; run it again before reading "
                "anything into it"
                if case["verdict"] == "no_signal" and _unsettled(case)
                else {
                    "harmful": "the skill steered the agent wrong here",
                    "unattributed": "something moved, but the agent never "
                    "opened the skill -- read the run before crediting it",
                }.get(
                    case["verdict"],
                    "the agent already did this unaided; the case is not "
                    "evidence about the skill",
                )
            )
            lines.append(
                f"| `{case['id']}` | {case['verdict']} | {_passes(case)} "
                f"| {_claims(case)} | {advice} |"
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
            f"claims met {_claims(case)}",
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
