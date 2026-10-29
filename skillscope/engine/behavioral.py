# Copyright Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: MIT

"""Behavioral evals under `inspect_ai`.

Shared by every engine that runs on the framework: the task, the scorer, the
sandbox and the reporting live here, and the caller supplies the solver that
drives the agent. `claude-code-no-sandbox` passes one; `claude-code` builds its own task in
`verify.py` because `inspect_swe` supplies the whole agent rather than a solver.

`run()` matches `behavior.run()` -- same arguments, same `BehaviorOutcome`
list -- so swapping engines is a one-line substitution in the CLI and every
report path downstream is untouched.
"""

from __future__ import annotations

import sys
from pathlib import Path

from .. import agent, config, deadline, routing as routing_core
from ..behavior import BehaviorOutcome
from ..datasets import Case
from . import convert, hooks, models, sandbox as sandbox_spec, scorers, stats

# An agent that never decides it is finished must still stop. The legacy engine
# bounded this with `--case-timeout` and a process kill; inspect expresses it
# declaratively, and a message cap catches the loop a wall-clock cap only ends
# after paying for it.
MESSAGE_LIMIT = 120

# A model that reaches no provider never calls the submit tool, so it loops to
# whatever cap it is given -- and every turn is a real sandbox round trip. The
# wiring run proves the machinery in a handful of turns; the rest is the mock
# failing to finish, slowly.
MOCK_MESSAGE_LIMIT = 6


# How much of the command's budget to keep back from inspect's own per-sample
# limit. The `--timeout` deadline ends the process with `os._exit`, which takes
# the report and the transcript with it -- so a run that overruns says only
# that it overran. Handing inspect the whole budget makes the two fire together
# and the hard kill wins the race. Stopping the sample early enough for inspect
# to score what exists and write the log turns a timeout into evidence: eight
# Instinct runs have now overrun and not one of them said where it got to.
TIMEOUT_RESERVE_S = 120


def task_time_limit(bound) -> int | None:
    """The per-sample limit to give inspect, inside the command's own deadline."""
    if bound is None:
        return None
    return max(60, int(bound.remaining() - TIMEOUT_RESERVE_S))


def realtime_logging() -> bool:
    """Whether inspect should keep its live sample buffer for this run.

    The buffer exists so `inspect view` can watch a run in progress, and it
    lives in a sqlite file under the user data directory, named after the task.
    On Windows that directory is the service account's profile, which is long
    enough that a task named after a longer skill crosses MAX_PATH -- sqlite
    then answers "unable to open database file" and the whole task dies. Two
    skills on the same runner passed and one did not, purely on the length of
    its name.

    Nothing watches a CI run live, and the `.eval` log is written either way,
    so the buffer is cost without benefit exactly where it breaks.
    """
    return not sys.platform.startswith("win")


def message_limit_for(model: str) -> int:
    """How many turns this model should be allowed before the case is stopped."""
    if model.lower().startswith(agent.NO_PROVIDER_PREFIXES):
        return MOCK_MESSAGE_LIMIT
    return MESSAGE_LIMIT


def build_task(
    skill: str,
    cases: list[Case],
    model: str,
    ctx: dict | None = None,
    *,
    solver_factory,
    baseline: bool = False,
    run_key: str | None = None,
):
    """One inspect `Task` per skill: its cases, its skill installed, its scorer.

    `solver_factory` is what drives the agent; everything around it -- scoring,
    judging, reporting -- stays the same whichever one is passed. It receives
    the skill's directory because staging is the driver's job: each driver puts
    the skill where the agent it runs will look for it.

    Required, and keyword-only. It was optional while a built-in react agent
    was the default, and a caller that forgot it silently graded a different
    agent than it asked for.

    `baseline` only names the task. Whether the skill is actually staged is
    the solver's business, so the flag and the factory have to agree -- the
    name is what makes a disagreement visible in the log directory rather
    than invisible in the numbers.
    """
    from inspect_ai import Task

    skill_dir = config.active().skill_path(skill)
    samples = [convert.sample_from_case(c, skill_dir, ctx) for c in cases]

    # `evals/hooks.py`, where the skill ships one. `cleanup` rather than a
    # trailing solver: inspect runs it in a `finally` under a shielded cancel
    # scope, so teardown still happens when the agent raises -- which is the
    # property a hook that removes containers exists for.
    hook = hooks.load(skill)

    bound = deadline.active()
    return Task(
        name=f"behavioral-{skill}" + ("-baseline" if baseline else ""),
        metadata={
            BASELINE_KEY: baseline,
            SKILL_KEY: skill,
            # Resolved here, where it is known to be right. See `SKILL_DIR_KEY`.
            SKILL_DIR_KEY: str(Path(skill_dir).resolve()),
            **({STALENESS_RUN_KEY: run_key} if run_key else {}),
        },
        tags=["skillscope", "baseline" if baseline else "graded"],
        dataset=samples,
        setup=hooks.setup_solver(hook, skill, cases),
        solver=solver_factory(skill_dir),
        cleanup=hooks.cleanup_fn(hook, skill, cases),
        scorer=scorers.expectations(),
        sandbox=sandbox_spec.for_skill(skill),
        message_limit=message_limit_for(model),
        time_limit=task_time_limit(bound),
    )


def _failed(skill: str, cases: list[Case], detail: str) -> list[BehaviorOutcome]:
    """One failed outcome per case, for a skill that could not be run at all."""
    return [
        BehaviorOutcome(
            id=case.id,
            skill=skill,
            prompt=case.prompt,
            passed=False,
            elapsed_s=0.0,
            error=detail,
            degraded=routing_core.is_provider_error(detail),
        )
        for case in cases
    ]


def _normalise(text: str) -> str:
    """Lowercased with separators flattened, so one comparison covers both OSes."""
    return text.replace("\\", "/").lower()


# Said instead of a contamination reason when there is no way to look for one.
# Phrased as what is missing rather than as a finding: it is not a claim that
# the baseline was dirty, it is a refusal to claim it was clean.
UNKNOWN_ROOM = (
    "could not tell whether the baseline read the skill: the run did not "
    "record where the skill was, and it is not on this machine"
)


def _skill_marks(skill: str, recorded: str | None = None) -> tuple[str, ...] | None:
    """The paths that reading makes a room dirty, lowercased and separator-flat.

    Two of them. The skill's own directory, absolute, which is where an agent
    that went looking would find it. And `.claude/skills/<name>`, the staged
    copy `no_sandbox.install_skill` writes under a workspace that changes per
    sample, so only its tail is knowable here.

    `recorded` is `SKILL_DIR_KEY` off the log: where the skill was on the
    machine that ran the agent. Preferred over resolving it here, because
    resolving it here resolves it against *this* process. A reader standing
    somewhere else is not evidence about where the agent stood.

    `None` when neither source can say where the skill was, which is an
    unknown room rather than a clean one. Returning the staged mark alone
    looked like an answer and was not: it is the subset of the check that
    cannot fail, so every tool call that reached the skill by its real path
    came back clean and the case was credited. See `SKILL_DIR_KEY`.

    The staged mark carries no leading separator on purpose. `no_sandbox` runs
    the agent with `cwd=workspace`, which makes `cat .claude/skills/foo/...`
    the natural shell form, and a mark anchored with a leading `/` missed
    exactly that. `_mentions` supplies the boundary instead, which is what
    keeps the mark from matching the middle of a longer name.
    """
    home = recorded
    if home is None:
        try:
            home = str(Path(config.active().skill_path(skill)).resolve())
        except (Exception, SystemExit):  # noqa: BLE001 -- see below
            # `skill_path` raises `SystemExit` rather than returning, so
            # catching `Exception` alone let a rebuild of someone else's logs
            # take the whole command down.
            return None
    return (_normalise(home), _normalise(f".claude/skills/{skill}"))


# Characters that continue a path component. A mark followed or preceded by
# one of these is part of a longer name, not the name itself.
_NAME_CHARS = "-_."


def _mentions(text: str, mark: str) -> bool:
    """Whether `text` names `mark` as a whole path component.

    Plain `in` was the bug twice over. It is what made `/{skill}/` match
    `/run/user/1000`, and it survived in the shell branch after the structured
    branch was fixed, where it read `.claude/skills/foo-extra` as skill `foo`'s
    own files. A sibling skill whose name extends this one's is not this one.

    So a hit has to end at a boundary -- end of string, or a character that
    cannot continue a component -- and start at one. `foo` matches in
    `/x/.claude/skills/foo/SKILL.md` and in `cat .claude/skills/foo/S`, and
    does not match in `foo-extra`, `foo.bak` or `myfoo`.
    """
    width = len(mark)
    start = 0
    while (at := text.find(mark, start)) >= 0:
        start = at + 1
        after = text[at + width : at + width + 1]
        if after and (after.isalnum() or after in _NAME_CHARS):
            continue
        before = text[at - 1 : at] if at else ""
        if before and (before.isalnum() or before in _NAME_CHARS):
            continue
        return True
    return False


def _strings(value) -> list[str]:
    """Every string anywhere in a tool call's arguments.

    Flat rather than by field name. The first version of this checked a fixed
    list of path-ish argument names, which is a closed set against an open tool
    surface: it had no entry for Glob's `pattern`, which is often the only
    argument a Glob call carries, so that call was invisible. MCP tools name
    their arguments whatever they like, so the list would have kept drifting.

    Checking every string is safe here only because the marks are boundary
    matched and absolute. That is what removed the false positives, not the
    field list.
    """
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        return [s for v in value.values() for s in _strings(v)]
    if isinstance(value, (list, tuple)):
        return [s for v in value for s in _strings(v)]
    return []


def _read_the_skill(call, marks: tuple[str, ...], skill: str) -> str | None:
    """Whether this call reached into the skill's own directory.

    Every string argument is tested as written, and a `~`-relative one is
    tested expanded as well. Expansion is skipped for anything with whitespace
    in it: that is a command line rather than a path, and it is matched as
    written.

    The graded side reads its own installed skill by design, so a hit there is
    expected. It is the baseline where this means the room was not empty.
    """
    arguments = getattr(call, "arguments", None) or {}
    if not isinstance(arguments, dict):
        return None

    reason = f"read the skill's own files ({skill})"
    for value in _strings(arguments):
        text = _normalise(value)
        if any(_mentions(text, mark) for mark in marks):
            return reason
        # `~/x` names the same file by another spelling, and expanding it is
        # context-free: `$HOME` is the agent's own, recorded in the
        # environment rather than inferred from where this is running.
        #
        # `./x` and `../x` are deliberately *not* resolved. Resolving them
        # needs the agent's cwd, which is `no_sandbox`'s per-sample workspace
        # and appears nowhere in the tool-call record -- so `resolve()` here
        # resolved them against skillscope's own cwd instead, and made the
        # verdict depend on which directory the report was run from. It never
        # could serve the case it was written for.
        if value.startswith("~") and not any(c.isspace() for c in value):
            try:
                expanded = _normalise(str(Path(value).expanduser()))
            except (Exception, SystemExit):  # noqa: BLE001 -- not a path, then
                continue
            if any(_mentions(expanded, mark) for mark in marks):
                return reason

    # The one mark that is not a path. `_bundled_skills` is the harness's own
    # staging directory, so its name in any argument is the agent having found
    # the staging area rather than a file of its own.
    if any("_bundled_skills" in _normalise(v) for v in _strings(arguments)):
        return reason
    return None


def _inspect_room(
    routing_engine, sample, skill: str, skill_dir: str | None = None
) -> tuple[str | None, str | None]:
    """`(the skill that fired, why the room was not what it claimed)`.

    `skill_dir` is `SKILL_DIR_KEY` off the log, passed by a reader rebuilding
    someone else's run. A live run leaves it `None` and resolves the skill
    where it already knows it is.

    One walk over the tool calls, answering both questions, because the second
    is only meaningful beside the first. `detect_activation` reports a skill
    nobody staged as `other:<name>`, which is exactly the contamination this
    looks for -- the runner's own `~/.claude/skills` are registered next to
    the staged ones, and a behavioral run passes no `config_dir` to keep them
    out.

    `fired` is `None` or exactly `skill`, never an `other:` marker. The two
    questions are answered by separate branches for that reason: when they
    shared one `if/elif`, a second `other:` hit arriving after `why` was
    already set fell through to the `elif` and was recorded as the skill
    having fired -- which reads downstream as "the skill activated" on exactly
    the contaminated cases that most need the opposite.

    Reading the skill's own source is the second route, and the one isolation
    only partly closes: `no_sandbox` runs the agent with `cwd=workspace`, but
    nothing stops it naming an absolute path, so a directory the staging step
    skipped is still on disk for it to find. The sandboxed `claude-code` leg
    does close it, by isolating the filesystem. A baseline that read it was
    not a baseline.

    Best effort: a shape this cannot walk is an unknown room, not a failed
    case.
    """
    fired: str | None = None
    why: str | None = None
    marks = _skill_marks(skill, skill_dir)
    # Nowhere to check against, so the read half of the check cannot run. Said
    # out loud rather than passed over: this lands the case `ungraded`, which
    # is what an unanswerable question is worth. Applied after the walk, not
    # before it, so a reason the walk *could* establish -- another skill
    # firing, which needs no path at all -- is reported in preference to it.
    unknown, marks = marks is None, marks or ()
    try:
        for call in routing_engine._tool_calls(sample):
            hit = routing_core.detect_activation(
                routing_engine._activation_event(call),
                [skill],
                allow_body_path=routing_engine.ALLOW_BODY_PATH,
            )
            if hit and hit.startswith("other:"):
                if why is None:
                    why = f"another skill fired: {hit.removeprefix('other:')}"
            elif hit and fired is None:
                fired = hit
            if why is None:
                why = _read_the_skill(call, marks, skill)
    except Exception:  # noqa: BLE001 -- see the docstring
        return fired, why or (UNKNOWN_ROOM if unknown else None)
    return fired, why or (UNKNOWN_ROOM if unknown else None)


def _outcomes(log, skill: str, cases: list[Case]) -> list[BehaviorOutcome]:
    """Map one inspect `EvalLog` back onto skillscope's outcome objects.

    A task that produced no samples at all reports one failed outcome per case
    rather than an empty list: an infrastructure failure must not render as
    "every expectation met".

    But a task that produced *some* samples keeps them, whatever its status.
    `log.status` goes to `error` when any sample raised, so failing the batch
    on it threw away every case the run had already graded and paid for --
    one container that would not start, and a skill's whole report read as a
    total failure. `engine/routing.py:_outcomes` has always done it this way,
    for the same reason; the two now agree. Cases with no sample are still
    errored below, so nothing goes quietly missing.
    """
    # Imported here rather than at module scope: `engine/routing.py` imports
    # this module, so a top-level import would close the cycle.
    from . import routing as routing_engine  # noqa: PLC0415

    prompts = {c.id: c.prompt for c in cases}
    outcomes: list[BehaviorOutcome] = []

    if not log.samples:
        detail = getattr(log.error, "message", None) or "the task produced no samples"
        return _failed(skill, cases, f"inspect task failed: {detail}")

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

        activated, contaminated = _inspect_room(routing_engine, sample, skill)

        outcomes.append(
            BehaviorOutcome(
                id=case_id,
                skill=skill,
                prompt=prompts.get(case_id, ""),
                passed=error is None and bool(checks) and all(c["passed"] for c in checks),
                elapsed_s=round(getattr(sample, "total_time", None) or 0.0, 2),
                checks=checks,
                error=error,
                degraded=routing_core.is_provider_error(error),
                activated=activated,
                contaminated=contaminated,
            )
        )

    # Samples inspect never reported back are still cases somebody asked for.
    # Now that a partial log is kept rather than discarded wholesale, the cases
    # missing from it are the ones that need saying -- silence here would read
    # as a smaller, cleaner run rather than an incomplete one.
    failure = getattr(getattr(log, "error", None), "message", None)
    missing = (
        f"inspect task failed before this case ran: {failure}"
        if failure
        else "inspect returned no sample for this case"
    )
    graded = {outcome.id for outcome in outcomes}
    outcomes.extend(
        _failed(skill, [c for c in cases if c.id not in graded], missing)
    )
    return outcomes


# Eval-level metadata naming the `run_eval` call that started an eval, so a
# stopped run reads back its own log and nobody else's.
CALL_TOKEN = "skillscope_call"

# Which half of a staleness comparison a log is, and whose skill it is about.
# The task name says the same thing, but a name is a string somebody has to
# parse; inspect filters on metadata and tags, and so can anything reading a
# log directory afterwards. Written by both engines, so a reader does not have
# to know which one produced the log it is holding.
BASELINE_KEY = "skillscope_baseline"
SKILL_KEY = "skillscope_skill"

# Which staleness invocation wrote this log. Only `cmd_staleness` sets it, and
# it mints one value per invocation, so two logs carrying the same value are
# the two halves of one comparison and nothing else is.
#
# `--from-logs` needs this and cannot be built without it. The log directory is
# shared by `behavioral`, `verify` and `routing`, is gitignored wholesale and
# is never pruned, and every ordinary `behavioral` run stamps
# `BASELINE_KEY: False` -- so grouping on `(skill, baseline)` alone splices an
# unrelated month-old run in as the "with skill" half, under a table that reads
# as confidently as a real one. `run_eval` reached the same conclusion for the
# same directory and solved it the same way, with `CALL_TOKEN`.
STALENESS_RUN_KEY = "skillscope_staleness_run"

# Where the skill was on the machine that ran the agent.
#
# `_skill_marks` needs an absolute directory to recognise a tool call that
# reached into the skill. Deriving it at read time derives it from whoever is
# *reading*: a rebuild in another directory, or on a machine where the skill
# is not checked out, built a quietly weaker mark set and reported a baseline
# the live run had voided as `ungraded` back as `no_signal`. That is the
# "a rebuild is more permissive than the run it rebuilds" failure this module
# closes everywhere else; contamination was the one place it was still open.
#
# So the run records where the skill actually was, and the reader uses that
# rather than its own cwd. Written by both engines for the same reason
# `SKILL_KEY` is: a reader should not have to know which one produced the log.
SKILL_DIR_KEY = "skillscope_skill_dir"


def stop_requested() -> bool:
    """Whether the deadline's graceful stage has asked the command to stop."""
    bound = deadline.active()
    return bound is not None and bound.interrupted


def run_eval(inspect_eval, task, *, log_dir: str, **kwargs):
    """`inspect_eval(task, ...)`, keeping the logs of a run the deadline stopped.

    When the graceful stage interrupts a run inside inspect's loop, inspect
    cancels the samples, runs `Task.cleanup`, writes the log as `cancelled` --
    and then returns `[]`. Measured with five of six samples finished: all five
    were in the log on disk, and none came back from `eval()`. Without this, a
    run that overran its wall clock dropped every case it had already graded
    and paid for, and the gate blamed "every case errored".

    The interrupt does not always land there. In `eval()`'s synchronous setup
    it surfaces as `KeyboardInterrupt`, and on Python 3.10 anyio can surface it
    as `CancelledError`; both are taken as the stop they are, and the readback
    runs the same way.

    So the log is read back from `log_dir`, and only this call's. The default
    directory accumulates across runs and can be shared by another command
    running beside this one -- including one running the very same task, since
    routing's is always `routing`. A task name or a time window cannot tell
    those apart, so each call tags its eval with a token of its own and reads
    back only what carries it. Files that predate the call are skipped before
    their headers are read, which is all that `before` is for. An empty
    result is the caller's to report, since only it knows which cases were
    asked for.
    """
    import asyncio
    import uuid

    from inspect_ai.log import read_eval_log

    token = uuid.uuid4().hex
    before = _log_files(log_dir)
    try:
        logs = inspect_eval(
            task, log_dir=log_dir, metadata={CALL_TOKEN: token}, **kwargs
        )
    except (KeyboardInterrupt, asyncio.CancelledError):
        if not stop_requested():
            raise
        logs = []
    if logs or not stop_requested():
        return logs
    ours = [
        uri for uri in sorted(_log_files(log_dir) - before)
        if (read_eval_log(uri, header_only=True).eval.metadata or {}).get(CALL_TOKEN)
        == token
    ]
    return [read_eval_log(uri) for uri in ours]


def _log_files(log_dir: str) -> set[str]:
    """The logs in `log_dir`, as URIs inspect can read back on every platform.

    Not `inspect_ai.log.list_eval_logs`: on Windows the names it returns do
    not survive the trip back into `read_eval_log`. It builds them as
    `file://C:...`, one slash short, so the drive letter parses as a host and
    is dropped -- every readback failed with FileNotFoundError on a path
    starting `\\Users`. `Path.as_uri()` gives `file:///C:/...`. Local only,
    which is every directory skillscope writes logs to.
    """
    root = Path(log_dir)
    if not root.is_dir():
        return set()
    return {
        path.resolve().as_uri()
        for path in root.iterdir()
        if path.suffix in (".eval", ".json")
    }


def run(
    skills: list[str],
    cases: list[Case],
    model: str,
    effort: str,
    *,
    solver_factory,
    log_dir: str | None = None,
    baseline: bool = False,
    run_key: str | None = None,
) -> list[BehaviorOutcome]:
    """Run every behavioral case, grouped by skill. Mirrors `behavior.run`.

    `baseline` is passed through to the task and used in the progress lines.
    It does not itself remove the skill: the caller's `solver_factory` does
    that, and this says so in the output.

    `run_key` is passed through to the task's metadata. See
    `STALENESS_RUN_KEY`: only a staleness run sets one, and it is what lets
    `--from-logs` pair two halves without pairing two unrelated runs.
    """
    from inspect_ai import eval as inspect_eval

    sandbox_spec.require_provider()

    outcomes: list[BehaviorOutcome] = []
    for skill in skills:
        skill_cases = [c for c in cases if c.skill == skill and c.has_behavior]
        if not skill_cases:
            continue

        if stop_requested():
            # The stop is for the command, not the task it landed in. Starting
            # the next skill would spend the reserve meant for cleanup and meet
            # the hard kill, which loses the report along with everything else.
            outcomes.extend(_failed(skill, skill_cases, deadline.active().message()))
            continue

        label = "baseline" if baseline else "behavioral"
        print(f"[{label}] {skill}: {len(skill_cases)} case(s)", flush=True)
        try:
            logs = run_eval(
                inspect_eval,
                build_task(
                    skill,
                    skill_cases,
                    model,
                    solver_factory=solver_factory,
                    baseline=baseline,
                    run_key=run_key,
                ),
                model=model,
                model_args=models.model_args(model),
                log_dir=log_dir or str(Path(".skillscope") / "logs"),
                log_realtime=realtime_logging(),
                # A sample that raises is that case's failure, not the batch's.
                # inspect's default ends the task on the first error, so one
                # container that would not start discarded every case already
                # graded -- `_outcomes` then reported the whole skill failed.
                # Errored samples still come back in the log and are still
                # graded as failures; what changes is that their neighbours
                # survive.
                fail_on_error=False,
                # skillscope's own progress lines are the report; inspect's rich
                # display takes over the terminal and produces nothing useful
                # when a CI job pipes stdout to a file.
                display="plain",
            )
        except SystemExit as exc:
            # One skill's broken setup is that skill's failure, not everybody's.
            # A malformed sandbox declaration used to abort the whole command,
            # throwing away results for skills already graded and paid for --
            # the same reason the structural gate reads only the skills a run
            # is about.
            outcomes.extend(_failed(skill, skill_cases, str(exc)))
            continue

        if not logs and stop_requested():
            # Stopped before inspect wrote anything. Failed rather than absent,
            # so the report still counts the cases it was asked for.
            outcomes.extend(_failed(skill, skill_cases, deadline.active().message()))
            continue

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
