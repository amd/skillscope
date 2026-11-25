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

import re
import sys
from pathlib import Path

from .. import agent, config, deadline, harness as harnesses, routing as routing_core
from ..behavior import BehaviorOutcome
from ..datasets import Case
from . import (
    convert,
    hooks,
    models,
    prerequisites,
    sandbox as sandbox_spec,
    scorers,
    stats,
)

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
    complete=None,
    message_limit: int | None = None,
    env: dict[str, str] | None = None,
):
    """One inspect `Task` per skill: its cases, its skill installed, its scorer.

    `solver_factory` is what drives the agent; everything around it -- scoring,
    judging, reporting -- stays the same whichever one is passed. It receives
    the skill's directory because staging is the driver's job: each driver puts
    the skill where the agent it runs will look for it. It also receives `env`,
    what the skill's prerequisites need set for the agent (see
    `prerequisites.on_host`), because the driver is what starts the agent.

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
        tags=["skillscope", "baseline" if baseline else "graded"],
        dataset=samples,
        setup=hooks.setup_solver(hook, skill, cases),
        solver=solver_factory(skill_dir, dict(env or {})),
        cleanup=hooks.cleanup_fn(hook, skill, cases),
        scorer=scorers.expectations(complete),
        sandbox=sandbox_spec.for_skill(skill),
        # The limit belongs to what drives the agent, not to the inspect model:
        # a harness whose agent is a subprocess hands inspect a placeholder
        # model, and that model's limit is a test value that a single real run
        # exceeds.
        message_limit=message_limit if message_limit is not None else message_limit_for(model),
        time_limit=task_time_limit(bound),
    )


def _failed(
    skill: str, cases: list[Case], detail: str, degraded: bool | None = None
) -> list[BehaviorOutcome]:
    """One failed outcome per case, for a skill that could not be run at all.

    `degraded` defaults to whether `detail` reads as a provider outage. Pass
    it where the cause is known: a failed install's output is a build log, and
    a build log is full of the words that heuristic looks for.
    """
    return [
        BehaviorOutcome(
            id=case.id,
            skill=skill,
            prompt=case.prompt,
            passed=False,
            elapsed_s=0.0,
            error=detail,
            degraded=(
                routing_core.is_provider_error(detail) if degraded is None else degraded
            ),
        )
        for case in cases
    ]


def _normalise(text: str) -> str:
    """Lowercased, separators flattened, no-op segments collapsed.

    The collapse is textual and needs no cwd: `a/b/../c` is `a/c` whoever is
    asking, and so are `a//c` and `a/./c`. Only a `..` with a real segment in
    front of it is removed, so a leading `../x` is left alone -- resolving
    that *would* need the agent's cwd, which no tool-call record carries.

    Without it, `<skills>/other/../run/SKILL.md` names the skill and matched
    nothing, and a baseline that read the skill by that spelling was reported
    as a clean room. `//` and `/./` are the same miss in a cheaper spelling:
    a shell that builds a path by joining two pieces produces them without
    anyone typing them, and both name the skill exactly.

    A trailing separator goes with them, which costs nothing: `_mentions`
    treats the end of the string as a boundary, so `<skills>/run/` and
    `<skills>/run` match the same mark.
    """
    flat = text.replace("\\", "/").lower()
    # `/.` covers `/./`, `/../` and a trailing `/.` or `/..`; `//` covers a
    # repeated separator anywhere.
    if "//" not in flat and "/." not in flat:
        return flat
    out: list[str] = []
    for index, part in enumerate(flat.split("/")):
        # A repeated or trailing separator. The leading empty segment of an
        # absolute path is kept: it is what makes the path absolute.
        if part == "" and index and out:
            continue
        if part == "." and out:
            continue
        if part == ".." and out and out[-1] not in ("", "..") and ":" not in out[-1]:
            out.pop()
            continue
        out.append(part)
    return "/".join(out)


# `~/x` and `$HOME/x` spell the same file as the absolute path, and a mark is
# always absolute. Anchored so it only fires where a path can start, and only
# where one actually follows.
_HOME = re.compile(r"(?<![\w/~$])(?:~|\$home|\$\{home\})(?=/)")


def _expand_home(text: str) -> str:
    """`text` again with `~/` and `$HOME/` written out, where they appear.

    Separate from `_normalise` because it is the one expansion that is not
    context-free. It resolves against *this* process's home, which is the
    agent's own on the live path -- `no_sandbox` runs it on this host -- and
    is a guess on a rebuild of logs written somewhere else. Worth making
    anyway: the alternative is reading `cat ~/dev/x/skills/run/SKILL.md` as
    a clean room, and the answer it buys is voided-or-unknown rather than a
    claim about the skill.

    Applied to whole command strings, not just to arguments that are paths.
    The expansion before this one fired only on a value that started with
    `~` and had no whitespace in it, which is a bare path -- so every `~`
    inside a Bash command line, which is where an agent actually types one,
    went unexpanded.
    """
    if "~" not in text and "$" not in text:
        return text
    try:
        home = _normalise(str(Path.home()))
    except (Exception, SystemExit):  # noqa: BLE001 -- no home, nothing to do
        return text
    return _HOME.sub(home, text) if home else text


# Shapes that read a whole directory rather than one named file. A call
# carrying one of these *and* naming a directory the skill sits in has swept
# the skill up with everything else, without ever spelling its name.
#
# Globs, as substrings: they are not words and can sit anywhere in a path.
_SWEEP_GLOBS = ("*", "?", "**", "rglob")

# The rest are words, and matching them as substrings was the bug twice over.
# `-r ` is a flag an agent almost never types alone: `grep -ri`, `grep -rn`
# and `ls -lR` all sweep and none of them contains `-r ` -- so the shapes the
# docstring below names as the realistic ones were matched only in their
# tidiest spelling. The flag is now any single-dash cluster containing `r`,
# which is what those all are, and the commands are matched on a word
# boundary so a sweep is a command somebody ran rather than two letters
# inside another word.
#
# Bare `grep` is deliberately not here. `grep foo <dir>` does not read the
# directory -- it is the recursive flag or a glob that does, and both are
# matched already -- so listing it would void verdicts on the strength of a
# command that never reached the skill.
#
# Lowercase only, deliberately: every caller passes `_normalise(...)`, so an
# upper-case spelling here can never match. `-R ` sat in the old tuple
# reading as case-sensitive coverage and was dead the day it was written;
# `\w*r\w*` covers it now because the text is lowercased first.
_SWEEP_WORDS = re.compile(
    r"(?:^|\s)(?:-\w*r\w*|--recursive|find|ls|rg|tree|fd)(?=\s|$)"
)


# Tools whose whole job is to walk a tree. A call to one of these is a sweep
# whatever its arguments say, because the recursion is the tool rather than
# something the agent spelled out: `Grep(pattern="deploy", path=".claude")`
# reads every file under `.claude` and carries no wildcard, no `-r` and no
# path that names the skill.
#
# Matched on the function name, loosely, because the same two tools are
# spelled differently by different harnesses and an MCP server may wrap them
# under a name of its own.
_SEARCH_TOOLS = ("glob", "grep", "search")


def _searches(call) -> bool:
    """Whether this call is a directory search by virtue of which tool it is."""
    name = _normalise(str(getattr(call, "function", "") or ""))
    return any(tool in name for tool in _SEARCH_TOOLS)


def _swept(text: str, marks: tuple[str, ...]) -> bool:
    """Whether this call read a directory containing the skill, wholesale.

    `find <skills> -name SKILL.md`, `grep -ri x <skills>/`, `rg x <skills>`
    and `cat <skills>/*/SKILL.md` all read the skill and none of them spells
    its directory, so matching the skill's own path missed every one. An
    agent that has been told nothing and is looking for something reaches for
    exactly these, which made the miss likeliest on the baseline -- the one
    side whose emptiness the whole command rests on.

    Reported as an unknown room rather than as contamination: the call could
    sweep the skill and the record does not prove it did. Unknown and dirty
    void the same verdicts, so the conservative answer costs nothing here.
    """
    if not any(glob in text for glob in _SWEEP_GLOBS) and not _SWEEP_WORDS.search(text):
        return False
    return any(_mentions(text, ancestor) for ancestor in _ancestors(marks))


def _ancestors(marks: tuple[str, ...]) -> set[str]:
    """Every directory the skill sits under, by its own spelling.

    The whole chain, not just the immediate parent: `find ~/.claude -name
    SKILL.md` reads the skill from two levels up and names neither the skill
    nor the directory holding it.

    Stops short of the top level, which is where the cost of being wrong
    turns over. A skill checked out under `/tmp/<something>` would otherwise
    contribute `/tmp` to this set, and then every `ls /tmp/...` in the run --
    including ones nowhere near the skill -- reads as a room nobody can
    vouch for. So `/home/u` is kept and `/home` is not. A sweep of a
    top-level directory does reach the skill, and flagging it would void
    verdicts across whole runs on the strength of a wildcard.

    Where that top level is depends on the mark, and the two marks do not
    have one in common. A single floor of three was calibrated for the
    absolute mark and left the staged one -- `<skills_dir>/<name>`, three
    segments on every registered harness -- with no ancestors at all, so
    `ls .claude/skills/`, `find .claude -name SKILL.md` and `grep -r x
    .claude/skills/` all came back as a clean room. That is the shape an
    agent told nothing reaches for, on the one side whose emptiness the
    whole command rests on.

    So the staged mark keeps every segment and the absolute one does not.
    The floor is not about depth, it is about what else lives there: `/home`
    holds every user on the machine and `/tmp` holds every run, while the
    staged mark is relative to a workspace made for one sample and each of
    its segments is the harness's own -- `.claude` holds skills and nothing
    else of ours. A sweep of it reaches the skill and almost nothing a
    baseline would touch for another reason, and the answer it buys is
    `UNKNOWN_ROOM` rather than contamination, so the conservative reading
    costs a verdict nobody could vouch for anyway.
    """
    found: set[str] = set()
    for mark in marks:
        parts = mark.split("/")
        # Absolute (`/x/...`, or a drive letter after `_normalise`) spends a
        # segment on the empty string before the root, so three there and
        # one here leave `/home/u` and `.claude` respectively.
        floor = 3 if (parts[0] == "" or ":" in parts[0]) else 1
        while len(parts) > floor:
            parts = parts[:-1]
            if (parent := "/".join(parts)):
                found.add(parent)
    return found


# Said instead of a contamination reason when there is no way to look for one.
# Phrased as what is missing rather than as a finding: it is not a claim that
# the baseline was dirty, it is a refusal to claim it was clean.
UNKNOWN_ROOM = (
    "could not tell whether the baseline read the skill: the run did not "
    "record where the skill was, and it is not on this machine"
)


def _skill_marks(skill: str) -> tuple[str, ...] | None:
    """The paths that reading makes a room dirty, lowercased and separator-flat.

    Two of them. The skill's own directory, absolute, which is where an agent
    that went looking would find it. And `<skills_dir>/<name>`, the staged copy
    `no_sandbox.install_skill` writes under a workspace that changes per
    sample, so only its tail is knowable here. The harness names that
    directory -- `.claude/skills` for Claude Code, its own for anything else --
    so this asks it rather than spelling one harness's answer in, which would
    read every other harness's staged copy as untouched.

    It asks the *default* harness, which is right while that is the only
    harness this command will run on -- `staleness` refuses any other, and
    `behavioral` on another harness stages its skill somewhere this would not
    recognise. Worth revisiting the day that refusal lifts.

    The absolute mark is resolved here, against this process. That is right
    for every caller there is: the check runs in the same process as the run
    it is judging. It would be wrong for a reader rebuilding somebody else's
    logs, which is why that reader will have to carry the path in the log
    rather than work it out -- but no such reader exists yet, and a parameter
    kept warm for one is a parameter nothing tests against reality.

    `None` when it cannot say where the skill was, which is an unknown room
    rather than a clean one. Returning the staged mark alone looked like an
    answer and was not: it is the subset of the check that cannot fail, so
    every tool call that reached the skill by its real path came back clean
    and the case was credited.

    The staged mark carries no leading separator on purpose. `no_sandbox` runs
    the agent with `cwd=workspace`, which makes `cat .claude/skills/foo/...`
    the natural shell form, and a mark anchored with a leading `/` missed
    exactly that. `_mentions` supplies the boundary instead, which is what
    keeps the mark from matching the middle of a longer name.
    """
    try:
        home = str(Path(config.active().skill_path(skill)).resolve())
    except (Exception, SystemExit):  # noqa: BLE001 -- see below
        # `skill_path` raises `SystemExit` rather than returning, so catching
        # `Exception` alone let a skill the config cannot place take the
        # whole command down.
        return None
    staged = f"{harnesses.get().skills_dir}/{skill}"
    return (_normalise(home), _normalise(staged))


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

    Every string argument is tested as written and again with `~/` and
    `$HOME/` written out, wherever in it they appear. The expansion before
    this one fired only on a value that started with `~` and had no
    whitespace -- a bare path -- so a `~` inside a Bash command line, which
    is where an agent actually types one, was never expanded at all.

    `./x` and `../x` are deliberately *not* resolved against a cwd. The agent's
    was `no_sandbox`'s per-sample workspace and appears nowhere in the
    tool-call record, so `resolve()` here resolved them against skillscope's
    own instead and made the verdict depend on which directory the report was
    run from. (`_normalise` does collapse an *interior* `/../` or `/./`,
    which needs no cwd to be right.)

    The graded side reads its own installed skill by design, so a hit there is
    expected. It is the baseline where this means the room was not empty.
    """
    arguments = getattr(call, "arguments", None) or {}
    if not isinstance(arguments, dict):
        return None

    reason = f"read the skill's own files ({skill})"
    spellings: list[str] = []
    for value in _strings(arguments):
        text = _normalise(value)
        spellings.append(text)
        if (expanded := _expand_home(text)) != text:
            spellings.append(expanded)

    for text in spellings:
        if any(_mentions(text, mark) for mark in marks):
            return reason

    # The one mark that is not a path. `_bundled_skills` is the harness's own
    # staging directory, so its name in any argument is the agent having found
    # the staging area rather than a file of its own.
    if any("_bundled_skills" in text for text in spellings):
        return reason

    # Last, and weaker on purpose: a call that read a directory the skill
    # sits in, wholesale. Answered after the exact matches above so a call
    # that provably named the skill keeps the reason that says so.
    #
    # Asked of the call, not of each argument on its own. A Bash sweep is one
    # string holding both halves -- the sweep word and the directory -- but
    # Glob and Grep take them as separate arguments, `pattern` and `path`,
    # and those are the tools an agent that has been told nothing reaches for
    # first. Tested per argument, `{"pattern": "**/SKILL.md", "path":
    # "<ancestor>"}` had a sweep word in one string and an ancestor in
    # another and matched in neither, so the same sweep that was caught as a
    # shell command came back as a clean room through the structured tool.
    # A call to a search tool needs no sweep word at all: the recursion is
    # the tool. `Grep` with `path` set to a directory walks everything under
    # it, and `{"pattern": "deploy", "path": ".claude"}` carries no marker
    # any string test could find -- the only evidence that this read the
    # whole tree is which tool was called.
    swept = _searches(call) or any(
        glob in text for text in spellings for glob in _SWEEP_GLOBS
    ) or any(_SWEEP_WORDS.search(text) for text in spellings)
    if swept and any(
        _mentions(text, ancestor)
        for text in spellings
        for ancestor in _ancestors(marks)
    ):
        return UNKNOWN_ROOM
    return None


def _inspect_room(
    routing_engine, sample, skill: str
) -> tuple[str | None, str | None]:
    """`(the skill that fired, why the room was not what it claimed)`.

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
    marks = _skill_marks(skill)
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
        # A walk cut short inspected only the calls it reached, so evidence on
        # any later call was never looked at. That is an unknown room whatever
        # `unknown` says: it was computed before the walk, from whether there
        # were marks to check against, and cannot speak for a walk that did
        # not finish. Returning the completed-walk expression here would
        # report a baseline nobody inspected as clean, which is the one
        # direction this check must never fail in.
        return fired, why or UNKNOWN_ROOM
    return fired, why or (UNKNOWN_ROOM if unknown else None)


# Limits that stop an agent mid-answer rather than deciding anything about
# it. `operator` and `custom` are left out deliberately: one is somebody
# cancelling and the other is a task's own rule, and neither is the clock.
_CUT_SHORT = {
    "time": "stopped at the sample's time limit",
    "working": "stopped at the sample's working-time limit",
    "message": "stopped at the message limit",
    "token": "stopped at the token limit",
}


def _cut_short(sample) -> str | None:
    """Why this sample was stopped before it finished, if it was.

    inspect records a limit as `sample.limit`, separately from `sample.error`,
    so a run the clock ended arrives looking exactly like one the agent
    finished and failed. Read defensively: `limit` is absent on older logs and
    its shape is inspect's rather than ours.
    """
    limit = getattr(sample, "limit", None)
    if limit is None:
        return None
    kind = str(getattr(limit, "type", None) or limit).lower()
    for name, reason in _CUT_SHORT.items():
        if name in kind:
            return reason
    return None


def _is_outage(sample_error, detail: str | None) -> bool:
    """Whether a case's error reads as the provider failing rather than the case.

    Not for a prerequisite that could not be provided in the case's container:
    its message carries the install's output, and a build log is full of the
    words the provider heuristic looks for. The same failure on the host is
    reported with `degraded=False` directly.
    """
    # Qualified: inspect has a `PrerequisiteError` of its own, which is not this.
    qualified = f"{prerequisites.PrerequisiteError.__module__}.PrerequisiteError"
    if qualified in (getattr(sample_error, "traceback", None) or ""):
        return False
    return routing_core.is_provider_error(detail)


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
        elif (cut := _cut_short(sample)) is not None:
            # An agent the clock stopped did not answer the question; it ran
            # out of room to answer it. inspect reports that as `sample.limit`
            # and leaves `error` unset, so the attempt arrived as an ordinary
            # failure -- which on the baseline is the dangerous direction. The
            # baseline runs second, on whatever budget the graded half left,
            # and is the half that explores longest, so it is the one the
            # limits cut. A case the skill never earned then reads `earns_it`,
            # off a baseline that was interrupted rather than beaten.
            error = cut
        elif not checks:
            error = "case has no behavioral assertions to grade"
        elif ungraded := [c for c in checks if not scorers.graded(c)]:
            # One check nobody could grade makes the whole attempt
            # unanswerable rather than partly answered: a side passes only by
            # meeting *every* positive claim, so a claim with no verdict
            # leaves the result undecidable, not negative. Reported as an
            # error so it lands in the errors column and gets triaged as
            # infrastructure, which is what it is -- and so that
            # `staleness.answered` drops the attempt instead of reading a
            # grader that never answered as evidence about a skill.
            error = "could not be graded: " + "; ".join(
                c["detail"] for c in ungraded[:3]
            )

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
                degraded=_is_outage(sample.error, error),
                activated=activated,
                contaminated=contaminated,
                # Which repeat this is, where `--attempts` asked for several.
                # 1 on every sample of an ordinary single-epoch run.
                attempt=int(getattr(sample, "epoch", 1) or 1),
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
    attempts: int = 1,
    concurrency: int = 0,
    complete=None,
    message_limit: int | None = None,
) -> list[BehaviorOutcome]:
    """Run every behavioral case, grouped by skill. Mirrors `behavior.run`.

    `baseline` is passed through to the task and used in the progress lines.
    It does not itself remove the skill: the caller's `solver_factory` does
    that, and this says so in the output.

    `attempts` runs every case that many times, as inspect epochs, and the
    epoch comes back on each sample as `BehaviorOutcome.attempt`. One attempt
    is the ordinary run and leaves the logs exactly as they were.

    `concurrency` caps how many cases are in flight at once; 0 leaves
    inspect's own default alone. Not raised by default: the agents these legs
    drive hold real sandboxes and real rate limits, and a default that
    multiplied either would be a change to every existing command rather than
    an option on a slow one.
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
        each = "" if attempts <= 1 else f", {attempts} attempts each"
        print(f"[{label}] {skill}: {len(skill_cases)} case(s){each}", flush=True)
        try:
            # Around the whole task rather than per case: on the host one
            # install serves every case, and what it built is removed once the
            # last of them is graded.
            with prerequisites.on_host(skill) as env:
                logs = run_eval(
                    inspect_eval,
                    build_task(
                        skill,
                        skill_cases,
                        model,
                        solver_factory=solver_factory,
                        baseline=baseline,
                        complete=complete,
                        message_limit=message_limit,
                        env=env,
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
                    # Repeats are inspect's epochs: one task, one log, each case
                    # sampled `attempts` times with the epoch on every sample.
                    # Running the leg twice over would have written two logs a
                    # side, which a reader pairing them cannot tell from two logs
                    # that collided -- so the repeats stay inside one log and the
                    # epoch on each sample is what tells them apart.
                    epochs=attempts,
                    **({"max_samples": concurrency} if concurrency else {}),
                    fail_on_error=False,
                    # skillscope's own progress lines are the report; inspect's
                    # rich display takes over the terminal and produces nothing
                    # useful when a CI job pipes stdout to a file.
                    display="plain",
                )
        except prerequisites.PrerequisiteError as exc:
            # The skill could not be given what it needs, so none of its cases
            # were run -- the same disposition as a broken sandbox declaration.
            # Not a provider outage, whatever a build log happens to contain.
            outcomes.extend(
                _failed(skill, skill_cases, f"{skill}: {exc}", degraded=False)
            )
            continue
        except KeyboardInterrupt:
            # The deadline's graceful stop landing in a host install, outside
            # inspect, which would otherwise have turned it into a cancel. Fail
            # this skill and keep everything already graded. A Ctrl-C from a
            # person still stops the command.
            if not stop_requested():
                raise
            outcomes.extend(_failed(skill, skill_cases, deadline.active().message()))
            continue
        except SystemExit as exc:
            # One skill's broken setup is that skill's failure, not everybody's.
            # A malformed sandbox declaration used to abort the whole command,
            # throwing away results for skills already graded and paid for --
            # the same reason the structural gate reads only the skills a run
            # is about. A declaration that does not parse is never a provider
            # outage, whatever the path in its message happens to contain.
            outcomes.extend(_failed(skill, skill_cases, str(exc), degraded=False))
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
