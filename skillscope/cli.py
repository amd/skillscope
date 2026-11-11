# Copyright Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: MIT

"""The one entry point for skill evals.

Every skill owns a single dataset at ``<skill>/evals/evals.json``. This runner
reads those datasets and grades them with four commands:

  * ``structural`` -- every skill folder, every dataset, and every reference
    the skill's markdown makes. No agent, no tokens, instant.
  * ``routing`` -- installs the skills named by ``--routing-room`` side by
    side and checks that the right one fires (and that nothing fires when
    nothing should). Cheap, no hardware, and it pools those skills' prompts so
    each one's positives are the others' negatives. A repo with a single skill
    need not name it: there is only one room its skill can be in.
  * ``behavioral`` -- installs one skill, runs the prompt to completion, and
    grades ``expected_behavior`` / ``unexpected_behavior`` / ``logs_contain``
    / ``files_exist``. Only runs evaluations that assert something beyond the
    routing decision.
  * ``staleness`` -- runs ``behavioral``'s cases twice, once with the skill
    installed and once without, and compares. Answers a question the other
    three cannot: whether the cases still need the skill, or whether the model
    has caught up and the suite is passing on its own.

A skill may also ship ``<skill>/evals/extended_evals.json``, in the same format
and under no coverage requirement of its own. The graded commands include it
by default; ``--no-extended`` grades the required dataset alone.

The prompt is written once and every graded command reads it, which is the
whole point: the alternative is a central routing prompt set plus a separate
per-skill test file that re-asserts routing with a substring match on the
transcript.

Usage::

    # structural checks only: no agent, no tokens, instant
    skillscope structural

    # the same, plus fetching every external URL the skills link to
    skillscope structural --external

    # what a skill does once it has fired
    skillscope behavioral --skill serving-llms-on-epyc

    # what CI runs: a routing miss fails the run, like a behavioral miss does
    skillscope routing --routing-room local-ai-use,tracelens --no-extended
    skillscope behavioral --skill local-ai-use --no-extended

    # a routing run that reports its score instead of gating on it
    skillscope routing --routing-room all --min-accuracy 0

    # a repo with one skill: the room is that skill, so nothing names it
    skillscope routing

    # one case, keeping the raw transcript
    skillscope routing --only qwen-on-mi300x --keep-logs eval-logs

    # whether the skill is earning its place: every case run with it and
    # without it, twice the agent passes of a behavioral run
    skillscope staleness --skill serving-llms-on-epyc

    # the same comparison again, read back off the logs it already wrote
    skillscope staleness --from-logs .skillscope/logs

    # what CI should run for a change
    skillscope select --since BASE HEAD

    # the same, from a list of paths worked out some other way
    git diff --name-only BASE...HEAD | skillscope select --changed

Reports go to stdout as markdown, to ``$GITHUB_STEP_SUMMARY`` under Actions,
and to a JSON artifact under ``.skillscope/runs/`` in the repo under test.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
import uuid
from pathlib import Path

from . import (
    behavior,
    config,
    datasets,
    deadline,
    engine,
    references,
    routing,
    staleness,
    structure,
    usage,
)
from . import selection as select_module
from . import harness as harnesses
from .agent import check_api_reachable

# The engines a graded run can be driven by.
#   claude-code             -- the real CLI in a sandbox, via inspect_swe (Linux only)
#   claude-code-no-sandbox  -- the real CLI on the host, under inspect_ai (any platform)
#
# Both drive the agent a skill is actually written for; what differs is where
# it runs. Two others have been removed. `inspect` drove a harness-independent
# agent, and carried a smaller tool set than the real harness, so a skill
# referring to a tool it did not have failed for a reason that was the agent's
# rather than the skill's. `legacy` drove the CLI as a bare subprocess, and
# `claude-code-no-sandbox` replaced it: measured per case against the run's own
# noise floor, the two agreed on every routing case outside that floor, across
# two runs, and on 12 of 13 behavioral cases.
ENGINES = ["claude-code", "claude-code-no-sandbox"]

# Every engine runs under inspect_ai now. Kept as its own name because the code
# that branches on it is about what an inspect-based engine needs -- a preflight
# of its own, a report that names the engine -- rather than about which engines
# happen to exist today.
INSPECT_ENGINES = tuple(ENGINES)

# The engines routing has a leg for: all of them. Derived rather than listed,
# because the last time two engine lists were kept by hand one fell behind and
# the runs silently went somewhere else.
ROUTING_ENGINES = tuple(ENGINES)

# What a run uses when nobody says. The host leg rather than the sandboxed one:
# it runs on every platform, where `claude-code` needs a POSIX guest and a
# container runtime, and a default that refuses on Windows is not a default.
DEFAULT_ENGINE = "claude-code-no-sandbox"

# The neutral names for where an agent runs. `host` and `sandbox` say only that.
# The older names also say which agent, and still work, meaning Claude Code.
NEUTRAL_ENGINES = {"host": "claude-code-no-sandbox", "sandbox": "claude-code"}


def resolve_leg(
    engine: str | None, harness_name: str | None
) -> tuple[str, harnesses.Harness]:
    """Settle which agent runs, and where, from whatever was said about each.

    The two are separate questions. `--engine` is where (on the host, or in a
    container) and `--harness` is what. The older engine names, which carry the
    agent in them, keep working and mean Claude Code; naming a different
    harness beside one is a contradiction and is refused rather than guessed.
    What comes back is the engine in the spelling the rest of the code branches
    on, and the harness itself.
    """
    explicit = harness_name or os.environ.get("SKILLSCOPE_HARNESS") or None
    subject = harnesses.get(explicit)
    # Unnamed means `host`, the neutral spelling of DEFAULT_ENGINE: the older
    # spelling names Claude Code, which would contradict any other harness.
    engine = engine or os.environ.get("SKILLSCOPE_ENGINE") or "host"

    if engine in NEUTRAL_ENGINES:
        leg = NEUTRAL_ENGINES[engine]
    elif engine in ENGINES:
        if explicit and subject.name != harnesses.DEFAULT:
            neutral = {v: k for k, v in NEUTRAL_ENGINES.items()}[engine]
            raise SystemExit(
                f"error: --engine {engine} runs Claude Code, which contradicts "
                f"--harness {subject.name}. Use --engine {neutral} to say where "
                "the agent runs without saying which one."
            )
        subject = harnesses.get(harnesses.DEFAULT)
        leg = engine
    else:
        raise SystemExit(
            f"error: unknown engine {engine!r}. Known engines: "
            f"{', '.join([*NEUTRAL_ENGINES, *ENGINES])}."
        )

    if leg == "claude-code" and not subject.supports_sandbox:
        raise SystemExit(
            f"error: the {subject.name} harness has no sandboxed leg. Use "
            "--engine host, which runs it on the host."
        )
    return leg, subject


def _engine_label(engine: str, subject: harnesses.Harness) -> str:
    """The engine as a report and the console spell it for this harness.

    Claude Code keeps the older name, so reports from before there was a choice
    stay comparable. Any other harness gets the neutral name: the old ones name
    an agent that did not run.
    """
    if subject.name == harnesses.DEFAULT:
        return engine
    return {v: k for k, v in NEUTRAL_ENGINES.items()}.get(engine, engine)


def _leg_meta(args: argparse.Namespace) -> dict:
    """Which agent graded, where it ran, and which build of it, for a report.

    The engine keeps its older spelling for Claude Code, so reports from before
    there was a choice stay comparable; any other harness is reported in the
    neutral names, because the old ones would name an agent that did not run.
    """
    subject = harnesses.get(getattr(args, "harness", None))
    return {
        "engine": _engine_label(args.engine, subject),
        "harness": subject.name,
        "harness_version": subject.version(),
    }

# Where JSON reports land inside the repo under test. One gitignored directory
# rather than a path per repo, so a report is always in the same place.
RUNS_DIRNAME = Path(".skillscope") / "runs"


def _selected_skills(names: str) -> list[str]:
    available = datasets.skills_with_datasets()
    if not names:
        return available
    wanted = [s.strip() for s in names.split(",") if s.strip()]
    unknown = [s for s in wanted if s not in available]
    if unknown:
        raise SystemExit(
            f"error: no eval dataset for {', '.join(unknown)}. Expected "
            f"<skill>/{datasets.DATASET_RELPATH.as_posix()}."
        )
    return wanted


def _write_report(summary: dict, report: str, args: argparse.Namespace, label: str) -> Path:
    if args.output:
        output = Path(args.output)
    else:
        output = (
            config.active().root / RUNS_DIRNAME / f"{label}-{time.strftime('%Y%m%d-%H%M%S')}.json"
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print()
    print(report)
    print(f"[evals] JSON report: {output}")

    summary_path = args.summary or os.environ.get("GITHUB_STEP_SUMMARY", "")
    if summary_path:
        with open(summary_path, "a", encoding="utf-8") as handle:
            handle.write(report)
    return output


def _report_failures(errors: list[str]) -> None:
    print("Structural checks failed:", file=sys.stderr)
    for err in errors:
        print(f"  - {err}", file=sys.stderr)


def _structural_or_exit(skills: list[str] | None = None) -> list[references.Reference]:
    """Fail before any tokens are spent if the skills are not in shape.

    The skill folders, the datasets, and the internal references: everything
    that costs nothing to check. External URLs are left out on purpose: they
    fail for reasons that have nothing to do with the run being gated.

    Given `skills`, only those are read. A graded run passes the skills it is
    about to install, because that is the run it is gating: a neighbour's
    malformed dataset says nothing about whether this run can proceed, and
    holding it back would make one skill's mistake everybody else's. The
    repo-wide answer is ``skillscope structural``, which passes nothing here
    and is the check that gates a merge.
    """
    found = references.collect(skills=skills)
    errors = (
        structure.errors(skills)
        + datasets.structural_errors(skills)
        + references.internal_errors(found)
    )
    if errors:
        _report_failures(errors)
        raise SystemExit(1)
    return found


def routing_gate(totals: dict, min_accuracy: float) -> str | None:
    """Why this routing run should fail, or ``None`` if it should not.

    The first two answers are infrastructure, not score, and hold whatever the
    bar is: a run where nothing was graded, or where no skill ever activated,
    has not measured routing at all.
    """
    if totals["graded"] == 0:
        return "every case errored; treating the run as a failure."
    if totals["activations"] == 0 and totals["activations_expected"]:
        return (
            "no skill activated in any case -- the skills were not installed, "
            "or activation detection is broken. Failing rather than reporting "
            "a 0% routing rate as if it were real."
        )
    if min_accuracy <= 0:
        return None
    # Compared against the exact ratio rather than the reported accuracy,
    # which is rounded for the report: at the default bar of 1, a single miss
    # in a large enough set rounds to 1.000 and would let itself through.
    if totals["passed"] / totals["graded"] < min_accuracy:
        return (
            f"{totals['passed']}/{totals['graded']} correct "
            f"(accuracy {totals['accuracy']}) is below the --min-accuracy bar "
            f"of {min_accuracy}."
        )
    return None


# --------------------------------------------------------------------------
# Commands
# --------------------------------------------------------------------------


def cmd_structural(args: argparse.Namespace) -> int:
    """Structural checks over every skill folder, dataset, and reference."""
    found = _structural_or_exit()
    skills = datasets.skills_with_datasets()
    # Extended datasets are checked regardless of --extended, so count them
    # here too rather than reporting fewer cases than were checked.
    cases = datasets.load_all_cases(extended=True)
    cfg = config.active()
    print(f"[evals] OK: {len(cfg.skills)} skill folder(s) in shape.")
    print(
        f"[evals] OK: {len(cases)} case(s) across {len(skills)} skill(s) "
        f"plus {len(datasets.load_shared_negatives())} shared negative(s)."
    )
    local = sum(1 for reference in found if reference.is_local)
    external = references.external_urls(found)
    print(
        f"[evals] OK: {local} internal reference(s) across "
        f"{len(references.markdown_files())} markdown file(s)."
    )

    if args.external:
        errors = references.external_errors(
            found, exclude=cfg.excluded_urls, jobs=args.jobs
        )
        if errors:
            _report_failures(errors)
            return 1
        print(f"[evals] OK: {len(external)} external URL(s) answered.")
    elif external:
        print(
            f"[evals] {len(external)} external URL(s) not fetched. "
            "`--external` checks them over the network."
        )

    print(f"[evals] repo: {cfg.root}  skills: {', '.join(cfg.skill_globs)}")
    return _fail_if_expired() or 0


def cmd_list_skills(args: argparse.Namespace) -> int:
    print(json.dumps(datasets.skills_with_datasets(), separators=(",", ":")))
    return 0


def cmd_template(args: argparse.Namespace) -> int:
    """Print the dataset template a new skill starts from."""
    sys.stdout.write(datasets.TEMPLATE.read_text(encoding="utf-8"))
    return 0


def _changed_for_select(args: argparse.Namespace) -> set[str]:
    """The changed paths to plan from, however the caller chose to say them."""
    if args.since:
        lines = select_module.changed_paths(*args.since)
        # The plan is the only thing on stdout, so what it was decided from
        # goes to stderr, where a CI log still shows it.
        print("Changed files:", file=sys.stderr)
        for path in lines or ["(none)"]:
            print(f"  {path}", file=sys.stderr)
    else:
        lines = sys.stdin.read().splitlines()
    return {line.strip().replace("\\", "/") for line in lines if line.strip()}


def cmd_select(args: argparse.Namespace) -> int:
    available = datasets.skills_with_datasets()
    if args.all:
        skills, needs_routing = available, True
    elif args.names is not None:
        requested = [n.strip() for n in args.names.split(",") if n.strip()]
        unknown = [n for n in requested if n not in available]
        if unknown:
            print(f"error: no eval dataset for: {', '.join(unknown)}", file=sys.stderr)
            return 1
        skills, needs_routing = requested, True
    else:
        changed = _changed_for_select(args)
        skills = select_module.select_from_changes(changed)
        needs_routing = select_module.routing_needed(changed, args.extended)

    labels = {token.strip() for token in args.labels.split(",") if token.strip()}
    print(
        json.dumps(
            select_module.plan(
                skills,
                routing=needs_routing,
                labels=labels,
                ignore_gates=args.ignore_gates,
                extended=args.extended,
            ),
            separators=(",", ":"),
        )
    )
    return 0


def _empty_room(args: argparse.Namespace) -> None:
    """A routing run was asked for with nobody in the room.

    ``--routing-room none`` on this command is a contradiction: the command
    is the routing run, and that flag empties the room. Skipping routing is
    done by not invoking ``routing`` -- ``select`` still takes ``none`` so CI
    can leave the job off the plan.

    Otherwise the repo has several skills with datasets and said nothing about
    which of them compete, and that is not a guess this harness makes: install
    every skill on disk and a work-in-progress drops everyone's number, install
    the one under review alone and it wins by walkover. A repo with a single
    skill is the exception, and never reaches this function -- there is only
    one room its skill can be in.
    """
    if config.wants_no_skills(args.routing_room):
        raise SystemExit(
            "error: `routing` asks for a routing run and "
            "`--routing-room none` empties the room. Name the skills that "
            "go in it, or skip this command."
        )

    available = ", ".join(datasets.skills_with_datasets()) or "(none)"
    raise SystemExit(
        "error: `routing` needs the skills that go in the room: "
        "`--routing-room a,b` or `all` for every skill with a dataset. "
        "Only a repo with one skill gets a default, because who a skill "
        "competes against is what its routing score means. Skills with a "
        f"dataset here: {available}."
    )


def _fail_if_expired() -> int | None:
    """Non-zero when the command's ``--timeout`` has elapsed or stopped the run.

    Both, because the graceful stage fires `GRACEFUL_RESERVE_S` *before* the
    wall. inspect then returns from `eval()` normally, so a run it stopped
    reaches the gate with the clock still short of `expired()` -- and was
    reported as "every case errored" instead of the overrun it was.
    """
    bound = deadline.active()
    if bound is None or not (bound.expired() or bound.interrupted):
        return None
    print(f"error: {bound.message()}", file=sys.stderr)
    return 1


def _prepare_graded_run(
    args: argparse.Namespace, scope: list[str] | None = None
) -> list[str]:
    """Structural checks, model pin, and API reachability. Shared by the graders.

    The structural gate reads `scope`, defaulting to the skills ``--skill``
    selected -- the ones about to be graded, and nobody else. A routing run
    passes the room instead, because the room is what it installs and what its
    score is about; ``--skill`` there only narrows which of the room's cases
    are reported on.
    """
    if (code := _fail_if_expired()) is not None:
        raise SystemExit(code)
    selected = _selected_skills(args.skill)
    _structural_or_exit(selected if scope is None else sorted(set(scope)))
    args.engine, subject = resolve_leg(
        getattr(args, "engine", None), getattr(args, "harness", None)
    )
    args.harness = subject.name
    if subject.name != harnesses.DEFAULT and getattr(args, "command", "") == "behavioral":
        # Behavioral cases are graded by an LLM judge, and that judge still goes
        # through an Anthropic model whatever agent produced the run. Saying so
        # beats running the agent and then failing every judged expectation.
        raise SystemExit(
            f"error: --harness {subject.name} cannot run behavioral yet: the "
            "judge that grades its expectations is not available through this "
            "harness. Routing runs on it."
        )
    # The model is the harness's to name and to pin: Claude Code keeps `opus`
    # and the CI pin it always had.
    args.model = subject.pin_model(subject.resolve_model(args.model))
    if getattr(args, "engine", DEFAULT_ENGINE) in INSPECT_ENGINES:
        # The CLI-based reachability probe tests something these engines do not
        # use, but they still need one of their own: a graded run starts
        # containers and installs skills before it first reaches a provider, so
        # without this a bad key surfaces as a task that failed after all that.
        engine.require(args.engine)
        # Before the model probe, because it costs nothing and a run that
        # cannot honour a skill's setup should not first spend a round trip
        # finding out the credentials are fine.
        _require_hook_support(
            args.engine, selected, getattr(args, "command", "")
        )
        # Also before the probe, and for the same reason: a malformed price
        # table should fail the command, not one sample into a graded run.
        # Registered here rather than in the engine because every graded
        # command comes through this function and inspect's registry is
        # process-global -- setting it twice would be the same work.
        from .engine import pricing as engine_pricing

        args.pricing_note = engine_pricing.apply()
        if not args.skip_preflight and subject.name != harnesses.DEFAULT:
            # The harness's own CLI is all a routing run reaches: its agent is a
            # subprocess and there is no judge, so no provider is probed.
            ok, detail = subject.preflight(args.model)
            if not ok:
                raise SystemExit(f"error: {subject.binary} not reachable -- {detail}")
        elif not args.skip_preflight:
            from .engine import models as engine_models

            ok, detail = engine_models.check_reachable(engine_models.resolve(args.model))
            if not ok:
                raise SystemExit(f"error: model not reachable -- {detail}")
            if args.engine == "claude-code-no-sandbox":
                # This engine reaches the provider for the judge but drives the
                # real CLI as the agent, so both credentials are load-bearing
                # and probing one proves nothing about the other.
                ok, detail = check_api_reachable(args.model)
                if not ok:
                    raise SystemExit(f"error: claude API not reachable -- {detail}")
        return selected
    if not args.skip_preflight:
        ok, detail = check_api_reachable(args.model)
        if not ok:
            raise SystemExit(f"error: claude API not reachable -- {detail}")
    return selected


def _require_hook_support(engine: str, skills: list[str], command: str) -> None:
    """Refuse a run whose hooks use entry points this engine cannot honour.

    `evals/hooks.py` is environment plumbing -- clearing stale containers,
    tearing down a service. `setup` and `teardown` run here: inspect's
    `Task.setup` and `Task.cleanup` are the same two shapes, and `cleanup`
    runs inside a `finally` under a shielded cancel scope, so teardown still
    happens when the agent raises.

    `check` and `setup_session` do not, and this refuses rather than skipping
    them. Skipping is the dangerous half: a hook that did not run leaves no
    trace in the report, the case is graded as though it had, and the failure
    surfaces later as a skill that mysteriously does not work on this runner.

    Refused by entry point, not by the file existing. An earlier version
    grounded any skill that shipped a hook at all -- which took the behavioral
    leg away from the one skill in the catalogue whose hook these engines can
    run perfectly well.

    Routing is exempt on every engine: it installs the skills, asks which one
    fires, and executes nothing. Staleness is not: it drives the same
    `build_task` on the same cases, twice, so a `check` it skipped is skipped
    on both halves and the comparison is between two runs that both lied.
    """
    if command not in ("behavioral", "staleness"):
        return

    from .engine import hooks as engine_hooks

    blocked: dict[str, list[str]] = {}
    for skill in skills:
        names = engine_hooks.unsupported_entry_points(engine_hooks.load(skill))
        if names:
            blocked[skill] = names
    if not blocked:
        return

    listed = "\n".join(
        f"    {skill}: {datasets.hooks_path(skill)} defines "
        f"{', '.join(names)}"
        for skill, names in blocked.items()
    )
    raise SystemExit(
        f"error: --engine {engine} cannot run every entry point these "
        f"skills' hooks define:\n{listed}\n"
        "    They would be skipped silently, and the cases graded as though "
        "they had run.\n"
        "    `setup` and `teardown` are supported. `check` needs the legacy "
        "engine's Run object, and `setup_session` returns template variables "
        "that are substituted before any hook runs -- neither has an "
        "equivalent here.\n"
        "    Move what they do into the dataset, or into setup/teardown."
    )


def _sandbox_meta(args: argparse.Namespace) -> dict:
    """What contained this run, recorded so the report does not have to imply it.

    Saying so in the artifact is the point: the same numbers mean different
    things depending on whether anything was isolated, and one engine here
    isolates nothing by construction. A report that left a reader to infer it
    from the engine name is how a table of host results came to be read as
    evidence about containers.
    """
    from .engine import sandbox as engine_sandbox

    return engine_sandbox.describe()


def _finish_routing(
    args: argparse.Namespace,
    outcomes: list,
    routing_set: dict,
    started: float,
    *,
    isolated: bool,
    sandbox: str,
    sandbox_isolated: bool,
    extra: dict | None = None,
) -> int:
    """Summarize, report, and gate a routing run. Shared by all three legs.

    What contained the run is required of the caller rather than worked out
    here. Deriving it from the engine's name is what let a run that started no
    container report `sandbox: docker, sandbox_isolated: true`, and hardcoding
    `host` was only right while no leg had a sandbox. Required keywords mean a
    leg added later that forgets fails at the call with a TypeError, instead of
    quietly reporting a containment it never had.
    """
    summary = routing.summarize(
        outcomes,
        list(routing_set),
        {
            "model": args.model,
            **_leg_meta(args),
            "effort": args.effort,
            "skills": list(routing_set),
            "extended": args.extended,
            "wall_time_s": round(time.time() - started, 1),
            "timeout": args.timeout,
            "github_run_id": os.environ.get("GITHUB_RUN_ID"),
            **usage.snapshot().as_meta(),
            **(extra or {}),
            # Last, so a leg's own meta cannot shadow what contained it.
            "isolated_config_dir": isolated,
            "sandbox": sandbox,
            "sandbox_isolated": sandbox_isolated,
        },
    )
    _write_report(summary, routing.render_markdown(summary), args, "routing")

    if (code := _fail_if_expired()) is not None:
        return code
    reason = routing_gate(summary["totals"], args.min_accuracy)
    if reason:
        print(f"[routing] {reason}", file=sys.stderr)
        return 1
    return 0


def cmd_routing(args: argparse.Namespace) -> int:
    # Who is in the room decides what the structural gate covers, so it is
    # settled before anything is checked or any token is spent.
    routing_set = config.active().routing_set
    if not routing_set:
        _empty_room(args)

    _prepare_graded_run(args, list(routing_set))
    subject = harnesses.get(args.harness)
    started = time.time()

    # Pool the routing set's cases: skill Y's positives are skill X's
    # negatives, which is where most of the false-trigger coverage comes
    # from. --skill narrows what is *reported on*, not what is installed,
    # unless the room was left unnamed, in which case it is both -- either
    # the skills --skill named, or the only skill here with a dataset.
    cases = datasets.routing_cases(list(routing_set), extended=args.extended)
    if args.only:
        cases = datasets.filter_cases(cases, args.only)
    elif args.skill:
        cases = datasets.filter_cases(cases, args.skill)

    print(f"[routing] installed together: {', '.join(routing_set)}")
    # No `jobs=`: it was printed but never passed on. inspect decides a routing
    # task's concurrency from its own `max_connections`, so naming a number the
    # run does not honour described a parallelism nobody chose. `--jobs` still
    # means what it always did for `structural --external`, which is the one
    # command that reads it.
    where = f"engine={_engine_label(args.engine, subject)}"
    if subject.name != harnesses.DEFAULT:
        where = f"harness={subject.name}, {where}"
    print(f"[routing] {len(cases)} cases, model={args.model}, {where}")

    from .engine import models as engine_models
    from .engine import routing as inspect_routing
    from .engine import sandbox as engine_sandbox

    outcomes = inspect_routing.run(
        cases,
        routing_set,
        args.model,
        args.effort,
        args.engine,
        case_timeout=args.case_timeout,
        max_tool_calls=args.max_tool_calls,
        max_inspection_calls=args.max_inspection_calls,
        max_budget_usd=args.max_budget_usd,
        jobs=args.jobs,
        log_dir=args.keep_logs or None,
        harness=subject,
    )
    if args.engine == "claude-code":
        # A container really was started: `verify.require()` refuses every
        # provider that shares the host's filesystem, so `describe()`
        # cannot report an isolation this leg did not have. The config dir
        # is isolated structurally -- the guest has no `~/.claude` to keep
        # out -- rather than conditionally on a credential.
        box = engine_sandbox.describe()
        isolated = True
    else:
        # The CLI on the host. `engine/routing.py` has already refused to
        # start without the credential that lets it redirect the config
        # dir, so reaching here means the room is the room that was asked
        # for -- but nothing was contained, and the report says so.
        box = {"sandbox": "host", "sandbox_isolated": False}
        isolated = True
    # Reported as enforced, per leg. Both legs stop at the decision now
    # -- the sandboxed one by declining the call, the host one by reading
    # the CLI's stream and killing the process group -- but they are held
    # to different numbers, so the report names the one that applied.
    #
    # `case_timeout` is the clipped bound, not the argument: `case_time_limit`
    # shortens it to whatever `--timeout` has left, so reporting the request
    # named a cap longer than the one a case was actually held to.
    extra: dict = {"case_timeout": inspect_routing.case_time_limit(args.case_timeout)}
    # Both legs, because both enforce these. The sandboxed leg declines the
    # call at the approver; the host leg reads the CLI's stream and stops the
    # process group (`host_stop_when_factory`) -- the same rule, applied one
    # call later. Recorded only for `claude-code` before, so `near_a_limit`
    # was handed a `meta` with no thresholds in it and the default engine
    # reported `near_limit: 0` for every run, including cases that stopped
    # *on* the cap. `budget_for` already returns the unscaled cap off-sandbox,
    # so one call covers both legs.
    extra["max_tool_calls"] = inspect_routing.budget_for(
        args.engine, args.max_tool_calls
    )
    extra["max_inspection_calls"] = inspect_routing.budget_for(
        args.engine, args.max_inspection_calls
    )
    if args.engine == "claude-code":
        extra["budget_scaled_by"] = inspect_routing.SANDBOX_BUDGET_FACTOR
        # This leg cannot give the CLI `--max-budget-usd` -- `inspect_swe`
        # takes no command-line passthrough -- so the cap goes to the task as
        # inspect's own per-sample `cost_limit` instead. Whether that can bind
        # depends on inspect being able to price the model, which it cannot do
        # for any model it currently ships. Both facts are recorded: a reader
        # who sees a budget and no limit event must be able to tell "stayed
        # under it" from "it could never fire".
        if inspect_routing.sandbox_cost_limit(
            args.engine, args.model, args.max_budget_usd
        ):
            extra["max_budget_usd"] = args.max_budget_usd
            extra["max_budget_enforced_by"] = "inspect cost_limit"
            extra["max_budget_can_bind"] = engine_models.cost_limit_binds(args.model)
            # Where the rates came from, when any did. A cap backed by an
            # operator's own figures and one backed by inspect's registry are
            # not the same claim, and only one of them is currently possible.
            from .engine import pricing as engine_pricing

            extra["max_budget_pricing"] = getattr(
                args, "pricing_note", engine_pricing.NO_PRICING
            )
    else:
        # The CLI's own cap, passed through and only when the build
        # advertises it -- so this records what was actually enforced.
        #
        # Keyed on the budget flag itself, not on the list being non-empty:
        # the list also carries `--no-session-persistence`, which says nothing
        # about spend, so `if flags:` would claim an enforced budget for a run
        # that passed none.
        flags = subject.cost_flags(args.max_budget_usd)
        if "--max-budget-usd" in flags:
            extra["max_budget_usd"] = args.max_budget_usd
            extra["max_budget_enforced_by"] = f"{subject.binary} --max-budget-usd"
            extra["max_budget_can_bind"] = True
        elif (
            subject.name != harnesses.DEFAULT
            and args.max_budget_usd
            and args.max_budget_usd > 0
        ):
            # Asked for and not available. Saying so is the difference between a
            # run that stayed under a cap and a run whose cap could never fire.
            extra["max_budget_usd"] = args.max_budget_usd
            extra["max_budget_enforced_by"] = None
            extra["max_budget_can_bind"] = False
        extra["optional_cli_flags_used"] = sorted(f for f in flags if f.startswith("--"))

    return _finish_routing(
        args,
        outcomes,
        routing_set,
        started,
        isolated=isolated,
        sandbox=box["sandbox"],
        sandbox_isolated=box["sandbox_isolated"],
        extra=extra,
    )


def _gradable_cases(args: argparse.Namespace, skills: list[str]) -> list:
    """The cases a behavioral-style run grades: selected, filtered, assertive.

    Shared so `behavioral` and `staleness` cannot drift apart about what
    `--only` or `--extended` select. A staleness run comparing a different set
    of cases than the behavioral run it is read beside would be the one bug
    neither report could show you.
    """
    cases = []
    for skill in skills:
        cases.extend(datasets.load_dataset(skill, extended=args.extended))
    if args.only:
        # Applied across every selected skill at once, because it refuses a
        # selection that matched nothing. Per skill, `--only` naming a case of
        # one skill would fail the run on the next skill's empty match.
        cases = datasets.filter_cases(cases, args.only)
    cases = [c for c in cases if c.has_behavior]

    # Capped here rather than through inspect's own `limit`, which applies per
    # task, and a run builds one task per skill: the same number would mean
    # "the first N cases" on a one-skill run and "N per skill" on a four-skill
    # one. Capped after the filters, so `--only` and `--max-cases` compose the
    # way a reader would expect.
    #
    # And capped once, in the shared selector, because the two halves of a
    # staleness run have to grade the same cases. A cap applied anywhere
    # downstream could hand the baseline a different set, and a comparison
    # between two runs that answered different questions is worse than no
    # comparison: nothing in the report would say so.
    #
    # Per skill rather than over the whole run, so a four-skill run does not
    # spend its whole budget on the first skill's dataset and report the other
    # three as having nothing to grade.
    cap = getattr(args, "max_cases", 0)
    if cap:
        kept: list = []
        for skill in skills:
            kept.extend([c for c in cases if c.skill == skill][:cap])
        return kept
    return cases


def _run_behavioral(
    args: argparse.Namespace,
    skills: list[str],
    gradable: list,
    *,
    install: bool = True,
    config_dir: Path | None = None,
    run_key: str | None = None,
) -> list:
    """Run the behavioral cases on the selected engine.

    `install` False runs them with no skill staged, which is the baseline half
    of `cmd_staleness`. Both halves go through here rather than through two
    similar call sites, so the only difference between the run being measured
    and the run it is measured against is the flag.

    `run_key` is stamped on the logs both halves write, so `--from-logs` can
    find this run's two halves and only this run's. Set by `cmd_staleness`
    alone -- see `engine/behavioral.py:STALENESS_RUN_KEY`.

    `attempts` and `concurrency` are read off `args` rather than passed, so
    the two halves cannot be given different ones. A baseline attempted fewer
    times than the graded half is not a baseline, and the counts the report
    prints would be comparing different denominators without saying so.
    """
    from .engine import behavioral as inspect_behavioral
    from .engine import models as engine_models

    attempts = getattr(args, "attempts", 1)
    concurrency = getattr(args, "concurrency", 0)

    if args.engine == "claude-code":
        from .engine import verify as runner

        return runner.run(
            skills,
            gradable,
            engine_models.resolve(args.model),
            args.effort,
            install=install,
            run_key=run_key,
            attempts=attempts,
            concurrency=concurrency,
        )
    if args.engine == "claude-code-no-sandbox":
        # The real CLI, driven on the host, inside inspect's framework.
        # `--model` stays the CLI's own alias here: this one does not go
        # through an inspect model provider.
        from .engine import no_sandbox

        # The harness is read here rather than passed in, for the reason
        # `attempts` and `concurrency` are: both halves of a staleness run go
        # through this function, so reading it once is what stops the two
        # halves being given different agents.
        subject = harnesses.get(getattr(args, "harness", None))
        no_sandbox.require_local(subject)
        return inspect_behavioral.run(
            skills,
            gradable,
            engine_models.resolve(args.model),
            args.effort,
            solver_factory=lambda d: no_sandbox.harness_solver(
                subject, args.model, args.effort, d, config_dir, install=install
            ),
            baseline=not install,
            run_key=run_key,
            attempts=attempts,
            concurrency=concurrency,
        )
    # Not a fallthrough. An engine in ENGINES with no branch here is a
    # bug this dispatch has had once already, and a silent default is
    # what let it survive: runs asked for one agent, got another, and
    # the report named the one they had asked for.
    raise SystemExit(
        f"error: --engine {args.engine} has no behavioral leg. "
        f"This is a bug in skillscope, not in how it was called."
    )


def cmd_behavioral(args: argparse.Namespace) -> int:
    skills = _prepare_graded_run(args)
    started = time.time()

    gradable = _gradable_cases(args, skills)

    if not gradable:
        print(
            "[behavioral] no evaluation in the selected skill(s) asserts "
            "anything beyond routing, so there is nothing to grade. Add "
            "`expected_behavior` / `unexpected_behavior` / `logs_contain` / "
            "`files_exist` to a triggering evaluation."
        )
        return 0

    outcomes = _run_behavioral(args, skills, gradable)

    summary = behavior.summarize(
        outcomes,
        {
            "model": args.model,
            **_leg_meta(args),
            "effort": args.effort,
            "skills": skills,
            "extended": args.extended,
            "wall_time_s": round(time.time() - started, 1),
            "timeout": args.timeout,
            "github_run_id": os.environ.get("GITHUB_RUN_ID"),
            **usage.snapshot().as_meta(),
            **_sandbox_meta(args),
        },
    )
    _write_report(summary, behavior.render_markdown(summary), args, "behavioral")
    if (code := _fail_if_expired()) is not None:
        return code
    if summary["totals"]["passed"] != summary["totals"]["cases"]:
        return 1
    return 0


def cmd_staleness(args: argparse.Namespace) -> int:
    """Run every behavioral case twice, with the skill and without it.

    Both passes run under the one `--timeout`, so a deadline that expires
    partway leaves the rest of the second pass unmeasured -- those cases are
    reported `ungraded` rather than as a skill that made no difference.

    Budget for a baseline that runs *longer than the graded pass*, not the
    same. An agent holding the skill goes straight to the work; an agent
    without it explores -- building throwaway virtualenvs, dry-running
    installs, reading around the problem. Measured on a two-case skill: the
    graded pass finished both cases in 2m23s and the baseline was still going
    on the second after 30 minutes. Twice a `behavioral` bound is the floor,
    not the figure -- which is why this parser defaults `--timeout` to twice
    what the single-pass commands use. On the shared 900s default those
    numbers left the slower half ~637s of a measured 1800s+ need, so
    truncation was the expected outcome on this command's own example rather
    than a risk it ran.

    Always exits 0 on a completed run. The comparison is nondeterministic on
    both sides, so a verdict here is somewhere to look rather than a build
    failure -- see `skillscope/staleness.py`. A deadline that expired still
    fails, because that is a statement about the run rather than about a skill.
    """
    started = time.time()

    asked = getattr(args, "harness", None) or os.environ.get("SKILLSCOPE_HARNESS")
    if not args.from_logs and asked and asked != harnesses.DEFAULT:
        # The one place this command cannot be harness-neutral yet. Its
        # baseline half is only a measurement if the room is empty, and the
        # check that establishes that -- `routing.can_isolate_config` below --
        # reads Claude Code's credential variables and no other harness's. Run
        # against a different agent with an Anthropic key happening to sit in
        # the environment, it would report an isolated room it had not
        # isolated, which is the one error this whole command exists to avoid
        # making. Refused rather than run on a baseline nobody can vouch for.
        raise SystemExit(
            f"error: --harness {asked} has no staleness leg yet. The baseline "
            "half has to be a room with no skills in it, and the check that "
            f"establishes that reads {harnesses.DEFAULT}'s credentials rather "
            "than this harness's -- so a run here would claim an empty room "
            f"it never checked. Use --harness {harnesses.DEFAULT}, or "
            "`routing` and `behavioral`, which are harness-neutral."
        )

    if args.run_id and not args.from_logs:
        # Same habit as `--unlabelled-logs` below. A live run mints its own
        # run id and stamps it on both halves, so there is no run here for a
        # reader to choose between.
        raise SystemExit(
            "error: --run-id only means something with --from-logs. A run "
            "started now mints its own id, and there is nothing to pick "
            "between."
        )

    if args.unlabelled_logs and not args.from_logs:
        # The repo's habit: refuse an argument that cannot do anything rather
        # than accept and ignore it. A live run writes its own run id on both
        # halves, so there is nothing unlabelled for this to reach.
        raise SystemExit(
            "error: --unlabelled-logs only means something with --from-logs. "
            "A run started now stamps a run id on both halves, so it has no "
            "unlabelled logs to read."
        )

    if args.attempts < 1:
        raise SystemExit(
            f"error: --attempts {args.attempts} would run nothing. The "
            "command compares a side that ran against a side that ran; one "
            "attempt each is the minimum that produces a comparison."
        )
    if args.max_cases < 0 or args.concurrency < 0:
        raise SystemExit(
            "error: --max-cases and --concurrency count things, so they "
            "cannot be negative. 0 means no cap on either."
        )

    # Same habit again. None of these reaches a rebuild: `--from-logs` starts
    # no agent, and the attempts, the cap, the concurrency and which evals
    # were graded are all properties of the run that wrote the logs rather
    # than of the reader. Accepting them silently would let somebody believe
    # they had re-measured something at three attempts when they had reread
    # one attempt.
    #
    # `--skill` and `--only` are not on this list, and used to be ignored like
    # it. They name a subset of what is already there, which a reader of a
    # shared log directory has every reason to want, so the rebuild filters on
    # them below instead.
    spent = [
        name
        for name, value in (
            ("--attempts", args.attempts > 1),
            ("--max-cases", args.max_cases),
            ("--concurrency", args.concurrency),
            ("--no-extended", not args.extended),
            # The flag, not `asked`: an ambient `SKILLSCOPE_HARNESS` is not an
            # argument to this invocation, and refusing a free rebuild over a
            # variable the shell happens to export would be a refusal nobody
            # asked for.
            ("--harness", args.harness),
        )
        if value
    ]
    if args.from_logs and spent:
        raise SystemExit(
            f"error: {', '.join(spent)} cannot change a --from-logs rebuild. "
            "It runs no agent: the attempts, the cases, the concurrency, the "
            "agent and which evals were graded are whatever the logs were "
            "written with. "
            "The report reads them back off the logs. Use --skill or --only "
            "to narrow what is reported."
        )

    if args.from_logs:
        # Nothing is run, so nothing is prepared: no structural gate, no model
        # pin, no reachability probe. Those guard a spend that is not about to
        # happen, and demanding them would make a free command fail for
        # reasons that cannot affect its answer.
        rebuild = staleness.from_logs(
            args.from_logs,
            unlabelled=args.unlabelled_logs,
            run_id=args.run_id or None,
        )
        comparisons = rebuild.comparisons
        known = sorted({run for ids in rebuild.runs.values() for run in ids})
        if args.run_id and not comparisons and known:
            # Told apart from "no staleness logs here at all", because they
            # are different mistakes: this one is a typo in an id, and the
            # answer to it is the list of ids that would have worked.
            raise SystemExit(
                f"error: no staleness run `{args.run_id}` in "
                f"{args.from_logs}. The runs there are: "
                + ", ".join(f"`{run}`" for run in known)
                + "."
            )
        if args.run_id:
            # `--run-id` is global and a run id is not. One invocation mints
            # one id and stamps it on whichever skills it was asked for, so
            # naming an id silently drops every skill that invocation did not
            # cover -- and the guard above cannot catch it, because it only
            # fires when *nothing* matched. Asking for one run and getting one
            # run is legitimate; losing a skill without being told is not.
            covered = {c.skill for c in comparisons}
            missing = sorted(set(rebuild.runs) - covered)
            if missing:
                print(
                    f"[staleness] run {args.run_id} covers "
                    f"{len(covered)} of the {len(rebuild.runs)} skill(s) in "
                    f"{args.from_logs}. Nothing is reported for "
                    + ", ".join(missing)
                    + ": that run did not cover them. Each has runs of its "
                    "own; drop --run-id to read the most recent of each.",
                    flush=True,
                )

        # `--skill` and `--only` narrow what is reported rather than what is
        # read. A log directory is shared and accumulates, so a reader asking
        # about one skill in it is the ordinary case; ignoring the flags, as
        # this used to, answered a question nobody asked.
        wanted = {s.strip() for s in args.skill.split(",") if s.strip()}
        cases = {c.strip() for c in args.only.split(",") if c.strip()}
        if wanted or cases:
            comparisons = [
                c
                for c in comparisons
                if (not wanted or c.skill in wanted)
                and (not cases or c.id in cases)
            ]
            if not comparisons:
                raise SystemExit(
                    "error: nothing in "
                    f"{args.from_logs} matches "
                    + " and ".join(
                        part
                        for part in (
                            f"--skill {args.skill}" if wanted else "",
                            f"--only {args.only}" if cases else "",
                        )
                        if part
                    )
                    + "."
                )
        if not comparisons:
            from .engine import behavioral as engine_behavioral

            raise SystemExit(
                f"error: no skillscope staleness logs in {args.from_logs}. "
                f"Both halves of a comparison carry `{engine_behavioral.SKILL_KEY}`, "
                f"`{engine_behavioral.BASELINE_KEY}` and "
                f"`{engine_behavioral.STALENESS_RUN_KEY}` in their eval "
                "metadata; logs written by another tool, or by a plain "
                "`behavioral` run, cannot be paired."
                + (
                    ""
                    if args.unlabelled_logs
                    else "\n    Logs written before the run id existed carry "
                    "none, and are passed over. `--unlabelled-logs` reads "
                    "them, where a baseline half is there to pair with."
                )
            )
        # No `model`, `engine` or `effort`. Nothing here read them off the
        # logs, and `args` holds this invocation's defaults -- which describe
        # the command that did the rebuilding, not the runs being rebuilt. The
        # report prints what it is given, so passing them through made it
        # assert a model the measurement was never taken on.
        summary = staleness.summarize(
            comparisons,
            {
                "skills": sorted({c.skill for c in comparisons}),
                "from_logs": args.from_logs,
                # The runs behind what is being reported, so the report can
                # name the one it read and say what it passed over. Narrowed
                # to the skills that survived the filters above: a skill the
                # reader excluded has runs that were not read, and counting
                # those as passed over would answer a question nobody asked
                # with a number nobody wanted. See `staleness.Rebuild`.
                "runs": {
                    skill: ids
                    for skill, ids in rebuild.runs.items()
                    if skill in {c.skill for c in comparisons}
                },
                "run_id": args.run_id or None,
                "wall_time_s": round(time.time() - started, 1),
                "github_run_id": os.environ.get("GITHUB_RUN_ID"),
            },
        )
        _write_report(summary, staleness.render_markdown(summary), args, "staleness")
        # The docstring says a deadline that expired still fails, and said it
        # of this branch too while this branch alone returned without asking.
        # Reading a large directory is not free, and a command that overran
        # the bound it was given should say so whatever it spent the time on.
        return _fail_if_expired() or 0

    skills = _prepare_graded_run(args)
    gradable = _gradable_cases(args, skills)

    if not gradable:
        print(
            "[staleness] no evaluation in the selected skill(s) asserts "
            "anything beyond routing, so there is nothing to compare. A "
            "staleness run needs cases that say what the skill should do."
        )
        return 0

    # The runner's own `~/.claude/skills` are registered beside the staged
    # ones, so without this the baseline is not an empty room -- it is this
    # machine's skills minus the one under test. `routing` has refused to run
    # without isolation for the same reason; staleness needs it more, because
    # the baseline being empty is the entire measurement rather than one
    # input to it.
    if args.engine == "claude-code-no-sandbox" and not routing.can_isolate_config():
        raise SystemExit(
            "error: staleness cannot run on --engine claude-code-no-sandbox "
            f"without {' or '.join(routing.ENV_CREDENTIALS)}. The baseline "
            "half has to be a room with no skills in it, and redirecting the "
            "CLI away from the runner's own config dir only works when auth "
            "comes from the environment. Without it every skill installed on "
            "this machine joins the baseline, and a case it solves there "
            "would be reported as one the skill under test was not needed "
            "for.\n    Set one of those, or use --engine claude-code, whose "
            "guest has no user-level skills at all."
        )

    # One token for this invocation, stamped on every log both halves write.
    # `--from-logs` has nothing else to pair on: the log directory is shared
    # and never pruned, and a plain `behavioral` run stamps the same
    # `BASELINE_KEY: False` the graded half does, so without this an unrelated
    # old run is a candidate for this comparison's "with skill" side.
    run_key = uuid.uuid4().hex

    passes = (
        "run twice" if args.attempts <= 1
        else f"run {args.attempts} time(s) on each side"
    )
    print(f"[staleness] {len(gradable)} case(s), {passes}", flush=True)
    # A directory per half, not one shared between them. The contamination
    # check reads what the agent did, so it cannot see state the CLI itself
    # writes into its config dir; were the graded half to leave anything
    # there that the baseline half could pick up, the baseline would no
    # longer be the empty room the whole command rests on. Nothing is known
    # to do this, but a second temporary directory costs nothing and the
    # question stops needing an answer.
    with (
        tempfile.TemporaryDirectory(prefix="skillscope-staleness-with-") as graded_dir,
        tempfile.TemporaryDirectory(prefix="skillscope-staleness-without-") as bare_dir,
    ):
        host = args.engine == "claude-code-no-sandbox"
        with_skill = _run_behavioral(
            args, skills, gradable,
            config_dir=Path(graded_dir) if host else None, run_key=run_key,
        )
        without = _run_behavioral(
            args, skills, gradable, install=False,
            config_dir=Path(bare_dir) if host else None, run_key=run_key,
        )

    comparisons = staleness.pair(with_skill, without)
    summary = staleness.summarize(
        comparisons,
        {
            "model": args.model,
            "engine": args.engine,
            "effort": args.effort,
            "skills": skills,
            "extended": args.extended,
            "attempts": args.attempts,
            # Recorded only when it bound. A `0` in the report would read as a
            # cap of nothing rather than as no cap, and `render_markdown`
            # warns about a partial run on the presence of this key.
            **({"max_cases": args.max_cases} if args.max_cases else {}),
            **({"concurrency": args.concurrency} if args.concurrency else {}),
            "wall_time_s": round(time.time() - started, 1),
            "timeout": args.timeout,
            "github_run_id": os.environ.get("GITHUB_RUN_ID"),
            **usage.snapshot().as_meta(),
            **_sandbox_meta(args),
        },
    )
    _write_report(summary, staleness.render_markdown(summary), args, "staleness")
    if (code := _fail_if_expired()) is not None:
        return code
    return 0


# --------------------------------------------------------------------------
# Argument parsing
# --------------------------------------------------------------------------


def _add_skills_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--skills-dir",
        default="",
        metavar="GLOB[,GLOB]",
        help=(
            "Globs naming the directories that are skills, relative to the "
            "repo root: 'skills/*' for a repo that keeps them together, or a "
            "path to a single one. Default: every directory in the one this "
            "command was run from."
        ),
    )


def _add_docs_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--docs",
        default="",
        metavar="GLOB[,GLOB]",
        help=(
            "Markdown outside the skills to check references in as well, such "
            "as a repo's README or docs tree. Default: the skills alone."
        ),
    )


def _add_routing_room_argument(
    parser: argparse.ArgumentParser, *, skip_allowed: bool = True
) -> None:
    if skip_allowed:
        help_text = (
            "Skills to install side by side for the routing run: a list, `all` "
            "for every skill with a dataset, or `none` to skip routing. Left "
            "out, a repo with one skill runs that skill and a repo with "
            "several stops -- who a skill competes against is what its routing "
            "score means, so there is nothing sensible to assume."
        )
    else:
        help_text = (
            "Skills to install side by side: a list, or `all` for every skill "
            "with a dataset. Left out, --skill is the room if it was given; "
            "otherwise a repo with one skill runs that skill and a repo with "
            "several stops -- who a skill competes against is what its routing "
            "score means, so there is nothing sensible to assume."
        )
    parser.add_argument(
        "--routing-room",
        default="",
        metavar="A,B,C",
        help=help_text,
    )


def _add_graded_arguments(
    parser: argparse.ArgumentParser, *, timeout: float = deadline.DEFAULT_TIMEOUT_S
) -> None:
    """Flags shared by the graded commands: routing, behavioral and staleness.

    `timeout` is threaded through rather than overridden afterwards so the
    `--help` text matches the default in force. See `_add_timeout_argument`.
    """
    _add_skills_argument(parser)
    parser.add_argument(
        "--skill",
        default="",
        help="Comma-separated skill names. Default: every skill with a dataset.",
    )
    parser.add_argument("--only", default="", help="Comma-separated case ids to run.")
    parser.add_argument(
        "--extended",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Also run each skill's optional evals/extended_evals.json, when it "
            "has one. On by default; `--no-extended` grades the required "
            "evals.json alone."
        ),
    )
    parser.add_argument(
        "--model",
        default=None,
        help=(
            "Model to run. Each harness has its own default and its own names "
            "(`opus`, `sonnet` and `haiku` are understood by all of them). CI "
            "pins each harness to one model. Default: opus for claude-code."
        ),
    )
    parser.add_argument(
        "--effort",
        default="high",
        choices=["low", "medium", "high", "max"],
        help="Reasoning effort. Default: high.",
    )
    parser.add_argument(
        "--output",
        default="",
        help=(
            "Write the JSON report here. Default: "
            ".skillscope/runs/<command>-<timestamp>.json."
        ),
    )
    parser.add_argument(
        "--summary",
        default="",
        help="Write the markdown report here (defaults to $GITHUB_STEP_SUMMARY when set).",
    )
    parser.add_argument(
        "--skip-preflight", action="store_true", help="Skip the API reachability check."
    )
    parser.add_argument(
        "--engine",
        default=None,
        choices=[*NEUTRAL_ENGINES, *ENGINES],
        help=(
            "Where the agent runs, under inspect_ai. `host` runs it on the "
            "host, on any platform. `sandbox` runs it inside a container via "
            "inspect_swe, which needs a POSIX guest and so is Linux only, and "
            "additionally needs `pip install 'skillscope[verify]'`; only a "
            "harness with a sandboxed leg can use it. `claude-code-no-sandbox` "
            "and `claude-code` are the older names for those two and mean "
            "Claude Code. Default: host, or $SKILLSCOPE_ENGINE."
        ),
    )
    parser.add_argument(
        "--harness",
        default=None,
        metavar="NAME",
        help=(
            "The agent that runs the cases: "
            + ", ".join(harnesses.names())
            + ". Each is reported as its own result; harnesses send different "
            "prompts and offer different tools, so their scores are not "
            "comparable. Default: claude-code, or $SKILLSCOPE_HARNESS."
        ),
    )
    _add_timeout_argument(parser, timeout)


def _add_timeout_argument(
    parser: argparse.ArgumentParser, default: float = deadline.DEFAULT_TIMEOUT_S
) -> None:
    """The whole-command deadline, with a help string that knows its default.

    `default` is a parameter rather than a `set_defaults` call afterwards
    because the help text is built here and argparse does not revisit it.
    `staleness` overrode the value and left the help saying 900, three lines
    under a description announcing that it defaults to twice the usual -- a
    screen of `--help` contradicting itself.
    """
    parser.add_argument(
        "--timeout",
        type=float,
        default=default,
        help=(
            "Seconds before the whole command is abandoned. Shared by "
            "structural, routing, behavioral and staleness. "
            f"Default: {default:g} ({default / 60:g} minutes). 0 disables."
        ),
    )


def _add_routing_arguments(parser: argparse.ArgumentParser) -> None:
    _add_graded_arguments(parser)
    _add_routing_room_argument(parser, skip_allowed=False)
    parser.add_argument(
        "--jobs",
        type=int,
        default=4,
        help="Routing cases to run concurrently. Default: 4.",
    )
    parser.add_argument(
        "--case-timeout",
        type=float,
        default=240.0,
        help=(
            "Seconds before a routing case is abandoned. Generous because it "
            "only bites when the agent neither activates a skill nor answers. "
            "Clipped to whatever --timeout has left. Default: 240."
        ),
    )
    parser.add_argument(
        "--max-tool-calls",
        type=int,
        default=4,
        help=(
            "Stop a routing case after this many non-skill tool calls. Agents "
            "often look around before invoking a skill, so this is not 1; it is "
            "small enough to cut the run off long before real work starts. "
            "Default: 4."
        ),
    )
    parser.add_argument(
        "--max-inspection-calls",
        type=int,
        default=8,
        help=(
            "Separate allowance for tool calls that only read the installed "
            "skills tree. Surveying what is installed is part of the routing "
            "decision, so it does not spend --max-tool-calls, but it is capped "
            "so a run cannot idle to timeout. Default: 8."
        ),
    )
    parser.add_argument(
        "--max-budget-usd",
        type=float,
        default=0.75,
        help=(
            "Per-case routing spend cap. 0 disables. Enforced by the CLI's own "
            "--max-budget-usd on claude-code-no-sandbox, and by inspect's "
            "per-sample cost_limit on claude-code, which binds only where "
            "inspect can price the model -- the report's "
            "`max_budget_can_bind` says which held. Default: 0.75."
        ),
    )
    parser.add_argument(
        "--keep-logs",
        default="",
        help=(
            "Directory for this run's inspect transcripts (`.eval` logs, one "
            "per task, readable with `inspect view`). Defaults to "
            ".skillscope/logs."
        ),
    )
    parser.add_argument(
        "--min-accuracy",
        type=float,
        default=1.0,
        help=(
            "Exit non-zero when routing accuracy falls below this (0-1). "
            "Default: 1 (every graded case has to be right). 0 reports the "
            "score without gating on it."
        ),
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="skillscope",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--repo",
        default="",
        metavar="PATH",
        help=(
            "Root of the repo to test. Default: $SKILLSCOPE_REPO, else the "
            "nearest enclosing git checkout."
        ),
    )
    commands = parser.add_subparsers(dest="command", required=True)

    structural_parser = commands.add_parser(
        "structural",
        help="Check every skill's structure, and exit.",
        description=(
            "The skill folders, their datasets, and the references their "
            "markdown makes. Relative paths and heading anchors are always "
            "checked; external URLs are fetched only with --external."
        ),
    )
    _add_skills_argument(structural_parser)
    _add_docs_argument(structural_parser)
    structural_parser.add_argument(
        "--skill-files",
        default="",
        metavar="PATH[,PATH]",
        help=(
            "Files every skill must ship beside its SKILL.md, relative to the "
            "skill folder. For whatever this repo requires of a skill on top "
            "of the format, such as a governance card. Default: none."
        ),
    )
    structural_parser.add_argument(
        "--skill-sections",
        default="",
        metavar="TITLE[,TITLE]",
        help=(
            "`##` headings each markdown file named by --skill-files must have "
            "something under, such as Description,Owner,License. Default: none."
        ),
    )
    structural_parser.add_argument(
        "--external",
        action="store_true",
        help=(
            "Also fetch every external URL. Off by default: it needs the "
            "network and fails for reasons unrelated to the change, so it "
            "belongs somewhere it cannot block a merge."
        ),
    )
    structural_parser.add_argument(
        "--exclude-url",
        dest="excluded_urls",
        default="",
        metavar="REGEX[,REGEX]",
        help=(
            "URLs --external leaves alone, as regexes. For hosts that are "
            "auth-gated or that answer a runner's IP with a 403. Keep them "
            "narrow, so real link rot keeps being caught."
        ),
    )
    structural_parser.add_argument(
        "--jobs",
        type=int,
        default=references.DEFAULT_JOBS,
        help=(
            "Hosts to fetch from concurrently under --external. One request "
            "at a time goes to any one host, whatever this is set to. "
            f"Default: {references.DEFAULT_JOBS}."
        ),
    )
    _add_timeout_argument(structural_parser)
    structural_parser.set_defaults(handler=cmd_structural)

    routing_parser = commands.add_parser(
        "routing",
        help="Grade which skill fires, with several installed together.",
        description=(
            "Install the skills named by --routing-room side by side and "
            "grade the trigger decision for every evaluation those skills own. "
            "A repo with one skill need not name it; a repo with several must, "
            "because who is in the room is what the score means."
        ),
    )
    _add_routing_arguments(routing_parser)
    routing_parser.set_defaults(handler=cmd_routing)

    behavioral_parser = commands.add_parser(
        "behavioral",
        help="Grade what a skill does once it has fired.",
        description=(
            "Install one skill, run the prompt to completion, and grade "
            "expected_behavior / unexpected_behavior / logs_contain / "
            "files_exist. Only evaluations that assert something beyond the "
            "trigger decision run."
        ),
    )
    _add_graded_arguments(behavioral_parser)
    behavioral_parser.set_defaults(handler=cmd_behavioral)

    staleness_parser = commands.add_parser(
        "staleness",
        help="Whether a skill's cases still need the skill.",
        description=(
            "Run every behavioral case twice -- once with the skill installed, "
            "once with nothing installed -- and report which cases moved. A "
            "case that passes either way is not evidence about the skill. "
            "Reports only: a verdict here never fails a run, though an "
            "expired deadline or a refused run still exits non-zero. It costs "
            "twice what `behavioral` costs and wants more than twice the "
            "--timeout, which is why --timeout defaults to twice the usual "
            "here: both passes share one bound, and the baseline is the "
            "slower half because an agent without the skill explores instead "
            "of going straight to the work."
        ),
    )
    # Twice the shared bound, because the command is two passes under one.
    # Passed into `_add_graded_arguments` rather than set afterwards, so the
    # `--help` text is built from the number actually in force: a
    # `set_defaults` override changes the value and leaves the help saying
    # whatever it was built with. The measurement behind the number is in
    # `cmd_staleness`: a two-case skill finished the graded pass in 2m23s and
    # had not finished the baseline after 30 minutes, so the shared 900s gave
    # the slower half ~637s of a measured 1800s+ need.
    _add_graded_arguments(staleness_parser, timeout=2 * deadline.DEFAULT_TIMEOUT_S)
    staleness_parser.add_argument(
        "--attempts",
        type=int,
        default=1,
        metavar="N",
        help=(
            "Run every case N times on each side instead of once. One pair of "
            "runs is a lead rather than a result -- an agent disagrees with "
            "itself between runs, so a case can pass with the skill and fail "
            "without it for no reason the skill is responsible for. Repeats "
            "are what let a reader weigh a verdict: the report prints how "
            "many attempts passed on each side, and 3/3 against 0/3 is not "
            "3/3 against 1/3. A side counts as having passed when any of its "
            "attempts did, which makes a higher N a higher bar rather than a "
            "steadier reading of the same one: earns_it at three attempts "
            "needs the baseline to have failed all three times, and a "
            "baseline that passes once takes the case to no_signal. So "
            "verdicts compare across runs at equal N and not otherwise. "
            "Costs N times what it costs now, on both sides. Default: 1."
        ),
    )
    staleness_parser.add_argument(
        "--max-cases",
        type=int,
        default=0,
        metavar="N",
        help=(
            "Grade only the first N cases of each selected skill, after "
            "--only and --extended have chosen them. For trying the command "
            "on a large suite without paying for all of it. Both sides run "
            "the same N, and the report says the run did not cover the whole "
            "suite. Default: 0, every case."
        ),
    )
    staleness_parser.add_argument(
        "--concurrency",
        type=int,
        default=0,
        metavar="N",
        help=(
            "How many cases may be in flight at once. This command's baseline "
            "half is the slow one -- an agent with no skill explores -- and "
            "running its cases one after another is where the wall clock "
            "goes. Raise it as far as the engine's sandboxes and the model's "
            "rate limits will take. Default: 0, inspect's own default."
        ),
    )
    staleness_parser.add_argument(
        "--from-logs",
        metavar="DIR",
        help=(
            "Rebuild the comparison from logs already on disk instead of "
            "running anything. Both halves carry the skill, which side they "
            "were, and the run that wrote them in their eval metadata, so a "
            "log directory is enough to pair them -- and to tell this run's "
            "halves from a neighbouring run of the same skill. Costs nothing "
            "and reaches no model. Logs carrying no run id are passed over; "
            "see --unlabelled-logs."
        ),
    )
    staleness_parser.add_argument(
        "--unlabelled-logs",
        action="store_true",
        help=(
            "With --from-logs, also read logs that carry no run id, which "
            "means logs written before this command stamped one. Off by "
            "default: an ordinary `behavioral` log is shaped exactly like a "
            "pre-id graded half, so reading them unasked inflates the counts "
            "with skills the run never covered, and can hand a comparison a "
            "'with skill' half from a different run. Passing this asserts the "
            "directory holds nothing else for these skills. A group still "
            "needs a baseline side, and a side holding several logs is "
            "reported `ungraded` rather than guessed at."
        ),
    )
    staleness_parser.add_argument(
        "--run-id",
        default="",
        metavar="ID",
        help=(
            "With --from-logs, rebuild a named run rather than the most "
            "recent. A log directory accumulates every staleness run done in "
            "it, and a rebuild reads one per skill: reading all of them would "
            "report the same case once per run and count it that many times "
            "in the totals. The report names the run it read and lists the "
            "ids it passed over, which is where this value comes from. An id "
            "belongs to the invocation that minted it and covers whichever "
            "skills that invocation was asked for, so naming one can leave a "
            "skill in the directory unreported; when it does, the command "
            "says which."
        ),
    )
    staleness_parser.set_defaults(handler=cmd_staleness)

    select_parser = commands.add_parser(
        "select",
        help="Emit the CI plan for a change, as JSON.",
        description=select_module.__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    _add_skills_argument(select_parser)
    _add_routing_room_argument(select_parser)
    mode = select_parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--all", action="store_true", help="Every skill with a dataset.")
    mode.add_argument("--changed", action="store_true", help="Read changed paths from stdin.")
    mode.add_argument(
        "--since",
        nargs=2,
        metavar=("BASE", "HEAD"),
        help=(
            "The two commits a pull request names. Changed paths are worked "
            "out from their merge base, so a branch is planned for what it "
            "changed rather than for how it differs from a base that has moved "
            "on. Both commits' history has to be in the checkout."
        ),
    )
    mode.add_argument(
        "--names", metavar="A,B,C", help="An explicit comma-separated skill list."
    )
    select_parser.add_argument(
        "--labels",
        default="",
        help="Comma-separated pull-request labels, used to satisfy runner gates.",
    )
    select_parser.add_argument(
        "--ignore-gates",
        action="store_true",
        help=(
            "Run gated skills without their label. For workflow_dispatch, which "
            "is already explicit human intent."
        ),
    )
    select_parser.add_argument(
        "--extended",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Count each skill's optional evals/extended_evals.json as a run "
            "input. On by default; must match the flag the runner is given."
        ),
    )
    select_parser.add_argument(
        "--infra-paths",
        default="",
        metavar="PATH[,PATH]",
        help=(
            "Files that change the harness rather than one skill; touching one "
            "re-runs every skill. The workflow that calls skillscope belongs "
            "here: it holds the routing set."
        ),
    )
    select_parser.add_argument(
        "--behavior-runner",
        default="",
        metavar="LABELS",
        help=(
            "`runs-on` labels for a behavioral leg, as a JSON array or a comma-"
            f"separated list. Default: {','.join(config.DEFAULT_BEHAVIOR_RUNNER)}."
        ),
    )
    select_parser.add_argument(
        "--behavior-os",
        default="",
        metavar="A,B",
        help=(
            "Platforms a skill runs on when its evals/machine.yml does not say. "
            f"Default: {','.join(config.DEFAULT_BEHAVIOR_OS)}."
        ),
    )
    select_parser.add_argument(
        "--scoped-runner",
        default="",
        metavar="LABELS",
        help=(
            "Base `runs-on` labels for a leg whose skill asks for extra labels "
            "in its evals/machine.yml. Defaults to --behavior-runner, which is "
            "right only for a repo with one pool."
        ),
    )
    select_parser.add_argument(
        "--scoped-gate",
        default="",
        metavar="LABEL",
        help=(
            "Pull-request label required before a leg on the scoped pool runs. "
            "For hardware too scarce to spend on every pull request; without "
            "it, those legs run like any other."
        ),
    )
    select_parser.add_argument(
        "--scoped-environment",
        default="",
        metavar="NAME",
        help=(
            "GitHub environment holding the credentials for scoped legs. Given "
            "one, those legs are planned as a separate matrix, because a job's "
            "credentials are fixed before its matrix expands."
        ),
    )
    select_parser.set_defaults(handler=cmd_select)

    list_parser = commands.add_parser(
        "list-skills", help="Print skills that have a dataset, as JSON."
    )
    _add_skills_argument(list_parser)
    list_parser.set_defaults(handler=cmd_list_skills)

    template_parser = commands.add_parser(
        "template", help="Print the dataset template a new skill starts from."
    )
    template_parser.set_defaults(handler=cmd_template)

    return parser


def _configure(args: argparse.Namespace) -> None:
    """Make the flags this subcommand was given the active config.

    Built twice for a subcommand that takes ``--routing-room``, because two
    of the answers that flag accepts -- ``all``, and the single skill a repo
    with one of them never had to name -- are questions about which skills
    ship a dataset, and that is itself answered through the config. The first
    pass is what makes the repo readable, the second records the answer.

    A blank room with ``--skill`` set is filled from ``--skill`` before that
    second pass: naming the skills to grade also names who they sit with,
    and an explicit ``--routing-room`` still wins.
    """
    root = Path(args.repo).expanduser() if args.repo else None
    settings = {
        name: getattr(args, name, None)
        for name in (
            "skills_dir",
            "routing_room",
            "infra_paths",
            "docs",
            "excluded_urls",
            "skill_files",
            "skill_sections",
            "behavior_runner",
            "behavior_os",
            "scoped_runner",
        )
    }
    settings["scoped_gate"] = getattr(args, "scoped_gate", "") or ""
    settings["scoped_environment"] = getattr(args, "scoped_environment", "") or ""

    config.use(config.build(root, **settings))
    if settings["routing_room"] is not None:
        if not str(settings["routing_room"]).strip():
            skill = getattr(args, "skill", "") or ""
            if skill.strip():
                settings["routing_room"] = skill
        config.use(
            config.build(root, **settings, dataset_skills=datasets.skills_with_datasets())
        )


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    _configure(args)
    bound = None
    timeout = getattr(args, "timeout", None)
    if timeout is not None and timeout > 0:
        bound = deadline.Deadline(timeout, command=args.command)
        deadline.use(bound)
        bound.arm()
    try:
        return args.handler(args)
    except KeyboardInterrupt:
        # A Ctrl-C or the deadline's graceful stage that landed outside an
        # `eval()` -- between skills, or while a report was being written.
        # Inside one, inspect absorbs it: it unwinds through `Task.cleanup`
        # and returns, `run_eval` reads a stopped run's log back, and
        # `_fail_if_expired` reports the overrun beside the results. All that
        # is left here is to say which it was, without a traceback.
        if bound is not None and bound.interrupted:
            print(f"error: {bound.message()}", file=sys.stderr)
        else:
            print("error: interrupted", file=sys.stderr)
        return 1
    finally:
        if bound is not None:
            bound.disarm()
            deadline.use(None)


if __name__ == "__main__":
    raise SystemExit(main())
