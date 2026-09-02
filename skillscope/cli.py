# Copyright Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: MIT

"""The one entry point for skill evals.

Every skill owns a single dataset at ``<skill>/evals/evals.json``. This runner
reads those datasets and grades them with three commands:

  * ``structural`` -- every skill folder, every dataset, and every reference
    the skill's markdown makes. No agent, no tokens, instant.
  * ``routing`` -- installs the skills named by ``--routing-skills`` side by
    side and checks that the right one fires (and that nothing fires when
    nothing should). Cheap, no hardware, and it pools those skills' prompts so
    each one's positives are the others' negatives. A repo with a single skill
    need not name it: there is only one room its skill can be in.
  * ``behavioral`` -- installs one skill, runs the prompt to completion, and
    grades ``expected_behavior`` / ``unexpected_behavior`` / ``logs_contain``
    / ``files_exist``. Only runs evaluations that assert something beyond the
    routing decision.

A skill may also ship ``<skill>/evals/extended_evals.json``, in the same format
and under no coverage requirement of its own. Both graded commands include it
by default; ``--no-extended`` grades the required dataset alone.

The prompt is written once and both graded commands read it, which is the whole
point: the alternative is a central routing prompt set plus a separate
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
    skillscope routing --routing-skills local-ai-use,tracelens --no-extended
    skillscope behavioral --skill local-ai-use --no-extended

    # a routing run that reports its score instead of gating on it
    skillscope routing --routing-skills all --min-accuracy 0

    # a repo with one skill: the room is that skill, so nothing names it
    skillscope routing

    # one case, keeping the raw transcript
    skillscope routing --only qwen-on-mi300x --keep-logs eval-logs

    # what CI should run for a change
    git diff --name-only BASE HEAD | skillscope select --changed

Reports go to stdout as markdown, to ``$GITHUB_STEP_SUMMARY`` under Actions,
and to a JSON artifact under ``.skillscope/runs/`` in the repo under test.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from . import behavior, config, datasets, deadline, references, routing, structure
from . import select as select_module
from .agent import check_api_reachable, enforce_model_policy

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
        changed = {
            line.strip().replace("\\", "/")
            for line in sys.stdin.read().splitlines()
            if line.strip()
        }
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

    ``--routing-skills none`` on this command is a contradiction: the command
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
    if config.wants_no_skills(args.routing_skills):
        raise SystemExit(
            "error: `routing` asks for a routing run and "
            "`--routing-skills none` empties the room. Name the skills that "
            "go in it, or skip this command."
        )

    available = ", ".join(datasets.skills_with_datasets()) or "(none)"
    raise SystemExit(
        "error: `routing` needs the skills that go in the room: "
        "`--routing-skills a,b` or `all` for every skill with a dataset. "
        "Only a repo with one skill gets a default, because who a skill "
        "competes against is what its routing score means. Skills with a "
        f"dataset here: {available}."
    )


def _fail_if_expired() -> int | None:
    """Non-zero when the command's ``--timeout`` has already elapsed."""
    bound = deadline.active()
    if bound is None or not bound.expired():
        return None
    print(f"error: {bound.message()}", file=sys.stderr)
    return 1


def _prepare_graded_run(
    args: argparse.Namespace, scope: list[str] | None = None
) -> list[str]:
    """Structural checks, model pin, and API reachability. Shared by both graders.

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
    args.model = enforce_model_policy(args.model) or args.model
    if not args.skip_preflight:
        ok, detail = check_api_reachable(args.model)
        if not ok:
            raise SystemExit(f"error: claude API not reachable -- {detail}")
    return selected


def cmd_routing(args: argparse.Namespace) -> int:
    # Who is in the room decides what the structural gate covers, so it is
    # settled before anything is checked or any token is spent.
    routing_set = config.active().routing_set
    if not routing_set:
        _empty_room(args)

    _prepare_graded_run(args, list(routing_set))
    started = time.time()

    if not args.routing_skills.strip():
        only = next(iter(routing_set))
        print(
            f"[routing] --routing-skills was not given, and {only} is the "
            "only skill here with a dataset, so it is the room."
        )

    # Pool the routing set's cases: skill Y's positives are skill X's
    # negatives, which is where most of the false-trigger coverage comes
    # from. --skill narrows what is *reported on*, not what is installed.
    cases = datasets.routing_cases(list(routing_set), extended=args.extended)
    if args.only:
        cases = datasets.filter_cases(cases, args.only)
    elif args.skill:
        cases = datasets.filter_cases(cases, args.skill)

    routing_config = routing.RoutingConfig(
        model=args.model,
        effort=args.effort,
        case_timeout=args.case_timeout,
        max_tool_calls=args.max_tool_calls,
        max_inspection_calls=args.max_inspection_calls,
        max_budget_usd=args.max_budget_usd,
        keep_logs=args.keep_logs,
        available_flags=routing.supported_flags(
            ["--no-session-persistence", "--max-budget-usd"]
        ),
        isolate_config=routing.can_isolate_config(),
    )
    if not routing_config.isolate_config:
        print(
            "[routing] warning: ANTHROPIC_API_KEY is not set, so the runner's "
            "own config dir is used and any user-level skill in it joins the "
            "room for every case. The report flags what was registered."
        )

    print(f"[routing] installed together: {', '.join(routing_set)}")
    print(f"[routing] {len(cases)} cases, model={args.model}, jobs={args.jobs}")
    if args.jobs > 1 and len(cases) > 1:
        with ThreadPoolExecutor(max_workers=args.jobs) as pool:
            outcomes = list(
                pool.map(lambda c: routing.run_case(c, routing_set, routing_config), cases)
            )
    else:
        outcomes = [routing.run_case(case, routing_set, routing_config) for case in cases]

    summary = routing.summarize(
        outcomes,
        list(routing_set),
        {
            "model": args.model,
            "effort": args.effort,
            "skills": list(routing_set),
            "extended": args.extended,
            "wall_time_s": round(time.time() - started, 1),
            "timeout": args.timeout,
            "case_timeout": args.case_timeout,
            "max_tool_calls": args.max_tool_calls,
            "max_inspection_calls": args.max_inspection_calls,
            "isolated_config_dir": routing_config.isolate_config,
            "max_budget_usd": args.max_budget_usd,
            "optional_cli_flags_used": sorted(routing_config.available_flags),
            "github_run_id": os.environ.get("GITHUB_RUN_ID"),
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


def cmd_behavioral(args: argparse.Namespace) -> int:
    skills = _prepare_graded_run(args)
    started = time.time()

    cases = []
    for skill in skills:
        cases.extend(datasets.load_dataset(skill, extended=args.extended))
    if args.only:
        cases = datasets.filter_cases(cases, args.only)
    gradable = [c for c in cases if c.has_behavior]

    if not gradable:
        print(
            "[behavioral] no evaluation in the selected skill(s) asserts "
            "anything beyond routing, so there is nothing to grade. Add "
            "`expected_behavior` / `unexpected_behavior` / `logs_contain` / "
            "`files_exist` to a triggering evaluation."
        )
        return 0

    outcomes = behavior.run(skills, gradable, args.model, args.effort)
    summary = behavior.summarize(
        outcomes,
        {
            "model": args.model,
            "effort": args.effort,
            "skills": skills,
            "extended": args.extended,
            "wall_time_s": round(time.time() - started, 1),
            "timeout": args.timeout,
            "github_run_id": os.environ.get("GITHUB_RUN_ID"),
        },
    )
    _write_report(summary, behavior.render_markdown(summary), args, "behavioral")
    if (code := _fail_if_expired()) is not None:
        return code
    if summary["totals"]["passed"] != summary["totals"]["cases"]:
        return 1
    return 0


# --------------------------------------------------------------------------
# Argument parsing
# --------------------------------------------------------------------------


def _add_skills_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--skills",
        default="",
        metavar="GLOB[,GLOB]",
        help=(
            "Globs naming the directories that hold skills. "
            f"Default: {','.join(config.DEFAULT_SKILL_GLOBS)}."
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


def _add_routing_skills_argument(
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
            "with a dataset. Left out, a repo with one skill runs that skill "
            "and a repo with several stops -- who a skill competes against is "
            "what its routing score means, so there is nothing sensible to "
            "assume."
        )
    parser.add_argument(
        "--routing-skills",
        default="",
        metavar="A,B,C",
        help=help_text,
    )


def _add_graded_arguments(parser: argparse.ArgumentParser) -> None:
    """Flags shared by the routing and behavioral commands."""
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
        "--model", default="opus", help="Model alias. CI pins this to opus. Default: opus."
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
    _add_timeout_argument(parser)


def _add_timeout_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--timeout",
        type=float,
        default=deadline.DEFAULT_TIMEOUT_S,
        help=(
            "Seconds before the whole command is abandoned. Shared by "
            "structural, routing, and behavioral. Default: 900 (15 minutes). "
            "0 disables."
        ),
    )


def _add_routing_arguments(parser: argparse.ArgumentParser) -> None:
    _add_graded_arguments(parser)
    _add_routing_skills_argument(parser, skip_allowed=False)
    parser.add_argument(
        "--jobs",
        type=int,
        default=4,
        help="Routing cases to run concurrently, each in its own workspace. Default: 4.",
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
        help="Per-case routing spend cap enforced by the CLI. 0 disables. Default: 0.75.",
    )
    parser.add_argument(
        "--keep-logs", default="", help="Directory for raw per-case stream-json transcripts."
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
            "Install the skills named by --routing-skills side by side and "
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

    select_parser = commands.add_parser(
        "select",
        help="Emit the CI plan for a change, as JSON.",
        description=select_module.__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    _add_skills_argument(select_parser)
    _add_routing_skills_argument(select_parser)
    mode = select_parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--all", action="store_true", help="Every skill with a dataset.")
    mode.add_argument("--changed", action="store_true", help="Read changed paths from stdin.")
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
            "here: it holds the routing set and the version pin."
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
    select_parser.add_argument(
        "--version",
        default=None,
        metavar="REF",
        help=(
            "The build of skillscope this run is, echoed into the plan so every "
            f"leg keeps using it. Default: ${config.VERSION_ENV}."
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

    Built twice for a subcommand that takes ``--routing-skills``, because two
    of the answers that flag accepts -- ``all``, and the single skill a repo
    with one of them never had to name -- are questions about which skills
    ship a dataset, and that is itself answered through the config. The first
    pass is what makes the repo readable, the second records the answer.
    """
    root = Path(args.repo).expanduser() if args.repo else None
    settings = {
        name: getattr(args, name, None)
        for name in (
            "skills",
            "routing_skills",
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
    settings["version"] = getattr(args, "version", None)

    config.use(config.build(root, **settings))
    if settings["routing_skills"] is not None:
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
    finally:
        if bound is not None:
            bound.disarm()
            deadline.use(None)


if __name__ == "__main__":
    raise SystemExit(main())
