# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.

"""The one entry point for skill evals.

Every skill owns a single dataset at ``<skill>/evals/evals.json``. This runner
reads those datasets and grades them in two modes:

  * ``routing``  -- installs the skills named by ``--routing-skills`` side by
    side and checks that the right one fires (and that nothing fires when
    nothing should). Cheap, no hardware, and it pools those skills' prompts so
    each one's positives are the others' negatives.
  * ``behavior`` -- installs one skill, runs the prompt to completion, and
    grades ``expected_behavior`` / ``unexpected_behavior`` / ``logs_contain``
    / ``files_exist``. Only runs evaluations that assert something beyond the
    routing decision.

A skill may also ship ``<skill>/evals/extended_evals.json``, in the same format
and under no coverage requirement of its own. Both modes include it by default;
``--no-extended`` grades the required dataset alone.

The prompt is written once and both modes read it, which is the whole point:
the alternative is a central routing prompt set plus a separate per-skill test
file that re-asserts routing with a substring match on the transcript.

Usage::

    # structural checks only: no agent, no tokens, instant
    skillscope structural

    # everything a skill owner needs before opening a pull request
    skillscope run --skill serving-llms-on-epyc

    # what CI runs
    skillscope run --mode routing --routing-skills local-ai-use,tracelens \
        --min-accuracy 0.9 --no-extended
    skillscope run --mode behavior --skill local-ai-use --no-extended

    # one case, keeping the raw transcript
    skillscope run --only qwen-on-mi300x --keep-logs eval-logs

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

from . import behavior, config, datasets, routing
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
    # `--output` names one file, so it only applies when one mode is running;
    # under `--mode both` the second report would otherwise clobber the first.
    if args.output and args.mode != "both":
        output = Path(args.output)
    elif args.output:
        named = Path(args.output)
        output = named.with_name(f"{named.stem}-{label}{named.suffix}")
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


def _structural_or_exit() -> None:
    """Fail before any tokens are spent if a dataset is malformed."""
    errors = datasets.structural_errors()
    if errors:
        print("Structural checks failed:", file=sys.stderr)
        for err in errors:
            print(f"  - {err}", file=sys.stderr)
        raise SystemExit(1)


# --------------------------------------------------------------------------
# Commands
# --------------------------------------------------------------------------


def cmd_structural(args: argparse.Namespace) -> int:
    """Structural checks over every dataset. No agent, no tokens."""
    _structural_or_exit()
    skills = datasets.skills_with_datasets()
    # Extended datasets are checked regardless of --extended, so count them
    # here too rather than reporting fewer cases than were checked.
    cases = datasets.load_all_cases(extended=True)
    cfg = config.active()
    print(
        f"[evals] OK: {len(cases)} case(s) across {len(skills)} skill(s) "
        f"plus {len(datasets.load_shared_negatives())} shared negative(s)."
    )
    print(f"[evals] repo: {cfg.root}  skills: {', '.join(cfg.skill_globs)}")
    return 0


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


def cmd_run(args: argparse.Namespace) -> int:
    _structural_or_exit()

    args.model = enforce_model_policy(args.model) or args.model
    skills = _selected_skills(args.skill)

    if not args.skip_preflight:
        ok, detail = check_api_reachable(args.model)
        if not ok:
            raise SystemExit(f"error: claude API not reachable -- {detail}")

    failed = False
    started = time.time()

    if args.mode in ("routing", "both"):
        routing_set = config.active().routing_set
        if not routing_set:
            raise SystemExit(
                "error: routing mode needs the skills that go in the room: "
                "`--routing-skills a,b`, or `--routing-skills all` for every "
                "skill with a dataset. There is no default, because who a "
                "skill competes against is what its routing score means."
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
            timeout=args.timeout,
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
                "max_tool_calls": args.max_tool_calls,
                "max_inspection_calls": args.max_inspection_calls,
                "isolated_config_dir": routing_config.isolate_config,
                "max_budget_usd": args.max_budget_usd,
                "optional_cli_flags_used": sorted(routing_config.available_flags),
                "github_run_id": os.environ.get("GITHUB_RUN_ID"),
            },
        )
        _write_report(summary, routing.render_markdown(summary), args, "routing")

        totals = summary["totals"]
        if totals["graded"] == 0:
            print("[routing] every case errored; treating the run as a failure.", file=sys.stderr)
            failed = True
        elif totals["activations"] == 0 and totals["activations_expected"]:
            print(
                "[routing] no skill activated in any case -- the skills were not "
                "installed, or activation detection is broken. Failing rather than "
                "reporting a 0% routing rate as if it were real.",
                file=sys.stderr,
            )
            failed = True
        elif args.min_accuracy > 0 and (totals["accuracy"] or 0) < args.min_accuracy:
            print(
                f"[routing] accuracy {totals['accuracy']} is below the "
                f"--min-accuracy bar of {args.min_accuracy}.",
                file=sys.stderr,
            )
            failed = True

    if args.mode in ("behavior", "both"):
        cases = []
        for skill in skills:
            cases.extend(datasets.load_dataset(skill, extended=args.extended))
        if args.only:
            cases = datasets.filter_cases(cases, args.only)
        gradable = [c for c in cases if c.has_behavior]

        if not gradable:
            print(
                "[behavior] no evaluation in the selected skill(s) asserts "
                "anything beyond routing, so there is nothing to grade. Add "
                "`expected_behavior` / `unexpected_behavior` / `logs_contain` / "
                "`files_exist` to a triggering evaluation."
            )
        else:
            outcomes = behavior.run(skills, gradable, args.model, args.effort)
            summary = behavior.summarize(
                outcomes,
                {
                    "model": args.model,
                    "effort": args.effort,
                    "skills": skills,
                    "extended": args.extended,
                    "wall_time_s": round(time.time() - started, 1),
                    "github_run_id": os.environ.get("GITHUB_RUN_ID"),
                },
            )
            _write_report(summary, behavior.render_markdown(summary), args, "behavior")
            if summary["totals"]["passed"] != summary["totals"]["cases"]:
                failed = True

    return 1 if failed else 0


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


def _add_routing_skills_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--routing-skills",
        default="",
        metavar="A,B,C",
        help=(
            "Skills to install side by side for the routing run, or `all` for "
            "every skill with a dataset. Empty means no routing run at all: "
            "who a skill competes against is what its routing score means, so "
            "there is nothing sensible to assume."
        ),
    )


def _add_run_arguments(parser: argparse.ArgumentParser) -> None:
    _add_skills_argument(parser)
    _add_routing_skills_argument(parser)
    parser.add_argument(
        "--mode",
        default="both",
        choices=["routing", "behavior", "both"],
        help="Which grader to run. Default: both.",
    )
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
        "--jobs",
        type=int,
        default=4,
        help="Routing cases to run concurrently, each in its own workspace. Default: 4.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=240.0,
        help=(
            "Seconds before a routing case is abandoned. Generous because it "
            "only bites when the agent neither activates a skill nor answers. "
            "Default: 240."
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
        "--output",
        default="",
        help="Write the JSON report here. Default: .skillscope/runs/<mode>-<timestamp>.json.",
    )
    parser.add_argument(
        "--summary",
        default="",
        help="Write the markdown report here (defaults to $GITHUB_STEP_SUMMARY when set).",
    )
    parser.add_argument(
        "--keep-logs", default="", help="Directory for raw per-case stream-json transcripts."
    )
    parser.add_argument(
        "--min-accuracy",
        type=float,
        default=0.0,
        help="Exit non-zero when routing accuracy falls below this (0-1). Default: 0 (report only).",
    )
    parser.add_argument(
        "--skip-preflight", action="store_true", help="Skip the API reachability check."
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

    run_parser = commands.add_parser(
        "run", help="Run the evals.", description="Run routing and/or behavior evals."
    )
    _add_run_arguments(run_parser)
    run_parser.set_defaults(handler=cmd_run)

    structural_parser = commands.add_parser(
        "structural", help="Check every dataset structurally and exit."
    )
    _add_skills_argument(structural_parser)
    structural_parser.set_defaults(handler=cmd_structural)

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
            "`runs-on` labels for a behavior leg, as a JSON array or a comma-"
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

    Built twice when ``--routing-skills all`` is asked for, because "every
    skill with a dataset" is itself a question answered through the config:
    the first pass is what makes the repo readable, the second records the
    answer.
    """
    root = Path(args.repo).expanduser() if args.repo else None
    settings = {
        name: getattr(args, name, None)
        for name in (
            "skills",
            "routing_skills",
            "infra_paths",
            "behavior_runner",
            "behavior_os",
            "scoped_runner",
        )
    }
    settings["scoped_gate"] = getattr(args, "scoped_gate", "") or ""
    settings["scoped_environment"] = getattr(args, "scoped_environment", "") or ""
    settings["version"] = getattr(args, "version", None)

    config.use(config.build(root, **settings))
    if config.wants_all_skills(settings["routing_skills"]):
        config.use(
            config.build(root, **settings, all_skills=datasets.skills_with_datasets())
        )


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    _configure(args)
    return args.handler(args)


if __name__ == "__main__":
    raise SystemExit(main())
