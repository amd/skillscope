#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: MIT

"""Compare two engines on the same dataset.

Answers the two questions a migration has to answer before it can be trusted.

**Does the new engine agree?** Per case, not in aggregate: an accuracy figure
can match exactly while individual cases flip in both directions and cancel
out. Flips are reported by direction, and against a measured noise floor --
routing and behavioral are both nondeterministic, so "these two runs differ"
means nothing until you know how much one engine differs from itself.

**Does it pay for itself?** Wall clock and tokens per run, from the report
`meta` both engines now populate.

Runs through the `skillscope` CLI rather than importing either engine, so what
is measured is what CI executes.

    tools/benchmark_engines.py routing --routing-room my-skill --noise
    tools/benchmark_engines.py behavioral --skill my-skill
    tools/benchmark_engines.py --compare legacy.json candidate.json

Which pair is compared is an argument, because the question changes over the
migration. `legacy` against `claude-code-no-sandbox` asks the narrow, sharp question: both
drive the same CLI, so agreement says the framework around the agent is
faithful, and disagreement is a defect in the crossing rather than a property
of a different agent. `claude-code-no-sandbox` against `claude-code` asks the other one --
same agent, host against container, which is where a contaminated runner shows
up as a disagreement neither engine could find alone.

    tools/benchmark_engines.py behavioral --candidate claude-code-no-sandbox --skill my-skill
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Taken from the CLI rather than restated, so a new engine is offered here the
# moment it is offered there.
from skillscope.cli import ENGINES  # noqa: E402

AGREE = "agree"
NEW_PASSES = "only the candidate passes"
NEW_FAILS = "only the baseline passes"


def run_leg(leg: str, engine: str, passthrough: list[str], label: str) -> dict:
    """Run one leg on one engine and return its JSON report."""
    out = Path(tempfile.mkdtemp(prefix="benchmark-")) / f"{label}.json"
    cmd = [
        sys.executable, "-m", "skillscope", leg,
        "--engine", engine, "--output", str(out), *passthrough,
    ]
    print(f"[benchmark] {label}: {' '.join(cmd)}", flush=True)
    # A failing leg is a result, not an error: a run where cases fail still
    # produces the report this compares.
    subprocess.run(cmd, check=False)
    if not out.is_file():
        raise SystemExit(f"error: {label} produced no report at {out}")
    return json.loads(out.read_text(encoding="utf-8"))


def cases_by_id(report: dict) -> dict[str, dict]:
    return {str(case["id"]): case for case in report.get("cases", [])}


def compare(baseline: dict, candidate: dict) -> dict:
    """Per-case comparison of two reports of the same dataset."""
    left, right = cases_by_id(baseline), cases_by_id(candidate)
    shared = sorted(set(left) & set(right))

    rows = []
    for case_id in shared:
        a, b = left[case_id], right[case_id]
        if a["passed"] == b["passed"]:
            direction = AGREE
        else:
            direction = NEW_PASSES if b["passed"] else NEW_FAILS
        rows.append(
            {
                "id": case_id,
                "direction": direction,
                "baseline_passed": a["passed"],
                "candidate_passed": b["passed"],
                # Routing carries the decision itself, which says more than
                # pass/fail: two engines can both fail a case for different
                # reasons, and that is not agreement.
                "baseline_observed": a.get("observed"),
                "candidate_observed": b.get("observed"),
                "baseline_verdict": a.get("verdict"),
                "candidate_verdict": b.get("verdict"),
            }
        )

    agreed = sum(1 for r in rows if r["direction"] == AGREE)
    return {
        "compared": len(rows),
        "agreed": agreed,
        "agreement": round(agreed / len(rows), 4) if rows else None,
        "flips": [r for r in rows if r["direction"] != AGREE],
        "only_in_baseline": sorted(set(left) - set(right)),
        "only_in_candidate": sorted(set(right) - set(left)),
        "rows": rows,
    }


def spend(report: dict, engine: str = "legacy") -> dict:
    meta = report.get("meta", {})
    return {
        "engine": meta.get("engine", engine),
        "wall_time_s": meta.get("wall_time_s"),
        "model_calls": meta.get("model_calls"),
        "total_tokens": meta.get("total_tokens"),
        "cost_usd": meta.get("cost_usd"),
    }


def _cell(value) -> str:
    return "n/a" if value is None else str(value)


def _spend_caveats(spend: dict) -> list[str]:
    """Say which columns are comparable, because not all of them are.

    The two engines count different things and silently tabulating them side by
    side invites the wrong conclusion. Wall time is always comparable. Tokens
    are not: the legacy engine reads them from assistant events, which exclude
    the system prompt and cached input, and a routing case is killed before the
    totals arrive -- so its figure is a floor, not a total. Cost is the legacy
    engine's trustworthy number, and the inspect_ai-backed engines only have
    one when the provider supplies pricing, which a gateway generally does not.
    """
    engines = {spend[label]["engine"] for label in ("baseline", "candidate")}
    notes = []
    if "legacy" in engines:
        notes.append(
            "> Legacy token counts are a floor: they omit the system prompt and "
            "cached input, and a killed case never reports its totals. Compare "
            "cost and wall time, not tokens."
        )
    if any(spend[label]["cost_usd"] is None for label in ("baseline", "candidate")):
        notes.append(
            "> One engine reported no cost -- the inspect_ai-backed engines only "
            "have one "
            "when the model provider supplies pricing, which a gateway generally "
            "does not. Wall time and model calls are comparable on both sides; "
            "model calls in particular is the like-for-like measure of how "
            "much work each engine asks of the model per case."
        )
    return notes


def render(result: dict) -> str:
    comparison = result["comparison"]
    base = result["spend"]["baseline"]["engine"]
    cand = result["spend"]["candidate"]["engine"]
    lines = [
        "## Engine benchmark",
        "",
        f"**{comparison['agreed']}/{comparison['compared']} cases agree** "
        f"between the `{base}` and `{cand}` engines.",
        "",
    ]

    noise = result.get("noise")
    if noise is not None:
        lines += [
            f"Noise floor: the `{base}` engine agrees with itself on "
            f"{noise['agreed']}/{noise['compared']} cases. Treat any difference "
            "at or below that as run-to-run variance rather than engine drift.",
            "",
        ]
    else:
        lines += [
            "_No noise floor measured; re-run with `--noise` before reading the "
            "flips below as engine differences._",
            "",
        ]

    lines += ["| Run | Wall time | Model calls | Tokens | Cost |", "| --- | --- | --- | --- | --- |"]
    for label in ("baseline", "candidate"):
        s = result["spend"][label]
        lines.append(
            f"| {label} (`{s['engine']}`) | {_cell(s['wall_time_s'])}s | "
            f"{_cell(s['model_calls'])} | {_cell(s['total_tokens'])} | "
            f"{_cell(s['cost_usd'])} |"
        )
    lines += ["", *_spend_caveats(result["spend"])]

    lines += ["", "### Cases that flipped", ""]
    if not comparison["flips"]:
        lines.append("None. Every shared case reached the same verdict on both engines.")
    else:
        lines += [
            f"| Case | Direction | `{base}` | `{cand}` |",
            "| --- | --- | --- | --- |",
        ]
        for flip in comparison["flips"]:
            left = flip["baseline_verdict"] or ("pass" if flip["baseline_passed"] else "fail")
            right = flip["candidate_verdict"] or ("pass" if flip["candidate_passed"] else "fail")
            lines.append(f"| `{flip['id']}` | {flip['direction']} | {left} | {right} |")

    for key, heading in (
        ("only_in_baseline", f"Only the `{base}` run produced these cases"),
        ("only_in_candidate", f"Only the `{cand}` run produced these cases"),
    ):
        missing = comparison[key]
        if missing:
            lines += ["", f"### {heading}", "", ", ".join(f"`{m}`" for m in missing)]

    return "\n".join(lines) + "\n"


def refuse_unrunnable_pair(parser, args) -> None:
    """Stop before the first leg when the pair cannot produce a comparison.

    Checked up front because the cost is not symmetric: the baseline leg runs
    first, and with `--noise` it runs twice, so a candidate the leg will refuse
    is discovered only after a full routing run has been paid for. The refusal
    then surfaces as `produced no report`, which blames a missing file rather
    than naming the engine that was never going to run.

    Every engine has a routing leg now, so in practice this catches the same
    engine named twice, which measures run-to-run variance rather than a
    difference between engines. `--noise` already reports that, and reports it
    as what it is.
    """
    if args.leg != "routing":
        return

    runnable = set(cli_routing_engines())
    unrunnable = sorted({args.baseline, args.candidate} - runnable)
    if unrunnable:
        parser.error(
            f"routing has no leg for {', '.join(unrunnable)}. It runs on "
            f"{', '.join(sorted(runnable))}."
        )
    if args.baseline == args.candidate:
        parser.error(
            f"--baseline and --candidate are both {args.baseline!r}, which "
            "measures run-to-run variance rather than a difference between "
            "engines. That is what --noise already reports, and it labels the "
            "result as the noise floor rather than as a flip list."
        )


def cli_routing_engines() -> tuple[str, ...]:
    """The engines the CLI will actually run a routing leg on."""
    from skillscope.cli import ROUTING_ENGINES

    return ROUTING_ENGINES


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("leg", nargs="?", choices=["routing", "behavioral"])
    parser.add_argument(
        "--compare",
        nargs=2,
        metavar=("BASELINE", "CANDIDATE"),
        help="Compare two reports that already exist instead of running the legs.",
    )
    parser.add_argument(
        "--baseline",
        default="legacy",
        choices=ENGINES,
        help="The engine to measure against. Default: legacy.",
    )
    parser.add_argument(
        "--candidate",
        default="claude-code-no-sandbox",
        choices=ENGINES,
        help="The engine under test. Default: claude-code-no-sandbox.",
    )
    parser.add_argument(
        "--noise",
        action="store_true",
        help=(
            "Run the baseline engine twice to measure how much it disagrees "
            "with itself. Without this the flip list cannot be read as engine "
            "drift."
        ),
    )
    parser.add_argument("--output", default="", help="Write the JSON result here.")
    args, passthrough = parser.parse_known_args(argv)

    if args.compare:
        baseline = json.loads(Path(args.compare[0]).read_text(encoding="utf-8"))
        candidate = json.loads(Path(args.compare[1]).read_text(encoding="utf-8"))
        noise = None
    else:
        if not args.leg:
            parser.error("give a leg to run (routing or behavioral), or --compare")
        refuse_unrunnable_pair(parser, args)
        baseline = run_leg(args.leg, args.baseline, passthrough, args.baseline)
        noise_run = (
            run_leg(args.leg, args.baseline, passthrough, f"{args.baseline}-again")
            if args.noise
            else None
        )
        candidate = run_leg(args.leg, args.candidate, passthrough, args.candidate)
        noise = compare(baseline, noise_run) if noise_run is not None else None

    result = {
        "comparison": compare(baseline, candidate),
        "noise": noise,
        "spend": {
            "baseline": spend(baseline, getattr(args, "baseline", "legacy")),
            "candidate": spend(candidate, getattr(args, "candidate", "claude-code-no-sandbox")),
        },
    }

    report = render(result)
    print(report)
    if args.output:
        Path(args.output).write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(f"[benchmark] JSON result: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
