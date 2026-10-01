# Copyright Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: MIT

"""Routing engine: does the right skill fire, and only then?

Behavioral asks "once this skill runs, does it do the job?". This asks the
question that comes first: **given several skills installed side by side, does
the agent pick the right one?** It grades the routing decision only, so it
catches the four failure modes a description can cause:

  * correct trigger -- the expected skill activated.
  * missed trigger  -- a skill was expected and none activated (under-triggering).
  * wrong skill     -- a skill activated, but not the expected one (two
                       descriptions overlap and the agent picked the wrong side).
  * false trigger   -- no skill was expected and one activated (over-triggering).

Which skills are in the room is the workflow's decision, passed in as
``--routing-room``. Wherever there is a choice it has to be a decision
someone makes deliberately, because that set is what the number means: a skill
tested alongside two neighbours is answering a harder question than one tested
alongside none. A repo with a single skill has no choice to make and so makes
none; it still gets the over- and under-triggering half of the answer, graded
against its own near misses and the shared negatives. Cases are pooled across
the room's datasets, so a positive case for skill Y is automatically a negative
for skill X and the confusion matrix fills itself in.

Cost control: each run is killed the moment the routing decision is observable
-- the first skill activation, the final result event, or a small budget of
tool calls that are neither bookkeeping nor a survey of the installed skills
-- so no case pays for the work the skill would have gone on to do.
``max_budget_usd`` is a second, independent backstop.

The CLI lives in ``skillscope/cli.py``; this module is the engine.
"""

from __future__ import annotations

import json
import os
import queue
import shutil
import signal
import subprocess
import tempfile
import threading
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path

from . import deadline, usage
from .agent import claude_env
from .datasets import Case

# Tools that carry no routing signal. An agent often opens with a todo list or
# a plan before deciding anything, and spending the non-skill tool budget on
# that would cut the run off before the real decision.
BOOKKEEPING_TOOLS = {"todowrite", "todoread", "exitplanmode"}

# Tool names Claude Code uses to activate a skill. `Skill` is current; older
# builds routed skills through the slash-command tool.
SKILL_TOOLS = {"skill", "slashcommand"}

# Where the staged skills live, as they appear in a tool argument.
STAGED_SKILLS_DIR = ".claude/skills"

# Signatures of a failure that belongs to the model provider rather than to the
# skill. Matched case-insensitively against whatever the run reported.
#
# Worth naming rather than leaving as prose: a gateway that answers 504 lands
# in a report as a lower score with nothing saying why, and a reader cannot
# tell it from the skill failing. At the rate these have been observed -- a
# third of runs on one catalogue -- an unmarked provider error is the single
# biggest reason two runs of the same engine disagree, which makes it the
# first thing to rule out before a difference between engines means anything.
PROVIDER_ERROR_SIGNS = (
    "api error",
    "overloaded",
    "rate limit",
    "429",
    "500",
    "502",
    "503",
    "504",
    "gateway",
    "upstream",
    "server-side",
    "connection error",
    "apiconnectionerror",
    "timed out",
    "timeout",
)


def is_provider_error(message: str | None) -> bool:
    """Whether this failure came from the provider rather than from the skill.

    Deliberately generous. A false positive marks a real skill failure as
    degraded, which makes a reader look twice at a run that was fine. A false
    negative lets a gateway outage score as a routing miss, which makes a
    reader trust a number that measured nothing. The costs are not symmetric.
    """
    if not message:
        return False
    lowered = str(message).lower()
    return any(sign in lowered for sign in PROVIDER_ERROR_SIGNS)


VERDICTS = ("correct_trigger", "true_negative", "missed_trigger", "wrong_skill", "false_trigger", "error")
PASSING_VERDICTS = {"correct_trigger", "true_negative"}

# Stop reasons that leave the routing decision unknown rather than observed.
INCONCLUSIVE_STOPS = {"completed", "timeout"}


@dataclass
class Outcome:
    id: str
    category: str
    skill: str | None
    prompt: str
    expect: str | None
    observed: str | None
    verdict: str
    passed: bool
    stop_reason: str
    elapsed_s: float
    tool_calls: int
    inspection_calls: int = 0
    visible_skills: list[str] = field(default_factory=list)
    extra_skills: list[str] = field(default_factory=list)
    error: str | None = None
    # Set when the failure was the provider's. Kept beside `error` rather than
    # folded into `verdict` so the verdict vocabulary stays about routing, and
    # so a reader can subtract these without re-parsing error strings.
    degraded: bool = False




def _capped_timeout(seconds: float) -> float:
    """``seconds``, or whatever the command's ``--timeout`` has left."""
    bound = deadline.active()
    return seconds if bound is None else bound.cap(seconds)


def supported_flags(flags: list[str]) -> set[str]:
    """Which of `flags` the installed `claude` build advertises in --help.

    The two cost-control flags this eval likes to pass are recent additions. An
    older CLI would reject them and every case would fail identically, which
    reads like a routing collapse rather than a flag problem -- so check once
    (free, no tokens) and drop what isn't there.
    """
    claude_bin = shutil.which("claude")
    if not claude_bin:
        return set()
    try:
        proc = subprocess.run(
            [claude_bin, "--help"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=_capped_timeout(60),
        )
    except (subprocess.SubprocessError, OSError):
        return set()
    text = (proc.stdout or "") + (proc.stderr or "")
    return {flag for flag in flags if flag in text}


# The credentials that live in the environment rather than in the CLI's own
# config dir. Either can be carried into a throwaway config dir; a login stored
# in the real one cannot, which is the whole distinction this turns on.
ENV_CREDENTIALS = ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN")


def can_isolate_config() -> bool:
    """Whether the runner's own ``~/.claude`` can be kept out of the session.

    User-level skills are registered next to the staged ones and change every
    routing decision, so the room has to hold exactly what was asked for.
    Pointing the CLI at a throwaway config dir achieves that, but only when
    auth comes from the environment -- if the login lives in the real config
    dir, hiding it means no case even starts.

    `ANTHROPIC_AUTH_TOKEN` counts for the same reason `ANTHROPIC_API_KEY`
    does: it is in the environment, so it survives the redirect. Testing only
    for the key refused every runner that authenticates by workload identity
    federation -- which holds no key at all, by design, and is what the
    reusable workflow offers downstream repos through `federation_rule_id`.
    Found by running the default engine the way a product repo would.
    """
    return any(os.environ.get(name, "").strip() for name in ENV_CREDENTIALS)


def _iter_tool_uses(obj) -> list[tuple[str, str]]:
    """Every (tool name, JSON-encoded tool input) pair nested anywhere in `obj`."""
    found: list[tuple[str, str]] = []

    def walk(node) -> None:
        if isinstance(node, dict):
            if node.get("type") == "tool_use":
                found.append(
                    (
                        str(node.get("name", "")),
                        json.dumps(node.get("input", {}), ensure_ascii=False),
                    )
                )
            for value in node.values():
                walk(value)
        elif isinstance(node, list):
            for value in node:
                walk(value)

    walk(obj)
    return found


def _match_skill(text: str, skills: list[str]) -> str | None:
    """Longest skill name mentioned in `text`, or None.

    Longest-first matters because one skill name can be a prefix of another
    (`local-ai-use` vs `local-ai-app-integration` share a stem today, and a
    future skill could nest outright).
    """
    lowered = text.lower()
    for skill in sorted(skills, key=len, reverse=True):
        if skill.lower() in lowered:
            return skill
    return None


def init_skills(event: dict, skills: list[str]) -> list[str] | None:
    """Skill names the CLI reported at session init, if this is that event.

    The room check: proof that the agent really saw the whole routing set. A
    case graded against a room it was never shown is not a routing result, and
    the failure looks exactly like a skill that failed to attract its prompt --
    `missed_trigger`, with nothing to say it was the installation rather than
    the description.

    `None` for any other event, so a caller can feed it the whole stream.
    """
    if event.get("type") != "system" or event.get("subtype") != "init":
        return None
    seen: list[str] = []
    for key in ("skills", "slash_commands", "slashCommands", "commands"):
        entries = event.get(key)
        if not isinstance(entries, list):
            continue
        for entry in entries:
            text = entry if isinstance(entry, str) else json.dumps(entry, ensure_ascii=False)
            hit = _match_skill(text, skills)
            if hit and hit not in seen:
                seen.append(hit)
    return seen


def init_extra_skills(event: dict, skills: list[str]) -> list[str] | None:
    """Skills the CLI reported at init that this eval did not install.

    The other half of the room check. A user-level skill on the runner --
    `~/.claude/skills` is the usual source -- is registered alongside the
    staged ones and competes for every prompt, so the numbers describe a room
    nobody asked for. Classifying an answer as `other:` notices such a skill
    only when it actually wins a case; this notices it being present at all,
    which is the difference between one odd result and a whole run measured in
    the wrong room.
    """
    if event.get("type") != "system" or event.get("subtype") != "init":
        return None
    entries = event.get("skills")
    if not isinstance(entries, list):
        return []
    known = {skill.lower() for skill in skills}
    extra: list[str] = []
    for entry in entries:
        if isinstance(entry, str):
            name = entry
        elif isinstance(entry, dict):
            name = str(entry.get("name") or "")
        else:
            continue
        name = name.strip().lstrip("/")
        if name and name.lower() not in known and name not in extra:
            extra.append(name)
    return extra


def _skill_from_body_path(text: str, skills: list[str]) -> str | None:
    """The skill whose own ``SKILL.md`` path appears in `text`, or None.

    Matching the joined ``skills/<name>/skill.md`` path -- never the bare
    filename, never the bare skill name -- is what separates "this skill's
    body was loaded" from "this text happens to mention the skill". Two or
    more matches mean the text enumerates the installed skills, which is a
    listing rather than a decision, so that is not an activation either.

    Reading one ``SKILL.md`` is only evidence of activation on a build that
    has no way to activate a skill except by reading it; see
    ``detect_activation``.
    """
    haystack = text.lower().replace("\\\\", "/").replace("\\", "/")
    hits = [skill for skill in skills if f"skills/{skill.lower()}/skill.md" in haystack]
    return hits[0] if len(hits) == 1 else None


def _is_skills_inspection(tool_input: str, skills: list[str]) -> bool:
    """True when a tool call is only looking at the installed skills tree.

    Surveying what is installed is part of making the routing decision, not
    the agent starting the work itself, so these calls must not spend the
    non-skill tool budget: ending a run mid-survey scored deliberation as a
    missed trigger.
    """
    haystack = tool_input.lower().replace("\\\\", "/").replace("\\", "/")
    if STAGED_SKILLS_DIR in haystack:
        return True
    return any(f"skills/{skill.lower()}/" in haystack for skill in skills)


def detect_activation(event: dict, skills: list[str], allow_body_path: bool = True) -> str | None:
    """The skill this event activates, or None.

    Only the agent's own tool calls count. Tool *results* and assistant prose
    are deliberately excluded: the staged workspace holds nothing but the
    skills tree, so any prompt that sends the agent looking for a file it
    cannot find gets a recursive listing of every ``SKILL.md`` back. Scoring
    that as an activation credited the longest installed skill name with
    a false trigger on unrelated prompts, and -- worse -- scored a correct
    trigger whenever an expected skill's prompt named a path that did not
    exist, hiding real misses behind the file hunt.

    ``allow_body_path`` carries the same distinction for tool *inputs*. On a
    build that exposes the ``Skill`` tool, an agent that opens a ``SKILL.md``
    is reading the installed skills to choose from them, so treating that as an
    activation just credits whichever skill the directory listing happened to
    put first. The path fallback therefore stays off unless the session has no
    skill tool at all, which is the only case it was written for.

    Returns ``"other:<name>"`` when a skill nobody installed fires -- that
    is a contaminated runner, not a routing result, and the report should say
    so rather than silently scoring it as a miss.
    """
    for name, tool_input in _iter_tool_uses(event):
        lowered = name.lower()
        if lowered in SKILL_TOOLS:
            hit = _match_skill(tool_input, skills)
            if hit:
                return hit
            try:
                parsed = json.loads(tool_input)
            except json.JSONDecodeError:
                parsed = {}
            invoked = ""
            for key in ("command", "skill", "name", "skill_name"):
                value = parsed.get(key) if isinstance(parsed, dict) else None
                if isinstance(value, str) and value.strip():
                    invoked = value.strip().lstrip("/")
                    break
            return f"other:{invoked or 'unknown'}"

        # Fallback for builds that load a skill body by reading the file
        # instead of going through the Skill tool. The call itself has to
        # target that skill's own SKILL.md; merely touching the skills
        # directory (`ls .claude/skills`) is not a routing decision.
        if allow_body_path:
            hit = _skill_from_body_path(tool_input, skills)
            if hit:
                return hit

    # Some builds announce an activation as a system event instead of a tool
    # call. Same joined-path rule, and init is excluded because it enumerates
    # every installed skill by design.
    if allow_body_path and event.get("type") == "system" and event.get("subtype") != "init":
        return _skill_from_body_path(json.dumps(event, ensure_ascii=False), skills)
    return None
















def classify(expect: str | None, observed: str | None) -> str:
    if observed is None:
        return "true_negative" if expect is None else "missed_trigger"
    if expect is None:
        return "false_trigger"
    return "correct_trigger" if observed == expect else "wrong_skill"




def near_a_limit(outcome: "Outcome", meta: dict) -> bool:
    """Whether this case stopped close enough to a cap to be decided by one.

    A case that used one call of a budget of four is measuring the agent. A
    case that used four is measuring the budget: ordinary run-to-run variation
    moves it across the line, and the verdict flips with it. Those two look
    identical in a report, and the difference is the whole of whether a flip
    between two runs means anything.

    Within one, because that is the resolution a single extra call has. Caps
    the run did not set are not caps: a leg that could not enforce a budget
    reports none, and nothing here should invent a threshold for it.
    """
    for used, cap in (
        (outcome.tool_calls, meta.get("max_tool_calls")),
        (outcome.inspection_calls, meta.get("max_inspection_calls")),
    ):
        if isinstance(cap, int) and cap > 0 and used >= cap - 1:
            return True
    return False


def summarize(outcomes: list[Outcome], skills: list[str], meta: dict) -> dict:
    verdicts = Counter(o.verdict for o in outcomes)
    graded = [o for o in outcomes if o.verdict != "error"]
    passed = [o for o in graded if o.passed]

    by_category: dict[str, dict] = {}
    for category in sorted({o.category for o in outcomes}):
        subset = [o for o in graded if o.category == category]
        hits = sum(1 for o in subset if o.passed)
        by_category[category] = {
            "graded": len(subset),
            "passed": hits,
            "accuracy": round(hits / len(subset), 3) if subset else None,
        }

    per_skill: dict[str, dict] = {}
    for skill in skills:
        expected = [o for o in graded if o.expect == skill]
        correct = sum(1 for o in expected if o.observed == skill)
        fired = [o for o in graded if o.observed == skill]
        false_fires = sum(1 for o in fired if o.expect != skill)
        per_skill[skill] = {
            "expected": len(expected),
            "correct": correct,
            "missed": sum(1 for o in expected if o.observed is None),
            "lost_to_other_skill": sum(
                1 for o in expected if o.observed is not None and o.observed != skill
            ),
            "fired_total": len(fired),
            "fired_when_not_expected": false_fires,
            "recall": round(correct / len(expected), 3) if expected else None,
            "precision": round(correct / len(fired), 3) if fired else None,
        }

    confusion: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for outcome in graded:
        confusion[outcome.expect or "(no skill)"][outcome.observed or "(no skill)"] += 1

    contaminated = sorted(
        {o.observed for o in outcomes if o.observed and o.observed.startswith("other:")}
    )
    # The room check, and whether it ran at all. `visible_skills` is populated
    # from the CLI's session-init event, which only the host leg can read:
    # `claude-code` drives the CLI through `inspect_swe` and never sees one. So
    # an empty list is ambiguous between "the agent reported no skills" and
    # "nobody could ask", and the two must not render the same way -- a report
    # with no missing-skill warning reads as a verified room.
    checked = [o for o in outcomes if o.visible_skills]
    missing = sorted(
        {skill for o in checked for skill in skills if skill not in o.visible_skills}
    )
    extras = sorted({skill for o in outcomes for skill in o.extra_skills})

    return {
        "meta": meta,
        "totals": {
            "cases": len(outcomes),
            "graded": len(graded),
            "passed": len(passed),
            "errors": verdicts.get("error", 0),
            "accuracy": round(len(passed) / len(graded), 3) if graded else None,
            # How many runs activated any skill at all. Zero across a set that
            # expects activations means the skills were never installed or the
            # activation detector no longer matches the CLI's output -- either
            # way the numbers are an artifact, not a result.
            "activations": sum(1 for o in graded if o.observed),
            "activations_expected": sum(1 for o in graded if o.expect),
            # Cases the provider failed rather than the skill. Reported beside
            # the score because it is the number that decides whether the
            # score can be read at all: a run with a third of its cases
            # degraded has measured the gateway, and comparing it against
            # another run attributes an outage to whatever changed in between.
            "degraded": sum(1 for o in outcomes if o.degraded),
            # Cases that stopped within one call of a cap. Not failures --
            # a flag on how much of this run measured the agent and how much
            # measured the budget. A flip on one of these between two runs is
            # a threshold artefact before it is anything else.
            "near_limit": sum(1 for o in outcomes if near_a_limit(o, meta)),
        },
        "verdicts": {name: verdicts.get(name, 0) for name in VERDICTS},
        "by_category": by_category,
        "per_skill": per_skill,
        "confusion": {k: dict(v) for k, v in confusion.items()},
        "unexpected_skills": contaminated,
        "skills_missing_from_session": missing,
        "extra_skills_in_session": extras,
        # How many cases the room check could actually be made for. Reported
        # rather than inferred, so "no missing skills" and "nobody looked" are
        # distinguishable in the JSON as well as in the markdown.
        "room_checked_cases": len(checked),
        "cases": [asdict(o) for o in outcomes],
    }


def render_markdown(summary: dict) -> str:
    totals = summary["totals"]
    verdicts = summary["verdicts"]
    meta = summary["meta"]
    accuracy = totals["accuracy"]
    lines = [
        "## Skill routing",
        "",
        f"**{totals['passed']}/{totals['graded']} correct "
        f"({'n/a' if accuracy is None else f'{accuracy:.1%}'})** across "
        f"{totals['cases']} prompts with {len(meta['skills'])} skills installed "
        f"together, on `{meta['model']}` (effort `{meta['effort']}`).",
        "",
        f"Installed together: {', '.join(f'`{s}`' for s in meta['skills'])}. "
        "Each skill's own prompts are graded against that room, so the score "
        "is only as meaningful as the room is realistic.",
        "",
    ]

    # Before the table, not after it. A reader who has already taken in the
    # score has formed a view, and a note underneath it does not undo that --
    # whereas a run with a tenth of its cases degraded is one whose score
    # should be read differently from the first glance.
    near = totals.get("near_limit", 0)
    if near:
        lines += [
            f"> **{near} of {totals['cases']} cases stopped within one call of "
            "a budget.** Those measured the budget as much as the agent: one "
            "more call either way moves them across the line and the verdict "
            "with them. Compare two runs on these last, and expect them to "
            "flip without meaning anything.",
            "",
        ]

    degraded = totals.get("degraded", 0)
    if degraded:
        share = degraded / totals["cases"] if totals["cases"] else 0
        lines += [
            f"> **{degraded} of {totals['cases']} cases failed at the model "
            f"provider, not in the skill** ({share:.0%}). A gateway error "
            "lands as a lower score with nothing in the verdict saying why, "
            "so treat this run as degraded rather than as a measurement: the "
            "difference between it and another run may be the provider's "
            "rather than the skill's or the engine's.",
            "",
        ]
    lines += [
        "| Verdict | Count | Meaning |",
        "| --- | --- | --- |",
        f"| correct_trigger | {verdicts['correct_trigger']} | expected skill activated |",
        f"| true_negative | {verdicts['true_negative']} | no skill expected, none activated |",
        f"| missed_trigger | {verdicts['missed_trigger']} | skill expected, nothing activated |",
        f"| wrong_skill | {verdicts['wrong_skill']} | a skill activated, but the wrong one |",
        f"| false_trigger | {verdicts['false_trigger']} | no skill expected, one activated |",
        f"| error | {verdicts['error']} | the run failed; excluded from accuracy |",
        "",
        "### By prompt category",
        "",
        "Categories are derived, not declared: `skill_should_trigger: true` is "
        "`positive`, `false` in a skill's own dataset is that skill's "
        "`near_miss`, and a prompt from the shared pool is `unrelated`.",
        "",
        "| Category | Graded | Correct | Accuracy |",
        "| --- | --- | --- | --- |",
    ]
    for category, stats in summary["by_category"].items():
        acc = stats["accuracy"]
        lines.append(
            f"| {category} | {stats['graded']} | {stats['passed']} | "
            f"{'n/a' if acc is None else f'{acc:.0%}'} |"
        )

    lines += [
        "",
        "### Per skill",
        "",
        "| Skill | Expected | Correct | Missed | Lost to another skill | Fired when not expected | Recall | Precision |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for skill, stats in summary["per_skill"].items():
        recall = stats["recall"]
        precision = stats["precision"]
        lines.append(
            f"| `{skill}` | {stats['expected']} | {stats['correct']} | {stats['missed']} | "
            f"{stats['lost_to_other_skill']} | {stats['fired_when_not_expected']} | "
            f"{'n/a' if recall is None else f'{recall:.0%}'} | "
            f"{'n/a' if precision is None else f'{precision:.0%}'} |"
        )

    # Errors get their own section: a crashed run says nothing about routing,
    # so listing it as a routing failure would be misleading.
    failures = [c for c in summary["cases"] if not c["passed"] and c["verdict"] != "error"]
    lines += ["", "### Routing failures", ""]
    if not failures:
        lines.append("None. Every graded prompt routed as expected.")
    else:
        lines += [
            "| Case | Category | Expected | Observed | Verdict | Prompt |",
            "| --- | --- | --- | --- | --- | --- |",
        ]
        for case in failures:
            prompt = case["prompt"].replace("|", "\\|")
            if len(prompt) > 110:
                prompt = prompt[:110] + "..."
            lines.append(
                f"| `{case['id']}` | {case['category']} | {case['expect'] or '(no skill)'} | "
                f"{case['observed'] or '(no skill)'} | {case['verdict']} | {prompt} |"
            )

    errored = [c for c in summary["cases"] if c["verdict"] == "error"]
    if errored:
        lines += [
            "",
            "### Errored cases (not graded)",
            "",
            "| Case | Stopped after | Error |",
            "| --- | --- | --- |",
        ]
        for case in errored:
            detail = (case["error"] or "unknown").replace("|", "\\|").replace("\n", " ")
            lines.append(f"| `{case['id']}` | {case['stop_reason']} | {detail[:160]} |")

    lines += [
        "",
        "<details><summary>All cases</summary>",
        "",
        "| Case | Expected | Observed | Verdict | Stopped after | Seconds |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for case in summary["cases"]:
        lines.append(
            f"| `{case['id']}` | {case['expect'] or '(no skill)'} | "
            f"{case['observed'] or '(no skill)'} | {case['verdict']} | "
            f"{case['stop_reason']} | {case['elapsed_s']} |"
        )
    lines += ["", "</details>"]

    if totals["activations"] == 0 and totals["activations_expected"]:
        lines += [
            "",
            "> **Not a valid result:** no skill activated in any case, including "
            f"the {totals['activations_expected']} that expected one. The skills "
            "were probably not installed for the session, or the activation "
            "detector no longer matches this `claude` build. Re-run with "
            "`--keep-logs` and inspect a transcript before trusting these numbers.",
        ]
    if summary["unexpected_skills"]:
        lines += [
            "",
            f"> **Warning:** a skill this run did not install activated "
            f"({', '.join(summary['unexpected_skills'])}). The runner has extra "
            f"skills installed, so these routing results are not trustworthy.",
        ]
    if summary["skills_missing_from_session"]:
        lines += [
            "",
            f"> **Warning:** the CLI did not report these installed skills at "
            f"session init: {', '.join(summary['skills_missing_from_session'])}. "
            f"They may not have been installed for the run.",
        ]
    if summary["extra_skills_in_session"]:
        extras = summary["extra_skills_in_session"]
        shown = ", ".join(f"`{name}`" for name in extras[:12])
        if len(extras) > 12:
            shown += f", and {len(extras) - 12} more"
        lines += [
            "",
            f"> **Warning:** {len(extras)} skill(s) beyond the routing set were "
            f"registered for these sessions ({shown}). They come from the "
            f"runner's own config (usually `~/.claude/skills`) and compete for "
            f"every prompt, so the room measured here is not the one that was "
            f"asked for. Set `ANTHROPIC_API_KEY` so the run can use an isolated "
            f"config dir, or remove them from the runner.",
        ]
    if not summary.get("room_checked_cases"):
        # Said once, and only when no case could be checked. Both warnings
        # above are silent on this leg whatever the room actually held, and a
        # silent warning reads as a passed check -- which is how a contaminated
        # runner would go unreported rather than unreportable.
        lines += [
            "",
            "> **Note:** the room was not verified. The check reads the skills "
            "the CLI announces at session init, which only "
            "`claude-code-no-sandbox` can see -- `claude-code` drives the CLI "
            "through `inspect_swe`, which does not surface that event. So the "
            "two warnings above cannot fire on this leg: absence of them is "
            "not evidence the agent saw the room this report describes.",
        ]
    return "\n".join(lines) + "\n"
