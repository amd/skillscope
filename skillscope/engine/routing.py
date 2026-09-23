# Copyright Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: MIT

"""Routing evals under `inspect_ai`, driving the real `claude` CLI.

`skillscope/routing.py` is the routing engine: it owns the vocabulary
(`Outcome`, `classify`, `detect_activation`), the report and the gate, and it
drives the CLI itself as a subprocess. This module gives the same question two
more legs, both of which reach the same CLI through `inspect_ai` instead:

  * `claude-code` -- `inspect_swe` runs the CLI *inside* the sandbox. Linux
    only, for the reasons `verify.require` spells out, and the only leg where
    the room is exactly the room that was asked for, because the guest has no
    `~/.claude` of its own to contribute to it.
  * `claude-code-no-sandbox` -- the CLI on the host, via the solver in
    `no_sandbox.py`. Runs anywhere, isolates nothing.

Everything downstream is untouched: this produces `routing.Outcome` objects, so
`routing.summarize`, `routing.render_markdown` and `cli.routing_gate` work as
they already do and a report from this leg is diffable against a legacy one.

**One detector, two shapes.** The two legs surface the agent's tool calls in
different formats. The host leg parses the CLI's stream-json and hands inspect
`ToolCall` objects; `inspect_swe` never produces stream-json at all -- the
bridged CLI's calls arrive as `ToolCall` objects on
`ChatMessageAssistant.tool_calls`, and there is no `ToolEvent` in the transcript
for a bridged scaffold to read instead. The tempting move is a second detector
that walks `ToolCall`s. That would be two answers to "did this skill activate?",
which drift, and the one in `skillscope/routing.py` is the one the legacy engine
and every recorded result were graded against. So the crossing happens in the
other direction: a `ToolCall` is re-wrapped into the one-line stream-json shape
`detect_activation` already reads (`_activation_event`), and the detector stays
the single answer. Converting data is cheap; keeping two graders agreeing is
not.

**Not yet: stopping at the decision.** The legacy engine kills the run the
moment a skill activates, because everything after that is work the routing
question does not ask for and does pay for. Without it a case here runs to
`ROUTING_MESSAGE_LIMIT`, which is the whole reason that constant is as tight as
it is.

Two different obstacles, and only one of them is permanent.

On `claude-code-no-sandbox` it cannot be done at all: the solver shells out
through `inspect_ai.util.subprocess`, which buffers the CLI's stdout until the
process exits, so the decision is only visible once the run is already over and
paid for. Recovering it means driving the CLI with a streaming reader, which is
what `routing.run_case` already does -- at which point inspect is supplying the
task and the log and not much else.

On `claude-code` it is available and not yet taken. An approval policy sees each
tool call the bridged CLI proposes *before* it runs, and returning `terminate`
ends the sample; that is an exact analogue of the legacy kill. What is not
established is whether the terminating call survives into the sample. Approval
runs inside `bridge_generate`, and the bridge adopts the assistant message into
the agent's state only after that call returns -- so a `terminate` on the very
tool call that reveals the decision may discard the observation it was triggered
by. That failure is silent and it is the worst shape available here: a
suppressed activation is indistinguishable from an agent that correctly declined
to route, and the run still reports a clean accuracy.

So it waits on one container run that answers whether the skill name is
recoverable after a terminate -- from the transcript, or failing that from the
`limit.reason` an approver can write. Until then the cost is bounded
declaratively and honestly, rather than optimised on an assumption.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

from .. import deadline, routing as routing_core
from ..datasets import Case
from . import behavioral, convert, models, no_sandbox, sandbox as sandbox_spec, stats

CLAUDE_CODE = "claude-code"
NO_SANDBOX = "claude-code-no-sandbox"

# The engines this module has a leg for. `legacy` is not one of them: it is the
# subprocess path in `skillscope/routing.py` and does not come through here.
ENGINES = (CLAUDE_CODE, NO_SANDBOX)

# Whether a tool call that merely opens a skill's own `SKILL.md` counts as an
# activation. Off, deliberately, and unlike the legacy engine -- which decides
# per session from the tool list in the CLI's `init` event, an event that does
# not survive the crossing into inspect messages on either leg.
#
# Both legs drive a current `claude` build, which activates a skill through the
# `Skill` tool; on such a build an agent that opens a `SKILL.md` is *reading the
# room to choose from it*, and scoring that as a decision credits whichever
# skill the directory listing happened to put first. The two ways to be wrong
# are not symmetric. Leaving it on invents activations that grade as
# `false_trigger` and `wrong_skill`, quietly, in a report that looks normal.
# Leaving it off on a build that has no `Skill` tool makes every case a
# `missed_trigger` -- which `cli.routing_gate` already refuses outright, loudly,
# as "no skill activated in any case". A failure the gate catches beats one it
# cannot see.
ALLOW_BODY_PATH = False

# How many messages a routing case may spend. Behavioral's 120 is sized for an
# agent doing a job; a routing case only has to reveal which skill it reaches
# for, and without the early stop described above every turn past that point is
# money spent on an answer nobody reads. The legacy budget is the reference
# point: 4 non-bookkeeping tool calls plus 8 inspections of the skills tree, so
# roughly a dozen calls and twice that in messages before it gives up waiting
# for a decision. This is that, rounded up -- generous enough that an agent
# which deliberates before choosing is not scored as one that never chose.
ROUTING_MESSAGE_LIMIT = 30


def require_isolated_room(engine: str) -> None:
    """Refuse a host-leg routing run that cannot keep the runner's room clean.

    The legacy engine warns here and carries on, because it can afford to: it
    reads the CLI's `init` event, so a user-level skill that joined the room is
    named in the report as an extra, and a reader can discount the run. Neither
    leg in this module gets that event -- it does not survive the crossing into
    inspect messages -- so the same contamination would be invisible.

    Invisible is the part that matters. A stray skill does not spoil one case's
    grade; it is offered for every prompt, so it changes every decision at
    once, and the run still reports a clean accuracy. Observed rather than
    feared: a probe of this leg on a developer machine put roughly forty
    user-level skills in the room and none of the three that were staged.

    So: isolate, or do not run. `claude-code` needs nothing here -- the guest
    has no `~/.claude` to contribute, which is the whole reason that leg is the
    one to prefer for routing.
    """
    if engine != NO_SANDBOX:
        return
    if routing_core.can_isolate_config():
        return
    raise SystemExit(
        "error: --engine claude-code-no-sandbox cannot run a routing leg "
        "without ANTHROPIC_API_KEY. Routing needs the room to hold exactly "
        "the skills that were asked for, and redirecting the CLI away from "
        "the runner's own config dir only works when auth comes from the "
        "environment. Without it every user-level skill on this machine joins "
        "the room for every case -- and unlike the legacy engine, this leg "
        "cannot see that happen or say so in the report.\n"
        "    Set ANTHROPIC_API_KEY, or use --engine claude-code, whose guest "
        "has no user-level skills at all."
    )


def install_room(skills: dict[str, Path], workspace: str) -> None:
    """Install every skill in the room into one workspace.

    `no_sandbox.install_skill` installs one, which is all a behavioral case
    needs. Routing needs all of them at once -- the whole question is which one
    the agent picks out of the set, and a room with a skill missing is a
    different, easier question that the report would describe as the one that
    was asked for.

    Plural lives here rather than beside the singular because the singular is
    the behavioral driver's, and a routing-shaped requirement has no business
    changing it.
    """
    for skill_dir in skills.values():
        no_sandbox.install_skill(skill_dir, workspace)


def _install_room_solver(skills: dict[str, Path]):
    """Solver that stages the whole room before the CLI is started.

    Chained *ahead* of `no_sandbox.claude_code_no_sandbox`, which stages one
    skill of its own and then runs the CLI in the same breath: by the time it
    starts there is nothing left to add. The overlap -- one skill installed
    twice, `copytree(dirs_exist_ok=True)` both times -- is deliberate and
    cheap. The alternative is teaching the behavioral driver about rooms, and
    staging is the driver's job precisely so that each driver can be wrong
    about only its own.
    """
    from inspect_ai.solver import solver

    @solver
    def _install():
        async def solve(state, generate):
            install_room(skills, await no_sandbox._workspace())
            return state

        return solve

    return _install()


def _sample_from_case(case: Case):
    """One inspect `Sample` per routing prompt.

    Not `convert.sample_from_case`: that carries a case's behavioral
    expectations and seeds its `workspace` fixture, and routing grades neither.
    The fixture especially -- the staged workspace holds the skills tree and
    nothing else, which is what the legacy engine has always done, because a
    file the agent can open is a file that can change the decision being
    measured. Two legs that seeded differently would be measuring two rooms.

    The prompt is used verbatim, again matching the legacy engine: routing
    prompts are read by the model for what they suggest, and `{}` expansion is
    a behavioral-case affordance.

    The metadata is for whoever opens the `.eval` transcript afterwards. Nothing
    grades from it -- the verdict is computed from the case in `_outcomes`,
    which is the object that knows what was expected.
    """
    from inspect_ai.dataset import Sample

    return Sample(
        id=case.id,
        input=case.prompt,
        metadata={
            convert.SKILL: case.skill,
            convert.SHOULD_TRIGGER: case.skill_should_trigger,
            convert.CATEGORY: case.category,
        },
    )


def _activation_event_from(function: str, arguments: dict) -> dict:
    """The stream-json shape `detect_activation` reads, from plain values.

    Split from `_activation_event` so the decision rule can be exercised
    without constructing an inspect `ToolCall`, which needs the extra the unit
    suite runs without.
    """
    return {
        "type": "assistant",
        "message": {
            "content": [
                {"type": "tool_use", "name": function, "input": arguments or {}}
            ]
        },
    }


def _activation_event(call) -> dict:
    """Re-wrap one inspect `ToolCall` as the stream-json event it would have been.

    The one place the two legs' shapes are reconciled. `detect_activation`
    takes a raw CLI event and walks it for `{"type": "tool_use", ...}` nodes;
    this builds the smallest event containing exactly one such node, so a
    bridged `ToolCall` is graded by the same code, with the same tool-name
    matching and the same `other:<name>` contamination check, as a line the CLI
    printed itself.
    """
    return _activation_event_from(call.function, call.arguments or {})


def _tool_calls(sample):
    """Every tool call the agent made, in the order it made them.

    Order is the whole point: the routing decision is the *first* skill the
    agent reaches for. Without the early stop this module does not have yet, a
    run continues past its decision and goes on to do the work, which can
    activate further skills -- grading the last one would report what the job
    needed rather than what the prompt routed to.
    """
    for message in getattr(sample, "messages", None) or []:
        for call in getattr(message, "tool_calls", None) or []:
            yield call


def _observe(sample, skills: list[str]) -> tuple[str | None, int, int]:
    """What this sample routed to, and what it spent getting there.

    Returns `(observed, tool_calls, inspection_calls)`. The two counters are
    the legacy engine's, computed with the legacy engine's own predicates, so
    the column means the same thing in both reports: surveying the installed
    skills is part of making the decision and is counted separately from the
    agent starting the work itself.
    """
    tool_calls = 0
    inspection_calls = 0
    for call in _tool_calls(sample):
        hit = routing_core.detect_activation(
            _activation_event(call), skills, allow_body_path=ALLOW_BODY_PATH
        )
        if hit:
            return hit, tool_calls, inspection_calls

        name = (getattr(call, "function", "") or "").lower()
        if name in routing_core.BOOKKEEPING_TOOLS:
            continue
        arguments = json.dumps(getattr(call, "arguments", None) or {}, ensure_ascii=False)
        if routing_core._is_skills_inspection(arguments, skills):
            inspection_calls += 1
        else:
            tool_calls += 1
    return None, tool_calls, inspection_calls


def _spoke(sample) -> bool:
    """Whether the agent produced anything at all in this sample.

    The distinction requirement 4 of the routing report rests on, and the one
    `cli.routing_gate` cannot make for itself. "No skill activated" is a
    finding -- a `true_negative` when nothing should have fired, a
    `missed_trigger` when something should have. "The agent never ran" is not a
    finding about routing at all, and grading it as a miss manufactures a
    routing result out of an infrastructure failure: a provider that refused,
    a sandbox that never came up, a CLI that printed nothing parseable. The
    legacy engine draws the same line with its `INCONCLUSIVE_STOPS` set.

    An assistant message is the cheapest honest evidence that the run got far
    enough to decide something. A sample with none of them said nothing, called
    nothing, and answered nothing.
    """
    return any(
        getattr(message, "role", None) == "assistant"
        for message in (getattr(sample, "messages", None) or [])
    )


def _error_outcome(case: Case, detail: str, stop_reason: str = "error") -> routing_core.Outcome:
    """A case that could not be graded, kept out of the accuracy it would skew."""
    return routing_core.Outcome(
        id=case.id,
        category=case.category,
        skill=case.skill,
        prompt=case.prompt,
        expect=case.expect_skill,
        observed=None,
        verdict="error",
        passed=False,
        stop_reason=stop_reason,
        elapsed_s=0.0,
        tool_calls=0,
        error=detail,
    )


def _limit_reason(sample) -> str | None:
    """The inspect limit this sample hit, if it hit one.

    Recorded in `stop_reason` rather than promoted to an error. An agent that
    burned `ROUTING_MESSAGE_LIMIT` messages without reaching for a skill has
    made its decision as surely as one that answered -- that is precisely the
    shape of an under-triggering skill, and the legacy engine grades its own
    `tool_budget` stop the same way. The reader still gets told which bound
    ended the run, because "missed, and also truncated" is worth knowing when a
    skill's recall looks worse here than on the legacy leg.
    """
    limit = getattr(sample, "limit", None)
    return f"limit:{getattr(limit, 'type', 'unknown')}" if limit is not None else None


def _outcomes(log, cases: list[Case], skills: list[str]) -> list[routing_core.Outcome]:
    """Map one inspect `EvalLog` back onto routing's outcome objects.

    Mirrors `behavioral._outcomes`, including its central rule: a task that
    failed outright reports one *errored* outcome per case rather than an empty
    list. Empty would render as a run with nothing wrong with it, and for
    routing that is worse than for behavioral -- `summarize` divides by the
    graded count, so a task that produced no samples would report an accuracy
    of `n/a` beside a clean-looking verdict table.

    It diverges on one point. `behavioral` discards a failed task's completed
    samples, which costs one skill's results; routing is a single task for the
    whole room, so the same rule would throw away every case the run already
    paid for because the last one raised -- observed on a task interrupted at
    sample three of three. So the samples that exist are graded whatever the
    task's status, and only the cases with no sample at all are errored.
    """
    by_id = {case.id: case for case in cases}
    failure = getattr(getattr(log, "error", None), "message", None)

    if not log.samples:
        detail = failure or "the task produced no samples"
        return [_error_outcome(case, f"inspect task failed: {detail}") for case in cases]

    outcomes: list[routing_core.Outcome] = []
    for sample in log.samples:
        case = by_id.get(str(sample.id))
        if case is None:
            # A sample nobody asked for cannot be graded against an
            # expectation, and inventing one would be worse than saying so.
            continue

        elapsed = round(getattr(sample, "total_time", None) or 0.0, 2)

        if sample.error is not None:
            outcome = _error_outcome(case, str(sample.error.message), "sample_error")
            outcome.elapsed_s = elapsed
        else:
            observed, tool_calls, inspection_calls = _observe(sample, skills)
            # The transcript first, the limit second. An approver that stopped
            # the case at the decision may have done so before the bridge
            # adopted the message carrying it, so the activation can be absent
            # from the messages and present in the limit it caused. Neither
            # source alone is reliable; the transcript is the richer one, so it
            # wins when both have an answer.
            if observed is None:
                observed = activation_from_limit(
                    getattr(getattr(sample, "limit", None), "reason", None), skills
                )
            if observed is None and not _spoke(sample):
                outcome = _error_outcome(
                    case,
                    "the sample produced no agent messages, so the run never "
                    "made a routing decision",
                    "no_output",
                )
                outcome.elapsed_s = elapsed
            else:
                verdict = routing_core.classify(case.expect_skill, observed)
                outcome = routing_core.Outcome(
                    id=case.id,
                    category=case.category,
                    skill=case.skill,
                    prompt=case.prompt,
                    expect=case.expect_skill,
                    observed=observed,
                    verdict=verdict,
                    passed=verdict in routing_core.PASSING_VERDICTS,
                    stop_reason=(
                        "skill_activated"
                        if observed
                        else (_limit_reason(sample) or "result")
                    ),
                    elapsed_s=elapsed,
                    tool_calls=tool_calls,
                    inspection_calls=inspection_calls,
                )
        outcomes.append(outcome)

    # Samples inspect never reported back are still cases somebody asked for.
    # Silence here reads as a smaller, cleaner run rather than an incomplete
    # one, which is the failure mode `behavioral._failed` exists to prevent.
    # The usual cause is the task dying partway, so the task's own error is the
    # useful thing to say about a case that never ran.
    missing = (
        f"inspect task failed before this case ran: {failure}"
        if failure
        else "inspect returned no sample for this case"
    )
    graded = {outcome.id for outcome in outcomes}
    outcomes.extend(
        _error_outcome(case, missing) for case in cases if case.id not in graded
    )
    return outcomes


def _solver(
    engine: str,
    routing_set: dict[str, Path],
    model: str,
    effort: str,
    config_dir: Path | None = None,
    max_budget_usd: float | None = None,
):
    """The agent that drives one routing case, for the leg that was asked for.

    Both legs install the whole room and then run the real CLI once. They
    differ only in where that CLI runs, which is the entire reason both exist:
    when two legs disagree about a routing decision, the disagreement is a fact
    about the machine rather than about the skill.
    """
    from inspect_ai.solver import chain

    room = list(routing_set.values())

    if engine == CLAUDE_CODE:
        from inspect_swe import claude_code

        from . import tools, verify

        # `skills=` takes the list, so the guest's own discovery machinery
        # installs and registers the room -- the point of this leg being that
        # its machinery, not ours, decides what happens. `cwd` is set for the
        # same reason `verify.build_task` sets it: a container starts at `/`.
        return chain(
            verify._ensure_workdir(),
            claude_code(skills=room, cwd=tools.workdir_path()),
        )

    # The host leg stages the room itself, because the CLI reads it off the
    # filesystem it is handed. `model` is the skillscope alias rather than the
    # resolved inspect name: this leg is the `claude` CLI's own `--model` flag,
    # not a provider lookup.
    return chain(
        _install_room_solver(routing_set),
        no_sandbox.claude_code_no_sandbox(
            model, effort, room[0], config_dir, host_cost_flags(max_budget_usd)
        ),
    )


# Written into an approver's explanation when it stops a case, and read back
# off the sample's limit. The transcript is the primary source for what a case
# observed; this is the one that survives a terminate, because approval runs
# before the bridge adopts the assistant message into the agent's state, so the
# very tool call that revealed the decision may not be there afterwards. Two
# sources, one of which cannot go missing.
ACTIVATION_MARK = "skillscope-activated:"
BUDGET_MARK = "skillscope-budget:"


class _Tally:
    """Tool calls and skills-tree inspections seen so far in one case."""

    def __init__(self) -> None:
        self.tools = 0
        self.inspections = 0


def routing_decision(
    function: str,
    arguments: dict,
    room: list[str],
    tally: _Tally,
    max_tool_calls: int | None,
    max_inspection_calls: int | None,
) -> tuple[str, str]:
    """Whether this tool call ends the case, and why. Returns (decision, reason).

    Pure, and deliberately so: it decides with nothing but the call, the room
    and a running count, which means the rule can be tested without inspect
    installed -- and the unit suite runs without the extras on purpose. The
    approver below is the thin wrapper that turns this into inspect's vocabulary.

    The rule is the legacy engine's, moved earlier. Legacy sees a call in the
    CLI's stream after it has run and then races to kill the process; this sees
    it proposed and declines it, so the work never happens.
    """
    hit = routing_core.detect_activation(
        _activation_event_from(function, arguments), room,
        allow_body_path=ALLOW_BODY_PATH,
    )
    if hit:
        # The decision. Everything after it is paid for and unread.
        return "terminate", f"{ACTIVATION_MARK}{hit}"

    if (function or "").lower() not in routing_core.BOOKKEEPING_TOOLS:
        tally.tools += 1
    if routing_core._is_skills_inspection(function, json.dumps(arguments or {})):
        tally.inspections += 1

    over = (max_tool_calls is not None and tally.tools > max_tool_calls) or (
        max_inspection_calls is not None and tally.inspections > max_inspection_calls
    )
    if over:
        # An agent still rummaging at this point is not about to choose, and
        # the run buys nothing by watching it.
        return "terminate", (
            f"{BUDGET_MARK}{tally.tools} tool call(s), "
            f"{tally.inspections} inspection(s)"
        )
    return "approve", "not a routing decision"


def routing_approver(
    room: list[str],
    max_tool_calls: int | None,
    max_inspection_calls: int | None,
):
    """Stop the case at the decision, and at the budget, the way legacy does.

    Sees each tool call the bridged CLI *proposes*, before it runs, and ends
    the sample as `EvalSampleLimit(type="operator")` carrying the reason
    `routing_decision` produced -- which is the same information legacy puts in
    `stop_reason`.

    Only the sandboxed leg gets one. The host leg's CLI is a subprocess whose
    stdout is buffered until exit, so nothing there can observe a decision
    while there is still a run to stop.
    """
    from inspect_ai.approval import Approval, approver

    tally = _Tally()

    @approver
    def _routing():
        async def approve(message, call, view, history) -> Approval:
            decision, reason = routing_decision(
                call.function, call.arguments or {}, room,
                tally, max_tool_calls, max_inspection_calls,
            )
            return Approval(decision=decision, explanation=reason)

        return approve

    return _routing()


def activation_from_limit(reason: str | None, room: list[str]) -> str | None:
    """The skill an approver named when it stopped the case, if it named one.

    The fallback half of the two-source rule above. Matched against the room so
    a reason that arrived from anywhere else cannot be read as an activation.
    """
    if not reason or ACTIVATION_MARK not in reason:
        return None
    named = reason.split(ACTIVATION_MARK, 1)[1].strip()
    if named in room or named.startswith("other:"):
        return named
    return None


def host_cost_flags(max_budget_usd: float | None) -> list[str]:
    """The CLI's own cost controls, for the leg that builds its command line.

    The host leg cannot count tool calls in time to stop a case -- its stdout
    is buffered until exit -- so the one bound it can enforce mid-run is the
    CLI's, passed straight through the way legacy does. Probed first, because
    an older build rejects an unknown flag and every case then fails the same
    way, which reads as a routing collapse rather than a missing flag.
    """
    if not max_budget_usd or max_budget_usd <= 0:
        return []
    if "--max-budget-usd" not in routing_core.supported_flags(["--max-budget-usd"]):
        return []
    return ["--max-budget-usd", str(max_budget_usd)]


def case_time_limit(case_timeout: float | None) -> int | None:
    """The per-sample bound: `--case-timeout`, inside the command's own deadline.

    inspect's `Task(time_limit=)` is per *sample*, which is per case -- the same
    unit `--case-timeout` has always meant. Without this the only bound is the
    whole command's remaining budget, so one hung prompt spends the run, which
    is the exact thing the flag exists to prevent (`deadline` says so in its own
    module docstring).

    Clipped to whatever `--timeout` has left, for the reason
    `behavioral.task_time_limit` keeps a reserve: the command deadline ends the
    process outright, taking the report and the transcript with it, so a
    per-case cap that outlives it turns a timeout into silence.
    """
    whole_run = behavioral.task_time_limit(deadline.active())
    if case_timeout is None or case_timeout <= 0:
        return whole_run
    if whole_run is None:
        return int(case_timeout)
    return max(1, min(int(case_timeout), whole_run))


def build_task(
    cases: list[Case],
    routing_set: dict[str, Path],
    model: str,
    effort: str,
    engine: str,
    config_dir: Path | None = None,
    case_timeout: float | None = None,
    max_tool_calls: int | None = None,
    max_inspection_calls: int | None = None,
    max_budget_usd: float | None = None,
):
    """One inspect `Task` for the *room*, with the cases as its samples.

    Not one per skill, which is how behavioral is organised and why the shapes
    diverge here. A routing case is a question about the whole set: skill Y's
    positive prompt is skill X's negative, the confusion matrix is built out of
    exactly that overlap, and a per-skill task would install one skill at a time
    and answer an easier question under the same name.

    The sandbox is the bare provider rather than `sandbox_spec.for_skill`. A
    compose file is one skill's declaration about the machine *its* work needs
    -- network egress, a device bound in -- and the room has many skills and no
    work: the agent is asked to choose, not to run anything. Honouring one
    member's machine would hand every other member's cases an environment
    nobody asked for, and picking which member to honour has no right answer.
    """
    from inspect_ai import Task
    from inspect_ai.approval import ApprovalPolicy

    return Task(
        name="routing",
        dataset=[_sample_from_case(case) for case in cases],
        solver=_solver(engine, routing_set, model, effort, config_dir, max_budget_usd),
        # Only the sandboxed leg. Its tool calls cross inspect's bridge, so an
        # approver sees each one before it runs; the host leg's CLI buffers its
        # stdout until exit, so there is nothing there to approve in time.
        approval=(
            [
                ApprovalPolicy(
                    approver=routing_approver(
                        list(routing_set), max_tool_calls, max_inspection_calls
                    ),
                    tools="*",
                )
            ]
            if engine == CLAUDE_CODE
            else None
        ),
        # No scorer. The verdict needs the case's expectation and the room's
        # membership, and `routing.classify` is already the grader for both
        # legacy and these legs; wrapping it in a scorer would put a second
        # copy of routing's vocabulary inside the task.
        sandbox=sandbox_spec.provider(),
        message_limit=min(
            behavioral.message_limit_for(models.resolve(model)), ROUTING_MESSAGE_LIMIT
        ),
        time_limit=case_time_limit(case_timeout),
    )


def run(
    cases: list[Case],
    routing_set: dict[str, Path],
    model: str,
    effort: str,
    engine: str,
    case_timeout: float | None = None,
    max_tool_calls: int | None = None,
    max_inspection_calls: int | None = None,
    max_budget_usd: float | None = None,
) -> list[routing_core.Outcome]:
    """Run every routing case against the room. Mirrors `routing.run_case`'s output.

    Shaped like `verify.run` and `behavioral.run` -- cases in, outcomes out,
    nothing about the caller assumed -- so `cli.cmd_routing` picks a leg and
    hands the result to the same `_finish_routing` the legacy path uses.

    `model` is the skillscope alias (`opus`, `mockllm/model`), not a resolved
    inspect model string. Unlike the behavioral legs, which are handed one or
    the other by the CLI, this function owns both legs and they want different
    spellings: inspect wants `anthropic/claude-opus-5` and the `claude` CLI
    wants `opus`. Resolving inside is the only place that knows which is which.
    """
    from inspect_ai import eval as inspect_eval

    if engine not in ENGINES:
        # Explicit, for the reason `cmd_behavioral` refuses rather than falling
        # through: a routing run that quietly got a different agent than it
        # asked for is exactly the bug that kept this module from existing.
        raise SystemExit(
            f"error: routing has no '{engine}' leg in the inspect engine. "
            f"This is a bug in skillscope, not in how it was called."
        )

    if not routing_set:
        # `cli.cmd_routing` already refuses an empty room with a much better
        # message. Repeated here because this is a public entry point and the
        # alternative failure is an IndexError from inside a solver factory.
        raise SystemExit(
            "error: routing needs at least one skill in the room; "
            "there is nothing to route between."
        )

    if engine == CLAUDE_CODE:
        from . import verify

        verify.require()
    else:
        no_sandbox.require_local()

    # Before the provider check and before a single token: an unisolated room
    # is not a worse run, it is a different question answered under this one's
    # name.
    require_isolated_room(engine)

    sandbox_spec.require_provider()

    skills = list(routing_set)
    print(
        f"[{engine}] routing: {len(cases)} case(s), "
        f"{len(skills)} skill(s) installed together",
        flush=True,
    )

    resolved = models.resolve(model)
    # A directory per run, not per case: the CLI writes session state here and
    # the room is the same for every case, so sharing it costs nothing and
    # saves re-registering the room once per prompt. Removed on the way out --
    # the point was to keep the runner's own config out, not to leave another
    # one behind.
    with tempfile.TemporaryDirectory(prefix="skillscope-routing-") as config_dir:
        logs = _evaluate(
            inspect_eval,
            cases,
            routing_set,
            model,
            effort,
            engine,
            resolved,
            Path(config_dir) if engine == NO_SANDBOX else None,
            case_timeout,
            max_tool_calls,
            max_inspection_calls,
            max_budget_usd,
        )

    outcomes: list[routing_core.Outcome] = []
    for log in logs:
        stats.record_log(log)
        outcomes.extend(_outcomes(log, cases, skills))

    for outcome in outcomes:
        print(
            f"  [{'PASS' if outcome.passed else 'FAIL'}] {outcome.id}: "
            f"expected {outcome.expect or 'no skill'} -> "
            f"got {outcome.observed or 'no skill'} "
            f"({outcome.verdict}, {outcome.stop_reason}, {outcome.elapsed_s}s)"
            + (f" -- {outcome.error}" if outcome.error else ""),
            flush=True,
        )
    return outcomes


def _evaluate(
    inspect_eval, cases, routing_set, model, effort, engine, resolved, config_dir,
    case_timeout=None, max_tool_calls=None, max_inspection_calls=None,
    max_budget_usd=None,
):
    """Run the task. Split out so `run` reads as a sequence of decisions."""
    return inspect_eval(
        build_task(
            cases, routing_set, model, effort, engine, config_dir, case_timeout,
            max_tool_calls, max_inspection_calls, max_budget_usd,
        ),
        model=resolved,
        model_args=models.model_args(resolved),
        log_dir=str(Path(".skillscope") / "logs"),
        log_realtime=behavioral.realtime_logging(),
        # skillscope's own progress lines are the report; inspect's rich display
        # takes over the terminal and produces nothing useful when a CI job
        # pipes stdout to a file.
        display="plain",
    )
