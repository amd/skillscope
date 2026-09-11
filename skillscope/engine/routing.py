# Copyright Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: MIT

"""Routing evals on the inspect engine.

Routing asks one question: with the whole catalog installed, does this prompt
activate the skill it should, and stay quiet when it should not? The room
matters -- a skill tested alone happily answers its neighbour's prompts -- so
every skill is offered on every case, exactly as `routing.stage_workspace` did.

Two things get much simpler than the legacy engine.

**Activation is observed, not inferred.** With the `skill()` tool the agent
names the skill it wants, so the decision is a tool call rather than something
reconstructed from a stream of events. `routing.detect_activation` and its
helpers exist because that signal was not available.

**Stopping is free.** A routing decision is visible in the first assistant turn,
so a case is exactly one model call: the skills are offered as tools and the
reply either names one or does not. Nothing is executed, so there is no agent
loop to bound and no sandbox to start -- the legacy engine's stream reader,
process group and SIGKILL all exist to end a turn it had already paid for.

Keeping the scaffolding out is also a measurement decision: no submit tool and
no agent system prompt sit between the descriptions and the decision, which is
what makes the result about the descriptions.

What this measures is how well a description discriminates against its
neighbours, which is the part a skill author controls. It is not a measurement
of any particular product harness's discovery machinery, and the numbers are
not interchangeable with one.
"""

from __future__ import annotations

import time
from pathlib import Path

from .. import config, deadline
from ..datasets import Case
from ..routing import PASSING_VERDICTS, Outcome, classify
from . import convert, models

SKILL_TOOL = "skill"


def activation_of(messages) -> str | None:
    """The skill the agent asked for, or None if it never asked for one."""
    for message in messages:
        for call in getattr(message, "tool_calls", None) or []:
            if call.function != SKILL_TOOL:
                continue
            named = (call.arguments or {}).get("command")
            if isinstance(named, str) and named.strip():
                return named.strip()
    return None


def tool_call_count(messages) -> int:
    return sum(len(getattr(m, "tool_calls", None) or []) for m in messages)


def decide(routing_set: dict[str, Path]):
    """Solver: offer the room as tools, take one turn, record what was named.

    The tool is never executed -- only its definition matters, which is the
    skill's name and description. That is the whole input to a routing decision.
    """
    from inspect_ai.solver import solver
    from inspect_ai.tool import skill

    @solver
    def _decide():
        tools = [skill(list(routing_set.values()))]

        async def solve(state, generate):
            from inspect_ai.model import get_model

            output = await get_model().generate(input=state.messages, tools=tools)
            state.messages.append(output.message)
            state.output = output
            return state

        return solve

    return _decide()


def build_task(cases: list[Case], routing_set: dict[str, Path]):
    """One task holding every routing case, with the whole room offered."""
    from inspect_ai import Task

    cfg = config.active()
    samples = [
        convert.sample_from_case(
            case, cfg.skill_path(case.skill) if case.skill else Path(".")
        )
        for case in cases
    ]

    bound = deadline.active()
    return Task(
        name="routing",
        dataset=samples,
        solver=decide(routing_set),
        # No sandbox: nothing is executed, so there is nothing to isolate.
        time_limit=int(bound.remaining()) if bound is not None else None,
    )


def _outcome(sample, case: Case, room: list[str]) -> Outcome:
    """Map one inspect sample onto the outcome `routing.summarize` expects."""
    messages = sample.messages or []
    observed = activation_of(messages)
    calls = tool_call_count(messages)

    if sample.error is not None:
        verdict, error, stop_reason = "error", sample.error.message, "error"
    else:
        verdict, error = classify(case.expect_skill, observed), None
        stop_reason = "decided" if observed else "completed"

    return Outcome(
        id=case.id,
        category=case.category,
        skill=case.skill,
        prompt=case.prompt,
        expect=case.expect_skill,
        observed=observed,
        verdict=verdict,
        passed=verdict in PASSING_VERDICTS,
        stop_reason=stop_reason,
        elapsed_s=round(getattr(sample, "total_time", None) or 0.0, 2),
        tool_calls=calls,
        # The legacy engine counted reads of a skill body separately because a
        # skill could be inspected without being activated. The tool makes that
        # distinction disappear: naming the skill *is* the activation.
        inspection_calls=0,
        visible_skills=room,
        # Nothing beyond the room can leak in: the tool is constructed from the
        # room, so there is no user-level config dir for a stray skill to arrive
        # from. That is why `routing.can_isolate_config` has no counterpart here.
        extra_skills=[],
        error=error,
    )


def run(cases: list[Case], routing_set: dict[str, Path], model: str) -> list[Outcome]:
    """Run every routing case against the whole room."""
    from inspect_ai import eval as inspect_eval

    if not cases:
        return []

    room = list(routing_set)
    print(f"[routing] installed together: {', '.join(room) or '(none)'}")
    print(f"[routing] {len(cases)} cases, model={model}", flush=True)

    started = time.perf_counter()
    logs = inspect_eval(
        build_task(cases, routing_set),
        model=model,
        model_args=models.model_args(),
        log_dir=str(Path(".skillscope") / "logs"),
        display="plain",
    )

    by_id = {case.id: case for case in cases}
    outcomes: list[Outcome] = []
    for log in logs:
        if log.status == "error" or not log.samples:
            detail = getattr(log.error, "message", None) or "no samples"
            raise SystemExit(f"error: routing task failed: {detail}")
        for sample in log.samples:
            case = by_id.get(str(sample.id))
            if case is not None:
                outcomes.append(_outcome(sample, case, room))

    elapsed = round(time.perf_counter() - started, 1)
    print(f"[routing] {len(outcomes)} decisions in {elapsed}s", flush=True)
    return outcomes
