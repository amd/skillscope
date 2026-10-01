# Copyright Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: MIT

"""What a graded run spent, recorded by whichever engine ran it.

Both engines know their own cost and neither reported it: the legacy engine
discards the `total_cost_usd` the CLI hands back with every result, and inspect
keeps usage in its own `.eval` log where skillscope's report never looks. That
was fine while there was one engine and nothing to compare it against.

Accumulated in module state rather than threaded through return values, because
the engines' entry points return outcome lists and that signature is what lets
the CLI swap one for the other in a single line. A run is a process, so the
scope is right even if the shape is blunt; `reset()` exists for tests.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Usage:
    """Totals for one graded run. Fields are None when the engine cannot say."""

    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float | None = None
    calls: int = 0

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    def as_meta(self) -> dict:
        """The shape that goes into a report's `meta`, omitting what is unknown."""
        meta: dict = {
            "model_calls": self.calls,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
        }
        if self.cost_usd is not None:
            meta["cost_usd"] = round(self.cost_usd, 4)
        return meta


_current: Usage = Usage()


def reset() -> None:
    global _current
    _current = Usage()


def snapshot() -> Usage:
    return _current


def record(
    *,
    input_tokens: int = 0,
    output_tokens: int = 0,
    cost_usd: float | None = None,
    calls: int = 1,
) -> None:
    """Add one model interaction's cost to the run."""
    _current.input_tokens += int(input_tokens or 0)
    _current.output_tokens += int(output_tokens or 0)
    _current.calls += int(calls or 0)
    if cost_usd is not None:
        _current.cost_usd = (_current.cost_usd or 0.0) + float(cost_usd)


def record_stream_event(event: dict) -> None:
    """Record what one `claude` stream-json event says the run has spent.

    Tokens come from assistant events, one per model reply, and are the only
    source this leg has: its CLI runs as a subprocess, so nothing reaches
    inspect's model layer and `log.stats.model_usage` is empty. Cost comes only
    from the result event, where it is a run total -- and a routing case is
    normally killed before that event arrives, so it reports responses with no
    cost. That is what driving a CLI can actually observe.

    **Calls are deliberately not counted here.** `engine.stats.record_log`
    counts one call per assistant message off the `EvalLog`, and it does that
    for every engine. This function's events become those same messages
    (`events_to_messages` in `engine/no_sandbox.py`), so counting in both
    places reported the host leg's `model_calls` at twice the sandboxed leg's
    for identical work -- and the two sit side by side in the benchmark, which
    is the comparison the column exists for.
    """
    kind = event.get("type")
    if kind == "assistant":
        message = event.get("message")
        counts = (message or {}).get("usage") if isinstance(message, dict) else None
        record(
            input_tokens=(counts or {}).get("input_tokens", 0),
            output_tokens=(counts or {}).get("output_tokens", 0),
            calls=0,
        )
    elif kind == "result":
        record(cost_usd=event.get("total_cost_usd"), calls=0)
