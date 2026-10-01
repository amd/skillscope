# Copyright Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: MIT

"""Operator-supplied model prices, so a spend cap can actually bind.

`--max-budget-usd` is enforced two different ways. The host leg hands
`--max-budget-usd` to the `claude` CLI, which knows what it bills and stops by
itself. The sandboxed leg cannot: `inspect_swe` builds its command line from a
closed set of arguments and takes no passthrough
(meridianlabs-ai/inspect_swe#178), so that leg uses inspect's own per-sample
`cost_limit` instead.

`cost_limit` is cooperative: inspect checks it from `record_model_usage`, but
only after computing a cost, and it computes one only when its model registry
supplies a rate. **The registry ships no rates at all** -- 796 entries across
ten provider files in inspect_ai 0.3.266, not one with a `cost`. So the cap is
inert out of the box, for every model, not merely for an unusual one.

This module is the way out, and it is opt-in on purpose. skillscope does not
ship a price list: rates change without notice, differ by contract and region,
and a stale table would hold a run to a number nobody agreed while the report
said the budget was enforced. That is a worse failure than no cap, and it is
the same shape as the defects this engine has already been corrected for.

So the operator supplies the rates, and the report records that they did --
`meta.max_budget_pricing` names the source, so a reader can tell a cap backed
by an operator's own figures from one backed by inspect's registry.

**Format.** `SKILLSCOPE_MODEL_PRICING` is either a path to a JSON file or the
JSON itself, mapping a model to its rates **per million tokens**:

    {
      "opus":                    {"input": 5.0, "output": 25.0,
                                  "cache_read": 0.5, "cache_write": 6.25},
      "anthropic/claude-haiku-4-5-20251001":
                                 {"input": 1.0, "output": 5.0,
                                  "cache_read": 0.1, "cache_write": 1.25}
    }

Keys may be skillscope aliases (`opus`) or inspect model strings; both are
resolved the same way `--model` is. `cache_read` and `cache_write` default to
0, which prices a cached read as free rather than refusing the entry -- a
deployment that does not bill separately for cache is a normal one.

**Why a file rather than a URL.** Fetching rates from a catalogue would mean
skillscope holding that service's credentials and reaching the network inside a
preflight, and would couple this tool to one deployment's API. Whoever knows
where their rates live can produce this file in one command; that keeps the
knowledge where it belongs and keeps skillscope's dependencies to a file read.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from . import models

PRICING_ENV = "SKILLSCOPE_MODEL_PRICING"

# What `meta.max_budget_pricing` says when nobody supplied rates. Spelled out
# rather than omitted: a missing key reads as an oversight, and this is a
# deliberate state with a consequence a reader needs to know about.
NO_PRICING = "none (inspect ships no rates, so a cost limit cannot bind)"


def _rates(entry: dict, model: str) -> dict:
    """One model's four rates, or raise with the key that is wrong."""
    try:
        return {
            "input": float(entry["input"]),
            "output": float(entry["output"]),
            "input_cache_read": float(entry.get("cache_read", 0) or 0),
            "input_cache_write": float(entry.get("cache_write", 0) or 0),
        }
    except KeyError as exc:
        raise SystemExit(
            f"error: {PRICING_ENV} entry for {model!r} is missing {exc.args[0]!r}. "
            "`input` and `output` are required, in dollars per million tokens; "
            "`cache_read` and `cache_write` default to 0."
        ) from exc
    except (TypeError, ValueError) as exc:
        raise SystemExit(
            f"error: {PRICING_ENV} entry for {model!r} has a non-numeric rate. "
            "Rates are dollars per million tokens."
        ) from exc


def load() -> dict | None:
    """The configured price table, or `None` when nobody configured one.

    Accepts a path or the JSON itself, because the two callers differ: a CI job
    writes a file next to its checkout, and a shell one-liner is easier to hand
    a string. Distinguished by trying the filesystem first -- a path is never
    valid JSON, and JSON is never an existing file.
    """
    raw = (os.environ.get(PRICING_ENV) or "").strip()
    if not raw:
        return None

    source = raw
    candidate = Path(raw)
    try:
        if candidate.is_file():
            raw = candidate.read_text(encoding="utf-8")
        else:
            source = "inline JSON"
    except OSError:
        # A path too long for the filesystem is JSON, not a missing file.
        source = "inline JSON"

    try:
        table = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise SystemExit(
            f"error: {PRICING_ENV} is neither a readable file nor valid JSON "
            f"({exc}). Give a path to a JSON file, or the JSON itself."
        ) from exc

    if not isinstance(table, dict) or not table:
        raise SystemExit(
            f"error: {PRICING_ENV} must be a non-empty object mapping a model "
            'to its rates, e.g. {"opus": {"input": 5.0, "output": 25.0}}.'
        )
    return {"source": source, "models": table}


def apply() -> str:
    """Register the configured rates with inspect. Returns what to report.

    Called once per graded run, before anything reaches a provider, so a
    malformed table fails the command rather than one sample deep into it.
    """
    loaded = load()
    if loaded is None:
        return NO_PRICING

    from inspect_ai.model import set_model_cost, set_model_info
    from inspect_ai.model._model_data.model_data import ModelCost, ModelInfo

    registered = 0
    for name, entry in loaded["models"].items():
        if not isinstance(entry, dict):
            raise SystemExit(
                f"error: {PRICING_ENV} entry for {name!r} is not an object. "
                'Each value is {"input": N, "output": N}, in dollars per '
                "million tokens."
            )
        resolved = models.resolve(name)
        cost = ModelCost(**_rates(entry, name))
        try:
            set_model_cost(resolved, cost)
        except ValueError:
            # Not in inspect's registry at all, which is ordinary for a model
            # reached through a gateway under a name of its own. Registering
            # the info outright is the documented route for exactly that.
            set_model_info(resolved, ModelInfo(cost=cost))
        registered += 1

    return f"{registered} model(s) from {loaded['source']}"
