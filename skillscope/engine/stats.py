# Copyright Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: MIT

"""Read what an inspect run spent out of its `EvalLog`.

inspect already counts this per model in `log.stats.model_usage`; skillscope
just has to move it somewhere the report can see. Kept apart from the engine
modules so `usage` stays the only shared vocabulary between the two engines.
"""

from __future__ import annotations

from .. import usage


def record_log(log) -> None:
    """Add one `EvalLog`'s token and cost totals to the run."""
    stats = getattr(log, "stats", None)
    for model_usage in (getattr(stats, "model_usage", None) or {}).values():
        usage.record(
            input_tokens=getattr(model_usage, "input_tokens", 0) or 0,
            output_tokens=getattr(model_usage, "output_tokens", 0) or 0,
            # Populated only when the provider supplies pricing; a gateway
            # generally does not, so this stays None and the report omits it.
            cost_usd=getattr(model_usage, "total_cost", None),
            calls=0,
        )

    # One "call" per sample is the comparable unit across engines: the legacy
    # engine reports once per case, and per-request counts are not visible on
    # both sides.
    usage.record(calls=len(getattr(log, "samples", None) or []))
