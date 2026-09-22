# Copyright Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: MIT

"""Model names: skillscope aliases to inspect model strings.

`--model opus` is the `claude` CLI's alias vocabulary. inspect wants a
provider-qualified name (`anthropic/claude-opus-5`), so the two have to be
translated at the boundary rather than either side changing its spelling --
`--model` is part of the frozen CLI surface.

Anything already carrying a provider prefix passes through untouched, which is
what makes `--model mockllm/model` work for the no-cost wiring runs.
"""

from __future__ import annotations

import os

from .. import deadline

# How long the reachability probe may take. The legacy probe has held the same
# bound since it was written, for the same reason: off-network is the ordinary
# way for this to fail, and a preflight that hangs is worse than no preflight.
PROBE_TIMEOUT_S = 60.0

# No retries, for the reason `claude_env` sets CLAUDE_CODE_MAX_RETRIES to 0:
# inspect's default is `None`, which retries a connection error without a
# limit, and a probe whose whole job is to fail fast must not be the one thing
# that hangs. Against a closed port this is the difference between answering in
# seconds and still running after four minutes.
PROBE_RETRIES = 0

ALIASES = {
    "opus": "anthropic/claude-opus-5",
    "sonnet": "anthropic/claude-sonnet-5",
    "haiku": "anthropic/claude-haiku-4-5-20251001",
}

# `claude` reads per-request headers from this; nothing in inspect does, so
# skillscope parses it and hands the result to the provider instead. An
# enterprise gateway in front of the Anthropic API is the reason it exists.
CUSTOM_HEADERS_ENV = "ANTHROPIC_CUSTOM_HEADERS"
AUTH_TOKEN_ENV = "ANTHROPIC_AUTH_TOKEN"


def resolve(model: str) -> str:
    """Translate a skillscope model alias into an inspect model string."""
    if "/" in model:
        return model
    return ALIASES.get(model.lower(), f"anthropic/{model}")


async def _probe(model: str, timeout: float):
    import anyio
    from inspect_ai.model import GenerateConfig, get_model

    resolved = get_model(
        model,
        # `timeout` bounds each attempt and PROBE_RETRIES stops it being
        # attempted again; the cancel scope below bounds the call as a whole,
        # since neither of those covers a connect that stalls before the
        # provider's own clock starts.
        config=GenerateConfig(max_retries=PROBE_RETRIES, timeout=max(1, int(timeout))),
        # Not memoized: this config exists for the probe, and a graded run that
        # later asked for the same model would otherwise inherit a retry
        # setting chosen for a one-shot check.
        memoize=False,
        **model_args(model),
    )
    with anyio.fail_after(timeout):
        return await resolved.generate("Reply with the single word: ok")


def probe_bound(timeout: float = PROBE_TIMEOUT_S) -> tuple[float | None, str]:
    """Seconds this probe may take, or ``(None, why)`` when there are none left.

    A preflight exists to save a run from a long confusing failure, so it must
    not become one: the bound is the smaller of its own and whatever the
    command's ``--timeout`` has left. The legacy probe has always clipped
    itself this way; this one did not, which is how a closed port kept it
    running long past the deadline that was supposed to cover it.
    """
    bound = deadline.active()
    if bound is None:
        return timeout, ""
    if bound.expired():
        return None, bound.message()
    return bound.cap(timeout), ""


def check_reachable(model: str, timeout: float = PROBE_TIMEOUT_S) -> tuple[bool, str]:
    """Confirm the model answers before anything expensive starts.

    A graded run starts containers and installs skills before it ever reaches a
    provider, so a misconfigured gateway surfaces as a task that failed after
    all that work rather than as a credentials problem. One tiny call up front
    turns a 401 buried in a sample error into a message on the first line.

    Bounded twice over, because an unreachable gateway is the common case and a
    preflight that outlives the command it protects helps nobody: `timeout`
    here, clipped to whatever ``--timeout`` has left.

    Costs a handful of tokens. `mockllm` reaches no provider, so it is skipped
    rather than charged for a round trip that proves nothing.
    """
    if model.startswith("mockllm"):
        return True, "mockllm (no provider)"

    timeout, expired = probe_bound(timeout)
    if timeout is None:
        return False, expired

    import anyio

    try:
        output = anyio.run(_probe, model, timeout)
    except TimeoutError:
        return False, (
            f"model preflight timed out after {timeout:g}s "
            "(is the network reachable?)"
        )
    except Exception as exc:  # noqa: BLE001 -- the reason is the return value
        exc = _underlying(exc)
        return False, f"{type(exc).__name__}: {exc}"[:400]
    return True, (output.completion or "").strip()[:40]


def _underlying(exc: BaseException) -> BaseException:
    """The error a retry wrapper is carrying, if it is carrying one.

    Turning off retries makes tenacity raise `RetryError` rather than what
    actually went wrong, and `RetryError[<Future at 0x7f...>]` is not a message
    anybody can act on. Reported as `APIConnectionError: ...` instead, which is
    the difference between this probe doing its job and merely failing.
    """
    attempt = getattr(exc, "last_attempt", None)
    if attempt is None:
        return exc
    try:
        return attempt.exception() or exc
    except Exception:  # noqa: BLE001 -- a probe never fails on its own reporting
        return exc


def custom_headers() -> dict[str, str]:
    """Parse ``ANTHROPIC_CUSTOM_HEADERS`` (newline-separated ``Key: value``)."""
    headers: dict[str, str] = {}
    for line in (os.environ.get(CUSTOM_HEADERS_ENV) or "").splitlines():
        if ":" not in line:
            continue
        name, _, value = line.partition(":")
        if name.strip():
            headers[name.strip()] = value.strip()
    return headers


def model_args(model: str) -> dict:
    """Provider arguments for the configured gateway, if any.

    inspect passes these straight to `AsyncAnthropic`, so custom headers ride in
    as `default_headers`. Empty when no gateway headers are configured, which is
    the ordinary api.anthropic.com case.

    Scoped to Anthropic models on purpose. The free `mockllm/model` wiring run
    reaches no provider at all, and refusing it because the shell happens to
    hold both Anthropic variables would break the one check that costs nothing
    -- on exactly the machines most likely to have an OAuth token lying around.
    """
    if not model.startswith("anthropic/"):
        return {}

    headers = custom_headers()
    if not headers:
        return {}

    if os.environ.get(AUTH_TOKEN_ENV):
        # The same rule `credentials.resolve` enforces when it hands a job its
        # environment: a federated token is only good at api.anthropic.com, so
        # it never travels with a gateway's base URL or headers. Caught here
        # too because an environment can be assembled by hand, and inspect's
        # OAuth path also sets `default_headers` itself -- passing ours would
        # surface as a duplicate keyword argument from inside the SDK.
        raise SystemExit(
            f"error: both {AUTH_TOKEN_ENV} and {CUSTOM_HEADERS_ENV} are set. "
            "A federated token only works at api.anthropic.com; reaching a "
            "gateway needs ANTHROPIC_API_KEY instead. Unset one of them."
        )

    return {"default_headers": headers}
