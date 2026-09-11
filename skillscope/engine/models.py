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
