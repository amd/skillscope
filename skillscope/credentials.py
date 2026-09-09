# Copyright Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: MIT

"""Resolve the model credentials one graded job runs with.

Either an API key, read from a secret the caller names and exported as
ANTHROPIC_API_KEY, or workload identity federation: the job trades its own
GitHub OIDC token for a short-lived Anthropic token, exported as
ANTHROPIC_AUTH_TOKEN. Federation wins when it is configured, and then no key is
read -- the CLI sends the two variables in different headers, so setting both
produces a 401 that reads like a bad key.
"""

from __future__ import annotations

import json
import os
import sys
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable, Mapping

TOKEN_URL = "https://api.anthropic.com/v1/oauth/token"
DEFAULT_AUDIENCE = "https://api.anthropic.com"
GRANT_TYPE = "urn:ietf:params:oauth:grant-type:jwt-bearer"
_DELIMITER = "__SKILLSCOPE_EOF__"

Fetch = Callable[..., bytes]


class CredentialError(RuntimeError):
    """Something the caller has to fix. Reported as a message, not a traceback."""


def _http(
    url: str, *, data: bytes | None = None, headers: Mapping[str, str] | None = None
) -> bytes:
    request = urllib.request.Request(url, data=data, headers=dict(headers or {}))
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            return response.read()
    except urllib.error.HTTPError as error:
        # The body carries the reason; the status line on its own does not.
        detail = error.read().decode("utf-8", "replace").strip()
        raise CredentialError(
            f"{url} returned {error.code}: {detail or error.reason}"
        ) from error
    except urllib.error.URLError as error:
        raise CredentialError(f"{url} could not be reached: {error.reason}") from error


def github_oidc_token(
    env: Mapping[str, str], audience: str, *, fetch: Fetch | None = None
) -> str:
    """Ask the runner's token endpoint for a JWT with this audience."""
    fetch = fetch or _http
    url = env.get("ACTIONS_ID_TOKEN_REQUEST_URL", "").strip()
    token = env.get("ACTIONS_ID_TOKEN_REQUEST_TOKEN", "").strip()
    if not url or not token:
        raise CredentialError(
            "this job cannot request a GitHub OIDC token, so federation has "
            "nothing to trade. Grant `id-token: write` under `permissions:` in "
            "the workflow that calls this one; a reusable workflow can only "
            "lower the permissions its caller passes down. GitHub also "
            "withholds the token from pull requests opened from forks."
        )

    query = urllib.parse.urlencode({"audience": audience})
    body = fetch(f"{url}&{query}", headers={"Authorization": f"Bearer {token}"})
    value = json.loads(body).get("value", "").strip()
    if not value:
        raise CredentialError("GitHub's OIDC endpoint answered without a token.")
    return value


def anthropic_access_token(
    assertion: str,
    *,
    rule_id: str,
    organization_id: str,
    service_account_id: str,
    workspace_id: str = "",
    token_url: str = TOKEN_URL,
    fetch: Fetch | None = None,
) -> str:
    """Trade a provider JWT for a short-lived Anthropic access token."""
    fetch = fetch or _http
    payload = {
        "grant_type": GRANT_TYPE,
        "assertion": assertion,
        "federation_rule_id": rule_id,
        "organization_id": organization_id,
        "service_account_id": service_account_id,
    }
    if workspace_id:
        payload["workspace_id"] = workspace_id

    try:
        body = fetch(
            token_url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"content-type": "application/json"},
        )
    except CredentialError as error:
        raise CredentialError(
            f"{error}\n"
            "A refused exchange is always this opaque. The reason is recorded "
            "in the Claude Console under Settings -> Workload identity -> the "
            "rule -> authentication history, as a check such as "
            "`match_subject_prefix` when `sub` did not match the rule."
        ) from error

    token = json.loads(body).get("access_token", "").strip()
    if not token:
        raise CredentialError("the token exchange returned no access_token.")
    return token


def resolve(env: Mapping[str, str], *, fetch: Fetch | None = None) -> dict[str, str]:
    """Work out what one job should export. Federation first, then the key."""
    fetch = fetch or _http
    rule_id = env.get("FEDERATION_RULE_ID", "").strip()
    base_url = env.get("API_BASE_URL", "").strip()
    headers = env.get("API_CUSTOM_HEADERS", "").strip()

    if rule_id:
        if base_url or headers:
            raise CredentialError(
                "federation cannot be combined with a custom base URL or "
                "custom headers: the token it mints is only good at "
                "api.anthropic.com. Leave both blank to federate, or use an "
                "API key for a gateway."
            )

        organization_id = env.get("FEDERATION_ORGANIZATION_ID", "").strip()
        service_account_id = env.get("FEDERATION_SERVICE_ACCOUNT_ID", "").strip()
        missing = [
            name
            for name, value in (
                ("federation_organization_id", organization_id),
                ("federation_service_account_id", service_account_id),
            )
            if not value
        ]
        if missing:
            raise CredentialError(
                "federation needs the whole triple. Set " + " and ".join(missing) + "."
            )

        audience = env.get("FEDERATION_AUDIENCE", "").strip() or DEFAULT_AUDIENCE
        token = anthropic_access_token(
            github_oidc_token(env, audience, fetch=fetch),
            rule_id=rule_id,
            organization_id=organization_id,
            service_account_id=service_account_id,
            workspace_id=env.get("FEDERATION_WORKSPACE_ID", "").strip(),
            token_url=env.get("FEDERATION_TOKEN_URL", "").strip() or TOKEN_URL,
            fetch=fetch,
        )
        return {"ANTHROPIC_AUTH_TOKEN": token}

    key = env.get("API_KEY", "").strip()
    if not key:
        name = env.get("SECRET_NAME", "the model API key secret")
        environment = env.get("ENVIRONMENT", "").strip()
        where = (
            f"Check that it is set on the '{environment}' environment"
            if environment
            else "Check that it is set"
        )
        raise CredentialError(
            f"the secret {name} resolved to an empty value. {where}, and that "
            "the caller passes `secrets: inherit`. GitHub withholds secrets "
            "from pull requests opened from forks, so re-run from a branch in "
            "the repository. To hold no key at all, set federation_rule_id."
        )

    exported = {"ANTHROPIC_API_KEY": key}
    if base_url:
        exported["ANTHROPIC_BASE_URL"] = base_url
    if headers:
        exported["ANTHROPIC_CUSTOM_HEADERS"] = headers.replace("$API_KEY", key)
    return exported


def main(argv: list[str] | None = None) -> int:
    try:
        exported = resolve(os.environ)
    except CredentialError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1

    # Minted during the run, so GitHub has never seen it and will not redact it.
    if "ANTHROPIC_AUTH_TOKEN" in exported:
        print(f"::add-mask::{exported['ANTHROPIC_AUTH_TOKEN']}")

    with open(os.environ["GITHUB_ENV"], "a", encoding="utf-8") as out:
        out.writelines(
            f"{name}<<{_DELIMITER}\n{value}\n{_DELIMITER}\n"
            for name, value in exported.items()
        )

    how = "federation" if "ANTHROPIC_AUTH_TOKEN" in exported else "an API key"
    print(f"Exported via {how}: " + ", ".join(exported))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
