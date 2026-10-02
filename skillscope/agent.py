# Copyright Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: MIT

"""What is left of the retired engine: helpers the inspect engines still use.

This was the legacy behavioral driver -- it staged a skill into a temp
workspace, ran the `claude` CLI there, and graded the result through a `Run`
object. The engines built on inspect_ai do all of that themselves, so the
driver is gone and what remains is the handful of functions they still call:

* `enforce_model_policy` pins the model under CI, so paid runs stay comparable.
* `claude_env` is the environment every `claude` subprocess gets, with the
  CLI's own retry loop disabled so an unreachable API fails fast.
* `check_api_reachable` is the preflight for the leg that drives that CLI.
* `_walk` and `_find_file` are read by `engine/no_sandbox.py` and
  `engine/scorers.py` respectively -- transcript flattening and the
  whole-segment path match that decides `files_exist`.

Kept here rather than scattered because they are the vocabulary the two
remaining engines share about the CLI, and moving them would make the diff
larger than the change.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys

from . import deadline

DEFAULT_MODEL = os.environ.get("SKILLSCOPE_MODEL", "opus")

# Automated runs are pinned to opus: a behavioral run makes real cloud calls
# (agent run + LLM judge), so pinning the model keeps CI results comparable
# between runs. No override -- the pin is non-negotiable in CI.
AUTOMATED_MODEL = "opus"
_TRUTHY = {"1", "true", "yes", "on"}


def _safe_print(text: str) -> None:
    """Print, falling back to `errors="replace"` if the console can't encode `text`."""
    try:
        print(text, flush=True)
    except UnicodeEncodeError:
        encoding = sys.stdout.encoding or "ascii"
        print(text.encode(encoding, errors="replace").decode(encoding), flush=True)


def is_automated_env() -> bool:
    """True under CI / an automated workflow (GitHub Actions sets both)."""
    return any(
        os.environ.get(var, "").strip().lower() in _TRUTHY
        for var in ("CI", "GITHUB_ACTIONS")
    )


# Model providers that reach no cloud service. The CI pin exists to keep paid
# runs comparable between runs; one of these grades nothing and costs nothing,
# so pinning it only turns a free wiring check into a run that needs a key.
NO_PROVIDER_PREFIXES = ("mockllm",)


def enforce_model_policy(model: str | None) -> str | None:
    """Coerce non-opus models to opus in CI; pass through otherwise."""
    if model is None or not is_automated_env() or "opus" in model.lower():
        return model
    if model.lower().startswith(NO_PROVIDER_PREFIXES):
        return model
    _safe_print(
        f"[skillscope] automated run: coercing model '{model}' -> "
        f"'{AUTOMATED_MODEL}' to pin the CI model."
    )
    return AUTOMATED_MODEL


def claude_env() -> dict[str, str]:
    """Environment for `claude` subprocesses.

    Disable the CLI's internal retry loop by default so a network/auth problem
    (e.g. not connected to the network that can reach the API) fails fast
    instead of being retried into a long, confusing hang. The caller can still
    override by exporting ``CLAUDE_CODE_MAX_RETRIES``.
    """
    env = dict(os.environ)
    env.setdefault("CLAUDE_CODE_MAX_RETRIES", "0")
    return env


def check_api_reachable(model: str | None = DEFAULT_MODEL, timeout: float = 60) -> tuple[bool, str]:
    """Preflight: confirm the `claude` CLI can actually reach the API.

    Runs a trivial prompt with retries disabled so an unreachable API fails
    fast. Returns ``(ok, detail)`` where ``detail`` is a short human-readable
    reason on failure. Called once before the (expensive) runs so a suite can
    fail cleanly when off-network.
    """
    claude_bin = shutil.which("claude")
    if not claude_bin:
        return False, "'claude' CLI not found on PATH"

    model = enforce_model_policy(model)
    cmd = [claude_bin, "-p", "--output-format", "json"]
    if model:
        cmd += ["--model", model]

    bound = deadline.active()
    if bound is not None:
        leftover = bound.remaining()
        if leftover <= 0:
            return False, bound.message()
        timeout = min(timeout, leftover)

    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True, encoding="utf-8",
            input="Reply with the single word: ok", timeout=timeout, env=claude_env(),
        )
    except subprocess.TimeoutExpired:
        return False, f"API preflight timed out after {timeout:g}s (is the network reachable?)"

    if proc.returncode != 0:
        detail = (proc.stderr or proc.stdout or f"exit code {proc.returncode}").strip()
        return False, detail[:500]
    return True, "ok"


def _walk(obj, tool_uses, tool_results) -> None:
    """Collect (tool name, tool input) pairs and tool-result text from events."""
    if isinstance(obj, dict):
        otype = obj.get("type")
        if otype == "tool_use":
            tool_uses.append((str(obj.get("name", "")), json.dumps(obj.get("input", {}), ensure_ascii=False)))
        elif otype == "tool_result":
            content = obj.get("content")
            if isinstance(content, str):
                tool_results.append(content)
            elif isinstance(content, list):
                for c in content:
                    if isinstance(c, dict) and isinstance(c.get("text"), str):
                        tool_results.append(c["text"])
        for v in obj.values():
            _walk(v, tool_uses, tool_results)
    elif isinstance(obj, list):
        for v in obj:
            _walk(v, tool_uses, tool_results)


def _find_file(files: list[str], expected: str) -> str | None:
    """Return the workspace file that satisfies ``expected``, or None.

    An expectation matches anywhere in the tree: ``analyze_plan.md`` is
    satisfied by ``examples/simple_hip_test/analyze_plan.md``, and
    ``out/report.md`` by ``run-1/out/report.md``. Only whole path segments
    count, so ``plan.md`` does not match ``analyze_plan.md``.

    A case asserts that an artifact was produced; which directory the agent
    chose for it is usually its own call, and a plan written beside the fixture
    it describes is not a failed run. Pin the location down in the prompt when
    it matters, and the judged expectations can grade whether it was honored.
    """
    wanted = expected.replace("\\", "/").strip("/")
    if wanted.startswith("./"):
        wanted = wanted[2:]
    for rel in files:
        if rel == wanted or rel.endswith("/" + wanted):
            return rel
    return None
