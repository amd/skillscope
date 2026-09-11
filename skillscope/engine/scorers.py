# Copyright Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: MIT

"""Grading for the inspect engine.

One scorer grades every expectation a case carries and reports them all, rather
than one scorer per kind. A behavioral run costs minutes and real tokens, so a
run that fails should not have to be repeated to discover the second thing wrong
with it -- the same reason the legacy `Run.evaluate` reports instead of raising.

The per-expectation results ride in `Score.metadata["checks"]` in the shape
`agent.Check` uses, so `behavior.render_markdown` keeps working unchanged.
"""

from __future__ import annotations

import os

from ..agent import _find_file
from . import convert, judge, tools

CHECKS = "checks"


def _check(kind: str, expectation: str, passed: bool, detail: str = "") -> dict:
    return {
        "kind": kind,
        "expectation": expectation,
        "passed": passed,
        "detail": detail,
    }


def searchable(state) -> str:
    """Everything in the run, for `logs_contain` to search.

    Deliberately broader than what the judge sees. The legacy engine searched
    the whole raw transcript, so a case can pin down a tool name, a command
    string, or a phrase the agent used -- and cases were written against that.
    `judge.transcript_of` is the narrower, prose-free view, because an agent
    *claiming* it avoided something is not evidence that it did.
    """
    parts: list[str] = []
    for message in state.messages:
        parts.append(f"{getattr(message, 'role', '')}:")
        for call in getattr(message, "tool_calls", None) or []:
            parts.append(f"{call.function} {call.arguments}")
        content = getattr(message, "content", None)
        if isinstance(content, str):
            parts.append(content)
        elif isinstance(content, list):
            for part in content:
                text = getattr(part, "text", None)
                if isinstance(text, str):
                    parts.append(text)
    return "\n".join(parts)


def expectations():
    """Grade every expectation on the case and report each one."""
    from inspect_ai.scorer import CORRECT, INCORRECT, Score, accuracy, scorer, stderr

    @scorer(metrics=[accuracy(), stderr()])
    def _expectations():
        async def score(state, target) -> "Score":
            meta = state.metadata or {}
            checks: list[dict] = []

            transcript = searchable(state)
            for text in meta.get(convert.LOGS_CONTAIN, []):
                checks.append(
                    _check("logs_contain", text, text.lower() in transcript.lower())
                )

            wanted = meta.get(convert.FILES_EXIST, [])
            if wanted:
                files = await tools.list_paths()
                for path in wanted:
                    found = _find_file(files, path)
                    detail = ""
                    if found is None:
                        detail = f"sandbox holds: {files or 'nothing'}"
                    elif found != path:
                        detail = f"found at {found}"
                    checks.append(
                        _check("files_exist", path, found is not None, detail)
                    )

            # Judged expectations last: the deterministic results are on screen
            # before the grader calls, which take a few seconds each, begin.
            for statement in meta.get(convert.EXPECTED, []):
                ok, reason = await judge.grade(statement, state, must_happen=True)
                checks.append(_check("expected_behavior", statement, ok, reason))

            for statement in meta.get(convert.UNEXPECTED, []):
                ok, reason = await judge.grade(statement, state, must_happen=False)
                checks.append(_check("unexpected_behavior", statement, ok, reason))

            passed = bool(checks) and all(c["passed"] for c in checks)
            return Score(
                value=CORRECT if passed else INCORRECT,
                answer=f"{sum(c['passed'] for c in checks)}/{len(checks)} checks",
                metadata={CHECKS: checks},
            )

        return score

    return _expectations()
