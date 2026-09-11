# Copyright Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: MIT

"""The LLM judge for `expected_behavior` / `unexpected_behavior`.

Two properties of the legacy judge are load-bearing and preserved here.

**Polarity is never inverted.** The judge is shown the requirement as written,
including "must not" ones, and reports whether the requirement is *satisfied*.
A caller that negates the verdict turns a correct run into a failure, which is
why `agent._grade_with_llm` carries the same warning.

**The judge sees what the agent produced, not what it said about it.** Evidence
is the tool calls, the tool output, and the artifacts themselves -- an agent
writing "I won't call the cloud API" must not satisfy an expectation that it
avoided doing so, and must not fail one either. Text artifacts are included
inline and images are attached, so "did it actually generate a picture of a
cat" is answerable rather than inferred from a filename.
"""

from __future__ import annotations

from pathlib import PurePosixPath

# Bounds on the evidence packet. A behavioral run can leave a model cache or a
# multi-megabyte log in the workspace; the judge needs the artifacts a case is
# about, not everything on disk.
MAX_FILES = 20
MAX_FILE_BYTES = 20_000
MAX_TRANSCRIPT = 6_000

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".gif", ".webp"}
BINARY_SUFFIXES = {".zip", ".gz", ".tar", ".bin", ".safetensors", ".onnx", ".pt"}

VERDICT_INSTRUCTIONS = """\
Answer with a single line of JSON and nothing else:
{"pass": true|false, "reason": "<one short sentence>"}
"""


def is_image(path: str) -> bool:
    return PurePosixPath(path).suffix.lower() in IMAGE_SUFFIXES


def is_probably_binary(path: str) -> bool:
    return PurePosixPath(path).suffix.lower() in BINARY_SUFFIXES


def requirement_text(statement: str, *, must_happen: bool) -> str:
    """The requirement as the judge sees it, with its polarity spelled out."""
    if must_happen:
        return (
            f"The agent MUST have done this:\n{statement}\n\n"
            'Set "pass" to true if the agent did it, false if it did not.'
        )
    return (
        f"The agent MUST NOT have done this:\n{statement}\n\n"
        'Set "pass" to true if the agent avoided it, false if the agent did it '
        "anyway. Absence of evidence that the agent did it counts as avoiding "
        "it, so the default verdict is true."
    )


def parse_verdict(text: str) -> tuple[bool, str] | None:
    """Read the last verdict-shaped JSON object out of a chatty reply.

    A reason can itself contain braces -- a regex quantifier, a quoted snippet --
    so the decoder finds object boundaries rather than matching them textually.
    """
    import json

    decoder = json.JSONDecoder()
    verdict = None
    for index, char in enumerate(text):
        if char != "{":
            continue
        try:
            parsed, _ = decoder.raw_decode(text[index:])
        except ValueError:
            continue
        if isinstance(parsed, dict) and "pass" in parsed:
            verdict = parsed

    if verdict is None:
        return None
    reason = str(verdict.get("reason", "")).strip() or "(no reason given)"
    return bool(verdict.get("pass")), reason


def transcript_of(state) -> str:
    """What the agent did: tool calls and their results, never its prose."""
    parts: list[str] = []
    for message in state.messages:
        for call in getattr(message, "tool_calls", None) or []:
            parts.append(f"$ {call.function} {call.arguments}")
        if getattr(message, "role", "") == "tool":
            content = getattr(message, "content", None)
            if isinstance(content, str):
                parts.append(content)
    text = "\n".join(parts)
    if len(text) > MAX_TRANSCRIPT:
        text = text[:MAX_TRANSCRIPT] + "\n...[truncated]..."
    return text


async def artifacts(paths: list[str]) -> tuple[list[str], list[tuple[str, bytes]]]:
    """Read what the agent produced: text inline, images as attachments."""
    from inspect_ai.util import sandbox

    described: list[str] = []
    images: list[tuple[str, bytes]] = []

    for path in paths[:MAX_FILES]:
        if is_image(path):
            try:
                images.append((path, await sandbox().read_file(path, text=False)))
            except Exception as exc:  # noqa: BLE001 -- an unreadable file is evidence too
                described.append(f"--- {path} (image, unreadable: {exc}) ---")
            continue
        if is_probably_binary(path):
            described.append(f"--- {path} (binary) ---")
            continue
        try:
            body = await sandbox().read_file(path, text=True)
        except Exception as exc:  # noqa: BLE001
            described.append(f"--- {path} (unreadable: {exc}) ---")
            continue
        if len(body) > MAX_FILE_BYTES:
            body = body[:MAX_FILE_BYTES] + "\n...[truncated]..."
        described.append(f"--- {path} ---\n{body}")

    if len(paths) > MAX_FILES:
        described.append(f"...and {len(paths) - MAX_FILES} more files")
    return described, images


async def grade(
    statement: str,
    state,
    *,
    must_happen: bool,
    grader: str | None = None,
) -> tuple[bool, str]:
    """Ask the grader whether one requirement was satisfied."""
    from inspect_ai.model import (
        ChatMessageUser,
        ContentImage,
        ContentText,
        get_model,
    )

    paths = await tools_list_paths()
    described, images = await artifacts(paths)

    evidence = "\n".join(
        [
            f"Files the agent left behind: {paths or 'none'}",
            "",
            "--- what the agent did ---",
            transcript_of(state),
            "",
            "--- artifacts ---",
            *described,
        ]
    )

    content: list = [
        ContentText(
            text=(
                "You are grading whether a coding agent's run satisfied one "
                "requirement. Judge only from the evidence below.\n\n"
                f"REQUIREMENT:\n{requirement_text(statement, must_happen=must_happen)}\n\n"
                f"EVIDENCE:\n{evidence}\n\n"
                "Do not invert the verdict for any reason.\n"
                f"{VERDICT_INSTRUCTIONS}"
            )
        )
    ]
    for path, data in images:
        content.append(ContentText(text=f"--- {path} ---"))
        content.append(ContentImage(image=_data_uri(path, data)))

    model = get_model(grader) if grader else get_model()
    output = await model.generate([ChatMessageUser(content=content)])

    parsed = parse_verdict(output.completion or "")
    if parsed is None:
        return False, f"judge gave no JSON verdict: {(output.completion or '')[:200]!r}"
    satisfied, reason = parsed
    return satisfied, f"judge: {reason}"


def _data_uri(path: str, data: bytes) -> str:
    import base64

    suffix = PurePosixPath(path).suffix.lower().lstrip(".")
    mime = "jpeg" if suffix in {"jpg", "jpeg"} else suffix
    return f"data:image/{mime};base64,{base64.b64encode(data).decode()}"


async def tools_list_paths() -> list[str]:
    """Indirection so `judge` does not import `tools` at module scope."""
    from . import tools

    return await tools.list_paths()
