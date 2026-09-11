# Copyright Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: MIT

"""Turn skillscope's dataset into inspect samples.

`evals.json` is the frozen contract: this module is the only place that knows
how a `Case` becomes an inspect `Sample`, so the dataset format and the engine
can move independently.

Expectations ride along in `Sample.metadata` rather than `Sample.target`. A case
asserts several unrelated things at once (files produced, phrases in the
transcript, judged behaviors), which is a poor fit for the single `target`
string inspect scorers conventionally compare against; the scorers in
`engine/scorers.py` read them back out by name.
"""

from __future__ import annotations

from pathlib import Path

from ..behavior import expand
from ..datasets import Case

# Keys written into `Sample.metadata`. Named here so scorers and tasks agree on
# the spelling without importing each other.
SKILL = "skill"
SHOULD_TRIGGER = "skill_should_trigger"
CATEGORY = "category"
EXPECTED = "expected_behavior"
UNEXPECTED = "unexpected_behavior"
LOGS_CONTAIN = "logs_contain"
FILES_EXIST = "files_exist"
EXTENDED = "extended"


def seed_files(seed: Path) -> dict[str, str]:
    """Map a case's ``workspace`` fixture directory onto `Sample.files`.

    The *contents* land at the sandbox working directory, matching what the
    legacy `_stage_workspace` did -- a case hands the agent a starting file to
    edit rather than describing one in prose.
    """
    if not seed.is_dir():
        raise FileNotFoundError(f"workspace fixture directory not found: {seed}")

    files: dict[str, str] = {}
    for path in sorted(seed.rglob("*")):
        if path.is_file():
            target = path.relative_to(seed).as_posix()
            files[target] = str(path)
    return files


def sample_from_case(case: Case, skill_dir: Path, ctx: dict | None = None) -> "object":
    """Build one inspect `Sample` from a `Case`.

    `ctx` supplies `{name}` template variables, expanded with the same
    substitution the legacy engine uses so a prompt containing literal braces
    (JSON snippets, regex quantifiers) survives unchanged.
    """
    from inspect_ai.dataset import Sample

    ctx = ctx or {}
    files = seed_files(skill_dir / case.workspace) if case.workspace else {}

    return Sample(
        id=case.id,
        input=expand(case.prompt, ctx),
        files=files or None,
        metadata={
            SKILL: case.skill,
            SHOULD_TRIGGER: case.skill_should_trigger,
            CATEGORY: case.category,
            EXPECTED: list(case.expected_behavior),
            UNEXPECTED: list(case.unexpected_behavior),
            LOGS_CONTAIN: [expand(t, ctx) for t in case.logs_contain],
            FILES_EXIST: [expand(p, ctx) for p in case.files_exist],
            EXTENDED: case.extended,
        },
    )


def samples_from_cases(
    cases: list[Case], skill_dir_for: "object", ctx: dict | None = None
) -> list:
    """Convert many cases. `skill_dir_for` maps a skill name to its directory."""
    return [
        sample_from_case(case, skill_dir_for(case.skill), ctx)
        for case in cases
    ]
