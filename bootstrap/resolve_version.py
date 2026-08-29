# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.

"""Work out which build of skillscope to run, before running any of it.

This is the whole of the `amd/skillscope@bootstrap` launcher's logic. Callers
reference `@bootstrap` forever; the version that actually grades their skills
is data, bumped in a reviewable one-line diff:

  1. an explicit `version` input, or `--version`
  2. `$SKILLSCOPE_VERSION`
  3. `skillscope_version` in the dataset of the skill being run
  4. the launcher's own ref, so a repo that pins nothing still runs

A skill's own dataset is the only pin a *repo* holds, because the harness
version belongs next to the prompts it grades. Everything above it comes from
the workflow, which is where the rest of the configuration lives too.

Deliberately dependency-free and deliberately ignorant of the harness: it
reads one JSON key and prints a string. Importing the package it is about to
fetch would make the launcher's behavior depend on the version being launched,
which is exactly the coupling `@bootstrap` exists to avoid.

Usage::

    python bootstrap/resolve_version.py --repo . --skill local-ai-use
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

DEFAULT_SKILL_GLOBS = ["skills/*"]
DATASET_RELPATH = "evals/evals.json"
VERSION_KEY = "skillscope_version"

# The result is interpolated into `uvx --from git+https://...@<ref>`, so it has
# to be a plausible git ref and nothing more. Anything with a shell
# metacharacter in it is refused rather than escaped: there is no legitimate
# ref that needs one.
REF_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._/+-]*$")


def _read_json(path: Path) -> dict:
    """Parse `path`, treating anything unreadable as absent.

    A malformed dataset is not this script's problem to report: the harness's
    structural checks say where and why. Failing here would replace that with
    a stack trace from the launcher.
    """
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def dataset_pin(root: Path, skill: str, globs: list[str]) -> str:
    """The `skillscope_version` in `skill`'s dataset, or "" if it has none."""
    for pattern in globs:
        for path in sorted(root.glob(pattern)):
            if path.name != skill or not path.is_dir():
                continue
            value = _read_json(path / DATASET_RELPATH).get(VERSION_KEY, "")
            if isinstance(value, str) and value.strip():
                return value.strip()
    return ""


def resolve(
    *,
    root: Path,
    requested: str = "",
    env: str = "",
    skill: str = "",
    default: str = "",
    globs: list[str] | None = None,
) -> str:
    """The ref to fetch the harness from. See the module docstring for the order."""
    candidates = [
        requested,
        env,
        dataset_pin(root, skill, globs or DEFAULT_SKILL_GLOBS) if skill else "",
        default,
    ]
    version = next((c.strip() for c in candidates if c and c.strip()), "")
    if not version:
        raise SystemExit(
            "error: no skillscope version to run and no default given. Pass "
            "`version` to the action, or pin one in the skill's dataset."
        )
    if not REF_PATTERN.match(version):
        raise SystemExit(
            f"error: {version!r} is not a usable git ref. The version is fetched "
            "as a git ref, so it must be a tag, a branch, or a commit."
        )
    return version


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", default=".", help="Root of the repo being tested.")
    parser.add_argument("--version", default="", help="Explicit version; wins outright.")
    parser.add_argument(
        "--skill", default="", help="The skill being run, whose dataset may pin a version."
    )
    parser.add_argument(
        "--default", default="", help="Fallback when nothing else pins a version."
    )
    parser.add_argument(
        "--skills",
        default="",
        help=(
            "Comma-separated globs naming the directories that hold skills, "
            "so a repo that keeps them somewhere unusual is still searched for "
            f"the pin. Default: {','.join(DEFAULT_SKILL_GLOBS)}."
        ),
    )
    args = parser.parse_args(argv)

    globs = [g.strip() for g in args.skills.split(",") if g.strip()]
    version = resolve(
        root=Path(args.repo).expanduser().resolve(),
        requested=args.version,
        env=os.environ.get("SKILLSCOPE_VERSION", ""),
        skill=args.skill,
        default=args.default,
        globs=globs or None,
    )
    print(version)

    # So the composite action can pass it to the next step.
    output = os.environ.get("GITHUB_OUTPUT", "")
    if output:
        with open(output, "a", encoding="utf-8") as handle:
            handle.write(f"version={version}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
