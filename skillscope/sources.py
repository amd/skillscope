# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.

"""Where a skill's evals get the source tree of the repo that owns the skill.

Some behavior cases need more than the skill folder: a fixture, a scoring
script, or the product itself, installed. Those live in the repo the skill is
authored in, and getting hold of that repo is this module's job rather than the
hook's.

The reason is provenance, not tidiness. A hook that clones its own repo can
only clone a *branch*, so when the product repo runs its own evals on a pull
request the eval grades whatever ``main`` holds -- never the change under
review. A pull request that breaks a fixture, a scorer, or the skill text then
passes its own eval. Resolving the checkout centrally lets each repo answer
with the tree it actually has:

1. ``SKILL_SOURCE_DIR``, when set, wins outright. The escape hatch: point the
   evals at a local clone, or at a product tree that is not the repo holding
   the skill.
2. Otherwise a vendored skill is pinned by its ``.federated.json``, which
   records the repo and the exact commit the skill was vendored from. Fetched
   into the run's cache directory, so the source matches the skill text beside
   it in the same pull request.
3. Otherwise the git repository the skill folder sits in. In a catalog that is
   the catalog; in a product repo running these evals over its own skills tree
   it is that repo's checkout -- which on a pull request is the merge commit,
   so the eval grades the change under test with nothing to configure.

Stdlib only, like the rest of the runner.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

from . import datasets, gitfetch

# Explicit override, checked before anything is inferred.
SOURCE_DIR_ENV = "SKILL_SOURCE_DIR"

# Written by a federation importer into every vendored skill. Carries the
# upstream repo and the resolved commit, which is all a pinned fetch needs.
MARKER_FILENAME = ".federated.json"

# Where a fetched tree lands inside the per-skill cache directory.
CLONE_DIRNAME = "source"


@dataclass(frozen=True)
class Source:
    """A resolved source tree, and where it came from.

    ``origin`` is for humans: it goes in the run log and the JSON report so a
    failure can be read without guessing which tree was graded.
    """

    path: Path
    origin: str


def read_marker(skill: str) -> dict | None:
    """The federation marker for `skill`, or None if it was authored here."""
    path = datasets.skill_path(skill) / MARKER_FILENAME
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise SystemExit(f"error: {path} is not valid JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise SystemExit(f"error: {path} must be a JSON object.")
    return data


def enclosing_repo(start: Path) -> Path | None:
    """The root of the git repository containing `start`, if any."""
    for candidate in (start, *start.parents):
        if (candidate / ".git").exists():
            return candidate
    return None


def resolve(skill: str, cache_dir: Path) -> Source:
    """The source tree `skill`'s hooks should be graded against.

    `cache_dir` is the runner's per-skill scratch directory, used only when a
    tree has to be fetched. See the module docstring for the order.
    """
    override = os.environ.get(SOURCE_DIR_ENV, "").strip()
    if override:
        path = Path(override).expanduser()
        if not path.is_dir():
            raise SystemExit(
                f"error: {SOURCE_DIR_ENV} is set to {override!r}, which is not a "
                "directory. It must point at a checkout of the repo that owns "
                f"the {skill} skill."
            )
        return Source(path.resolve(), f"${SOURCE_DIR_ENV}")

    marker = read_marker(skill)
    if marker is not None:
        repo, commit = marker.get("repo"), marker.get("commit")
        if not repo or not commit:
            raise SystemExit(
                f"error: {datasets.skill_path(skill) / MARKER_FILENAME} is "
                "missing `repo` or `commit`, so the source tree this skill was "
                f"vendored from cannot be identified. Re-import the skill, or set "
                f"{SOURCE_DIR_ENV} to a checkout of it."
            )
        dest = cache_dir / CLONE_DIRNAME
        gitfetch.fetch_ref(repo, commit, dest)
        return Source(dest.resolve(), f"{repo}@{commit[:12]}")

    skill_dir = datasets.skill_path(skill)
    root = enclosing_repo(skill_dir.resolve())
    if root is None:
        raise SystemExit(
            f"error: {skill} is not a vendored skill and {skill_dir} is not "
            f"inside a git repository, so there is no source tree to grade "
            f"against. Set {SOURCE_DIR_ENV} to the checkout to use."
        )
    return Source(root, "enclosing checkout")
