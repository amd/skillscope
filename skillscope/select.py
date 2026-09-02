# Copyright Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: MIT

"""Decide what a CI run should actually run for a given change.

Emits one JSON object, which the reusable workflow reads once and every job
downstream consumes::

    {
      "routing": true,
      "extended": false,
      "version": "v1.2.0",
      "default": [
        {"skill": "local-ai-use", "os": "Linux",
         "runner": "[\\"self-hosted\\",\\"strix_halo\\",\\"Linux\\"]",
         "gate": "", "version": "v1.2.0"}
      ],
      "scoped": [
        {"skill": "serving-llms-on-instinct", "os": "Linux",
         "runner": "[\\"self-hosted\\",\\"mi300x\\",\\"Linux\\"]",
         "environment": "behavioral-instinct", "gate": "enable_mi_ci",
         "version": "v1.2.0"}
      ],
      "skipped": [{"skill": "serving-llms-on-instinct", "gate": "enable_mi_ci"}],
      "gates": ["enable_mi_ci"]
    }

``default`` and ``scoped`` are GitHub Actions matrices. A leg's ``runs-on``
labels are the base labels the workflow passes plus whatever the skill's
``evals/machine.yml`` asked for, so adding a skill that needs unusual hardware
never means editing this file or the workflow. A hardcoded list of which skills
need which runner lives in the wrong place: the person who knows about the
hardware is the skill's owner, not whoever last touched CI, and the two drift
silently.

The split between ``default`` and ``scoped`` is by credentials, not by
hardware. A skill that asks for labels lands on a pool the repo may hold behind
a pull-request label and pay for out of a separate environment, and the two
matrices have to be separate jobs because a job's credentials are fixed before
its matrix expands. A repo that declares no scoped environment gets one matrix,
labels and all.

``version`` is which build of the harness grades the leg, so a skill can pin
the harness in its own dataset and have CI honor it (see
``datasets.pinned_version``). The top-level one covers everything that is not
one skill's behavioral run.

``extended`` echoes back whether the optional ``evals/extended_evals.json``
datasets are in play, so the workflow decides that once and every job reads the
same answer. It has to match what the runner is passed: selecting a skill whose
only graded prompts live in a dataset the run then ignores schedules a job with
nothing to do.

``skipped`` holds legs whose gate label is missing from the pull request. They
are reported so the gate job can warn that a change shipped without ever
running on real hardware, rather than failing a pull request for a test it
deliberately did not request.
"""

from __future__ import annotations

import json
from pathlib import Path

from . import config, datasets

# The two dataset files a change can touch, as `git diff --name-only` spells
# them. Derived rather than written out, so renaming one in `datasets` is not a
# silent way to stop selecting runs for it.
DATASET_SUFFIX = "/" + datasets.DATASET_RELPATH.as_posix()
EXTENDED_SUFFIX = "/" + datasets.EXTENDED_DATASET_RELPATH.as_posix()


def infra_paths() -> set[str]:
    """Paths that change the shared engine rather than one skill.

    Touching one re-runs everything rather than guessing at the blast radius.
    The workflow names them with ``--infra-paths``, and the workflow file
    itself is the usual entry: it now holds the harness pin and the routing
    set, so a change to it can move any result.
    """
    return set(config.active().infra_paths)


def owning_skill(path: str, skills: dict[str, Path], root: Path) -> str | None:
    """Which declared skill a changed path belongs to, if any."""
    for name, folder in skills.items():
        try:
            prefix = folder.relative_to(root).as_posix() + "/"
        except ValueError:  # a skill outside the repo root; not a change target
            continue
        if path.startswith(prefix):
            return name
    return None


def has_behavior_cases(skill: str, extended: bool = True) -> bool:
    """Whether this skill asserts anything a behavioral run could grade."""
    return any(
        case.has_behavior for case in datasets.load_dataset(skill, extended=extended)
    )


def runs_on(plan: dict, os_name: str) -> list[str]:
    """The ``runs-on`` labels for one leg of a skill's behavioral matrix.

    Base labels from the workflow, then what the skill asked for, then the
    platform -- each added only once, so a pool registered with the platform in
    its label set keeps exactly the labels it has.
    """
    cfg = config.active()
    labels = cfg.base_labels(plan["labels"])
    for label in [*plan["labels"], os_name]:
        if label not in labels:
            labels.append(label)
    return labels


def matrix_entries(
    skills: list[str],
    labels: set[str],
    ignore_gates: bool = False,
    extended: bool = True,
) -> tuple[list[dict], list[dict]]:
    """Split `skills` into matrix legs to run and legs held back by a gate."""
    cfg = config.active()
    include: list[dict] = []
    skipped: list[dict] = []

    for skill in skills:
        if not has_behavior_cases(skill, extended):
            continue
        plan = datasets.machine_plan(skill)
        # Asking for a runner label is what makes a leg scoped: it is going on
        # hardware the repo singled out, which is also the hardware it rations
        # and pays for separately.
        scoped = bool(plan["labels"])
        gate = cfg.scoped_gate if scoped else ""
        if gate and not ignore_gates and gate not in labels:
            skipped.append({"skill": skill, "gate": gate})
            continue

        for os_name in plan["os"]:
            leg = {
                "skill": skill,
                "os": os_name,
                "runner": json.dumps(runs_on(plan, os_name)),
                "gate": gate,
                # Per leg, because a skill owner pins the harness in the same
                # file as the prompts it grades.
                "version": datasets.pinned_version(skill),
            }
            if scoped and cfg.scoped_environment:
                leg["environment"] = cfg.scoped_environment
            include.append(leg)
    return include, skipped


def routing_needed(changed: set[str], extended: bool = True) -> bool:
    """Whether the change can move a routing decision.

    Nothing to move if the routing set came out empty, which is a repo with
    several skills that never said which of them compete, or one that said
    ``none``.

    Otherwise, a skill's description and its prompts are the only inputs, so a
    pull request that only edits a reference file or a helper script under a
    skill does not need to pay for a run. A skill outside the routing set is
    not an input either: it is not in the room to win or lose a prompt, and its
    prompts are not graded there. An extended dataset counts only where it
    runs.
    """
    cfg = config.active()
    routing_set = set(cfg.routing_room)
    if not routing_set:
        return False
    if changed & infra_paths():
        return True

    skills = cfg.skills
    for path in changed:
        if not (
            path.endswith(DATASET_SUFFIX)
            or path.endswith("/SKILL.md")
            or (extended and path.endswith(EXTENDED_SUFFIX))
        ):
            continue
        owner = owning_skill(path, skills, cfg.root)
        # An unrecognized skill file is treated as an input: better one extra
        # routing run than a silently unmeasured description.
        if owner is None or owner in routing_set:
            return True
    return False


def select_from_changes(changed: set[str]) -> list[str]:
    """Skills whose behavioral tests should re-run for a set of changed paths."""
    available = datasets.skills_with_datasets()
    if changed & infra_paths():
        return available
    cfg = config.active()
    skills = cfg.skills
    touched = {owning_skill(path, skills, cfg.root) for path in changed}
    return [skill for skill in available if skill in touched]


def plan(
    skills: list[str],
    *,
    routing: bool,
    labels: set[str],
    ignore_gates: bool = False,
    extended: bool = True,
) -> dict:
    """The selection object CI consumes."""
    include, skipped = matrix_entries(
        skills, labels, ignore_gates=ignore_gates, extended=extended
    )
    return {
        "routing": routing and bool(config.active().routing_room),
        "extended": extended,
        # Routing installs several skills in one session, so it runs at the
        # version this run is already using; a per-skill pin governs that
        # skill's behavioral leg.
        "version": datasets.pinned_version(),
        "default": [leg for leg in include if "environment" not in leg],
        "scoped": [leg for leg in include if "environment" in leg],
        "skipped": skipped,
        "gates": sorted({leg["gate"] for leg in skipped}),
    }
