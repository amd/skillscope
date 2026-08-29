# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.

"""Per-skill eval datasets: discovery, parsing, and structural checks.

Every skill owns one dataset at ``<skill>/evals/evals.json``, holding an
``evaluations`` array. Each evaluation is a user prompt, a yes/no answer to
"should this skill fire?", and -- when the answer is yes -- what should be
true once it has::

    {
      "id": "epyc-vllm-zentorch",
      "skill_should_trigger": true,
      "prompt": "Serve Llama 3.1 8B with zentorch."
    }

A skill may ship a second dataset in the same format at
``<skill>/evals/extended_evals.json``. It is optional, it carries no coverage
requirement of its own -- the tier-0 bar below is counted over ``evals.json``
alone -- and whether it runs at all is the caller's decision
(``extended=True`` here, ``--extended`` on the runner). It is where a product
repo keeps the prompts it wants graded in its own CI without every consumer of
its skills paying for them.

There are two graded commands to satisfy:

  * **routing** -- the skills a workflow lists are installed side by side and
    only the trigger decision is graded ("did the right skill fire, and only
    then?"). Every evaluation of a listed skill runs here.
  * **behavioral** -- just this skill is installed, the run goes to completion,
    and ``expected_behavior`` / ``unexpected_behavior`` / ``logs_contain`` /
    ``files_exist`` are graded ("once it fired, did it do the job?"). Only a
    triggering evaluation can run here, and only if it asserts something.

One prompt graded by both is the point: a routing prompt that nothing grades
is a prompt nobody maintains, and a behavioral test that re-asserts routing
with a substring match is a worse version of a check this module already
models as a field.

``skill_should_trigger: false`` makes the evaluation routing-only. No skill
loads for it, so there is no behavioral phase to hang an assertion or a staged
workspace off, and those fields are rejected rather than silently ignored:
such an evaluation is an ``id``, a ``prompt``, the flag, and maybe a ``note``.

The folder is the identity, so no evaluation names a skill;
``serving-llms-on-epyc/evals/evals.json`` is about ``serving-llms-on-epyc``
and ``skill_should_trigger`` refers to it. A prompt that should trigger a
*different* skill belongs in that skill's dataset: routing installs those
skills side by side, so it is the same assertion either way, and filing it
under the neighbour keeps ``false`` meaning "nothing fires".

Prompt categories are derived rather than declared, because the flag and the
file a prompt lives in already carry the distinction:

  * ``skill_should_trigger: true``               -> ``positive``
  * ``false`` in a skill's own dataset           -> ``near_miss`` (its owner
    wrote it precisely because it sits close to that skill)
  * an evaluation in the shared pool             -> ``unrelated`` (belongs to
    no skill's domain)

Stdlib only, so a run needs no wheels beyond this package. ``machine_plan`` is
the one exception and imports PyYAML lazily; nothing on the run path calls it.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from . import config

PACKAGE_DIR = Path(__file__).resolve().parent

# One dataset per skill, beside the skill it describes.
DATASET_RELPATH = Path("evals") / "evals.json"
# The optional second dataset: same format, no coverage bar, only run when the
# caller asks for it.
EXTENDED_DATASET_RELPATH = Path("evals") / "extended_evals.json"
HOOKS_RELPATH = Path("evals") / "hooks.py"
MACHINE_RELPATH = Path("evals") / "machine.yml"

# Prompts that belong to no skill's domain: ordinary programming work and
# general questions. They are the "unrelated" control group for every skill at
# once, and they are shipped with the harness rather than written per repo
# because nothing in them is specific to a product.
SHARED_NEGATIVES = PACKAGE_DIR / "data" / "negatives.json"

# What a new skill copies to start.
TEMPLATE = PACKAGE_DIR / "data" / "TEMPLATE.json"

# Tier 0, the bar every skill clears before it can ship. Cheap to meet (five
# prompts, no hardware, no assertions) and enforced structurally so a thin
# dataset fails the structural checks without spending a single token.
MIN_POSITIVE_CASES = 3
MIN_NEGATIVE_CASES = 2

# A dataset is one array of evaluations, and every evaluation answers the
# routing question outright rather than leaving it to be inferred.
EVALUATIONS_KEY = "evaluations"
TRIGGER_KEY = "skill_should_trigger"

# Which build of this harness grades the skill. Optional, and a skill owner's
# call: pinning it here means the version that runs a dataset is bumped in the
# same file, and the same review, as the prompts it runs. See `pinned_version`.
VERSION_KEY = "skillscope_version"

# Anything git can resolve: a tag, a branch, a commit. Checked only for shape,
# because whether the ref exists is the launcher's problem and saying so twice
# would mean two places to fix when a ref format changes.
_REF_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._/+-]*$")

# `additionalProperties: false`, by hand. A mistyped key would otherwise be
# silently dropped, quietly turning an expectation into no expectation at all.
#
# The two shapes take different fields, and the difference is not a style
# choice. An evaluation with `skill_should_trigger: false` is graded on exactly
# one thing -- that nothing fired -- and no skill is ever loaded for it, so
# there is no behavioral phase to hang an assertion or a staged workspace off.
# Those are a prompt and nothing more.
TRIGGER_CASE_KEYS = {
    "id",
    "prompt",
    TRIGGER_KEY,
    "expected_behavior",
    "unexpected_behavior",
    "logs_contain",
    "files_exist",
    "workspace",
    "note",
}
NO_TRIGGER_CASE_KEYS = {"id", "prompt", TRIGGER_KEY, "note"}

DATASET_KEYS = {EVALUATIONS_KEY, VERSION_KEY, "comment"}

# JSON has no comments, so `note` is the sanctioned place for one. The runner
# ignores it; without it owners annotate fields that are not free text.
_STRING_LISTS = ("expected_behavior", "unexpected_behavior", "logs_contain", "files_exist")


@dataclass
class Case:
    """One prompt and everything that should be true after the agent sees it."""

    id: str
    prompt: str
    # The skill whose dataset this came from; None for the shared pool.
    skill: str | None
    skill_should_trigger: bool
    expected_behavior: list[str] = field(default_factory=list)
    unexpected_behavior: list[str] = field(default_factory=list)
    logs_contain: list[str] = field(default_factory=list)
    files_exist: list[str] = field(default_factory=list)
    # Directory (relative to the skill root) whose contents seed the workspace.
    workspace: str | None = None
    note: str = ""
    # Came from `evals/extended_evals.json` rather than the required dataset,
    # so it does not count towards our requirements bar and only runs when asked for.
    extended: bool = False

    @property
    def expect_skill(self) -> str | None:
        """The skill that must activate, or None when nothing should.

        Derived, never written down: the owning folder names the skill and
        `skill_should_trigger` says whether it should fire.
        """
        return self.skill if self.skill_should_trigger else None

    @property
    def category(self) -> str:
        """Reporting bucket, derived from the flag and the source file.

        Kept out of the file format on purpose: an owner who has to classify a
        prompt will eventually classify one wrong, and every input needed to
        do it correctly is already here.
        """
        if self.skill_should_trigger:
            return "positive"
        return "near_miss" if self.skill else "unrelated"

    @property
    def has_behavior(self) -> bool:
        """Whether this case grades anything beyond the routing decision."""
        return bool(
            self.expected_behavior
            or self.unexpected_behavior
            or self.logs_contain
            or self.files_exist
        )


def skill_path(skill: str) -> Path:
    """The folder holding `skill`, in the repo under test."""
    return config.active().skill_path(skill)


def dataset_path(skill: str) -> Path:
    return skill_path(skill) / DATASET_RELPATH


def extended_dataset_path(skill: str) -> Path:
    return skill_path(skill) / EXTENDED_DATASET_RELPATH


def hooks_path(skill: str) -> Path:
    return skill_path(skill) / HOOKS_RELPATH


def machine_path(skill: str) -> Path:
    return skill_path(skill) / MACHINE_RELPATH


def declared_skills() -> list[str]:
    """Every skill the repo under test declares, in or out of a routing run.

    This is the set the structural checks cover, and the set behavioral runs are
    planned over. Which of them compete in a routing run is a separate, and
    much smaller, question: see ``config.Config.routing_set``.
    """
    return sorted(config.active().skills)


def skills_with_datasets() -> list[str]:
    """Skills that ship an eval dataset (and so can be run or gated on)."""
    return [skill for skill in declared_skills() if dataset_path(skill).is_file()]


def routing_cases(
    skills: list[str] | tuple[str, ...], errors: list[str] | None = None, *, extended: bool = False
) -> list[Case]:
    """Every prompt a routing run over `skills` grades.

    Their own datasets, pooled, plus the shared pool. Pooling is where the
    coverage comes from: a positive case for skill Y is an implicit negative
    for skill X, so N owners each writing about their own domain produce
    N-squared routing coverage without coordinating. A skill outside the list
    contributes nothing here, because it is not in the room -- neither to win
    a prompt of its own nor to lose one to a neighbour.
    """
    cases: list[Case] = []
    for skill in skills:
        if dataset_path(skill).is_file():
            cases.extend(load_dataset(skill, errors, extended=extended))
    cases.extend(load_shared_negatives(errors))
    return cases


def _parse_case(
    entry: object,
    skill: str | None,
    label: str,
    errors: list[str],
    extended: bool = False,
) -> Case | None:
    """Turn one array element into a Case, appending any problems found."""
    if not isinstance(entry, dict):
        errors.append(f"{label} must be an object.")
        return None

    should_trigger = entry.get(TRIGGER_KEY)
    if not isinstance(should_trigger, bool):
        errors.append(
            f"{label} needs `{TRIGGER_KEY}`: true if this prompt should activate "
            "the skill that owns this file, false if no skill should fire at all."
        )
        return None

    if skill is None and should_trigger:
        errors.append(
            f"{label}: the shared pool belongs to no skill, so every evaluation "
            f"in it needs `{TRIGGER_KEY}: false`. A prompt that should trigger a "
            "skill belongs in that skill's dataset."
        )
        return None

    allowed = TRIGGER_CASE_KEYS if should_trigger else NO_TRIGGER_CASE_KEYS
    unknown = set(entry) - allowed
    # Called out separately from a plain typo: these are real fields on the
    # wrong kind of evaluation, and the reason they are rejected is worth saying.
    misplaced = sorted(unknown & (TRIGGER_CASE_KEYS - NO_TRIGGER_CASE_KEYS))
    if misplaced:
        errors.append(
            f"{label} uses {', '.join(f'`{k}`' for k in misplaced)}, which only "
            f"apply when `{TRIGGER_KEY}` is true. An evaluation expecting nothing "
            "to fire is graded on that alone -- no skill is ever loaded for it, "
            "so there is no behavioral phase to assert anything about."
        )
    unknown = sorted(unknown - set(misplaced))
    if unknown:
        errors.append(
            f"{label} has unknown key(s): {', '.join(unknown)}. "
            f"Allowed here: {', '.join(sorted(allowed))}."
        )

    case_id = entry.get("id")
    if not isinstance(case_id, str) or not case_id.strip():
        errors.append(f"{label} is missing a non-empty string `id`.")
        return None
    case_id = case_id.strip()

    prompt = entry.get("prompt")
    if not isinstance(prompt, str) or not prompt.strip():
        errors.append(f"{label} (`{case_id}`) is missing a non-empty string `prompt`.")
        return None

    lists: dict[str, list[str]] = {}
    for key in _STRING_LISTS:
        value = entry.get(key, [])
        if not isinstance(value, list) or not all(
            isinstance(item, str) and item.strip() for item in value
        ):
            errors.append(
                f"{label} (`{case_id}`): `{key}` must be an array of non-empty strings."
            )
            return None
        lists[key] = [item.strip() for item in value]

    workspace = entry.get("workspace")
    if workspace is not None and (not isinstance(workspace, str) or not workspace.strip()):
        errors.append(f"{label} (`{case_id}`): `workspace` must be a directory path.")
        return None

    note = entry.get("note", "")
    if not isinstance(note, str):
        errors.append(f"{label} (`{case_id}`): `note` must be a string.")
        return None

    return Case(
        id=case_id,
        prompt=prompt.strip(),
        skill=skill,
        skill_should_trigger=should_trigger,
        expected_behavior=lists["expected_behavior"],
        unexpected_behavior=lists["unexpected_behavior"],
        logs_contain=lists["logs_contain"],
        files_exist=lists["files_exist"],
        workspace=workspace.strip() if isinstance(workspace, str) else None,
        note=note,
        extended=extended,
    )


def _parse_version(payload: dict, where: str, errors: list[str]) -> str:
    """The harness version this dataset pins, or "" when it pins none."""
    value = payload.get(VERSION_KEY, "")
    if not isinstance(value, str):
        errors.append(
            f"{where}: `{VERSION_KEY}` must be a string naming a skillscope "
            "tag, branch, or commit."
        )
        return ""
    value = value.strip()
    if value and not _REF_PATTERN.match(value):
        errors.append(
            f"{where}: `{VERSION_KEY}` is {value!r}, which is not a usable git "
            "ref. Use a tag (`v1.2.0`), a branch, or a commit."
        )
        return ""
    return value


def _parse_cases(
    payload: object,
    skill: str | None,
    source: Path,
    errors: list[str],
    extended: bool = False,
) -> list[Case]:
    """Turn one parsed dataset file into cases, appending any problems found."""
    where = source.name

    if not isinstance(payload, dict):
        errors.append(f"{where}: top level must be an object with an `{EVALUATIONS_KEY}` array.")
        return []

    unknown = sorted(set(payload) - DATASET_KEYS)
    if unknown:
        errors.append(f"{where}: unknown top-level key(s): {', '.join(unknown)}.")

    _parse_version(payload, where, errors)

    raw = payload.get(EVALUATIONS_KEY)
    if not isinstance(raw, list) or not raw:
        errors.append(f"{where}: `{EVALUATIONS_KEY}` must be a non-empty array.")
        return []

    cases: list[Case] = []
    for index, entry in enumerate(raw):
        case = _parse_case(
            entry, skill, f"{where}: {EVALUATIONS_KEY}[{index}]", errors, extended
        )
        if case is not None:
            cases.append(case)
    return cases


def _read_dataset(
    skill: str, path: Path, collected: list[str], extended: bool
) -> list[Case]:
    """Parse one dataset file, reporting a read or JSON problem as an error."""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        collected.append(f"{skill}/{path.name}: invalid JSON: {exc}")
        return []
    return _parse_cases(payload, skill, path, collected, extended)


def load_dataset(
    skill: str, errors: list[str] | None = None, *, extended: bool = False
) -> list[Case]:
    """Cases for one skill. Raises SystemExit on error unless collecting.

    The required dataset always, plus ``evals/extended_evals.json`` when
    `extended` and the skill ships one. An absent extended dataset is the
    common case and not a problem: the file is optional by design.
    """
    collected: list[str] = [] if errors is None else errors
    path = dataset_path(skill)
    if not path.is_file():
        collected.append(f"{skill}: missing {DATASET_RELPATH.as_posix()}.")
        cases: list[Case] = []
    else:
        cases = _read_dataset(skill, path, collected, False)

    if extended:
        extra = extended_dataset_path(skill)
        if extra.is_file():
            cases.extend(_read_dataset(skill, extra, collected, True))

    if errors is None and collected:
        raise SystemExit("error: " + "\n       ".join(collected))
    return cases


def dataset_version(skill: str) -> str:
    """The harness version pinned in `skill`'s dataset, or "" when unpinned.

    Read straight from the file rather than carried on every Case: it is a
    property of the dataset, not of a prompt, and the caller that needs it (CI
    planning) has no cases in hand yet.
    """
    path = dataset_path(skill)
    if not path.is_file():
        return ""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return ""
    if not isinstance(payload, dict):
        return ""
    return _parse_version(payload, path.name, [])


def pinned_version(skill: str | None = None) -> str:
    """Which build of the harness should grade `skill`.

    A skill's own dataset wins, because the owner who writes the prompts is
    the one who knows which harness they were written against. Falling back to
    the version this run is already using covers everything that is not one
    skill's run -- including routing, which installs several skills in one
    session and so cannot honor several pins at once.
    """
    if skill is not None:
        pin = dataset_version(skill)
        if pin:
            return pin
    return config.active().version


def load_shared_negatives(errors: list[str] | None = None) -> list[Case]:
    """The repo-wide `unrelated` control group."""
    collected: list[str] = [] if errors is None else errors
    if not SHARED_NEGATIVES.is_file():
        return []
    try:
        payload = json.loads(SHARED_NEGATIVES.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        collected.append(f"{SHARED_NEGATIVES.name}: invalid JSON: {exc}")
        payload = None
    cases = _parse_cases(payload, None, SHARED_NEGATIVES, collected) if payload is not None else []

    if errors is None and collected:
        raise SystemExit("error: " + "\n       ".join(collected))
    return cases


def load_all_cases(errors: list[str] | None = None, *, extended: bool = False) -> list[Case]:
    """Every case in the repo: each skill's dataset plus the shared pool.

    What the structural checks survey. A routing run grades a subset of it
    -- the listed skills' prompts, see ``routing_cases`` -- but every prompt
    in the repo has to be well-formed whether or not it is in the room.
    """
    cases: list[Case] = []
    for skill in skills_with_datasets():
        cases.extend(load_dataset(skill, errors, extended=extended))
    cases.extend(load_shared_negatives(errors))
    return cases


def duplicate_ids(cases: list[Case]) -> list[str]:
    """Case ids used more than once. Ids are repo-wide because routing pools them."""
    return sorted(cid for cid, count in Counter(c.id for c in cases).items() if count > 1)


def filter_cases(cases: list[Case], only: str) -> list[Case]:
    """Narrow `cases` to a comma-separated list of case ids or skill names."""
    if not only.strip():
        return cases
    wanted = {token.strip() for token in only.split(",") if token.strip()}
    selected = [case for case in cases if case.id in wanted or case.skill in wanted]
    if not selected:
        raise SystemExit(f"error: --only '{only}' matched no cases")
    return selected


def structural_errors() -> list[str]:
    """Every structural problem across every dataset, as human-readable strings.

    Run by CI before any tokens are spent, so a malformed dataset fails in
    seconds rather than halfway through a paid run.
    """
    errors: list[str] = []
    cases = load_all_cases(errors, extended=True)
    declared = set(declared_skills())

    for case_id in duplicate_ids(cases):
        errors.append(
            f"duplicate case id `{case_id}`. Ids are repo-wide because routing "
            "pools every skill's cases into one run."
        )

    for case in cases:
        # Only reachable for a triggering evaluation, which is the only kind
        # that owns a skill and the only kind allowed to stage anything.
        if case.workspace and case.skill:
            if not (skill_path(case.skill) / case.workspace).is_dir():
                errors.append(
                    f"case `{case.id}`: `workspace` points at "
                    f"`{case.skill}/{case.workspace}`, which is not a directory."
                )

    for skill in sorted(declared):
        errors.extend(
            tier0_errors(skill, [c for c in cases if c.skill == skill and not c.extended])
        )
        # A malformed machine.yml would schedule a job onto a pool with no
        # runners, which hangs rather than failing. Surface it here, where it
        # costs seconds.
        try:
            machine_plan(skill)
        except SystemExit as exc:
            errors.append(str(exc).removeprefix("error: "))
    return errors


def tier0_errors(skill: str, cases: list[Case]) -> list[str]:
    """Whether `skill` meets the mandatory coverage bar.

    Counted over the required dataset alone (evals.json).
    """
    if not dataset_path(skill).is_file():
        return [
            f"{skill}: no eval dataset. Every skill needs "
            f"`{DATASET_RELPATH.as_posix()}` in its folder with at least "
            f"{MIN_POSITIVE_CASES} evaluations where `{TRIGGER_KEY}` is true "
            f"and {MIN_NEGATIVE_CASES} where it is false. "
            "Copy the template from `skillscope template` to start."
        ]

    errors: list[str] = []
    positive = sum(1 for c in cases if c.skill_should_trigger)
    negative = len(cases) - positive
    if positive < MIN_POSITIVE_CASES:
        errors.append(
            f"{skill}: {positive} evaluation(s) with `{TRIGGER_KEY}: true`; "
            f"Tier 0 needs at least {MIN_POSITIVE_CASES}. Add prompts a real "
            "user would type."
        )
    if negative < MIN_NEGATIVE_CASES:
        errors.append(
            f"{skill}: {negative} evaluation(s) with `{TRIGGER_KEY}: false`; "
            f"Tier 0 needs at least {MIN_NEGATIVE_CASES}. Add prompts close to "
            "this skill's domain that should NOT trigger it."
        )
    return errors


MACHINE_KEYS = {"os", "labels"}


def _read_machine(skill: str) -> dict:
    """The raw ``evals/machine.yml`` for `skill`, or ``{}`` when it has none.

    PyYAML is imported here rather than at module scope: nothing on the run
    path needs this, so a run stays dependency-free.
    """
    path = machine_path(skill)
    if not path.is_file():
        return {}
    import yaml  # noqa: PLC0415 -- keeps the run path stdlib-only

    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise SystemExit(f"error: {path} must be a YAML mapping.")
    return data


def _string_list(path: Path, key: str, value: object) -> list[str]:
    if (
        not isinstance(value, list)
        or not value
        or not all(isinstance(item, str) and item.strip() for item in value)
    ):
        raise SystemExit(
            f"error: {path}: `{key}` must be a non-empty list of strings; "
            f"got {value!r}."
        )
    return [item.strip() for item in value]


def machine_plan(skill: str) -> dict:
    """What kind of machine `skill`'s behavioral cases need.

    An absent ``evals/machine.yml`` is the common case: the everyday runners,
    on the platforms the repo runs on by default. A skill ships one to drop a
    platform it cannot support (``os``) or to ask for a runner label its work
    requires (``labels``)::

        os: [Linux]
        labels: [mi300x]

    Labels rather than a class name, because a class name has to be defined
    somewhere and that somewhere is a second file to keep in step. A label is
    the thing GitHub actually matches a runner on, so a skill saying `mi300x`
    says everything: it lands on a pool carrying that label, and if no pool
    does, that is visible as a queued job rather than hidden behind a mapping
    that quietly resolved to nothing.

    What follows from asking for hardware -- which base labels to start from,
    which pull-request label rations the pool, which environment holds its
    credentials -- is the repo's business and comes from the workflow, not from
    here. Returns ``{os, labels}``, raising SystemExit on a malformed file so
    CI stops at planning.
    """
    path = machine_path(skill)
    data = _read_machine(skill)

    unknown = sorted(set(data) - MACHINE_KEYS)
    if unknown:
        raise SystemExit(
            f"error: {path}: unknown key(s): {', '.join(unknown)}. "
            f"A machine.yml holds only {' and '.join(sorted(MACHINE_KEYS))}."
        )

    platforms = (
        _string_list(path, "os", data["os"])
        if "os" in data
        else list(config.active().behavior_os)
    )
    labels = _string_list(path, "labels", data["labels"]) if "labels" in data else []
    return {"os": platforms, "labels": labels}
