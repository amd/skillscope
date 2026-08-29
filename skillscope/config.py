# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.

"""Which repository skillscope is testing, and how that repository is wired.

The harness is skill-agnostic, so everything about the repo under test is
data: where its skills live, which of them compete in a routing run, what its
runners are labelled, and which of its files are infrastructure rather than
content.

That data arrives as command-line flags, and the caller that passes them is
the repo's own workflow::

    skillscope select --changed \\
      --skills 'skills/*' \\
      --routing-skills local-ai-use,serving-llms-on-instinct \\
      --behavior-runner '["self-hosted", "strix_halo"]' \\
      --infra-paths .github/workflows/evals.yml

Deliberately not a config file in the repo under test. A repo that runs these
evals already has a workflow saying when to run them, on what, and with which
credentials; splitting the other half of the same decision into a second file
means two places to read, two places to change, and a file whose only reader
is the workflow next to it. Every flag has a default, so a repo that keeps its
skills in ``skills/`` and grades nothing on special hardware passes none of
them.

The defaults are modest rather than clever. Guessing where skills live by
scanning a whole tree finds vendored copies, fixtures, and a contributor's
local install, and each of those silently changes a routing score; naming the
directory costs one flag and cannot drift.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

# The environment contract with the launcher, checked before anything is
# inferred. `SKILLSCOPE_REPO` points at the repo under test (CI runs from its
# checkout, so this is rarely needed). `SKILLSCOPE_VERSION` is the build of the
# harness that is running, so a plan can tell CI to keep using it.
#
# `SKILLSCOPE_SKILLS` is where the skills are, and it is an environment
# variable rather than only a flag because the launcher needs the same answer
# this does: it looks in a skill's dataset for the version pin before it has
# fetched the harness that could parse a flag. `--skills` still wins.
REPO_ENV = "SKILLSCOPE_REPO"
VERSION_ENV = "SKILLSCOPE_VERSION"
SKILLS_ENV = "SKILLSCOPE_SKILLS"

DEFAULT_SKILL_GLOBS = ("skills/*",)

# GitHub-hosted Linux, which is what a repo with no runners of its own has.
DEFAULT_BEHAVIOR_RUNNER = ("ubuntu-latest",)
DEFAULT_BEHAVIOR_OS = ("Linux",)

# The two answers `--routing-skills` takes instead of a list of names: every
# skill in the repo that ships a dataset, and no routing run at all. Both are
# words rather than shapes of an empty flag, because saying nothing means "work
# it out" -- a repo with one skill has only one room its skill can be in -- and
# "no routing at all" is a real answer that has to stay sayable.
ALL_SKILLS = "all"
NO_SKILLS = "none"


@dataclass(frozen=True)
class Config:
    """A resolved view of the repo under test."""

    root: Path

    # Globs naming the directories that hold skills.
    skill_globs: tuple[str, ...] = DEFAULT_SKILL_GLOBS

    # The skills a routing run installs side by side, in the order given, and
    # already resolved: `all` has become the list, a repo whose only skill was
    # never named has become that skill, and `none` -- or a repo with several
    # skills that said nothing about which of them compete -- has become empty,
    # which is no routing run.
    routing_skills: tuple[str, ...] = ()

    # Paths that change the harness rather than one skill, so touching one
    # re-runs every skill instead of guessing at the blast radius.
    infra_paths: frozenset[str] = frozenset()

    # Markdown outside the skills that should have its references checked
    # anyway: a repo's README, its docs tree. Empty by default, because the
    # structure this harness grades is a skill's.
    doc_globs: tuple[str, ...] = ()

    # Regexes matching URLs the external reference check leaves alone. For
    # hosts that are auth-gated or that answer a runner's IP with a 403: real
    # link rot keeps getting caught, and a known liar stops crying wolf.
    excluded_urls: tuple[str, ...] = ()

    # What this repo asks of every skill beyond the format itself: files each
    # one ships beside its SKILL.md, and the `##` headings those of them that
    # are markdown must have something under. A governance card naming an owner
    # and a license is the usual reason. Empty by default, because this is a
    # repo's policy rather than a skill's format.
    skill_files: tuple[str, ...] = ()
    skill_sections: tuple[str, ...] = ()

    # `runs-on` labels for a behavior leg, and the platforms a skill runs on
    # when it does not say. A skill that asks for extra labels in its
    # evals/machine.yml gets `scoped_runner` as its base instead, because the
    # pool that has an MI300X in it is not the pool that answers to the
    # everyday labels.
    behavior_runner: tuple[str, ...] = DEFAULT_BEHAVIOR_RUNNER
    behavior_os: tuple[str, ...] = DEFAULT_BEHAVIOR_OS
    scoped_runner: tuple[str, ...] = ()

    # A pull-request label rationing the scoped pool, and the environment
    # holding its credentials. Both belong to the repo that owns the machines:
    # a skill says what hardware it needs, not who pays for it.
    scoped_gate: str = ""
    scoped_environment: str = ""

    # The build of the harness this run is. Echoed into a CI plan so every leg
    # keeps using it unless the skill's own dataset pins another.
    version: str = ""

    @property
    def skills(self) -> dict[str, Path]:
        """Every skill this repo declares, as ``{name: folder}``.

        A skill is a directory holding a ``SKILL.md``, and its directory name
        is its identity -- the same rule every agent harness applies when it
        loads one.
        """
        found: dict[str, Path] = {}
        for pattern in self.skill_globs:
            for path in sorted(self.root.glob(pattern)):
                if not (path.is_dir() and (path / "SKILL.md").is_file()):
                    continue
                previous = found.get(path.name)
                if previous is not None and previous != path:
                    raise SystemExit(
                        f"error: two skills are both named '{path.name}' "
                        f"({previous} and {path}). A skill's directory name is "
                        "its identity, so the names have to be unique across "
                        "the globs passed to --skills."
                    )
                found[path.name] = path
        return found

    def skill_path(self, skill: str) -> Path:
        """Where `skill` lives. Raises SystemExit when this repo has no such skill."""
        try:
            return self.skills[skill]
        except KeyError:
            known = ", ".join(sorted(self.skills)) or "(none found)"
            raise SystemExit(
                f"error: no skill named '{skill}' under "
                f"{', '.join(self.skill_globs)} in {self.root}. Found: {known}."
            ) from None

    @property
    def routing_set(self) -> dict[str, Path]:
        """The skills to install together for a routing run, as ``{name: folder}``.

        Order is the order they were listed, which is the workflow author's
        order; nothing downstream depends on it, and rewriting it would make
        the report harder to compare against the flag that produced it.
        """
        skills = self.skills
        unknown = [name for name in self.routing_skills if name not in skills]
        if unknown:
            raise SystemExit(
                f"error: --routing-skills names {', '.join(unknown)}, which "
                f"{'is' if len(unknown) == 1 else 'are'} not in this repo. "
                f"Found: {', '.join(sorted(skills)) or '(none)'}."
            )
        return {name: skills[name] for name in self.routing_skills}

    def base_labels(self, extra: list[str] | tuple[str, ...]) -> list[str]:
        """The `runs-on` labels a leg starts from, given what it asked for.

        A leg that names no extra labels belongs on the everyday pool. One that
        does belongs on whatever pool has that hardware, so it starts from
        `scoped_runner` -- falling back to the everyday labels for a repo that
        never declared a second pool, which is better than emitting a bare
        label set that matches nothing.
        """
        if not extra:
            return list(self.behavior_runner)
        return list(self.scoped_runner or self.behavior_runner)


def find_root(start: Path | None = None) -> Path:
    """The repo under test: `$SKILLSCOPE_REPO`, else the enclosing checkout.

    Walking up for a ``.git`` directory means running the CLI from a
    subdirectory works the way every other repo-scoped tool does.
    """
    override = os.environ.get(REPO_ENV, "").strip()
    if override:
        path = Path(override).expanduser().resolve()
        if not path.is_dir():
            raise SystemExit(
                f"error: {REPO_ENV} is set to {override!r}, which is not a directory."
            )
        return path

    here = (start or Path.cwd()).resolve()
    for candidate in (here, *here.parents):
        if (candidate / ".git").exists():
            return candidate
    return here


def _items(value: object, flag: str) -> tuple[str, ...]:
    """Parse a list-valued flag: a JSON array, or comma-separated values.

    JSON is accepted because ``runs-on`` labels are a JSON array everywhere
    else in a workflow, and making the caller translate that into a different
    syntax on the way in is how a label ends up with a stray bracket in it.
    """
    if value is None:
        return ()
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return ()
        if text[0] in "[{":
            try:
                parsed = json.loads(text)
            except json.JSONDecodeError as exc:
                raise SystemExit(f"error: {flag}: invalid JSON: {exc}") from exc
            if not isinstance(parsed, list):
                raise SystemExit(
                    f"error: {flag}: expected a JSON array, got "
                    f"{type(parsed).__name__}."
                )
            items = parsed
        else:
            items = text.split(",")
    else:
        items = list(value)

    resolved: list[str] = []
    for item in items:
        if not isinstance(item, str) or not item.strip():
            raise SystemExit(f"error: {flag}: every entry must be a non-empty string.")
        resolved.append(item.strip())
    return tuple(resolved)


def _is_sentinel(value: object, word: str) -> bool:
    items = _items(value, "--routing-skills")
    return len(items) == 1 and items[0].lower() == word


def wants_all_skills(value: object) -> bool:
    """Whether a ``--routing-skills`` value is the ``all`` shorthand."""
    return _is_sentinel(value, ALL_SKILLS)


def wants_no_skills(value: object) -> bool:
    """Whether a ``--routing-skills`` value is the ``none`` shorthand."""
    return _is_sentinel(value, NO_SKILLS)


def build(
    root: Path | None = None,
    *,
    skills: object = None,
    routing_skills: object = None,
    infra_paths: object = None,
    docs: object = None,
    excluded_urls: object = None,
    skill_files: object = None,
    skill_sections: object = None,
    behavior_runner: object = None,
    behavior_os: object = None,
    scoped_runner: object = None,
    scoped_gate: str = "",
    scoped_environment: str = "",
    version: str | None = None,
    dataset_skills: list[str] | None = None,
) -> Config:
    """A Config from loose values: what the CLI hands over after parsing.

    One place converts strings into the shapes the rest of the harness reads,
    so a flag, a workflow input, and a test fixture cannot disagree about what
    ``"a,b"`` means. `dataset_skills` is every skill in the repo that ships a
    dataset, which resolves both of the routing answers this module cannot
    reach on its own: ``all``, and the room a repo with a single skill never
    had to spell out. The caller supplies it because having a dataset is a
    question about datasets, and this module only knows about directories.
    """
    root = (root or find_root()).resolve()

    globs = (
        _items(skills, "--skills")
        or _items(os.environ.get(SKILLS_ENV, ""), SKILLS_ENV)
        or DEFAULT_SKILL_GLOBS
    )
    routing = _items(routing_skills, "--routing-skills")
    if wants_all_skills(routing):
        routing = tuple(dataset_skills if dataset_skills is not None else ())
    elif wants_no_skills(routing):
        routing = ()
    elif not routing and dataset_skills is not None and len(dataset_skills) == 1:
        # One skill with a dataset is the whole room. Naming it would be the
        # caller repeating back the only answer there is, and the score still
        # measures something: whether that skill fires on its own prompts and
        # stays quiet on its near misses and the shared negatives.
        routing = tuple(dataset_skills)

    return Config(
        root=root,
        skill_globs=globs,
        routing_skills=routing,
        infra_paths=frozenset(_items(infra_paths, "--infra-paths")),
        doc_globs=_items(docs, "--docs"),
        excluded_urls=_items(excluded_urls, "--exclude-url"),
        skill_files=_items(skill_files, "--skill-files"),
        skill_sections=_items(skill_sections, "--skill-sections"),
        behavior_runner=_items(behavior_runner, "--behavior-runner")
        or DEFAULT_BEHAVIOR_RUNNER,
        behavior_os=_items(behavior_os, "--behavior-os") or DEFAULT_BEHAVIOR_OS,
        scoped_runner=_items(scoped_runner, "--scoped-runner"),
        scoped_gate=(scoped_gate or "").strip(),
        scoped_environment=(scoped_environment or "").strip(),
        version=(
            version if version is not None else os.environ.get(VERSION_ENV, "")
        ).strip(),
    )


_active: Config | None = None


def active() -> Config:
    """The config for this process, defaulted on first use."""
    global _active
    if _active is None:
        _active = build()
    return _active


def use(config: Config | None) -> Config | None:
    """Install `config` as the active one and return the previous one.

    For the CLI, which resolves its flags before anything else, and for tests,
    which point the harness at a fixture repo.
    """
    global _active
    previous, _active = _active, config
    return previous
