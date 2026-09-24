# Copyright Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: MIT

"""Tests for the harness itself. No agent, no tokens, no network.

    python -m unittest discover -s tests -t .

Three jobs. First, guard the parts that decide whether a paid run is
trustworthy: routing verdicts, activation detection, and the rules that reject
a malformed dataset. Second, keep the JSON Schema in lockstep with the parser
-- the schema is the field reference skill owners read, and one that has
quietly drifted from what the runner enforces is worse than no schema at all.
Third, hold the harness to being repo-agnostic, which is the whole reason it
lives in its own repo: every test that needs a repo builds a throwaway one in a
temp directory rather than reading whatever happens to be checked out here.
"""

from __future__ import annotations

import argparse
import sys
import asyncio
import inspect
import contextlib
import io
import json
import os
import re
import runpy
import subprocess
import tempfile
import time
import unittest
import urllib.error
from pathlib import Path
from unittest import mock

import skillscope
from skillscope import (
    agent,
    behavior,
    cli,
    config,
    credentials,
    datasets,
    deadline,
    engine as engine_module,
    references,
    routing,
    structure,
)
from skillscope import selection as select_module
from skillscope.datasets import EVALUATIONS_KEY, TRIGGER_KEY
from skillscope.engine import behavioral as engine_behavioral
from skillscope.engine import no_sandbox as engine_no_sandbox
from skillscope.engine import behavioral as engine_behavioral
from skillscope.engine import judge as engine_judge
from skillscope.engine import models as engine_models
from skillscope.engine import sandbox as engine_sandbox
from skillscope.engine import verify as engine_verify
from skillscope.engine import routing as engine_routing
from skillscope.engine import no_sandbox as engine_no_sandbox
from skillscope.engine import behavioral as engine_behavioral
from skillscope.engine import tools as engine_tools

REPO_ROOT = datasets.PACKAGE_DIR.parent
SCHEMA_DIR = datasets.PACKAGE_DIR / "schema"
TRIGGERING = "triggeringEvaluation"
NON_TRIGGERING = "nonTriggeringEvaluation"


def parse(
    payload: dict, skill: str | None = "demo-skill", extended: bool = False
) -> tuple[list, list[str]]:
    """Run the dataset parser over an in-memory payload."""
    errors: list[str] = []
    with tempfile.TemporaryDirectory() as tmp:
        name = "extended_evals.json" if extended else "evals.json"
        source = Path(tmp) / name
        source.write_text(json.dumps(payload), encoding="utf-8")
        cases = datasets._parse_cases(payload, skill, source, errors, extended)
    return cases, errors


def triggers(**case) -> dict:
    """A dataset holding one evaluation that should fire the skill."""
    return {EVALUATIONS_KEY: [{TRIGGER_KEY: True, **case}]}


def triggers_nothing(**case) -> dict:
    """A dataset holding one evaluation where nothing should fire."""
    return {EVALUATIONS_KEY: [{TRIGGER_KEY: False, **case}]}


def tier0_dataset(slug: str, **extra) -> dict:
    """The smallest dataset that clears the mandatory coverage bar."""
    evaluations = [
        {"id": f"{slug}-yes-{i}", TRIGGER_KEY: True, "prompt": f"please do {slug} work {i}"}
        for i in range(datasets.MIN_POSITIVE_CASES)
    ]
    evaluations += [
        {"id": f"{slug}-no-{i}", TRIGGER_KEY: False, "prompt": f"nothing to do with {slug} {i}"}
        for i in range(datasets.MIN_NEGATIVE_CASES)
    ]
    return {EVALUATIONS_KEY: evaluations, **extra}


class Repo:
    """A throwaway repo laid out the way skillscope expects one.

    Every test that needs "a repo with skills in it" builds one of these. The
    harness is supposed to work against any repo, so reading the checkout it
    happens to be running from would test one repo's contents instead of the
    harness -- and would break the moment this code moved.
    """

    def __init__(self, test: unittest.TestCase) -> None:
        tmp = tempfile.TemporaryDirectory()
        test.addCleanup(tmp.cleanup)
        # Resolved, because the config does the same to whatever root it is
        # given: a Windows runner's temp directory arrives in its 8.3 form
        # (`RUNNER~1`), so an unresolved root here would compare unequal to the
        # very path the harness derived from it.
        self.root = Path(tmp.name).resolve()
        self.test = test
        self.settings: dict = {}

    def skill(
        self,
        name: str,
        *,
        dataset: dict | None = None,
        extended: dict | None = None,
        machine: str | None = None,
        hooks: str | None = None,
        workspace: dict[str, str] | None = None,
        description: str = "",
        where: str = ".",
    ) -> Path:
        folder = self.root / where / name
        (folder / "evals").mkdir(parents=True, exist_ok=True)
        (folder / "SKILL.md").write_text(
            f"---\nname: {name}\ndescription: {description or f'Does {name} things.'}\n---\n",
            encoding="utf-8",
        )
        if dataset is not None:
            (folder / "evals" / "evals.json").write_text(
                json.dumps(dataset, indent=2), encoding="utf-8"
            )
        if extended is not None:
            (folder / "evals" / "extended_evals.json").write_text(
                json.dumps(extended, indent=2), encoding="utf-8"
            )
        if machine is not None:
            (folder / "evals" / "machine.yml").write_text(machine, encoding="utf-8")
        if hooks is not None:
            (folder / "evals" / "hooks.py").write_text(hooks, encoding="utf-8")
        for relative, text in (workspace or {}).items():
            path = folder / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text, encoding="utf-8")
        # Skills added after activate() should be visible without re-activating:
        # the config resolves its globs on every read, and several tests grow
        # the repo mid-test to see what the structural checks make of the result.
        return folder

    def chdir(self, relative: str = ".") -> None:
        """Run the rest of the test from a directory inside this repo.

        Registered after the temp directory's own cleanup, so it is undone
        first: Windows will not remove a directory that is the cwd.
        """
        self.test.addCleanup(os.chdir, os.getcwd())
        os.chdir(self.root / relative)

    def activate(self, **settings) -> config.Config:
        """Configure the harness for this repo the way the CLI's flags would."""
        self.settings = settings
        cfg = config.build(self.root, **settings)
        previous = config.use(cfg)
        self.test.addCleanup(config.use, previous)
        return cfg

    def reactivate(self, **overrides) -> config.Config:
        """Re-activate with `overrides` applied, for a test that changes one setting."""
        return self.activate(**{**self.settings, **overrides})


class TestTheVersionIsOneNumber(unittest.TestCase):
    """The version a run reports has to be the version that was packaged.

    The `uses:` pin decides which harness runs, and the workflows derive that
    pin rather than repeating it, so there is no list of refs to keep in step.
    What is left is the label: the action reads `__version__` out of the
    checkout it ran from and reports it, and that is how a caller confirms the
    pin did what they meant. If `__version__` and the packaging metadata can
    disagree, the label is a guess.
    """

    def test_the_packaged_version_matches_the_module(self) -> None:
        declared = re.search(
            r'(?m)^version\s*=\s*"([^"]+)"',
            (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"),
        )
        self.assertIsNotNone(declared, "pyproject.toml declares no version")
        self.assertEqual(declared.group(1), skillscope.__version__)

    def test_the_launcher_reads_the_version_it_reports(self) -> None:
        # The launcher cannot import skillscope -- it runs before anything is
        # installed -- so it scrapes `__version__` with a regex. This is what
        # says the regex still matches the file it is aimed at. Run without
        # `__main__`, so loading it does not launch anything.
        launcher = runpy.run_path(str(REPO_ROOT / "bootstrap" / "launch.py"))
        self.assertEqual(launcher["packaged_version"](REPO_ROOT), skillscope.__version__)


class TestEveryRunStepIsPortable(unittest.TestCase):
    """Nothing that grades a skill may assume the runner's shell.

    A workflow step with no `shell:` gets the runner's default -- bash on Linux
    and macOS, PowerShell on Windows -- so it runs anywhere. A composite action
    has no such option: GitHub requires a shell on every `run:`, and naming
    `bash` is a bet that every runner has it. Hosted Windows does, having Git
    Bash on PATH; a self-hosted Windows box need not, and there the bet loses
    as `bash: command not found` on a step that had nothing to do with bash.

    Which platforms a run lands on is the caller's to decide -- `runner` here,
    `behavior_os` and a skill's own `machine.yml` in the full pipeline -- so
    every step that can reach one of them either names Python as its shell or,
    in a workflow, names none. That is not a taste in scripting languages; it
    is the only shell all three platforms are guaranteed to agree on, and this
    test is what keeps the next step honest.
    """

    # Every runner in these three is a caller's input -- `runner`,
    # `behavior_runner`, `scoped_runner`, and `coordinator_runner` all default
    # to something but none is ours to assume. selftest.yml is left out on
    # purpose: it runs on this repo's own matrix, so it may use bash.
    CI_FILES = (
        Path("action.yml"),
        Path(".github") / "workflows" / "reusable.yml",
        Path(".github") / "workflows" / "skill-evals.yml",
    )

    def steps(self, document: object) -> list[dict]:
        """Every step in a workflow or an action, wherever it is nested."""
        if isinstance(document, dict):
            found = []
            for key, value in document.items():
                if key == "steps" and isinstance(value, list):
                    found += [item for item in value if isinstance(item, dict)]
                found += self.steps(value)
            return found
        if isinstance(document, list):
            return [step for item in document for step in self.steps(item)]
        return []

    def test_no_run_step_names_a_shell_a_runner_might_not_have(self) -> None:
        import yaml

        for relative in self.CI_FILES:
            path = REPO_ROOT / relative
            self.assertTrue(path.is_file(), f"{relative} is missing")
            # A workflow step may leave `shell:` unset and get the runner's
            # default; a composite action step has no default to fall back on.
            portable = {"python"} if relative.name == "action.yml" else {"python", ""}
            document = yaml.safe_load(path.read_text(encoding="utf-8"))
            for step in self.steps(document):
                if "run" not in step:
                    continue
                shell = step.get("shell", "")
                name = step.get("name", step.get("id", "<unnamed>"))
                self.assertIn(
                    shell,
                    portable,
                    f"{relative}: step {name!r} runs under {shell!r}. A step that "
                    f"can land on a runner a caller chose must use one of {sorted(portable)}.",
                )


class TestSchemaStaysInSyncWithParser(unittest.TestCase):
    """The schema is documentation; these tests stop it becoming fiction."""

    def setUp(self) -> None:
        self.schema = json.loads(
            (SCHEMA_DIR / "evals.schema.json").read_text(encoding="utf-8")
        )

    def defs(self, name: str) -> dict:
        return self.schema["$defs"][name]

    def test_top_level_properties_match_parser(self) -> None:
        self.assertEqual(set(self.schema["properties"]), datasets.DATASET_KEYS)

    def test_a_removed_harness_pin_is_an_unknown_key(self) -> None:
        _, errors = parse(tier0_dataset("demo", skillscope_version="v1.2.0"))
        self.assertTrue(any("unknown top-level key" in e for e in errors), errors)

    def test_triggering_properties_match_parser(self) -> None:
        self.assertEqual(
            set(self.defs(TRIGGERING)["properties"]), datasets.TRIGGER_CASE_KEYS
        )

    def test_non_triggering_properties_match_parser(self) -> None:
        self.assertEqual(
            set(self.defs(NON_TRIGGERING)["properties"]), datasets.NO_TRIGGER_CASE_KEYS
        )

    def test_the_flag_is_required_and_discriminates_the_two_shapes(self) -> None:
        for name, value in ((TRIGGERING, True), (NON_TRIGGERING, False)):
            with self.subTest(name):
                self.assertEqual(self.defs(name)["required"], ["id", "prompt", TRIGGER_KEY])
                self.assertEqual(self.defs(name)["properties"][TRIGGER_KEY]["const"], value)

    def test_unknown_keys_are_rejected_by_both(self) -> None:
        for name in (TRIGGERING, NON_TRIGGERING):
            with self.subTest(name):
                self.assertFalse(self.defs(name)["additionalProperties"])
        _, errors = parse(triggers(id="a", prompt="p", expect_skill="demo-skill"))
        self.assertTrue(any("unknown key" in e for e in errors), errors)


class TestMachineSchema(unittest.TestCase):
    """A bad machine.yml means a job that never schedules, so catch it here."""

    def setUp(self) -> None:
        self.schema = json.loads(
            (SCHEMA_DIR / "machine.schema.json").read_text(encoding="utf-8")
        )
        self.repo = Repo(self)
        self.repo.skill("plain-skill", dataset=tier0_dataset("plain"))
        self.repo.skill(
            "gpu-skill",
            dataset=tier0_dataset("gpu"),
            machine="os: [Linux]\nlabels: [mi300x]\n",
        )
        self.repo.activate(
            behavior_runner=["self-hosted", "strix_halo"],
            behavior_os=["Linux", "Windows"],
            scoped_runner=["self-hosted"],
            scoped_gate="enable_mi_ci",
            scoped_environment="behavioral-instinct",
        )

    def test_documented_keys_match_the_parser(self) -> None:
        self.assertEqual(set(self.schema["properties"]), datasets.MACHINE_KEYS)

    def test_no_list_key_is_enumerated_in_the_schema(self) -> None:
        # Neither `os` nor `labels` can be: a label means whatever a repo
        # registered its runners with, so the schema documents what the key is
        # for and the workflow supplies the labels around it. Scoped to the
        # list-valued keys, since `sandbox` names a file rather than a set.
        for key, spec in self.schema["properties"].items():
            if spec.get("type") != "array":
                continue
            with self.subTest(key=key):
                self.assertNotIn("enum", spec["items"])

    def test_every_machine_yml_in_the_repo_resolves(self) -> None:
        for skill in datasets.declared_skills():
            with self.subTest(skill=skill):
                self.assertTrue(datasets.machine_plan(skill)["os"])

    def test_a_skill_without_the_file_gets_the_everyday_runners(self) -> None:
        plan = datasets.machine_plan("plain-skill")
        self.assertEqual(plan["os"], ["Linux", "Windows"])
        self.assertEqual(plan["labels"], [])
        self.assertEqual(
            select_module.runs_on(plan, "Windows"),
            ["self-hosted", "strix_halo", "Windows"],
        )

    def test_asking_for_a_label_lands_the_leg_on_the_scoped_pool(self) -> None:
        # The skill names the hardware it needs and nothing else. The base
        # labels, the label rationing the pool, and the environment holding its
        # key belong to the repo that owns the machines.
        self.assertEqual(
            datasets._read_machine("gpu-skill"), {"os": ["Linux"], "labels": ["mi300x"]}
        )
        plan = datasets.machine_plan("gpu-skill")
        self.assertEqual(plan["os"], ["Linux"])
        self.assertEqual(
            select_module.runs_on(plan, "Linux"), ["self-hosted", "mi300x", "Linux"]
        )

    def test_a_label_is_not_repeated(self) -> None:
        # A pool registered with the platform in its label set keeps exactly the
        # labels it has.
        plan = {"os": ["Linux"], "labels": ["Linux", "mi300x"]}
        self.assertEqual(
            select_module.runs_on(plan, "Linux"), ["self-hosted", "Linux", "mi300x"]
        )


class TestMachineRejections(unittest.TestCase):
    """Failing at planning beats scheduling a job onto a pool that has no runners."""

    def setUp(self) -> None:
        self.repo = Repo(self)
        self.repo.skill("demo-skill", dataset=tier0_dataset("demo"))
        self.repo.activate(behavior_os=["Linux", "Windows"])

    def plan(self, text: str) -> dict:
        path = self.repo.root / "demo-skill" / "evals" / "machine.yml"
        path.write_text(text, encoding="utf-8")
        return datasets.machine_plan("demo-skill")

    def test_a_retired_key_is_rejected_rather_than_ignored(self) -> None:
        # `runner_type`, `runner`, `gate`, `environment`, and `reason` all used
        # to live here. Silently dropping one would leave a skill on the wrong
        # hardware, or run scarce hardware with no gate.
        for text in (
            "runner_type: instinct\n",
            "gate: enable_mi_ci\n",
            "environment: behavioral-instinct\n",
            "reason: because\n",
            "runner: [a, b]\n",
        ):
            with self.subTest(text.strip()), self.assertRaises(SystemExit) as caught:
                self.plan(text)
            self.assertIn("unknown key", str(caught.exception))

    def test_an_empty_os_list(self) -> None:
        with self.assertRaises(SystemExit) as caught:
            self.plan("os: []\n")
        self.assertIn("`os`", str(caught.exception))

    def test_an_empty_label_list(self) -> None:
        # Asking for hardware and naming none is a leg that would silently land
        # back on the everyday pool.
        with self.assertRaises(SystemExit) as caught:
            self.plan("labels: []\n")
        self.assertIn("`labels`", str(caught.exception))

    def test_a_label_that_is_not_a_string(self) -> None:
        with self.assertRaises(SystemExit) as caught:
            self.plan("labels: mi300x\n")
        self.assertIn("`labels`", str(caught.exception))

    def test_the_minimum_useful_files(self) -> None:
        self.assertEqual(self.plan("os: [Linux]\n")["os"], ["Linux"])
        labelled = self.plan("labels: [mi300x]\n")
        self.assertEqual(labelled["labels"], ["mi300x"])
        # Saying nothing about platforms still means the repo's platforms.
        self.assertEqual(labelled["os"], ["Linux", "Windows"])

    def test_structural_checks_report_a_broken_machine_file_rather_than_raising(self) -> None:
        # `skillscope structural` has to survey every skill, so one bad file is
        # a reported error, not an abandoned run.
        (self.repo.root / "demo-skill" / "evals" / "machine.yml").write_text(
            "labels: nope\n", encoding="utf-8"
        )
        self.assertTrue(any("`labels`" in e for e in datasets.structural_errors()))


class TestConfig(unittest.TestCase):
    """The repo under test is data, and this is how the flags describe it."""

    def test_a_repo_that_configures_nothing_still_works(self) -> None:
        repo = Repo(self)
        repo.skill("demo-skill", dataset=tier0_dataset("demo"))
        repo.chdir()
        cfg = repo.activate()
        self.assertEqual(cfg.skill_globs, config.DEFAULT_SKILL_GLOBS)
        self.assertEqual(cfg.routing_room, ())
        self.assertEqual(datasets.declared_skills(), ["demo-skill"])

    def test_skills_can_live_anywhere_the_globs_say(self) -> None:
        repo = Repo(self)
        repo.skill("shipped", dataset=tier0_dataset("shipped"), where="agents/skills")
        repo.activate(skills_dir="agents/skills/*")
        self.assertEqual(datasets.declared_skills(), ["shipped"])
        self.assertTrue(datasets.dataset_path("shipped").is_file())

    def test_the_default_looks_one_level_down_and_no_further(self) -> None:
        # Deep enough for a repo that keeps its skills where you are standing,
        # shallow enough that a vendored copy further down is not found and
        # silently graded.
        repo = Repo(self)
        repo.skill("at-the-root", dataset=tier0_dataset("root"))
        repo.skill("buried", dataset=tier0_dataset("buried"), where="vendor/skills")
        repo.chdir()
        repo.activate()
        self.assertEqual(datasets.declared_skills(), ["at-the-root"])

    def test_the_default_is_the_directory_the_command_was_run_from(self) -> None:
        # A repo whose skills sit a level down is the usual layout, and
        # `cd skills && skillscope structural` is what a person does about it.
        # The glob is still reported against the root, which is the base every
        # other path in a run is measured from.
        repo = Repo(self)
        repo.skill("shipped", dataset=tier0_dataset("shipped"), where="skills")
        repo.chdir("skills")
        cfg = repo.activate()
        self.assertEqual(cfg.skill_globs, ("skills/*",))
        self.assertEqual(datasets.declared_skills(), ["shipped"])

    def test_a_command_run_from_outside_the_repo_gets_that_repos_root(self) -> None:
        # `--repo somewhere-else` is not standing anywhere in it, so the only
        # directory it can mean is the root it was handed.
        repo = Repo(self)
        repo.skill("at-the-root", dataset=tier0_dataset("root"))
        self.assertEqual(repo.activate().skill_globs, config.DEFAULT_SKILL_GLOBS)
        self.assertEqual(datasets.declared_skills(), ["at-the-root"])

    def test_a_glob_that_was_passed_is_relative_to_the_root_not_the_cwd(self) -> None:
        # Every other path flag is root-relative, and a workflow that names
        # `skills/*` means the same thing wherever its runner happens to be.
        repo = Repo(self)
        repo.skill("shipped", dataset=tier0_dataset("shipped"), where="skills")
        repo.chdir("skills")
        repo.activate(skills_dir="skills/*")
        self.assertEqual(datasets.declared_skills(), ["shipped"])

    def test_a_directory_without_a_skill_file_is_not_a_skill(self) -> None:
        repo = Repo(self)
        repo.skill("real-skill", dataset=tier0_dataset("real"))
        (repo.root / "notes").mkdir(parents=True)
        repo.activate()
        self.assertEqual(datasets.declared_skills(), ["real-skill"])

    def test_an_unknown_skill_names_what_is_available(self) -> None:
        repo = Repo(self)
        repo.skill("real-skill", dataset=tier0_dataset("real"))
        repo.activate()
        with self.assertRaises(SystemExit) as caught:
            datasets.skill_path("ghost")
        self.assertIn("real-skill", str(caught.exception))

    def test_a_list_can_be_json_or_comma_separated(self) -> None:
        # runs-on labels are a JSON array everywhere else in a workflow, so
        # making the caller translate them on the way in is how a label ends up
        # with a stray bracket in it.
        repo = Repo(self)
        for value in ('["self-hosted", "strix_halo"]', "self-hosted,strix_halo"):
            with self.subTest(value=value):
                cfg = config.build(repo.root, behavior_runner=value)
                self.assertEqual(cfg.behavior_runner, ("self-hosted", "strix_halo"))

    def test_a_malformed_list_names_the_flag_that_was_wrong(self) -> None:
        repo = Repo(self)
        for value in ("[oops", '{"a": 1}', "[1, 2]", "a,,b"):
            with self.subTest(value=value):
                with self.assertRaises(SystemExit) as caught:
                    config.build(repo.root, behavior_runner=value)
                self.assertIn("--behavior-runner", str(caught.exception))

    def test_the_skill_globs_can_come_from_the_environment(self) -> None:
        # The action passes the same globs the CLI would take as --skills-dir.
        repo = Repo(self)
        repo.skill("shipped", dataset=tier0_dataset("shipped"), where="agents/skills")
        with mock.patch.dict(os.environ, {config.SKILLS_ENV: "agents/skills/*"}):
            self.assertEqual(config.build(repo.root).skill_globs, ("agents/skills/*",))
            self.assertEqual(
                config.build(repo.root, skills_dir="skills/*").skill_globs,
                ("skills/*",),
            )


class TestRoutingSet(unittest.TestCase):
    """Who a skill competes against is listed, never inferred."""

    def setUp(self) -> None:
        self.repo = Repo(self)
        self.repo.skill("one", dataset=tier0_dataset("one"))
        self.repo.skill("two", dataset=tier0_dataset("two"))

    def test_the_listed_skills_are_what_gets_installed(self) -> None:
        cfg = self.repo.activate(routing_room="one,two")
        self.assertEqual(list(cfg.routing_set), ["one", "two"])
        self.assertEqual(cfg.routing_set["one"], self.repo.root / "one")

    def test_an_unlisted_skill_is_not_in_the_room(self) -> None:
        cfg = self.repo.activate(routing_room="one")
        self.assertEqual(list(cfg.routing_set), ["one"])

    def test_listing_nothing_where_there_is_a_choice_means_no_routing_run(self) -> None:
        cfg = self.repo.activate(dataset_skills=["one", "two"])
        self.assertEqual(cfg.routing_set, {})
        plan = select_module.plan(["one"], routing=True, labels=set())
        self.assertFalse(plan["routing"])

    def test_all_stands_for_every_skill_with_a_dataset(self) -> None:
        # Resolved by the CLI, which is where "has a dataset" can be answered.
        cfg = config.build(
            self.repo.root,
            routing_room="all",
            dataset_skills=["one", "two"],
        )
        self.assertEqual(cfg.routing_room, ("one", "two"))

    def test_none_says_no_routing_run_outright(self) -> None:
        # The one way to turn routing off, and it survives a repo having only
        # one skill -- which is otherwise enough to infer a room.
        for available in (["one"], ["one", "two"]):
            with self.subTest(available=available):
                cfg = config.build(
                    self.repo.root, routing_room="none", dataset_skills=available
                )
                self.assertEqual(cfg.routing_room, ())

    def test_a_skill_that_does_not_exist_is_refused(self) -> None:
        cfg = self.repo.activate(routing_room="one,ghost")
        with self.assertRaises(SystemExit) as caught:
            cfg.routing_set  # noqa: B018 -- the property is the assertion
        self.assertIn("ghost", str(caught.exception))

    def test_the_listed_order_is_kept(self) -> None:
        cfg = self.repo.activate(routing_room="two,one")
        self.assertEqual(list(cfg.routing_set), ["two", "one"])


class TestASingleSkillIsItsOwnRoom(unittest.TestCase):
    """One skill with a dataset is not a choice, so nothing has to be made."""

    def setUp(self) -> None:
        self.repo = Repo(self)
        self.repo.skill("only-skill", dataset=tier0_dataset("only"))

    def resolve(self, **settings) -> config.Config:
        """Configure in two passes, the way the CLI does.

        Which skills ship a dataset is a question about the repo, so it can
        only be answered once the repo is readable.
        """
        self.repo.activate(**settings)
        return self.repo.reactivate(dataset_skills=datasets.skills_with_datasets())

    def test_the_only_skill_with_a_dataset_is_the_room(self) -> None:
        self.assertEqual(list(self.resolve().routing_set), ["only-skill"])

    def test_it_is_the_room_for_planning_too(self) -> None:
        # Otherwise the flag would be redundant on the runner and still
        # required for CI to schedule the job that runs it.
        self.resolve(infra_paths=".github/workflows/evals.yml")
        self.assertTrue(select_module.routing_needed({"only-skill/SKILL.md"}))
        self.assertTrue(
            select_module.plan(["only-skill"], routing=True, labels=set())["routing"]
        )

    def test_a_second_skill_makes_it_a_choice_again(self) -> None:
        self.repo.skill("neighbour", dataset=tier0_dataset("neighbour"))
        self.assertEqual(self.resolve().routing_set, {})

    def test_a_skill_without_a_dataset_is_not_a_candidate(self) -> None:
        # It has no prompts, so it could not be graded in the room it would
        # otherwise make ambiguous.
        self.repo.skill("undocumented")
        self.assertEqual(list(self.resolve().routing_set), ["only-skill"])

    def test_saying_none_still_turns_routing_off(self) -> None:
        cfg = self.resolve(routing_room="none")
        self.assertEqual(cfg.routing_set, {})
        self.assertFalse(
            select_module.plan(["only-skill"], routing=True, labels=set())["routing"]
        )

    def test_the_cli_resolves_the_room_before_the_command_runs(self) -> None:
        # Through the flags, because the two-pass configure in the CLI is what
        # turns "said nothing" into the one skill there is.
        self.repo.activate()  # registers the cleanup that restores the config
        cli._configure(
            cli.build_parser().parse_args(
                ["--repo", str(self.repo.root), "routing"]
            )
        )
        self.assertEqual(config.active().routing_room, ("only-skill",))


class TestARoutingRunWithNobodyInTheRoom(unittest.TestCase):
    """What `routing` does when the routing set comes out empty."""

    def setUp(self) -> None:
        self.repo = Repo(self)
        self.repo.skill("one", dataset=tier0_dataset("one"))
        self.repo.skill("two", dataset=tier0_dataset("two"))
        self.repo.activate()

    def args(self, *argv) -> argparse.Namespace:
        return cli.build_parser().parse_args(["routing", *argv])

    def test_a_repo_with_a_choice_to_make_is_told_what_its_options_are(self) -> None:
        with self.assertRaises(SystemExit) as caught:
            cli._empty_room(self.args())
        message = str(caught.exception)
        for expected in ("all", "one, two"):
            self.assertIn(expected, message)

    def test_asking_for_routing_and_emptying_the_room_is_a_contradiction(self) -> None:
        with self.assertRaises(SystemExit) as caught:
            cli._empty_room(self.args("--routing-room", "none"))
        self.assertIn("--routing-room none", str(caught.exception))

    def configure(self, *argv) -> None:
        cli._configure(
            cli.build_parser().parse_args(
                ["--repo", str(self.repo.root), "routing", *argv]
            )
        )

    def test_skill_names_the_room_when_the_room_was_not(self) -> None:
        self.configure("--skill", "one")
        self.assertEqual(config.active().routing_room, ("one",))

    def test_skill_can_name_several_as_the_room(self) -> None:
        self.configure("--skill", "two,one")
        self.assertEqual(config.active().routing_room, ("two", "one"))

    def test_an_explicit_room_wins_over_skill(self) -> None:
        self.configure("--routing-room", "two", "--skill", "one")
        self.assertEqual(config.active().routing_room, ("two",))

    def test_none_is_an_explicit_room_and_stays_empty(self) -> None:
        self.configure("--routing-room", "none", "--skill", "one")
        self.assertEqual(config.active().routing_room, ())


class TestCommands(unittest.TestCase):
    """The three graders are commands, not modes of run."""

    def test_structural_routing_and_behavioral_are_commands(self) -> None:
        parser = cli.build_parser()
        for command in ("structural", "routing", "behavioral"):
            with self.subTest(command):
                self.assertEqual(parser.parse_args([command]).command, command)

    def test_run_is_not_a_command(self) -> None:
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                cli.build_parser().parse_args(["run"])

    def test_neither_grader_has_a_mode_flag(self) -> None:
        for command in ("routing", "behavioral"):
            with self.subTest(command):
                self.assertFalse(hasattr(cli.build_parser().parse_args([command]), "mode"))

    def test_behavioral_does_not_take_routing_flags(self) -> None:
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                cli.build_parser().parse_args(["behavioral", "--routing-room", "all"])
            with self.assertRaises(SystemExit):
                cli.build_parser().parse_args(["behavioral", "--min-accuracy", "0"])

    def test_the_three_graders_share_a_timeout(self) -> None:
        parser = cli.build_parser()
        for command in ("structural", "routing", "behavioral"):
            with self.subTest(command):
                args = parser.parse_args([command])
                self.assertEqual(args.timeout, 900.0)

    def test_routing_keeps_a_separate_case_timeout(self) -> None:
        args = cli.build_parser().parse_args(["routing"])
        self.assertEqual(args.timeout, 900.0)
        self.assertEqual(args.case_timeout, 240.0)

    def test_select_does_not_take_a_timeout(self) -> None:
        args = cli.build_parser().parse_args(["select", "--all"])
        self.assertFalse(hasattr(args, "timeout"))

    def test_select_takes_a_commit_pair_or_a_list_of_paths_but_not_both(self) -> None:
        args = cli.build_parser().parse_args(["select", "--since", "base", "head"])
        self.assertEqual(args.since, ["base", "head"])
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                cli.build_parser().parse_args(
                    ["select", "--since", "base", "head", "--changed"]
                )


class TestDeadline(unittest.TestCase):
    """The command-level --timeout, distinct from a routing case's own cap."""

    def test_cap_is_the_tighter_bound(self) -> None:
        bound = deadline.Deadline(10, command="routing", start=time.perf_counter() - 6)
        self.assertAlmostEqual(bound.cap(30), 4, places=1)
        self.assertAlmostEqual(bound.cap(2), 2, places=1)

    def test_an_elapsed_bound_is_expired(self) -> None:
        bound = deadline.Deadline(1, command="routing", start=time.perf_counter() - 2)
        self.assertTrue(bound.expired())
        self.assertEqual(bound.cap(240), 0.0)
        self.assertIn("--timeout of 1s", bound.message())

    def test_an_expired_deadline_does_not_start_a_routing_case(self) -> None:
        cases, errors = parse(triggers(id="hung", prompt="go"))
        self.assertEqual(errors, [])
        bound = deadline.Deadline(1, command="routing", start=time.perf_counter() - 2)
        previous = deadline.use(bound)
        try:
            outcome = routing.run_case(
                cases[0], {"demo-skill": Path(".")}, routing.RoutingConfig()
            )
        finally:
            deadline.use(previous)
        self.assertEqual(outcome.verdict, "error")
        self.assertEqual(outcome.stop_reason, "timeout")
        self.assertIn("routing exceeded --timeout", outcome.error)

    def test_an_expired_deadline_does_not_start_a_behavioral_case(self) -> None:
        cases, errors = parse(triggers(id="hung", prompt="go", logs_contain=["x"]))
        self.assertEqual(errors, [])
        bound = deadline.Deadline(1, command="behavioral", start=time.perf_counter() - 2)
        previous = deadline.use(bound)
        try:
            outcome = behavior.run_case(cases[0], {}, None, "opus", "high")
        finally:
            deadline.use(previous)
        self.assertFalse(outcome.passed)
        self.assertIn("behavioral exceeded --timeout", outcome.error)


class TestCredentialResolution(unittest.TestCase):
    """Which credential a graded job runs with, and what it refuses to do."""

    # What a runner exposes to a job that was granted `id-token: write`.
    RUNNER = {
        "ACTIONS_ID_TOKEN_REQUEST_URL": "https://pipelines.example/idtoken?api-version=2.0",
        "ACTIONS_ID_TOKEN_REQUEST_TOKEN": "runner-token",
    }
    RULE = {
        "FEDERATION_RULE_ID": "fdrl_rule",
        "FEDERATION_ORGANIZATION_ID": "0000-org",
        "FEDERATION_SERVICE_ACCOUNT_ID": "svac_account",
    }

    def recorder(self, calls: list[dict]):
        """A stand-in for both HTTP calls the federation path makes."""

        def fetch(url, *, data=None, headers=None):
            calls.append({"url": url, "data": data, "headers": dict(headers or {})})
            if "idtoken" in url:
                return json.dumps({"value": "the.jwt.value"}).encode("utf-8")
            return json.dumps(
                {"access_token": "sk-ant-oat01-minted", "expires_in": 600}
            ).encode("utf-8")

        return fetch

    def test_a_key_is_exported_as_the_key(self) -> None:
        exported = credentials.resolve({"API_KEY": " sk-ant-a-key ", "SECRET_NAME": "K"})
        self.assertEqual(exported, {"ANTHROPIC_API_KEY": "sk-ant-a-key"})

    def test_a_gateway_key_carries_its_base_url_and_headers(self) -> None:
        exported = credentials.resolve(
            {
                "API_KEY": "the-key",
                "SECRET_NAME": "K",
                "API_BASE_URL": "https://gateway.example/Anthropic",
                "API_CUSTOM_HEADERS": "Ocp-Apim-Subscription-Key: $API_KEY",
            }
        )
        self.assertEqual(exported["ANTHROPIC_BASE_URL"], "https://gateway.example/Anthropic")
        self.assertEqual(
            exported["ANTHROPIC_CUSTOM_HEADERS"], "Ocp-Apim-Subscription-Key: the-key"
        )

    def test_an_empty_secret_names_the_secret_and_the_environment(self) -> None:
        with self.assertRaises(credentials.CredentialError) as raised:
            credentials.resolve(
                {"API_KEY": "", "SECRET_NAME": "MY_KEY", "ENVIRONMENT": "behavioral-instinct"}
            )
        self.assertIn("MY_KEY", str(raised.exception))
        self.assertIn("behavioral-instinct", str(raised.exception))

    def test_federation_trades_an_oidc_token_for_a_bearer_token(self) -> None:
        calls: list[dict] = []
        exported = credentials.resolve(
            # A caller that federates one leg keeps `api_key_secret` at its
            # default, so the key is present and must still be ignored.
            {**self.RUNNER, **self.RULE, "API_KEY": "a-key-that-must-not-win"},
            fetch=self.recorder(calls),
        )
        self.assertEqual(exported, {"ANTHROPIC_AUTH_TOKEN": "sk-ant-oat01-minted"})

        asked, traded = calls
        self.assertIn("audience=https%3A%2F%2Fapi.anthropic.com", asked["url"])
        self.assertEqual(asked["headers"]["Authorization"], "Bearer runner-token")
        self.assertEqual(traded["url"], credentials.TOKEN_URL)
        body = json.loads(traded["data"])
        self.assertEqual(body["grant_type"], credentials.GRANT_TYPE)
        self.assertEqual(body["assertion"], "the.jwt.value")
        self.assertEqual(body["federation_rule_id"], "fdrl_rule")

    def test_a_workspace_is_sent_only_when_the_caller_named_one(self) -> None:
        calls: list[dict] = []
        credentials.resolve({**self.RUNNER, **self.RULE}, fetch=self.recorder(calls))
        self.assertNotIn("workspace_id", json.loads(calls[1]["data"]))

        calls.clear()
        credentials.resolve(
            {**self.RUNNER, **self.RULE, "FEDERATION_WORKSPACE_ID": "wrkspc_one"},
            fetch=self.recorder(calls),
        )
        self.assertEqual(json.loads(calls[1]["data"])["workspace_id"], "wrkspc_one")

    def test_a_job_without_the_permission_is_told_which_one(self) -> None:
        with self.assertRaises(credentials.CredentialError) as raised:
            credentials.resolve(dict(self.RULE), fetch=self.recorder([]))
        self.assertIn("id-token: write", str(raised.exception))

    def test_federation_needs_the_whole_triple(self) -> None:
        with self.assertRaises(credentials.CredentialError) as raised:
            credentials.resolve(
                {**self.RUNNER, "FEDERATION_RULE_ID": "fdrl_rule"}, fetch=self.recorder([])
            )
        self.assertIn("federation_organization_id", str(raised.exception))
        self.assertIn("federation_service_account_id", str(raised.exception))

    def test_federation_refuses_to_hand_a_gateway_an_anthropic_token(self) -> None:
        for contradiction in ("API_BASE_URL", "API_CUSTOM_HEADERS"):
            with self.subTest(contradiction=contradiction):
                with self.assertRaises(credentials.CredentialError) as raised:
                    credentials.resolve(
                        {**self.RUNNER, **self.RULE, contradiction: "https://gateway.example"},
                        fetch=self.recorder([]),
                    )
                self.assertIn("api.anthropic.com", str(raised.exception))

    def test_a_refused_exchange_says_where_the_reason_is_recorded(self) -> None:
        def refuse(url, *, data=None, headers=None):
            if "idtoken" in url:
                return json.dumps({"value": "the.jwt.value"}).encode("utf-8")
            raise credentials.CredentialError(f"{url} returned 401: Authentication failed")

        with self.assertRaises(credentials.CredentialError) as raised:
            credentials.resolve({**self.RUNNER, **self.RULE}, fetch=refuse)
        self.assertIn("authentication history", str(raised.exception))
        self.assertIn("match_subject_prefix", str(raised.exception))

    def test_the_minted_token_is_masked_before_anything_can_log_it(self) -> None:
        calls: list[dict] = []
        with tempfile.TemporaryDirectory() as tmp:
            github_env = Path(tmp) / "github.env"
            github_env.touch()
            environment = {**self.RUNNER, **self.RULE, "GITHUB_ENV": str(github_env)}
            stdout = io.StringIO()
            with mock.patch.dict(os.environ, environment, clear=True), mock.patch.object(
                credentials, "_http", self.recorder(calls)
            ), contextlib.redirect_stdout(stdout):
                self.assertEqual(credentials.main(), 0)

            printed = stdout.getvalue()
            self.assertIn("::add-mask::sk-ant-oat01-minted", printed)
            written = github_env.read_text(encoding="utf-8")
            self.assertIn("ANTHROPIC_AUTH_TOKEN", written)
            self.assertIn("sk-ant-oat01-minted", written)
            self.assertNotIn("ANTHROPIC_API_KEY", written)

    def test_a_gateway_key_reaches_the_env_file_but_never_stdout(self) -> None:
        # GitHub masks the key only because it arrived as a secret, so nothing
        # printed may carry it, substituted into a header or otherwise.
        with tempfile.TemporaryDirectory() as tmp:
            github_env = Path(tmp) / "github.env"
            github_env.touch()
            environment = {
                "EXPECT_KEY": "true",
                "ANTHROPIC_API_KEY": "the-gateway-key",
                "API_BASE_URL": "https://gateway.example/anthropic",
                "API_CUSTOM_HEADERS": "Ocp-Apim-Subscription-Key: $API_KEY",
                "GITHUB_ENV": str(github_env),
            }
            stdout, stderr = io.StringIO(), io.StringIO()
            with mock.patch.dict(os.environ, environment, clear=True), contextlib.redirect_stdout(
                stdout
            ), contextlib.redirect_stderr(stderr):
                self.assertEqual(credentials.main(["--reusable"]), 0)

            self.assertNotIn("the-gateway-key", stdout.getvalue() + stderr.getvalue())
            written = github_env.read_text(encoding="utf-8")
            self.assertIn("ANTHROPIC_BASE_URL<<", written)
            self.assertIn("Ocp-Apim-Subscription-Key: the-gateway-key", written)

    def test_a_gateway_without_a_key_is_refused(self) -> None:
        with self.assertRaises(credentials.CredentialError) as raised:
            credentials.resolve_for_reusable(
                {"EXPECT_KEY": "false", "API_BASE_URL": "https://gateway.example"}
            )
        self.assertIn("no key is configured", str(raised.exception))
        self.assertEqual(credentials.resolve_for_reusable({"EXPECT_KEY": "false"}), {})

    def test_running_credentials_as_a_script_does_not_shadow_stdlib_select(self) -> None:
        # Graded jobs run this file by path, which puts the package directory on
        # sys.path. A module named select.py would shadow the stdlib and this
        # import would die.
        import sys

        script = Path(credentials.__file__).resolve()
        with tempfile.TemporaryDirectory() as tmp:
            github_env = Path(tmp) / "github.env"
            github_env.touch()
            completed = subprocess.run(
                [sys.executable, str(script)],
                capture_output=True,
                text=True,
                encoding="utf-8",
                env={
                    **os.environ,
                    "API_KEY": "sk-ant-a-key",
                    "SECRET_NAME": "K",
                    "GITHUB_ENV": str(github_env),
                    "FEDERATION_RULE_ID": "",
                },
            )
        self.assertEqual(
            completed.returncode, 0, completed.stderr or completed.stdout
        )


class TestActionLauncher(unittest.TestCase):
    """The composite action runs the checkout it was pinned at."""

    def setUp(self) -> None:
        import importlib.util

        path = Path(__file__).resolve().parent.parent / "bootstrap" / "launch.py"
        spec = importlib.util.spec_from_file_location("skillscope_launch", path)
        self.module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.module)

    def test_it_installs_from_the_action_checkout(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            (root / "pyproject.toml").write_text("[project]\nname = 'x'\n", encoding="utf-8")
            with mock.patch.dict(os.environ, {"SKILLSCOPE_ACTION_PATH": str(root)}):
                self.assertEqual(self.module.action_root(), root)


class TestSelection(unittest.TestCase):
    """What CI runs for a change, and what it is right to skip."""

    def setUp(self) -> None:
        self.repo = Repo(self)
        self.graded = tier0_dataset("alpha")
        self.graded[EVALUATIONS_KEY].append(
            {
                "id": "alpha-graded",
                TRIGGER_KEY: True,
                "prompt": "do it",
                "logs_contain": ["alpha.py"],
            }
        )
        self.repo.skill("alpha", dataset=self.graded)
        self.repo.skill("beta", dataset=tier0_dataset("beta"))
        self.repo.activate(
            routing_room="alpha,beta",
            infra_paths=".github/workflows/evals.yml",
        )

    def gpu(self, skill: str = "alpha") -> None:
        """Give `skill` a machine.yml asking for hardware this repo rations."""
        (self.repo.root / skill / "evals" / "machine.yml").write_text(
            "labels: [gpu]\n", encoding="utf-8"
        )
        self.repo.reactivate(
            scoped_runner="self-hosted",
            scoped_gate="enable_gpu_ci",
            scoped_environment="gpu-evals",
        )

    def test_only_skills_with_gradeable_behavior_get_a_leg(self) -> None:
        plan = select_module.plan(["alpha", "beta"], routing=True, labels=set())
        self.assertEqual([leg["skill"] for leg in plan["default"]], ["alpha"])
        self.assertNotIn("version", plan)
        for leg in plan["default"]:
            self.assertNotIn("version", leg)

    def test_a_touched_skill_is_selected(self) -> None:
        self.assertEqual(
            select_module.select_from_changes({"alpha/SKILL.md"}), ["alpha"]
        )

    def test_an_infra_path_selects_everything(self) -> None:
        # The workflow holds the routing set, so a change to it can move any
        # result.
        self.assertEqual(
            select_module.select_from_changes({".github/workflows/evals.yml"}),
            ["alpha", "beta"],
        )

    def test_an_unrelated_change_selects_nothing(self) -> None:
        self.assertEqual(select_module.select_from_changes({"README.md"}), [])

    def test_a_description_change_buys_a_routing_run(self) -> None:
        self.assertTrue(select_module.routing_needed({"alpha/SKILL.md"}))

    def test_a_dataset_change_buys_a_routing_run(self) -> None:
        self.assertTrue(select_module.routing_needed({"beta/evals/evals.json"}))

    def test_a_reference_file_under_a_skill_does_not(self) -> None:
        self.assertFalse(select_module.routing_needed({"alpha/reference.md"}))

    def test_an_unlisted_skills_description_is_not_a_routing_input(self) -> None:
        self.repo.skill("draft", dataset=tier0_dataset("draft"))
        self.assertFalse(select_module.routing_needed({"draft/SKILL.md"}))

    def test_with_no_routing_set_nothing_buys_a_routing_run(self) -> None:
        self.repo.reactivate(routing_room="")
        self.assertFalse(select_module.routing_needed({"alpha/SKILL.md"}))
        self.assertFalse(
            select_module.routing_needed({".github/workflows/evals.yml"})
        )

    def test_a_gated_leg_is_reported_rather_than_run(self) -> None:
        self.gpu()

        held = select_module.plan(["alpha"], routing=False, labels=set())
        self.assertEqual(held["default"], [])
        self.assertEqual(held["scoped"], [])
        self.assertEqual(held["skipped"], [{"skill": "alpha", "gate": "enable_gpu_ci"}])
        self.assertEqual(held["gates"], ["enable_gpu_ci"])

        labelled = select_module.plan(["alpha"], routing=False, labels={"enable_gpu_ci"})
        self.assertEqual(labelled["skipped"], [])
        self.assertEqual(len(labelled["scoped"]), 1)
        self.assertEqual(labelled["scoped"][0]["environment"], "gpu-evals")
        self.assertEqual(
            json.loads(labelled["scoped"][0]["runner"]),
            ["self-hosted", "gpu", "Linux"],
        )

        forced = select_module.plan(["alpha"], routing=False, labels=set(), ignore_gates=True)
        self.assertEqual(len(forced["scoped"]), 1)

    def test_credentials_split_the_two_matrices(self) -> None:
        # A job's credentials are fixed before its matrix expands, so legs that
        # read a scoped environment cannot share a job with legs that do not.
        self.gpu()
        plan = select_module.plan(["alpha", "beta"], routing=True, labels={"enable_gpu_ci"})
        self.assertTrue(all("environment" not in leg for leg in plan["default"]))
        self.assertTrue(all("environment" in leg for leg in plan["scoped"]))
        self.assertEqual([leg["skill"] for leg in plan["scoped"]], ["alpha"])

    def test_hardware_with_no_environment_stays_in_one_matrix(self) -> None:
        # Only credentials force a second job. A repo that rations a pool but
        # pays for it out of the same key should not get an extra one.
        (self.repo.root / "alpha" / "evals" / "machine.yml").write_text(
            "labels: [gpu]\n", encoding="utf-8"
        )
        self.repo.reactivate(scoped_runner="self-hosted", scoped_gate="enable_gpu_ci")
        plan = select_module.plan(["alpha"], routing=False, labels={"enable_gpu_ci"})
        self.assertEqual(plan["scoped"], [])
        self.assertEqual(len(plan["default"]), 1)
        self.assertEqual(
            json.loads(plan["default"][0]["runner"]), ["self-hosted", "gpu", "Linux"]
        )


class TestWhatABranchChanged(unittest.TestCase):
    """Selection is planned from the merge base, not from the base branch tip.

    A pull request names two commits, and the difference between their trees is
    not the same question as what the branch did. As soon as the base moves on
    without the branch, everything merged into it comes back in that diff, in
    reverse -- so a change to one skill re-runs its neighbours, and a base
    commit that touched an infra path re-runs the whole catalog.
    """

    def setUp(self) -> None:
        self.repo = Repo(self)
        self.repo.skill("alpha", dataset=tier0_dataset("alpha"))
        self.repo.skill("beta", dataset=tier0_dataset("beta"))
        self.repo.activate(
            routing_room="alpha,beta",
            infra_paths=".github/workflows/evals.yml",
        )
        self.git("init", "--quiet", "--initial-branch", "main")
        self.git("config", "user.email", "selftest@example.invalid")
        self.git("config", "user.name", "skillscope selftest")
        self.base = self.commit("everything so far")

    def git(self, *args: str) -> str:
        done = subprocess.run(
            ["git", "-C", str(self.repo.root), *args],
            capture_output=True,
            text=True,
            encoding="utf-8",
            check=True,
        )
        return done.stdout.strip()

    def commit(self, message: str) -> str:
        self.git("add", "--all")
        self.git("commit", "--quiet", "--allow-empty", "-m", message)
        return self.git("rev-parse", "HEAD")

    def write(self, relative: str, text: str) -> None:
        path = self.repo.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")

    def branch_behind_an_advanced_base(self) -> tuple[str, str]:
        """A branch off `main`, with `main` moving on after it left.

        Returns the base branch's new tip and the branch's head -- the pair a
        pull request event hands CI.
        """
        self.git("checkout", "--quiet", "-b", "touches-alpha")
        self.write("alpha/SKILL.md", "---\nname: alpha\ndescription: Does alpha things now.\n---\n")
        head = self.commit("edit one skill")

        self.git("checkout", "--quiet", "main")
        self.write(".github/workflows/evals.yml", "name: evals\n")
        base = self.commit("something else lands on the base branch")
        return base, head

    def test_a_branch_is_planned_for_its_own_commits(self) -> None:
        base, head = self.branch_behind_an_advanced_base()
        self.assertEqual(select_module.changed_paths(base, head), ["alpha/SKILL.md"])

    def test_the_base_branchs_own_commits_are_not_this_branchs(self) -> None:
        # The regression this pins: the plain diff between the two commits
        # names a file only the base branch touched, and that file is an infra
        # path, so planning from it re-runs every skill in the repo.
        base, head = self.branch_behind_an_advanced_base()
        self.assertIn(
            ".github/workflows/evals.yml",
            self.git("diff", "--name-only", base, head).splitlines(),
        )
        self.assertEqual(
            select_module.select_from_changes(set(select_module.changed_paths(base, head))),
            ["alpha"],
        )

    def test_a_branch_that_does_touch_an_infra_path_still_re_runs_everything(self) -> None:
        self.git("checkout", "--quiet", "-b", "touches-the-harness")
        self.write(".github/workflows/evals.yml", "name: evals\n")
        head = self.commit("edit the harness")
        self.assertEqual(
            select_module.select_from_changes(
                set(select_module.changed_paths(self.base, head))
            ),
            ["alpha", "beta"],
        )

    def test_unrelated_histories_fall_back_to_the_plain_diff(self) -> None:
        # No common ancestor to diff from -- a clone too shallow to hold one,
        # or histories that really are unrelated. Selecting too much costs a
        # slow run; selecting too little ships an untested change.
        self.git("checkout", "--quiet", "--orphan", "elsewhere")
        self.git("rm", "-rq", "--cached", ".")
        self.write("beta/SKILL.md", "---\nname: beta\ndescription: Does beta things now.\n---\n")
        head = self.commit("a history of its own")

        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr):
            changed = select_module.changed_paths(self.base, head)
        self.assertIn("no merge base", stderr.getvalue())
        self.assertIn("beta/SKILL.md", changed)


class TestCaseExpectations(unittest.TestCase):
    """`skill_should_trigger` is the whole expectation."""

    def test_a_triggering_evaluation_targets_the_owning_skill(self) -> None:
        cases, errors = parse(triggers(id="a", prompt="p"))
        self.assertEqual(errors, [])
        self.assertEqual(cases[0].expect_skill, "demo-skill")
        self.assertEqual(cases[0].category, "positive")

    def test_a_non_triggering_evaluation_is_a_near_miss_for_the_owning_skill(self) -> None:
        cases, errors = parse(triggers_nothing(id="a", prompt="p"))
        self.assertEqual(errors, [])
        self.assertIsNone(cases[0].expect_skill)
        self.assertEqual(cases[0].category, "near_miss")

    def test_shared_pool_cases_are_unrelated(self) -> None:
        cases, errors = parse(triggers_nothing(id="a", prompt="p"), skill=None)
        self.assertEqual(errors, [])
        self.assertIsNone(cases[0].expect_skill)
        self.assertEqual(cases[0].category, "unrelated")

    def test_both_kinds_live_in_one_array(self) -> None:
        cases, errors = parse(
            {
                EVALUATIONS_KEY: [
                    {"id": "a", TRIGGER_KEY: True, "prompt": "p"},
                    {"id": "b", TRIGGER_KEY: False, "prompt": "q"},
                ]
            }
        )
        self.assertEqual(errors, [])
        self.assertEqual([c.skill_should_trigger for c in cases], [True, False])

    def test_has_behavior_only_when_something_is_asserted(self) -> None:
        cases, _ = parse(
            {
                EVALUATIONS_KEY: [
                    {"id": "a", TRIGGER_KEY: True, "prompt": "p"},
                    {
                        "id": "b",
                        TRIGGER_KEY: True,
                        "prompt": "p",
                        "expected_behavior": ["do the thing"],
                    },
                ]
            }
        )
        self.assertFalse(cases[0].has_behavior)
        self.assertTrue(cases[1].has_behavior)


class TestDatasetRejections(unittest.TestCase):
    def test_missing_id(self) -> None:
        _, errors = parse(triggers(prompt="p"))
        self.assertTrue(any("`id`" in e for e in errors), errors)

    def test_missing_prompt(self) -> None:
        _, errors = parse(triggers(id="a"))
        self.assertTrue(any("`prompt`" in e for e in errors), errors)

    def test_the_trigger_flag_is_required(self) -> None:
        # Defaulting it would recreate the hazard the flag exists to remove:
        # an omitted field silently deciding the routing expectation.
        _, errors = parse({EVALUATIONS_KEY: [{"id": "a", "prompt": "p"}]})
        self.assertTrue(any(TRIGGER_KEY in e for e in errors), errors)

    def test_the_trigger_flag_must_be_a_boolean(self) -> None:
        for value in ("yes", "true", 1, None):
            with self.subTest(value=value):
                _, errors = parse(
                    {EVALUATIONS_KEY: [{"id": "a", "prompt": "p", TRIGGER_KEY: value}]}
                )
                self.assertTrue(any(TRIGGER_KEY in e for e in errors), errors)

    def test_an_empty_dataset(self) -> None:
        _, errors = parse({EVALUATIONS_KEY: []})
        self.assertTrue(any("non-empty array" in e for e in errors), errors)

    def test_evaluations_must_be_an_array(self) -> None:
        _, errors = parse({EVALUATIONS_KEY: {"id": "a", "prompt": "p"}})
        self.assertTrue(any("non-empty array" in e for e in errors), errors)

    def test_a_non_triggering_evaluation_takes_a_prompt_and_nothing_else(self) -> None:
        # No skill is ever loaded for these, so there is no behavioral phase for
        # an assertion to be graded in or a workspace to be staged into.
        for key, value in (
            ("expected_behavior", ["x"]),
            ("unexpected_behavior", ["x"]),
            ("logs_contain", ["x"]),
            ("files_exist", ["x"]),
            ("workspace", "evals/files/thing"),
        ):
            with self.subTest(key):
                _, errors = parse(triggers_nothing(id="a", prompt="p", **{key: value}))
                self.assertTrue(
                    any(f"`{key}`" in e and TRIGGER_KEY in e for e in errors), errors
                )

    def test_a_non_triggering_evaluation_never_reaches_behavioral(self) -> None:
        cases, errors = parse(triggers_nothing(id="a", prompt="p", note="why"))
        self.assertEqual(errors, [])
        self.assertFalse(cases[0].has_behavior)

    def test_the_shared_pool_cannot_expect_a_trigger(self) -> None:
        _, errors = parse(triggers(id="a", prompt="p"), skill=None)
        self.assertEqual(len(errors), 1, errors)
        self.assertIn("belongs to no skill", errors[0])

    def test_string_lists_reject_a_bare_string(self) -> None:
        _, errors = parse(triggers(id="a", prompt="p", expected_behavior="do the thing"))
        self.assertTrue(any("array of non-empty strings" in e for e in errors), errors)

    def test_duplicate_ids_are_found(self) -> None:
        cases, _ = parse(
            {
                EVALUATIONS_KEY: [
                    {"id": "a", TRIGGER_KEY: True, "prompt": "p"},
                    {"id": "a", TRIGGER_KEY: False, "prompt": "q"},
                ]
            }
        )
        self.assertEqual(datasets.duplicate_ids(cases), ["a"])


class TestTier0(unittest.TestCase):
    def setUp(self) -> None:
        self.repo = Repo(self)
        self.repo.skill("demo-skill", dataset=tier0_dataset("demo"))
        self.repo.activate()

    def test_thin_dataset_is_rejected(self) -> None:
        cases, _ = parse(triggers(id="a", prompt="p"))
        errors = datasets.tier0_errors("demo-skill", cases)
        self.assertTrue(any(f"{TRIGGER_KEY}: true" in e for e in errors), errors)
        self.assertTrue(any(f"{TRIGGER_KEY}: false" in e for e in errors), errors)

    def test_the_minimum_dataset_passes(self) -> None:
        cases, errors = parse(tier0_dataset("demo"), skill="demo-skill")
        self.assertEqual(errors, [])
        self.assertEqual(datasets.tier0_errors("demo-skill", cases), [])

    def test_a_skill_with_no_dataset_is_reported(self) -> None:
        self.repo.skill("bare-skill")
        errors = datasets.tier0_errors("bare-skill", [])
        self.assertEqual(len(errors), 1)
        self.assertIn("no eval dataset", errors[0])


class TestExtendedDataset(unittest.TestCase):
    """The optional second dataset: same format, no bar, opt-in run."""

    def setUp(self) -> None:
        self.repo = Repo(self)
        self.repo.activate()

    def extra(self) -> dict:
        return {
            EVALUATIONS_KEY: [
                {
                    "id": "extended-one",
                    TRIGGER_KEY: True,
                    "prompt": "p",
                    "expected_behavior": ["do the thing"],
                }
            ]
        }

    def load(self, payload: dict | None, *, extended: bool) -> list:
        name = f"skill-{int(payload is not None)}{int(extended)}"
        self.repo.skill(name, dataset=tier0_dataset(name), extended=payload)
        return datasets.load_dataset(name, extended=extended)

    def test_cases_are_added_only_when_asked_for(self) -> None:
        required = {c.id for c in self.load(self.extra(), extended=False)}
        both = {c.id for c in self.load(self.extra(), extended=True)}
        self.assertNotIn("extended-one", required)
        self.assertIn("extended-one", both)

    def test_extended_cases_are_marked_and_others_are_not(self) -> None:
        cases = {c.id: c for c in self.load(self.extra(), extended=True)}
        self.assertTrue(cases["extended-one"].extended)
        self.assertFalse(any(c.extended for c in cases.values() if c.id != "extended-one"))

    def test_a_missing_file_is_not_an_error(self) -> None:
        self.assertTrue(self.load(None, extended=True))

    def test_extended_cases_do_not_count_towards_tier0(self) -> None:
        # Otherwise a skill could clear the mandatory bar with prompts this
        # repo never runs.
        self.repo.skill("demo-skill", dataset=tier0_dataset("demo"))
        cases, errors = parse(tier0_dataset("demo"), skill="demo-skill", extended=True)
        self.assertEqual(errors, [])
        self.assertTrue(all(c.extended for c in cases))
        self.assertTrue(
            datasets.tier0_errors("demo-skill", [c for c in cases if not c.extended])
        )

    def test_the_format_is_the_same_one(self) -> None:
        # No separate parser, so an extended dataset is rejected for the same
        # reasons the required one is.
        _, errors = parse(triggers(prompt="p"), extended=True)
        self.assertTrue(any("`id`" in e for e in errors), errors)


class TestWholeRepoStructure(unittest.TestCase):
    """What `skillscope structural` guarantees about a repo, whichever repo it is."""

    def setUp(self) -> None:
        self.repo = Repo(self)
        self.hooks = "def check(run, case, ctx):\n    pass\n"
        self.repo.skill(
            "alpha",
            dataset=tier0_dataset("alpha"),
            hooks=self.hooks,
            workspace={"evals/files/stub/main.py": "print('hi')\n"},
        )
        self.repo.skill("beta", dataset=tier0_dataset("beta"))
        self.repo.activate(routing_room="alpha,beta")

    def test_a_healthy_repo_checks_out_clean(self) -> None:
        self.assertEqual(datasets.structural_errors(), [])

    def test_a_skill_without_a_dataset_fails_the_structural_checks(self) -> None:
        self.repo.skill("undocumented")
        self.assertTrue(any("undocumented" in e for e in datasets.structural_errors()))

    def test_duplicate_ids_across_skills_are_caught(self) -> None:
        # Ids are repo-wide because routing pools every listed skill's cases.
        self.repo.skill("gamma", dataset=tier0_dataset("alpha"))
        self.assertTrue(any("duplicate case id" in e for e in datasets.structural_errors()))

    def test_a_workspace_pointing_nowhere_is_caught(self) -> None:
        dataset = tier0_dataset("delta")
        dataset[EVALUATIONS_KEY].append(
            {
                "id": "delta-staged",
                TRIGGER_KEY: True,
                "prompt": "edit it",
                "workspace": "evals/files/missing",
                "files_exist": ["main.py"],
            }
        )
        self.repo.skill("delta", dataset=dataset)
        self.assertTrue(any("`workspace`" in e for e in datasets.structural_errors()))

    def test_every_skill_with_a_dataset_is_a_declared_skill(self) -> None:
        self.assertEqual(
            sorted(datasets.declared_skills()), sorted(datasets.skills_with_datasets())
        )

    def test_every_listed_skill_brings_prompts_to_the_routing_run(self) -> None:
        # A skill in the room with no gradeable prompt of its own would silently
        # drop out of the score rather than failing.
        listed = config.active().routing_room
        cases = datasets.routing_cases(list(listed))
        covered = {case.expect_skill for case in cases if case.expect_skill}
        self.assertEqual(sorted(covered), sorted(listed))

    def test_hooks_are_importable_and_expose_known_entry_points(self) -> None:
        known = {"setup_session", "setup", "teardown", "check"}
        for skill in datasets.skills_with_datasets():
            if not datasets.hooks_path(skill).is_file():
                continue
            with self.subTest(skill=skill):
                module = behavior.load_hooks(skill)
                exported = {
                    name
                    for name in dir(module)
                    if not name.startswith("_") and callable(getattr(module, name))
                }
                self.assertTrue(exported & known, f"{skill} hooks export nothing usable")

    def test_the_shipped_negatives_pool_parses(self) -> None:
        shared = datasets.load_shared_negatives()
        self.assertTrue(shared)
        self.assertTrue(all(c.category == "unrelated" for c in shared))

    def test_template_is_a_valid_dataset(self) -> None:
        # New owners copy this file, so a template the parser rejects would
        # greet every one of them with an error they did not cause.
        template = json.loads(datasets.TEMPLATE.read_text(encoding="utf-8"))
        cases, errors = parse(template, skill="alpha")
        self.assertEqual(errors, [])
        self.assertEqual(datasets.tier0_errors("alpha", cases), [])


class TestTheGateAPaidRunPassesFirst(unittest.TestCase):
    """Which skills the structural checks cover before any tokens are spent.

    A run asked for one skill is gated on that skill. Holding it back for a
    neighbour's mistake would make one skill's malformed file everybody else's
    problem, and the repo-wide answer is what `structural` is for.
    """

    def setUp(self) -> None:
        self.repo = Repo(self)
        self.repo.skill("alpha", dataset=tier0_dataset("alpha"))
        # The neighbour, with a machine.yml naming a key that does not exist.
        self.repo.skill(
            "beta", dataset=tier0_dataset("beta"), machine="runner_type: mi300x\n"
        )
        self.repo.activate(routing_room="alpha,beta")

    def args(self, *argv) -> argparse.Namespace:
        return cli.build_parser().parse_args([*argv, "--skip-preflight"])

    def fails(self, call, *arguments) -> str:
        """The stderr of a gate that stopped the run."""
        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr), self.assertRaises(SystemExit):
            call(*arguments)
        return stderr.getvalue()

    def test_a_neighbours_mistake_does_not_stop_a_run_of_one_skill(self) -> None:
        selected = cli._prepare_graded_run(self.args("behavioral", "--skill", "alpha"))
        self.assertEqual(selected, ["alpha"])

    def test_the_selected_skills_own_mistake_still_stops_it(self) -> None:
        stderr = self.fails(
            cli._prepare_graded_run, self.args("behavioral", "--skill", "beta")
        )
        self.assertIn("runner_type", stderr)

    def test_routing_is_gated_on_everyone_in_the_room(self) -> None:
        # A routing score is about all of them, so all of them are read --
        # whichever one --skill narrows the report to.
        stderr = self.fails(
            cli._prepare_graded_run,
            self.args("routing", "--skill", "alpha"),
            ["alpha", "beta"],
        )
        self.assertIn("runner_type", stderr)

    def test_routing_reads_the_room_and_not_the_repo(self) -> None:
        # A routing run leaves --skill off, so the gate has to take its scope
        # from the room rather than from "every skill that has a dataset".
        self.repo.reactivate(routing_room="alpha")
        args = self.args("routing", "--routing-room", "alpha")
        self.assertEqual(cli._prepare_graded_run(args, ["alpha"]), ["alpha", "beta"])

    def test_the_repo_wide_check_still_covers_the_repo(self) -> None:
        self.assertIn("runner_type", self.fails(cli._structural_or_exit))

    def test_a_neighbours_broken_link_is_not_this_runs_problem(self) -> None:
        (self.repo.root / "beta" / "reference.md").write_text(
            "[gone](./nowhere.md)\n", encoding="utf-8"
        )
        self.assertIn("nowhere.md", self.fails(cli._structural_or_exit))
        self.assertEqual(cli._structural_or_exit(["alpha"]), [])

    def test_the_docs_tree_waits_for_the_repo_wide_check(self) -> None:
        # --docs is a repo's own markdown, so it belongs to no skill's run.
        self.repo.reactivate(docs="*.md")
        (self.repo.root / "README.md").write_text("[gone](./nowhere.md)\n", encoding="utf-8")
        self.assertIn("README.md", self.fails(cli._structural_or_exit))
        self.assertEqual(cli._structural_or_exit(["alpha"]), [])

    def test_a_duplicate_id_outside_the_scope_is_the_repo_wide_checks_business(self) -> None:
        self.repo.skill("gamma", dataset=tier0_dataset("alpha"))
        self.assertEqual(datasets.structural_errors(["alpha"]), [])
        self.assertTrue(
            any("duplicate case id" in e for e in datasets.structural_errors())
        )


class TestSkillStructure(unittest.TestCase):
    """The skill folder itself: what the format requires, and what a repo adds."""

    def setUp(self) -> None:
        self.repo = Repo(self)
        self.folder = self.repo.skill("demo-skill", dataset=tier0_dataset("demo"))
        self.repo.activate()

    def write(self, text: str, skill: str = "demo-skill") -> None:
        (self.repo.root / skill / "SKILL.md").write_text(text, encoding="utf-8")

    def declares(self, frontmatter: str, body: str = "\n# Demo\n") -> None:
        self.write(f"---\n{frontmatter}\n---\n{body}")

    def test_a_skill_in_shape_reports_nothing(self) -> None:
        self.assertEqual(structure.errors(), [])

    def test_a_skill_md_without_frontmatter_is_never_loaded_by_an_agent(self) -> None:
        self.write("# Demo\n\nNo frontmatter at all.\n")
        errors = structure.errors()
        self.assertEqual(len(errors), 1, errors)
        self.assertIn("frontmatter", errors[0])

    def test_frontmatter_that_is_not_yaml_is_reported_as_such(self) -> None:
        self.declares("name: [demo-skill\ndescription: broken")
        self.assertTrue(any("not valid YAML" in e for e in structure.errors()))

    def test_frontmatter_that_is_not_a_mapping_is_reported(self) -> None:
        self.declares("- demo-skill\n- a list, not a mapping")
        self.assertTrue(any("must be a mapping" in e for e in structure.errors()))

    def test_a_missing_name_or_description_is_reported_separately(self) -> None:
        self.declares("summary: neither field is here")
        errors = structure.errors()
        self.assertEqual(len(errors), 2, errors)
        self.assertTrue(any("`name`" in e for e in errors))
        self.assertTrue(any("`description`" in e for e in errors))

    def test_a_name_that_disagrees_with_the_folder_is_reported(self) -> None:
        # The folder name is the skill's identity in its dataset, in a routing
        # verdict, and in the report, so a frontmatter name that differs makes
        # every one of those about a skill that does not exist.
        self.declares("name: other-skill\ndescription: Does things.")
        self.assertTrue(any("`demo-skill`" in e for e in structure.errors()))

    def test_a_name_that_is_not_lowercase_with_hyphens_is_reported(self) -> None:
        self.repo.skill("Demo_Skill", dataset=tier0_dataset("odd"))
        self.assertTrue(
            any("lowercase-with-hyphens" in e for e in structure.errors())
        )

    def test_a_name_longer_than_the_format_allows_is_reported(self) -> None:
        long_name = "a" * (structure.MAX_NAME_LENGTH + 1)
        self.repo.skill(long_name, dataset=tier0_dataset("long"))
        self.assertTrue(
            any(f"{len(long_name)} characters" in e for e in structure.errors())
        )

    def test_a_name_claiming_a_reserved_word_is_reported(self) -> None:
        self.repo.skill("claude-helper", dataset=tier0_dataset("helper"))
        self.assertTrue(any("`claude`" in e for e in structure.errors()))

    def test_a_description_longer_than_the_format_allows_is_reported(self) -> None:
        self.declares(
            f"name: demo-skill\ndescription: {'d' * (structure.MAX_DESCRIPTION_LENGTH + 1)}"
        )
        self.assertTrue(any("`description` is" in e for e in structure.errors()))

    def test_a_body_past_the_limit_is_reference_material_in_the_wrong_file(self) -> None:
        lines = "\n".join(f"line {i}" for i in range(structure.MAX_BODY_LINES + 1))
        self.declares("name: demo-skill\ndescription: Does things.", f"\n{lines}\n")
        self.assertTrue(any("sibling files" in e for e in structure.errors()))

    def test_blank_lines_around_the_body_do_not_count_against_it(self) -> None:
        lines = "\n".join(f"line {i}" for i in range(structure.MAX_BODY_LINES))
        self.declares("name: demo-skill\ndescription: Does things.", f"\n\n{lines}\n\n\n")
        self.assertEqual(structure.errors(), [])

    def test_a_globbed_directory_with_no_skill_file_is_passed_over(self) -> None:
        # A directory holding no SKILL.md is not a skill, and the default glob
        # matches every directory in the repo, so reporting one would be a line
        # per README folder.
        (self.repo.root / "notes").mkdir(parents=True)
        self.assertEqual(structure.errors(), [])

    def test_a_repo_that_requires_nothing_extra_requires_nothing_extra(self) -> None:
        self.assertEqual(structure.errors(), [])

    def test_a_file_the_repo_requires_of_every_skill_has_to_be_there(self) -> None:
        self.repo.reactivate(skill_files="skill-card.md")
        errors = structure.errors()
        self.assertEqual(len(errors), 1, errors)
        self.assertIn("skill-card.md", errors[0])

    def test_a_required_card_has_to_carry_the_sections_it_is_for(self) -> None:
        self.repo.reactivate(
            skill_files="skill-card.md", skill_sections="Description,Owner,License"
        )
        (self.folder / "skill-card.md").write_text(
            "# Skill Card\n\n## Description\n\nWhat it does.\n\n## Owner\n\n", encoding="utf-8"
        )
        errors = structure.errors()
        self.assertEqual(len(errors), 2, errors)
        self.assertTrue(any("`## Owner` section is empty" in e for e in errors))
        self.assertTrue(any("no `## License` section" in e for e in errors))

    def test_a_complete_card_passes(self) -> None:
        self.repo.reactivate(
            skill_files="skill-card.md", skill_sections="Description,Owner,License"
        )
        (self.folder / "skill-card.md").write_text(
            "# Skill Card\n\n## Description\n\nWhat it does.\n\n"
            "## Owner\n\nA team.\n\n## License\n\nMIT\n",
            encoding="utf-8",
        )
        self.assertEqual(structure.errors(), [])

    def test_a_required_file_that_is_not_markdown_only_has_to_exist(self) -> None:
        self.repo.reactivate(
            skill_files="scripts/detect.py", skill_sections="Description"
        )
        path = self.folder / "scripts" / "detect.py"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("print('hi')\n", encoding="utf-8")
        self.assertEqual(structure.errors(), [])

    def test_a_malformed_skill_stops_a_run_before_it_spends_anything(self) -> None:
        self.declares("name: demo-skill")
        with self.assertRaises(SystemExit):
            cli._structural_or_exit()


class TestARepoWhereNoSkillWasFound(unittest.TestCase):
    """Grading nothing is reported, because a green check for it would lie."""

    def test_finding_no_skill_is_itself_the_finding(self) -> None:
        repo = Repo(self)
        repo.activate()
        errors = structure.errors()
        self.assertEqual(len(errors), 1, errors)
        self.assertIn("no skill found", errors[0])

    def test_the_globs_that_matched_nothing_are_named(self) -> None:
        # A glob pointing somewhere the skills are not is the usual cause, so
        # the message has to say which one was used.
        repo = Repo(self)
        repo.skill("shipped", dataset=tier0_dataset("shipped"), where="agents/skills")
        repo.activate(skills_dir="skills/*")
        self.assertIn("skills/*", structure.errors()[0])

    def test_an_example_glob_is_offered_only_to_a_caller_who_passed_none(self) -> None:
        # Suggesting one to a caller who just passed one would be suggesting
        # the glob that found nothing.
        repo = Repo(self)
        repo.activate()
        self.assertIn("such as", structure.errors()[0])
        repo.reactivate(skills_dir="skills/*")
        self.assertNotIn("such as", structure.errors()[0])

    def test_a_run_asked_for_docs_still_has_work_to_do(self) -> None:
        # --docs is a repo checking its own prose, which is a real run in a
        # repo that ships no skill at all.
        repo = Repo(self)
        repo.activate(docs="*.md")
        self.assertEqual(structure.errors(), [])


def targets(text: str) -> list[str]:
    """Every reference the extractor finds in one markdown document."""
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "doc.md"
        path.write_text(text, encoding="utf-8")
        return [reference.target for reference in references.collect([path])]


class TestReferenceExtraction(unittest.TestCase):
    """What counts as a reference, and what is only a picture of one."""

    def test_every_way_markdown_spells_a_link(self) -> None:
        found = targets(
            "[inline](./a.md) and ![image](img/b.png)\n"
            '[titled](./c.md "why")\n'
            "[spaced](<./d e.md>)\n"
            "<https://example.com/auto>\n"
            'Raw <a href="./f.md">html</a> and <img src="g.png">\n'
            "Bare https://example.com/bare in a sentence.\n"
            "[style][ref]\n"
            "\n"
            "[ref]: ./h.md\n"
        )
        self.assertEqual(
            sorted(found),
            sorted(
                [
                    "./a.md",
                    "img/b.png",
                    "./c.md",
                    "./d e.md",
                    "https://example.com/auto",
                    "./f.md",
                    "g.png",
                    "https://example.com/bare",
                    "./h.md",
                ]
            ),
        )

    def test_code_and_comments_are_illustrations_not_promises(self) -> None:
        # A link in a code sample is showing you what a link looks like; a
        # commented-out one was deliberately taken out of the document.
        found = targets(
            "```markdown\n[fenced](./fenced.md)\n```\n"
            "~~~\n[tilde](./tilde.md)\n~~~\n"
            "Inline `[code](./code.md)` span.\n"
            "<!-- [comment](./comment.md) -->\n"
            "<!--\n[multi](./multi.md)\n-->\n"
            "[real](./real.md)\n"
        )
        self.assertEqual(found, ["./real.md"])

    def test_a_reference_remembers_where_it_was_written(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "doc.md"
            path.write_text("first\n\n[link](./a.md)\n", encoding="utf-8")
            reference = references.collect([path])[0]
        self.assertEqual((reference.source, reference.line), (path, 3))


class TestAnchors(unittest.TestCase):
    """Heading slugs, by the rules a repository host renders them with."""

    def test_headings_become_the_anchors_they_render_as(self) -> None:
        found = references.anchors(
            "# Title Here\n"
            "## Punctuation: it's dropped!\n"
            "### `code` in a heading\n"
            "#### [linked](https://example.com) heading\n"
            '<a id="hand-written"></a>\n'
        )
        self.assertEqual(
            found,
            {
                "title-here",
                "punctuation-its-dropped",
                "code-in-a-heading",
                "linked-heading",
                "hand-written",
            },
        )

    def test_a_repeated_heading_is_suffixed(self) -> None:
        self.assertEqual(references.anchors("## Dup\n## Dup\n## Dup\n"), {"dup", "dup-1", "dup-2"})

    def test_a_heading_inside_a_fence_is_a_comment_not_a_heading(self) -> None:
        self.assertEqual(references.anchors("```\n# Shell Comment\n```\n"), set())


class TestInternalReferences(unittest.TestCase):
    """Relative paths and anchors, resolved against the repo under test."""

    def setUp(self) -> None:
        self.repo = Repo(self)
        self.folder = self.repo.skill("demo", dataset=tier0_dataset("demo"))
        self.repo.activate()

    def write(self, relative: str, text: str) -> Path:
        path = self.folder / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        return path

    def errors(self) -> list[str]:
        return references.internal_errors(references.collect())

    def test_a_link_to_a_file_that_exists_is_fine(self) -> None:
        self.write("reference.md", "# Reference\n")
        self.write("scripts/detect.py", "print('hi')\n")
        self.write("SKILL.md", "[ref](./reference.md) [script](scripts/detect.py) [dir](scripts)\n")
        self.assertEqual(self.errors(), [])

    def test_a_link_to_a_file_that_does_not_exist_is_reported(self) -> None:
        self.write("SKILL.md", "\n[gone](./reference.md)\n")
        errors = self.errors()
        self.assertEqual(len(errors), 1, errors)
        self.assertIn("SKILL.md:2", errors[0])
        self.assertIn("`./reference.md`", errors[0])

    def test_an_anchor_is_checked_against_the_file_it_points_into(self) -> None:
        self.write("reference.md", "# Reference\n\n## Known Section\n")
        self.write(
            "SKILL.md",
            "# Demo\n\n## Local Section\n\n"
            "[here](#local-section) [there](./reference.md#known-section)\n"
            "[nowhere](#missing) [neither](./reference.md#missing)\n",
        )
        errors = self.errors()
        self.assertEqual(len(errors), 2, errors)
        self.assertTrue(all("#missing" in error for error in errors))

    def test_a_fragment_on_something_that_is_not_markdown_is_left_alone(self) -> None:
        # `#L20` is rendered by the host out of the file's line numbers, and
        # nothing in the file declares it.
        self.write("scripts/detect.py", "print('hi')\n")
        self.write("SKILL.md", "[line](scripts/detect.py#L1)\n")
        self.assertEqual(self.errors(), [])

    def test_root_relative_links_resolve_from_the_repo_root(self) -> None:
        self.write("SKILL.md", "[ok](/demo/SKILL.md) [no](/demo/gone.md)\n")
        errors = self.errors()
        self.assertEqual(len(errors), 1, errors)
        self.assertIn("/demo/gone.md", errors[0])

    def test_percent_encoding_is_decoded_before_the_file_is_looked_for(self) -> None:
        self.write("a file.md", "# Spaced\n")
        self.write("SKILL.md", "[spaced](./a%20file.md)\n")
        self.assertEqual(self.errors(), [])

    def test_addresses_and_urls_are_not_paths(self) -> None:
        self.write(
            "SKILL.md",
            "[mail](mailto:someone@example.com) [call](tel:+1234) "
            "[web](https://example.com/nope)\n",
        )
        self.assertEqual(self.errors(), [])

    def test_only_skill_markdown_is_read_until_docs_says_otherwise(self) -> None:
        (self.repo.root / "README.md").write_text("[gone](./nowhere.md)\n", encoding="utf-8")
        self.assertEqual(self.errors(), [])
        self.repo.reactivate(docs="*.md")
        errors = self.errors()
        self.assertEqual(len(errors), 1, errors)
        self.assertIn("README.md:1", errors[0])

    def test_a_broken_reference_stops_a_run_before_it_spends_anything(self) -> None:
        self.write("SKILL.md", "[gone](./reference.md)\n")
        with self.assertRaises(SystemExit):
            cli._structural_or_exit()


class TestExternalReferences(unittest.TestCase):
    """Fetching URLs: which ones, how the answer is judged, and what is said."""

    def setUp(self) -> None:
        self.repo = Repo(self)
        self.folder = self.repo.skill("demo", dataset=tier0_dataset("demo"))
        self.repo.activate()

    def markdown(self, text: str) -> list:
        (self.folder / "SKILL.md").write_text(text, encoding="utf-8")
        return references.collect()

    def test_only_external_urls_are_fetched_and_each_one_only_once(self) -> None:
        found = self.markdown(
            "[a](https://example.com/page#one) [b](https://example.com/page#two)\n"
            "[c](./local.md) [d](mailto:someone@example.com)\n"
        )
        asked: list[str] = []
        references.external_errors(found, probe=lambda url: asked.append(url) or "")
        self.assertEqual(asked, ["https://example.com/page"])

    def test_an_unreachable_url_says_where_it_was_written(self) -> None:
        found = self.markdown("line one\n[dead](https://example.com/gone)\n")
        errors = references.external_errors(found, probe=lambda url: "HTTP 404")
        self.assertEqual(len(errors), 1, errors)
        self.assertIn("https://example.com/gone", errors[0])
        self.assertIn("HTTP 404", errors[0])
        self.assertIn("SKILL.md:2", errors[0])

    def test_an_excluded_url_is_never_asked_about(self) -> None:
        found = self.markdown(
            "[gated](https://intranet.example.com/x) [open](https://example.com/y)\n"
        )
        asked: list[str] = []
        errors = references.external_errors(
            found,
            exclude=[r"^https://intranet\.example\.com/"],
            probe=lambda url: asked.append(url) or "HTTP 403",
        )
        self.assertEqual(asked, ["https://example.com/y"])
        self.assertEqual(len(errors), 1, errors)


class TestExternalProbe(unittest.TestCase):
    """When an answer from a server counts as "the reference is fine"."""

    URL = "https://example.com/thing"

    def opener(self, **outcomes):
        """A stand-in for urlopen answering per HTTP method."""

        def open_url(request, timeout=None):
            outcome = outcomes[request.get_method()]
            if isinstance(outcome, Exception):
                raise outcome
            response = mock.MagicMock()
            response.__enter__.return_value.status = outcome
            return response

        return open_url

    def probe(self, **outcomes) -> str:
        with mock.patch("urllib.request.urlopen", self.opener(**outcomes)):
            return references._probe(self.URL, timeout=1.0, retries=0)

    def http_error(self, code: int) -> urllib.error.HTTPError:
        return urllib.error.HTTPError(self.URL, code, "nope", None, None)

    def test_a_plain_success(self) -> None:
        self.assertEqual(self.probe(HEAD=200), "")

    def test_rate_limiting_means_the_host_is_there(self) -> None:
        self.assertEqual(self.probe(HEAD=self.http_error(429)), "")

    def test_a_refused_head_is_asked_again_as_a_get(self) -> None:
        # Plenty of servers answer HEAD with 403 or 405 and the same URL with
        # 200 on GET. That is a server preference, not link rot.
        self.assertEqual(self.probe(HEAD=self.http_error(405), GET=200), "")

    def test_a_head_that_never_comes_back_is_asked_again_as_a_get(self) -> None:
        # The failure that matters most: a host that black-holes HEAD would
        # otherwise be reported as rotten on the strength of a method it
        # simply does not serve.
        self.assertEqual(self.probe(HEAD=TimeoutError("timed out"), GET=200), "")

    def test_what_is_reported_is_what_a_reader_following_the_link_would_get(self) -> None:
        detail = self.probe(HEAD=self.http_error(405), GET=self.http_error(410))
        self.assertEqual(detail, "HTTP 410")

    def test_a_url_nobody_serves(self) -> None:
        detail = self.probe(HEAD=self.http_error(404), GET=self.http_error(404))
        self.assertEqual(detail, "HTTP 404")

    def test_a_host_that_does_not_resolve(self) -> None:
        unresolvable = urllib.error.URLError("Name or service not known")
        detail = self.probe(HEAD=unresolvable, GET=unresolvable)
        self.assertIn("Name or service not known", detail)


class TestRoutingClassification(unittest.TestCase):
    def test_verdicts(self) -> None:
        cases = [
            ("skill-a", "skill-a", "correct_trigger"),
            (None, None, "true_negative"),
            ("skill-a", None, "missed_trigger"),
            ("skill-a", "skill-b", "wrong_skill"),
            (None, "skill-a", "false_trigger"),
        ]
        for expect, observed, verdict in cases:
            with self.subTest(expect=expect, observed=observed):
                self.assertEqual(routing.classify(expect, observed), verdict)

    def test_only_correct_and_true_negative_pass(self) -> None:
        self.assertEqual(routing.PASSING_VERDICTS, {"correct_trigger", "true_negative"})


class TestRoutingGate(unittest.TestCase):
    """What turns a routing run red. By default: any wrong decision."""

    def totals(self, passed: int, graded: int, **extra) -> dict:
        return {
            "passed": passed,
            "graded": graded,
            "accuracy": round(passed / graded, 3) if graded else None,
            "activations": graded,
            "activations_expected": graded,
            **extra,
        }

    def gate(self, passed: int, graded: int, bar: float = 1.0, **extra) -> str | None:
        return cli.routing_gate(self.totals(passed, graded, **extra), bar)

    def test_the_default_bar_is_every_graded_case(self) -> None:
        self.assertEqual(cli.build_parser().parse_args(["routing"]).min_accuracy, 1.0)

    def test_a_clean_sweep_passes(self) -> None:
        self.assertIsNone(self.gate(12, 12))

    def test_one_wrong_decision_fails(self) -> None:
        reason = self.gate(11, 12)
        self.assertIn("11/12", reason)
        self.assertIn("--min-accuracy", reason)

    def test_an_accuracy_that_rounds_up_does_not_slip_through(self) -> None:
        # The reported figure is rounded to three places, so one miss in a big
        # enough set prints as a clean 1.0. The bar is held against the ratio.
        totals = self.totals(3999, 4000)
        self.assertEqual(totals["accuracy"], 1.0)
        self.assertIsNotNone(cli.routing_gate(totals, 1.0))

    def test_zero_reports_the_score_without_gating(self) -> None:
        self.assertIsNone(self.gate(1, 12, bar=0))

    def test_a_bar_short_of_perfect_is_kept_to_the_letter(self) -> None:
        self.assertIsNone(self.gate(9, 10, bar=0.9))
        self.assertIsNotNone(self.gate(8, 10, bar=0.9))

    def test_a_case_that_errored_is_outside_the_bar(self) -> None:
        # Errors are excluded from accuracy -- a timeout is not a routing
        # verdict -- so a perfect score over what was graded still passes.
        self.assertIsNone(self.gate(11, 11, cases=12, errors=1))

    def test_nothing_graded_fails_however_low_the_bar(self) -> None:
        totals = {
            "passed": 0,
            "graded": 0,
            "accuracy": None,
            "activations": 0,
            "activations_expected": 0,
        }
        self.assertIsNotNone(cli.routing_gate(totals, 0.0))

    def test_a_run_where_no_skill_activated_fails_however_low_the_bar(self) -> None:
        self.assertIsNotNone(self.gate(0, 12, bar=0, activations=0))


class TestActivationDetection(unittest.TestCase):
    SKILLS = ["local-ai-use", "local-ai-app-integration", "serving-llms-on-instinct"]

    def event(self, tool: str, tool_input: dict) -> dict:
        return {
            "type": "assistant",
            "message": {"content": [{"type": "tool_use", "name": tool, "input": tool_input}]},
        }

    def test_skill_tool_call_is_an_activation(self) -> None:
        event = self.event("Skill", {"command": "local-ai-use"})
        self.assertEqual(routing.detect_activation(event, self.SKILLS), "local-ai-use")

    def test_longest_name_wins_when_one_is_a_prefix_of_another(self) -> None:
        event = self.event("Skill", {"command": "local-ai-app-integration"})
        self.assertEqual(
            routing.detect_activation(event, self.SKILLS), "local-ai-app-integration"
        )

    def test_a_skill_nobody_installed_is_flagged_not_scored(self) -> None:
        event = self.event("Skill", {"command": "somebody-elses-skill"})
        self.assertEqual(
            routing.detect_activation(event, self.SKILLS), "other:somebody-elses-skill"
        )

    def test_listing_the_installed_skills_is_not_an_activation(self) -> None:
        event = self.event("Bash", {"command": "ls .claude/skills"})
        self.assertIsNone(routing.detect_activation(event, self.SKILLS))

    def test_reading_a_skill_body_counts_only_without_a_skill_tool(self) -> None:
        event = self.event("Read", {"file_path": "/tmp/x/.claude/skills/local-ai-use/SKILL.md"})
        self.assertEqual(
            routing.detect_activation(event, self.SKILLS, allow_body_path=True),
            "local-ai-use",
        )
        self.assertIsNone(
            routing.detect_activation(event, self.SKILLS, allow_body_path=False)
        )

    def test_a_tool_result_listing_every_skill_is_not_an_activation(self) -> None:
        # An empty workspace answers a file hunt with a recursive listing of
        # every SKILL.md; scoring that credited whichever name sorted first.
        event = {
            "type": "user",
            "message": {
                "content": [
                    {
                        "type": "tool_result",
                        "content": "skills/local-ai-use/SKILL.md\nskills/serving-llms-on-instinct/SKILL.md",
                    }
                ]
            },
        }
        self.assertIsNone(routing.detect_activation(event, self.SKILLS))

    def test_inspecting_the_installed_skills_is_recognized(self) -> None:
        self.assertTrue(
            routing._is_skills_inspection('{"path": ".claude/skills"}', self.SKILLS)
        )
        self.assertFalse(routing._is_skills_inspection('{"path": "src/main.py"}', self.SKILLS))


class TestRoutingStaging(unittest.TestCase):
    def test_the_routing_set_lands_in_the_workspace_and_nothing_else(self) -> None:
        repo = Repo(self)
        repo.skill("one", dataset=tier0_dataset("one"))
        repo.skill("two", dataset=tier0_dataset("two"))
        repo.skill("unlisted", dataset=tier0_dataset("unlisted"))
        cfg = repo.activate(routing_room="one,two")
        workspace = routing.stage_workspace(cfg.routing_set)
        try:
            staged = sorted(p.name for p in (workspace / ".claude" / "skills").iterdir())
            self.assertEqual(staged, ["one", "two"])
            self.assertTrue(
                (workspace / ".claude" / "skills" / "one" / "SKILL.md").is_file()
            )
        finally:
            import shutil

            shutil.rmtree(workspace, ignore_errors=True)


class TestPromptTemplating(unittest.TestCase):
    def test_placeholders_are_substituted(self) -> None:
        self.assertEqual(
            behavior.expand("trace: {trace_path}", {"trace_path": "/tmp/a.json"}),
            "trace: /tmp/a.json",
        )

    def test_literal_braces_survive(self) -> None:
        # Prompts routinely contain JSON snippets and regex quantifiers, which
        # str.format would choke on.
        text = 'produce {"a": 1} and match \\d{3}'
        self.assertEqual(behavior.expand(text, {"x": "y"}), text)


def stream(*tool_calls: tuple[str, dict], result: str = "done") -> list[dict]:
    """Synthetic stream-json events, shaped like the CLI's output."""
    events: list[dict] = [{"type": "system", "subtype": "init", "tools": ["Bash", "Skill"]}]
    for name, tool_input in tool_calls:
        events.append(
            {
                "type": "assistant",
                "message": {"content": [{"type": "tool_use", "name": name, "input": tool_input}]},
            }
        )
    events.append({"type": "result", "result": result})
    return events


class TestRunGrading(unittest.TestCase):
    """Deterministic grading only; the judged fields need a live judge."""

    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.workspace = Path(self.tmp.name)
        self.addCleanup(self.tmp.cleanup)

    def make_run(self, events: list[dict]) -> agent.Run:
        return agent.Run(workspace=self.workspace, events=events, judge_model=None)

    def test_transcript_and_tools_are_captured(self) -> None:
        run = self.make_run(stream(("Bash", {"command": "python detect.py"})))
        self.assertIn("Bash", run.tool_names)
        self.assertIn("detect.py", run.logs)
        self.assertEqual(run.result_text, "done")

    def test_logs_contain_is_case_insensitive(self) -> None:
        run = self.make_run(stream(("Bash", {"command": "python DETECT.py"})))
        checks = run.evaluate(logs_contain=["detect.py"])
        self.assertTrue(checks[0].passed)

    def test_logs_contain_reports_a_miss(self) -> None:
        run = self.make_run(stream(("Bash", {"command": "ls"})))
        checks = run.evaluate(logs_contain=["detect.py"])
        self.assertFalse(checks[0].passed)

    def test_files_exist(self) -> None:
        (self.workspace / "out.png").write_bytes(b"x")
        checks = self.make_run(stream()).evaluate(files_exist=["out.png", "missing.txt"])
        self.assertTrue(checks[0].passed)
        self.assertFalse(checks[1].passed)

    def test_files_exist_finds_the_artifact_in_a_subdirectory(self) -> None:
        # Where a plan lands is the agent's call; asking for `plan.md` and
        # getting `examples/fixture/plan.md` is a pass, not a defect.
        nested = self.workspace / "examples" / "fixture"
        nested.mkdir(parents=True)
        (nested / "plan.md").write_text("x", encoding="utf-8")
        checks = self.make_run(stream()).evaluate(files_exist=["plan.md"])
        self.assertTrue(checks[0].passed)
        self.assertIn("examples/fixture/plan.md", checks[0].detail)

    def test_files_exist_matches_whole_segments_only(self) -> None:
        (self.workspace / "analyze_plan.md").write_text("x", encoding="utf-8")
        checks = self.make_run(stream()).evaluate(files_exist=["plan.md"])
        self.assertFalse(checks[0].passed)

    def test_files_exist_keeps_the_directory_context_it_was_given(self) -> None:
        deep = self.workspace / "run-1" / "analysis_output"
        deep.mkdir(parents=True)
        (deep / "analysis.md").write_text("x", encoding="utf-8")
        (self.workspace / "analysis.md").write_text("x", encoding="utf-8")
        run = self.make_run(stream())
        self.assertTrue(run.evaluate(files_exist=["analysis_output/analysis.md"])[0].passed)
        self.assertFalse(run.evaluate(files_exist=["other_output/analysis.md"])[0].passed)

    def test_files_exist_ignores_a_directory_of_the_wanted_name(self) -> None:
        (self.workspace / "out.png").mkdir()
        checks = self.make_run(stream()).evaluate(files_exist=["out.png"])
        self.assertFalse(checks[0].passed)

    def test_every_expectation_is_reported_not_just_the_first(self) -> None:
        # A run that cost minutes should not have to be repeated to discover
        # the second thing wrong with it.
        checks = self.make_run(stream()).evaluate(
            logs_contain=["nope"], files_exist=["also-nope"]
        )
        self.assertEqual(len(checks), 2)
        self.assertFalse(any(c.passed for c in checks))

    def test_dot_claude_is_excluded_from_workspace_listing(self) -> None:
        staged = self.workspace / ".claude" / "skills" / "demo"
        staged.mkdir(parents=True)
        (staged / "SKILL.md").write_text("x", encoding="utf-8")
        (self.workspace / "out.png").write_bytes(b"x")
        self.assertEqual(self.make_run(stream()).files, ["out.png"])


class FakeAgent:
    """Stands in for a real agent session so the flow can be tested offline."""

    def __init__(self, events: list[dict], seed: Path | None) -> None:
        self.events = events
        self.seed = seed
        self.workspace: Path | None = None
        self.prompts: list[str] = []
        self._tmp: tempfile.TemporaryDirectory | None = None

    def __enter__(self) -> "FakeAgent":
        self._tmp = tempfile.TemporaryDirectory()
        self.workspace = Path(self._tmp.name)
        if self.seed is not None:
            for path in self.seed.iterdir():
                (self.workspace / path.name).write_bytes(path.read_bytes())
        return self

    def __exit__(self, *exc) -> None:
        if self._tmp is not None:
            self._tmp.cleanup()

    def prompt(self, text: str):
        self.prompts.append(text)
        return agent.Run(workspace=self.workspace, events=self.events, judge_model=None)


class TestBehaviorCaseFlow(unittest.TestCase):
    """The hook contract and prompt templating, without spending tokens."""

    def setUp(self) -> None:
        self.repo = Repo(self)
        self.repo.skill(
            "demo-skill",
            dataset=tier0_dataset("demo"),
            workspace={"evals/files/stub/main.py": "print('hi')\n"},
        )
        self.repo.activate()

    def run_case(self, case_payload: dict, hooks=None, events=None, skill="demo-skill"):
        cases, errors = parse(triggers(**case_payload), skill=skill)
        self.assertEqual(errors, [])
        made: list[FakeAgent] = []

        def fake_claude(model, *, skill, effort, seed=None):
            made.append(FakeAgent(events or stream(), seed))
            return made[-1]

        original = behavior.claude
        behavior.claude = fake_claude
        try:
            outcome = behavior.run_case(cases[0], {}, hooks, "opus", "high")
        finally:
            behavior.claude = original
        return outcome, made[0]

    def test_a_passing_case(self) -> None:
        outcome, session = self.run_case(
            {"id": "a", "prompt": "run it", "logs_contain": ["detect.py"]},
            events=stream(("Bash", {"command": "detect.py"})),
        )
        self.assertTrue(outcome.passed)
        self.assertEqual(session.prompts, ["run it"])

    def test_a_failing_expectation_fails_the_case(self) -> None:
        outcome, _ = self.run_case({"id": "a", "prompt": "run it", "logs_contain": ["nope"]})
        self.assertFalse(outcome.passed)

    def test_hooks_run_in_order_and_can_template_the_prompt(self) -> None:
        calls: list[str] = []

        class Hooks:
            @staticmethod
            def setup(workspace, case, ctx):
                calls.append("setup")
                return {"output_dir": workspace / "out"}

            @staticmethod
            def check(run, case, ctx):
                calls.append("check")

            @staticmethod
            def teardown(workspace, case, ctx):
                calls.append("teardown")

        outcome, session = self.run_case(
            {"id": "a", "prompt": "write to {output_dir}", "logs_contain": ["detect"]},
            hooks=Hooks,
            events=stream(("Bash", {"command": "detect"})),
        )
        self.assertEqual(calls, ["setup", "check", "teardown"])
        self.assertNotIn("{output_dir}", session.prompts[0])
        self.assertTrue(outcome.passed)

    def test_a_raising_hook_check_fails_the_case_without_killing_the_run(self) -> None:
        class Hooks:
            @staticmethod
            def check(run, case, ctx):
                raise AssertionError("scorer reported 3 failures")

        outcome, _ = self.run_case({"id": "a", "prompt": "p", "logs_contain": []}, hooks=Hooks)
        self.assertFalse(outcome.passed)
        self.assertTrue(any("scorer reported" in c["detail"] for c in outcome.checks))

    def test_teardown_runs_even_when_the_agent_raises(self) -> None:
        calls: list[str] = []

        class Hooks:
            @staticmethod
            def teardown(workspace, case, ctx):
                calls.append("teardown")

        class Exploding(FakeAgent):
            def prompt(self, text):
                raise RuntimeError("claude produced no output")

        cases, _ = parse(triggers(id="a", prompt="p", unexpected_behavior=["x"]))
        original = behavior.claude
        behavior.claude = lambda model, *, skill, effort, seed=None: Exploding(stream(), seed)
        try:
            outcome = behavior.run_case(cases[0], {}, Hooks, "opus", "high")
        finally:
            behavior.claude = original
        self.assertEqual(calls, ["teardown"])
        self.assertFalse(outcome.passed)
        self.assertIn("claude produced no output", outcome.error)

    def test_workspace_fixtures_are_staged(self) -> None:
        outcome, _ = self.run_case(
            {
                "id": "a",
                "prompt": "edit it",
                "workspace": "evals/files/stub",
                "files_exist": ["main.py"],
            }
        )
        self.assertTrue(outcome.passed, outcome.checks)


class TestBehaviorReporting(unittest.TestCase):
    def test_summary_counts_cases_and_expectations(self) -> None:
        outcomes = [
            behavior.BehaviorOutcome(
                id="a",
                skill="s",
                prompt="p",
                passed=True,
                elapsed_s=1.0,
                checks=[
                    {"kind": "logs_contain", "expectation": "x", "passed": True, "detail": ""}
                ],
            ),
            behavior.BehaviorOutcome(
                id="b",
                skill="s",
                prompt="p",
                passed=False,
                elapsed_s=1.0,
                checks=[
                    {
                        "kind": "expected_behavior",
                        "expectation": "y",
                        "passed": False,
                        "detail": "no",
                    }
                ],
            ),
        ]
        summary = behavior.summarize(outcomes, {"model": "opus", "effort": "high"})
        self.assertEqual(
            summary["totals"],
            {
                "cases": 2,
                "passed": 1,
                "checks": 2,
                "checks_passed": 1,
                "errors": 0,
                "degraded": 0,
            },
        )
        report = behavior.render_markdown(summary)
        self.assertIn("1/2 cases passed", report)
        self.assertIn("`b`", report)


class TestCaseFiltering(unittest.TestCase):
    def setUp(self) -> None:
        self.cases, _ = parse(
            {
                EVALUATIONS_KEY: [
                    {"id": "a", TRIGGER_KEY: True, "prompt": "p"},
                    {"id": "b", TRIGGER_KEY: True, "prompt": "q"},
                ]
            },
            skill="demo-skill",
        )

    def test_filter_by_id(self) -> None:
        self.assertEqual([c.id for c in datasets.filter_cases(self.cases, "a")], ["a"])

    def test_filter_by_skill(self) -> None:
        self.assertEqual(len(datasets.filter_cases(self.cases, "demo-skill")), 2)

    def test_empty_filter_keeps_everything(self) -> None:
        self.assertEqual(len(datasets.filter_cases(self.cases, "")), 2)

    def test_no_match_is_an_error(self) -> None:
        with self.assertRaises(SystemExit):
            datasets.filter_cases(self.cases, "nope")


class TestRoutingCasePooling(unittest.TestCase):
    """A routing run grades the listed skills' prompts, and the shared pool."""

    def setUp(self) -> None:
        self.repo = Repo(self)
        self.repo.skill("listed", dataset=tier0_dataset("listed"))
        self.repo.skill("unlisted", dataset=tier0_dataset("unlisted"))
        self.repo.activate(routing_room="listed")

    def test_a_listed_skill_brings_both_kinds_of_prompt(self) -> None:
        # Its positives are the room's positives; its near misses assert that
        # nothing in the room grabs them.
        ids = {case.id for case in datasets.routing_cases(["listed"])}
        self.assertIn("listed-yes-0", ids)
        self.assertIn("listed-no-0", ids)

    def test_an_unlisted_skill_brings_none(self) -> None:
        # It is not in the room, so a prompt expecting it could only ever lose,
        # and its near misses assert nothing about the skills that are there.
        cases = datasets.routing_cases(["listed"])
        self.assertFalse(any(case.skill == "unlisted" for case in cases))

    def test_the_shared_pool_is_always_in(self) -> None:
        cases = datasets.routing_cases(["listed"])
        self.assertTrue(any(case.category == "unrelated" for case in cases))

    def test_an_empty_room_leaves_only_the_shared_pool(self) -> None:
        cases = datasets.routing_cases([])
        self.assertTrue(cases)
        self.assertTrue(all(case.skill is None for case in cases))


class TestCiModelPin(unittest.TestCase):
    """The pin keeps paid runs comparable; a mock is neither paid nor graded."""

    def test_a_real_model_is_pinned_under_ci(self) -> None:
        with mock.patch.dict(os.environ, {"CI": "true"}):
            self.assertEqual(agent.enforce_model_policy("sonnet"), "opus")

    def test_a_mock_is_left_alone_under_ci(self) -> None:
        # Otherwise the free wiring run becomes a run that needs a key, in the
        # one place where not needing a key is the whole point.
        with mock.patch.dict(os.environ, {"CI": "true"}):
            self.assertEqual(
                agent.enforce_model_policy("mockllm/model"), "mockllm/model"
            )

    def test_nothing_is_pinned_outside_ci(self) -> None:
        with mock.patch.dict(os.environ, {"CI": "", "GITHUB_ACTIONS": ""}):
            self.assertEqual(agent.enforce_model_policy("sonnet"), "sonnet")


class TestEngineMessageLimit(unittest.TestCase):
    """A model that cannot finish should not be given a hundred turns to prove it."""

    def test_a_real_model_gets_the_full_budget(self) -> None:
        self.assertEqual(
            engine_behavioral.message_limit_for("anthropic/claude-opus-5"),
            engine_behavioral.MESSAGE_LIMIT,
        )

    def test_a_mock_gets_a_short_one(self) -> None:
        # It never calls submit, so it loops to whatever cap it is given, and
        # every turn is a real sandbox round trip.
        self.assertEqual(
            engine_behavioral.message_limit_for("mockllm/model"),
            engine_behavioral.MOCK_MESSAGE_LIMIT,
        )


class TestEngineSkillFailureIsContained(unittest.TestCase):
    """One skill's broken setup is that skill's failure, not everybody's."""

    def test_a_skill_that_cannot_run_becomes_failed_outcomes(self) -> None:
        cases = [
            datasets.Case(id="a", prompt="p", skill="broken", skill_should_trigger=True),
            datasets.Case(id="b", prompt="q", skill="broken", skill_should_trigger=True),
        ]
        outcomes = engine_behavioral._failed(cases[0].skill, cases, "no compose file")
        self.assertEqual([o.id for o in outcomes], ["a", "b"])
        self.assertTrue(all(not o.passed for o in outcomes))
        self.assertTrue(all("no compose file" in (o.error or "") for o in outcomes))
        # Not silence: an unreported skill would let a run that graded nothing
        # call itself green.
        self.assertTrue(all(o.checks == [] for o in outcomes))


class TestBehavioralEngineDispatch(unittest.TestCase):
    """Only `legacy` may bypass the inspect path.

    `claude-code-no-sandbox` was added to the dispatch chain but not to the guard around
    it, so it fell through to the legacy engine: runs that asked for one agent
    silently got another, while the reports said `engine: claude-code-no-sandbox`
    throughout. Nothing in the suite noticed, because nothing asserted which
    runner a flag actually reaches.
    """

    def test_the_guard_covers_every_engine_but_legacy(self) -> None:
        self.assertEqual(set(cli.INSPECT_ENGINES), set(cli.ENGINES) - {"legacy"})

    def test_the_dispatch_reads_that_set_rather_than_a_literal(self) -> None:
        # The bug was a literal tuple that fell behind the choices list. A
        # literal here is the defect itself, so the source is what to assert.
        source = inspect.getsource(cli.cmd_behavioral)
        self.assertIn("if args.engine in INSPECT_ENGINES:", source)

    def test_the_preflight_uses_the_same_set(self) -> None:
        # These disagreed: the preflight demanded the inspect extra for
        # claude-code-no-sandbox while the dispatch sent it to an engine that never uses
        # it, which is how a Windows job failed on a flag it had not passed.
        self.assertIn("INSPECT_ENGINES", inspect.getsource(cli._prepare_graded_run))


class TestEngineInstallHint(unittest.TestCase):
    """The hint has to name the engine the user actually asked for."""

    def test_it_names_the_requested_engine(self) -> None:
        # Naming `inspect` regardless sent a Windows CI job looking for a flag
        # it had never passed -- it had asked for claude-code-no-sandbox.
        self.assertIn("--engine claude-code-no-sandbox", engine_module.install_hint("claude-code-no-sandbox"))

    def test_it_still_points_at_the_one_extra_that_fixes_all_of_them(self) -> None:
        self.assertIn("skillscope[inspect]", engine_module.install_hint("claude-code"))


class TestEngineWorkdirPath(unittest.TestCase):
    """The working directory has to be knowable before a sandbox exists."""

    def setUp(self) -> None:
        self.addCleanup(os.environ.pop, engine_sandbox.SANDBOX_ENV, None)

    def test_a_container_run_names_the_workdir_without_creating_it(self) -> None:
        os.environ[engine_sandbox.SANDBOX_ENV] = "podman"
        self.assertEqual(engine_tools.workdir_path(), engine_tools.WORKDIR)

    def test_a_local_run_has_none_so_the_agent_keeps_its_own(self) -> None:
        # Creating /workspace on somebody's laptop is not ours to do, and the
        # harness's own directory is already where the scorers look.
        os.environ[engine_sandbox.SANDBOX_ENV] = "local"
        self.assertIsNone(engine_tools.workdir_path())


class TestEngineNoSandboxInstallsSkill(unittest.TestCase):
    """A driver that replaces the react agent must stage the skill itself.

    It did not, so the CLI ran with no skill and answered from the prompt
    alone -- scoring 4/21 where every other engine scored 21/21, while
    finishing faster, which was the only visible sign.
    """

    def test_the_skill_lands_where_the_harness_looks(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / "my-skill"
            (src / "scripts").mkdir(parents=True)
            (src / "SKILL.md").write_text("# my-skill", encoding="utf-8")
            (src / "scripts" / "validate.py").write_text("x = 1", encoding="utf-8")
            workspace = Path(tmp) / "ws"
            workspace.mkdir()

            engine_no_sandbox.install_skill(src, str(workspace))

            staged = workspace / ".claude" / "skills" / "my-skill"
            self.assertTrue((staged / "SKILL.md").is_file())
            # The whole tree, not just the manifest: skills ship validators and
            # references the agent is expected to run.
            self.assertTrue((staged / "scripts" / "validate.py").is_file())


class TestTaskTimeLimit(unittest.TestCase):
    """A run that overruns should still say where it got to."""

    def test_inspect_stops_before_the_hard_deadline_does(self) -> None:
        bound = deadline.Deadline(1800, command="behavioral")
        limit = engine_behavioral.task_time_limit(bound)
        self.assertLess(limit, bound.remaining())
        # Enough room for inspect to score what exists and write the log.
        self.assertGreaterEqual(bound.remaining() - limit, 60)

    def test_a_tiny_budget_still_gets_a_usable_limit(self) -> None:
        # Never negative, never zero: a nonsense limit would fail the sample
        # instantly and look like the agent doing nothing.
        self.assertGreaterEqual(engine_behavioral.task_time_limit(deadline.Deadline(5)), 60)

    def test_no_deadline_means_no_limit(self) -> None:
        self.assertIsNone(engine_behavioral.task_time_limit(None))


class TestRealtimeLogging(unittest.TestCase):
    """The live sample buffer is what MAX_PATH kills on Windows."""

    def test_windows_runs_without_the_buffer(self) -> None:
        with mock.patch.object(engine_behavioral.sys, "platform", "win32"):
            self.assertFalse(engine_behavioral.realtime_logging())

    def test_posix_keeps_it(self) -> None:
        with mock.patch.object(engine_behavioral.sys, "platform", "linux"):
            self.assertTrue(engine_behavioral.realtime_logging())


class TestShellPrefixProbe(unittest.TestCase):
    """A guest without bash raises rather than answering."""

    def _probe(self, exc: Exception | None):
        from skillscope.engine import tools as t

        class _Result:
            success = True

        class _Sandbox:
            async def exec(self, *a, **k):
                if exc is not None:
                    raise exc
                return _Result()

        store: dict = {}

        class _Store:
            def get(self, k, default=None):
                return store.get(k, default)

            def set(self, k, v):
                store[k] = v

        with mock.patch.dict(
            sys.modules,
            {"inspect_ai.util": mock.MagicMock(sandbox=lambda: _Sandbox(), store=_Store)},
        ):
            return asyncio.run(t.shell_prefix())

    def test_a_missing_bash_selects_powershell_rather_than_failing(self) -> None:
        # WinError 2 here took the whole task down and reported 0/0
        # expectations, which reads as the harness being broken.
        self.assertEqual(
            self._probe(FileNotFoundError(2, "The system cannot find the file specified")),
            engine_tools.WINDOWS_SHELL,
        )

    def test_a_working_bash_still_selects_posix(self) -> None:
        self.assertEqual(self._probe(None), engine_tools.POSIX_SHELL)


class TestClaudeCodeRefusesTheHostsFilesystem(unittest.TestCase):
    """`claude-code` needs a container, and not for isolation's sake.

    `inspect_swe` prepares the guest by writing `$HOME/.claude/settings.json`
    outright. In a container that file belongs to nobody. Under the `local`
    provider `$HOME` is the developer's own, and the same write silently
    destroys their real configuration -- permissions, model, gateway
    environment -- with no backup. This cost one settings.json before the
    guard existed, which is why the guard is a refusal rather than a warning.
    """

    def setUp(self) -> None:
        self.addCleanup(os.environ.pop, engine_sandbox.SANDBOX_ENV, None)
        os.environ.pop(engine_sandbox.SANDBOX_ENV, None)
        # Two different levers, and this class is about the second one.
        # `provider()` asks `is_windows()`; `require()` reads `sys.platform`
        # itself and reads it first. Patching only the former left these
        # tests measuring the platform guard on a Windows runner and the
        # sandbox guard on a Linux one, under the same names.
        for target, attr, value in (
            (engine_sandbox, "is_windows", lambda: False),
            (engine_verify.sys, "platform", "linux"),
        ):
            patch = mock.patch.object(target, attr, value)
            patch.start()
            self.addCleanup(patch.stop)

    def test_a_host_sharing_provider_is_refused(self) -> None:
        os.environ[engine_sandbox.SANDBOX_ENV] = "local"
        with self.assertRaises(SystemExit) as caught:
            engine_verify.require()
        message = str(caught.exception)
        self.assertIn("settings.json", message)
        self.assertIn("--engine claude-code-no-sandbox", message)

    def test_every_unisolated_provider_is_refused(self) -> None:
        # Keyed off the same set `describe()` reports from, so a provider that
        # is added as unisolated cannot quietly stay allowed here.
        for provider in engine_sandbox.NOT_ISOLATED:
            with self.subTest(provider=provider):
                os.environ[engine_sandbox.SANDBOX_ENV] = provider
                with self.assertRaises(SystemExit):
                    engine_verify.require()

    def test_a_container_provider_is_not_refused_for_being_a_container(self) -> None:
        # Asserted as "not this refusal" rather than "no refusal at all": the
        # unit suite runs without the inspect extra on purpose, so `require`
        # may still stop on the missing wheel. That is a different answer to a
        # different question, and conflating them made this pass locally and
        # fail in CI.
        os.environ[engine_sandbox.SANDBOX_ENV] = "docker"
        try:
            engine_verify.require()
        except SystemExit as exc:
            # Whatever it stopped on, it was not the sandbox.
            self.assertNotIn("settings.json", str(exc))
            self.assertIn("pip install", str(exc))


class TestEngineNoSandboxGuard(unittest.TestCase):
    """The CLI runs on the host, so the sandbox has to be the host."""

    def setUp(self) -> None:
        self.addCleanup(os.environ.pop, engine_sandbox.SANDBOX_ENV, None)

    def test_a_container_provider_is_refused_with_the_alternative(self) -> None:
        # Otherwise the CLI would work in the host's filesystem while the
        # scorers read a container, and every expectation would fail for a
        # reason nothing in the report explains.
        os.environ[engine_sandbox.SANDBOX_ENV] = "podman"
        with self.assertRaises(SystemExit) as caught:
            engine_no_sandbox.require_local()
        message = str(caught.exception)
        self.assertIn("podman", message)
        self.assertIn("--engine claude-code", message)


class TestEngineModelNames(unittest.TestCase):
    """`--model` speaks the claude CLI's aliases; inspect wants provider names."""

    def test_an_alias_becomes_a_provider_qualified_name(self) -> None:
        self.assertEqual(engine_models.resolve("opus"), "anthropic/claude-opus-5")

    def test_an_alias_is_case_insensitive(self) -> None:
        self.assertEqual(engine_models.resolve("Opus"), "anthropic/claude-opus-5")

    def test_a_qualified_name_passes_through(self) -> None:
        # What makes `--model mockllm/model` work for the no-cost wiring runs.
        self.assertEqual(engine_models.resolve("mockllm/model"), "mockllm/model")

    def test_an_unknown_bare_name_is_assumed_to_be_anthropic(self) -> None:
        self.assertEqual(engine_models.resolve("claude-x"), "anthropic/claude-x")


class TestEngineGatewayHeaders(unittest.TestCase):
    """`ANTHROPIC_CUSTOM_HEADERS` is a claude CLI variable; inspect ignores it."""

    def setUp(self) -> None:
        for var in (engine_models.CUSTOM_HEADERS_ENV, engine_models.AUTH_TOKEN_ENV):
            self.addCleanup(os.environ.pop, var, None)
            os.environ.pop(var, None)

    def test_no_headers_configured_means_no_provider_arguments(self) -> None:
        self.assertEqual(engine_models.model_args("anthropic/claude-opus-5"), {})

    def test_headers_are_parsed_into_default_headers(self) -> None:
        os.environ[engine_models.CUSTOM_HEADERS_ENV] = (
            "X-Subscription-Key: secret\nuser: ci-runner\n"
        )
        self.assertEqual(
            engine_models.model_args("anthropic/claude-opus-5"),
            {
                "default_headers": {
                    "X-Subscription-Key": "secret",
                    "user": "ci-runner",
                }
            },
        )

    def test_a_value_containing_a_colon_survives(self) -> None:
        os.environ[engine_models.CUSTOM_HEADERS_ENV] = "Referer: https://example.com/x"
        self.assertEqual(
            engine_models.custom_headers(), {"Referer": "https://example.com/x"}
        )

    def test_blank_and_malformed_lines_are_skipped(self) -> None:
        os.environ[engine_models.CUSTOM_HEADERS_ENV] = "\nnot-a-header\n\nk: v\n"
        self.assertEqual(engine_models.custom_headers(), {"k": "v"})

    def test_a_non_anthropic_model_needs_no_gateway_arguments(self) -> None:
        # The free wiring run reaches no provider, so a shell that happens to
        # hold both Anthropic variables must not break the one check that costs
        # nothing -- and those are exactly the machines that have an OAuth token.
        os.environ[engine_models.CUSTOM_HEADERS_ENV] = "k: v"
        os.environ[engine_models.AUTH_TOKEN_ENV] = "token"
        self.assertEqual(engine_models.model_args("mockllm/model"), {})

    def test_oauth_and_gateway_headers_together_are_refused(self) -> None:
        # inspect's OAuth path sets `default_headers` itself, so ours would be a
        # duplicate keyword argument deep inside the SDK. Fail with the reason.
        os.environ[engine_models.CUSTOM_HEADERS_ENV] = "k: v"
        os.environ[engine_models.AUTH_TOKEN_ENV] = "token"
        with self.assertRaises(SystemExit) as caught:
            engine_models.model_args("anthropic/claude-opus-5")
        self.assertIn(engine_models.AUTH_TOKEN_ENV, str(caught.exception))


class TestEngineListingNormalisation(unittest.TestCase):
    """`find` and `Get-ChildItem` disagree about separators and prefixes."""

    def test_posix_output(self) -> None:
        listing = "./out.png\n./docs/plan.md\n"
        self.assertEqual(
            engine_tools.normalize_listing(listing), ["docs/plan.md", "out.png"]
        )

    def test_windows_output(self) -> None:
        listing = ".\\out.png\r\n.\\docs\\plan.md\r\n"
        self.assertEqual(
            engine_tools.normalize_listing(listing), ["docs/plan.md", "out.png"]
        )

    def test_the_installed_skill_does_not_satisfy_files_exist(self) -> None:
        # The harness put it there, so a case asserting SKILL.md was produced
        # would otherwise pass without the agent doing anything.
        listing = "./skills/demo/SKILL.md\n./.claude/settings.json\n./out.png\n"
        self.assertEqual(engine_tools.normalize_listing(listing), ["out.png"])

    def test_blank_lines_are_dropped(self) -> None:
        self.assertEqual(engine_tools.normalize_listing("\n\n  \n"), [])


class TestEngineJudgeVerdicts(unittest.TestCase):
    """A grader is chatty and its reasons contain punctuation."""

    def test_a_bare_verdict(self) -> None:
        self.assertEqual(
            engine_judge.parse_verdict('{"pass": true, "reason": "it did"}'),
            (True, "it did"),
        )

    def test_a_verdict_wrapped_in_prose(self) -> None:
        text = 'Looking at the evidence...\n{"pass": false, "reason": "no file"}\nDone.'
        self.assertEqual(engine_judge.parse_verdict(text), (False, "no file"))

    def test_a_reason_containing_braces(self) -> None:
        # A regex quantifier or a quoted snippet in the reason must not confuse
        # the scan, which is why boundaries are decoded rather than matched.
        text = '{"pass": true, "reason": "matched a{2,3} in the output"}'
        self.assertEqual(
            engine_judge.parse_verdict(text), (True, "matched a{2,3} in the output")
        )

    def test_the_last_verdict_wins(self) -> None:
        text = '{"pass": true, "reason": "first"}\n{"pass": false, "reason": "second"}'
        self.assertEqual(engine_judge.parse_verdict(text), (False, "second"))

    def test_no_verdict_at_all(self) -> None:
        self.assertIsNone(engine_judge.parse_verdict("I could not decide."))

    def test_a_missing_reason_still_yields_a_verdict(self) -> None:
        self.assertEqual(
            engine_judge.parse_verdict('{"pass": true}'), (True, "(no reason given)")
        )


class TestEngineJudgePolarity(unittest.TestCase):
    """The judge grades the requirement; callers must never negate the verdict."""

    def test_a_must_requirement_asks_whether_it_happened(self) -> None:
        text = engine_judge.requirement_text("generate an image", must_happen=True)
        self.assertIn("MUST have done this", text)
        self.assertIn("true if the agent did it", text)

    def test_a_must_not_requirement_asks_whether_it_was_avoided(self) -> None:
        # Read as a pass when the agent avoided it: negating this verdict is
        # what turns a correct run into a failure.
        text = engine_judge.requirement_text("call a cloud API", must_happen=False)
        self.assertIn("MUST NOT have done this", text)
        self.assertIn("true if the agent avoided it", text)
        self.assertIn("default verdict is true", text)


class TestEngineJudgeTruncation(unittest.TestCase):
    """What settles a check is usually the last thing the agent did."""

    def test_short_transcripts_are_untouched(self) -> None:
        self.assertEqual(engine_judge._elide_middle("abc", 100), "abc")

    def test_the_end_survives(self) -> None:
        # Cutting the tail would drop the validator run that a "did it verify
        # its work" expectation turns on, making the agent look like it lied.
        text = "START" + ("x" * 5000) + "VALIDATED"
        trimmed = engine_judge._elide_middle(text, 400)
        self.assertTrue(trimmed.startswith("START"))
        self.assertTrue(trimmed.endswith("VALIDATED"))
        self.assertIn("elided", trimmed)
        self.assertLess(len(trimmed), 600)


class _State:
    def __init__(self, messages, output=None) -> None:
        self.messages = messages
        self.output = output


class _Output:
    def __init__(self, completion: str) -> None:
        self.completion = completion


class _Assistant:
    role = "assistant"

    def __init__(self, content: str) -> None:
        self.content = content


class TestEngineJudgeFinalMessage(unittest.TestCase):
    """A `react` agent answers through submit, not through a chat message."""

    def test_the_submitted_answer_is_included_and_marked(self) -> None:
        state = _State(
            [_Assistant("Here are the commands you need:")],
            _Output("curl -X POST /api/v1/pull -d '{...}'"),
        )
        said = engine_judge.final_message_of(state)
        self.assertIn("curl -X POST", said)
        self.assertIn("[submitted answer]", said)

    def test_an_earlier_turn_still_counts_as_having_told_the_user(self) -> None:
        # The user sees every assistant turn, so an agent that prints the
        # commands mid-run and then submits a summary did tell them. Reading
        # only the last turn credited the summary and called the commands
        # missing.
        state = _State(
            [
                _Assistant("Run: curl -X POST /api/v1/pull"),
                _Assistant("Done -- commands delivered above."),
            ],
            _Output("Done -- commands delivered above."),
        )
        self.assertIn("curl -X POST", engine_judge.final_message_of(state))

    def test_it_works_without_a_submit_tool(self) -> None:
        state = _State([_Assistant("no submit tool in this agent")], None)
        self.assertEqual(
            engine_judge.final_message_of(state), "no submit tool in this agent"
        )

    def test_silence_is_reported_rather_than_guessed_at(self) -> None:
        self.assertEqual(
            engine_judge.final_message_of(_State([], None)), "(the agent said nothing)"
        )


class TestEngineJudgeArtifacts(unittest.TestCase):
    def test_images_are_recognised_by_suffix(self) -> None:
        self.assertTrue(engine_judge.is_image("out.PNG"))
        self.assertTrue(engine_judge.is_image("art/cat.jpeg"))
        self.assertFalse(engine_judge.is_image("notes.md"))

    def test_known_binaries_are_not_read_as_text(self) -> None:
        self.assertTrue(engine_judge.is_probably_binary("model.safetensors"))
        self.assertFalse(engine_judge.is_probably_binary("report.md"))


class TestEngineSandboxSelection(unittest.TestCase):
    """The provider is the machine's choice; the compose file is the skill's."""

    def setUp(self) -> None:
        self.addCleanup(os.environ.pop, engine_sandbox.SANDBOX_ENV, None)
        os.environ.pop(engine_sandbox.SANDBOX_ENV, None)
        self.repo = Repo(self)
        # Pinned, because the answer depends on the platform and the suite runs
        # on both. Without this these assertions quietly mean something
        # different on a Windows runner than on a Linux one.
        self._posix_host()

    def _posix_host(self) -> None:
        patch = mock.patch.object(engine_sandbox, "is_windows", lambda: False)
        patch.start()
        self.addCleanup(patch.stop)

    def _windows_host(self) -> None:
        patch = mock.patch.object(engine_sandbox, "is_windows", lambda: True)
        patch.start()
        self.addCleanup(patch.stop)

    def _skill(self, machine: str | None = None, compose: bool = False) -> None:
        folder = self.repo.skill(
            "boxed", dataset=tier0_dataset("boxed"), machine=machine
        )
        if compose:
            # Beside machine.yml, not at the skill root: the skill root is what
            # gets published, and eval infrastructure does not belong there.
            (folder / "evals" / "compose.yaml").write_text(
                "services: {}\n", encoding="utf-8"
            )
        self.repo.activate()

    def test_docker_by_default(self) -> None:
        self._skill()
        self.assertEqual(engine_sandbox.for_skill("boxed"), "docker")

    def test_the_env_var_selects_the_provider(self) -> None:
        self._skill()
        os.environ[engine_sandbox.SANDBOX_ENV] = "podman"
        self.assertEqual(engine_sandbox.for_skill("boxed"), "podman")

    def test_a_declared_compose_file_rides_along(self) -> None:
        self._skill(machine="sandbox: compose.yaml\n", compose=True)
        provider, config = engine_sandbox.for_skill("boxed")
        self.assertEqual(provider, "docker")
        self.assertTrue(config.endswith("compose.yaml"))

    def test_selecting_a_provider_keeps_the_skill_s_compose_file(self) -> None:
        # The skill asked for network egress; choosing podman must not drop it,
        # or the case runs without what it needs and fails unexplainably.
        self._skill(machine="sandbox: compose.yaml\n", compose=True)
        os.environ[engine_sandbox.SANDBOX_ENV] = "podman"
        provider, config = engine_sandbox.for_skill("boxed")
        self.assertEqual(provider, "podman")
        self.assertTrue(config.endswith("compose.yaml"))

    def test_local_takes_no_configuration(self) -> None:
        self._skill(machine="sandbox: compose.yaml\n", compose=True)
        os.environ[engine_sandbox.SANDBOX_ENV] = "local"
        self.assertEqual(engine_sandbox.for_skill("boxed"), "local")

    def test_windows_has_no_sandbox_available(self) -> None:
        # inspect's sandbox layer assumes a POSIX guest, so those legs run
        # unsandboxed -- and a compose file the skill declared cannot apply,
        # because there is no container to apply it to.
        self._windows_host()
        self._skill(machine="sandbox: compose.yaml\n", compose=True)
        self.assertEqual(engine_sandbox.for_skill("boxed"), "local")

    def test_an_unresolvable_provider_says_what_to_install(self) -> None:
        # The binary being present proves nothing: inspect resolves a
        # third-party provider through an entry point, so the Python package
        # has to be installed too. Its own error names neither the variable
        # nor the package.
        os.environ[engine_sandbox.SANDBOX_ENV] = "podman"

        def unresolvable(name: str):
            raise ValueError(f"SandboxEnvironment type {name!r} not recognized.")

        with self.assertRaises(SystemExit) as caught:
            engine_sandbox.require_provider(resolve=unresolvable)
        message = str(caught.exception)
        self.assertIn(engine_sandbox.SANDBOX_ENV, message)
        self.assertIn("skillscope[podman]", message)

    def test_a_named_compose_file_that_is_missing_is_an_error(self) -> None:
        self._skill(machine="sandbox: nope.yaml\n")
        with self.assertRaises(SystemExit) as caught:
            engine_sandbox.for_skill("boxed")
        self.assertIn("nope.yaml", str(caught.exception))


class TestRoutingRunsOnEveryEngine(unittest.TestCase):
    """Routing has a leg for all three engines, and reaches the one asked for.

    It used to have one. `claude-code` and `claude-code-no-sandbox` were
    refused, because they named the real CLI as the agent and routing had no
    path that drove it. Now they do, and the assertion that matters is no
    longer the refusal -- it is that asking for a leg reaches that leg. The
    defect this replaces was exactly the opposite: a run asked for one engine
    silently got another, and the report named the one it had asked for.
    """

    def setUp(self) -> None:
        self.repo = Repo(self)
        self.repo.skill("alpha", dataset=tier0_dataset("alpha"))
        self.repo.activate(routing_room="alpha")
        self.reached: list[str] = []

        def record_inspect(cases, routing_set, model, effort, engine, **kwargs):
            # The cap is recorded, not merely tolerated: a leg that never
            # receives it bounds a case by the whole command's budget while the
            # report claims otherwise.
            self.reached.append(f"inspect:{engine}")
            self.passed_kwargs = kwargs
            return []

        def record_legacy(case, routing_set, cfg):
            self.reached.append("legacy")
            raise AssertionError("the legacy leg ran for an inspect engine")

        for target, attr, fn in (
            ("skillscope.engine.routing", "run", record_inspect),
            (None, "run_case", record_legacy),
        ):
            patch = (
                mock.patch(f"{target}.{attr}", fn)
                if target
                else mock.patch.object(routing, attr, fn)
            )
            patch.start()
            self.addCleanup(patch.stop)

        for name in ("_write_report", "_prepare_graded_run"):
            patch = mock.patch.object(cli, name, lambda *a, **k: None)
            patch.start()
            self.addCleanup(patch.stop)

    def run_routing(self, engine: str) -> None:
        args = cli.build_parser().parse_args(
            ["routing", "--engine", engine, "--skip-preflight", "--model", "mockllm/model"]
        )
        cli.cmd_routing(args)

    def test_every_engine_now_has_a_routing_leg(self) -> None:
        self.assertEqual(set(cli.ROUTING_ENGINES), set(cli.ENGINES))

    def test_asking_for_the_sandboxed_leg_reaches_it(self) -> None:
        self.run_routing("claude-code")
        self.assertEqual(self.reached, ["inspect:claude-code"])

    def test_asking_for_the_host_leg_reaches_it(self) -> None:
        # The assertion that would have caught the original fallthrough: the
        # legacy runner is patched to fail loudly if it is reached.
        self.run_routing("claude-code-no-sandbox")
        self.assertEqual(self.reached, ["inspect:claude-code-no-sandbox"])

    def test_the_leg_is_given_the_per_case_cap(self) -> None:
        self.run_routing("claude-code")
        self.assertIn("case_timeout", self.passed_kwargs)

    def test_the_guard_reads_the_engine_set_rather_than_a_literal(self) -> None:
        # A literal tuple is the defect itself, so the source is what to assert.
        self.assertIn(
            "if args.engine in INSPECT_ENGINES:", inspect.getsource(cli.cmd_routing)
        )

    def test_an_engine_that_no_longer_exists_is_refused_by_the_parser(self) -> None:
        self.assertNotIn("inspect", cli.ENGINES)
        with self.assertRaises(SystemExit), contextlib.redirect_stderr(io.StringIO()):
            cli.build_parser().parse_args(["routing", "--engine", "inspect"])


class TestTheTranscriptIsReadTooNotJustTheMessages(unittest.TestCase):
    """A bridged agent's work does not always land in `sample.messages`.

    inspect adopts one conversation onto the sample, and for a bridged scaffold
    that adoption follows heuristics about which thread is the main one -- a
    run ending inside a sub-agent can leave it holding the wrong thread or
    none. The transcript is strictly more complete: every bridged generation
    emits a `ModelEvent`, sub-agents included.

    Measured rather than reasoned about. The first sandboxed routing run on a
    real container graded 24 of 67 cases as "the agent never ran", while the
    legacy engine saw those same cases activate a skill. Reading only
    `sample.messages` was the whole of the bug.
    """

    class Call:
        def __init__(self, id, function, arguments):
            self.id, self.function, self.arguments = id, function, arguments

    class Msg:
        def __init__(self, role="assistant", tool_calls=None, id=None):
            self.role, self.tool_calls, self.id = role, tool_calls, id

    class Event:
        def __init__(self, message):
            self.output = type("O", (), {"message": message})()

    class Sample:
        def __init__(self, messages=None, events=None):
            self.messages, self.events = messages, events

    ROOM = ["alpha", "beta"]

    def skill_call(self):
        return self.Call("t0", "Skill", {"command": "alpha"})

    def test_an_activation_only_in_the_messages_is_seen(self) -> None:
        sample = self.Sample(messages=[self.Msg(tool_calls=[self.skill_call()])])
        self.assertEqual(engine_routing._observe(sample, self.ROOM)[0], "alpha")

    def test_an_activation_only_in_the_transcript_is_seen(self) -> None:
        # The case that was being missed.
        sample = self.Sample(events=[self.Event(self.Msg(tool_calls=[self.skill_call()]))])
        self.assertEqual(engine_routing._observe(sample, self.ROOM)[0], "alpha")

    def test_the_same_turn_as_two_objects_is_counted_once(self) -> None:
        # What the bridge actually produces: the turn adopted onto the sample
        # and the turn carried by the transcript event are different objects
        # with the same id. De-duplicating by identity missed that and counted
        # every call twice -- a sandboxed run reported ten tool calls for
        # cases the approver had terminated at five, halving the effective
        # budget.
        calls = [self.Call("t1", "Bash", {"command": "ls"})]
        adopted = self.Msg(tool_calls=calls)
        adopted.id = "m1"
        in_event = self.Msg(tool_calls=calls)
        in_event.id = "m1"
        sample = self.Sample(messages=[adopted], events=[self.Event(in_event)])
        self.assertEqual(engine_routing._observe(sample, self.ROOM)[1], 1)

    def test_two_genuinely_different_turns_are_both_counted(self) -> None:
        first = self.Msg(tool_calls=[self.Call("t1", "Bash", {"command": "ls"})])
        first.id = "m1"
        second = self.Msg(tool_calls=[self.Call("t2", "Bash", {"command": "pwd"})])
        second.id = "m2"
        sample = self.Sample(messages=[first, second])
        self.assertEqual(engine_routing._observe(sample, self.ROOM)[1], 2)

    def test_a_run_that_chose_nothing_is_still_a_run(self) -> None:
        sample = self.Sample(messages=[self.Msg(tool_calls=[])])
        self.assertIsNone(engine_routing._observe(sample, self.ROOM)[0])
        self.assertTrue(engine_routing._spoke(sample))

    class Limit:
        def __init__(self, type):
            self.type = type

    def test_a_sample_that_hit_a_limit_counts_as_having_run(self) -> None:
        # When the message cap trips, what survives on the sample can be the
        # prompt and nothing else. Calling that "never ran" is wrong and the
        # opposite of useful: an agent that spent its whole budget without
        # reaching for a skill is the clearest kind of missed trigger, which
        # is how legacy grades its own `tool_budget` stop.
        sample = self.Sample(messages=[self.Msg(role="user")])
        sample.limit = self.Limit("message")
        self.assertTrue(engine_routing._spoke(sample))

    def test_a_prompt_with_no_limit_and_no_reply_still_never_ran(self) -> None:
        sample = self.Sample(messages=[self.Msg(role="user")])
        self.assertFalse(engine_routing._spoke(sample))

    def test_a_sample_with_neither_record_never_ran(self) -> None:
        # Preserved: this is the infrastructure failure the guard exists for,
        # and grading it as a miss would invent a routing result.
        self.assertFalse(engine_routing._spoke(self.Sample()))

    def test_the_transcript_alone_counts_as_having_run(self) -> None:
        sample = self.Sample(events=[self.Event(self.Msg(tool_calls=[]))])
        self.assertTrue(engine_routing._spoke(sample))


class TestAnEmptyAnswerIsStillAnAnswer(unittest.TestCase):
    """A run that finished saying nothing is a routing result, not a failure.

    The routing mapper has to tell "the agent reached for no skill" from "the
    agent never ran", and it draws that line at whether the sample produced any
    messages. The host driver appended a closing message only when the CLI had
    something to say, so a run that finished quietly left none -- and two cases
    on the real runner were graded as infrastructure failures where the legacy
    engine graded them `correct_trigger` and `true_negative`.

    The distinction is still drawn, just in the right place: a stream carrying
    no result event at all is a CLI that never finished.

    Exercised through `run_completed`, which is pure. The unit suite runs
    without the inspect extra, and a rule reachable only through inspect's
    message objects would go untested in CI -- which is where it matters.
    """

    def test_a_run_that_finished_with_an_answer(self) -> None:
        done, final = engine_no_sandbox.run_completed(
            [{"type": "result", "result": "no skill needed"}]
        )
        self.assertTrue(done)
        self.assertEqual(final, "no skill needed")

    def test_a_run_that_finished_saying_nothing_still_counts_as_finished(self) -> None:
        # The case this fixes. An empty answer is the agent declining to route.
        done, final = engine_no_sandbox.run_completed([{"type": "result", "result": ""}])
        self.assertTrue(done)
        self.assertEqual(final, "")

    def test_a_result_event_with_no_text_at_all_still_counts(self) -> None:
        self.assertTrue(engine_no_sandbox.run_completed([{"type": "result"}])[0])

    def test_a_stream_that_never_finished_does_not_count(self) -> None:
        # Preserved deliberately: no result event means the CLI did not reach
        # the end, which is an infrastructure failure and must stay one.
        self.assertFalse(
            engine_no_sandbox.run_completed(
                [{"type": "assistant", "message": {"content": []}}]
            )[0]
        )

    def test_the_last_result_event_wins(self) -> None:
        _, final = engine_no_sandbox.run_completed(
            [{"type": "result", "result": "first"}, {"type": "result", "result": "last"}]
        )
        self.assertEqual(final, "last")

    def test_the_placeholder_matches_what_the_output_already_used(self) -> None:
        # The message list was the inconsistent half; `state.output` has always
        # substituted this for an empty answer.
        source = inspect.getsource(engine_no_sandbox)
        self.assertEqual(source.count('"(no final message)"'), 2)


class TestHooksCannotBeSilentlySkipped(unittest.TestCase):
    """A skill's setup either runs, or the run stops. Not skipped quietly.

    `evals/hooks.py` is environment plumbing and only the legacy engine
    executes it -- no inspect-backed engine builds the `ctx` those hooks take.
    Skipping it leaves no trace: the case is graded as though the setup
    happened, and the failure surfaces later as a skill that mysteriously does
    not work on this runner.

    Not hypothetical. The catalogue this was written against ships a hook whose
    `setup` clears stale vLLM containers and whose `teardown` removes them, so
    skipping it leaks containers holding GPU memory into whatever runs next --
    the same contamination the sandboxed engines exist to prevent.
    """

    def setUp(self) -> None:
        self.repo = Repo(self)
        self.repo.skill("plain", dataset=tier0_dataset("plain"))
        self.repo.skill(
            "hooked",
            dataset=tier0_dataset("hooked"),
            hooks="def setup(workspace, case, ctx):\n    pass\n",
        )
        self.repo.activate()

    def refuse(self, engine: str, skills: list[str], command: str = "behavioral"):
        return cli._require_hook_support(engine, skills, command)

    def test_an_inspect_engine_refuses_a_skill_that_ships_a_hook(self) -> None:
        with self.assertRaises(SystemExit) as caught:
            self.refuse("claude-code-no-sandbox", ["hooked"])
        message = str(caught.exception)
        self.assertIn("hooks.py", message)
        self.assertIn("hooked", message)
        self.assertIn("--engine legacy", message)

    def test_the_refusal_names_every_skill_that_would_be_skipped(self) -> None:
        # Naming one of three sends someone round the loop twice.
        self.repo.skill(
            "also-hooked",
            dataset=tier0_dataset("also"),
            hooks="def teardown(workspace, case, ctx):\n    pass\n",
        )
        with self.assertRaises(SystemExit) as caught:
            self.refuse("claude-code", ["hooked", "also-hooked", "plain"])
        message = str(caught.exception)
        self.assertIn("hooked", message)
        self.assertIn("also-hooked", message)

    def test_legacy_runs_hooks_so_it_is_not_refused(self) -> None:
        self.assertIsNone(self.refuse("legacy", ["hooked"]))

    def test_a_skill_without_a_hook_is_not_refused(self) -> None:
        self.assertIsNone(self.refuse("claude-code", ["plain"]))

    def test_routing_is_exempt_because_it_never_reads_hooks(self) -> None:
        # A routing run installs the skills and asks which one fires. It
        # executes nothing, so there is no setup to skip.
        self.assertIsNone(self.refuse("claude-code", ["hooked"], command="routing"))

    def test_every_inspect_engine_is_covered(self) -> None:
        for engine in cli.INSPECT_ENGINES:
            with self.subTest(engine=engine):
                with self.assertRaises(SystemExit):
                    self.refuse(engine, ["hooked"])

    def test_no_inspect_engine_has_quietly_gained_hook_support(self) -> None:
        # If one ever does, this guard becomes wrong rather than merely
        # unnecessary, and the failure would be a refusal nobody can explain.
        import skillscope.engine.behavioral as eb
        import skillscope.engine.verify as ev

        for module in (eb, ev):
            with self.subTest(module=module.__name__):
                self.assertNotIn("load_hooks", inspect.getsource(module))


class TestCasesDecidedByTheBudgetAreFlagged(unittest.TestCase):
    """A case that stopped on its cap measured the cap, not the agent.

    Two runs of one engine disagreed on six cases where the noise floor was
    one, and the explanation was that 26 of 67 cases sat within a single call
    of a budget. Those are coin-flips: one more call either way moves them
    across the line and takes the verdict with them. They look identical to
    cases decided on their merits, which is what made the disagreement
    unreadable.

    Reported rather than corrected. The budget is doing its job; the reader
    just has to know how much of the run it decided.
    """

    def outcome(self, tool_calls=0, inspection_calls=0):
        return routing.Outcome(
            id="a", category="c", skill="s", prompt="p", expect=None, observed=None,
            verdict="true_negative", passed=True, stop_reason="result",
            elapsed_s=0.0, tool_calls=tool_calls, inspection_calls=inspection_calls,
        )

    CAPS = {"max_tool_calls": 4, "max_inspection_calls": 8}

    def test_a_case_that_stopped_on_the_cap_is_flagged(self) -> None:
        self.assertTrue(routing.near_a_limit(self.outcome(tool_calls=4), self.CAPS))

    def test_a_case_one_below_the_cap_is_flagged(self) -> None:
        # One call is the resolution the threshold has.
        self.assertTrue(routing.near_a_limit(self.outcome(tool_calls=3), self.CAPS))

    def test_a_case_well_clear_of_the_cap_is_not(self) -> None:
        self.assertFalse(routing.near_a_limit(self.outcome(tool_calls=1), self.CAPS))

    def test_the_inspection_budget_counts_too(self) -> None:
        self.assertTrue(routing.near_a_limit(self.outcome(inspection_calls=8), self.CAPS))

    def test_a_cap_the_run_never_set_is_not_a_threshold(self) -> None:
        # A leg that cannot enforce a budget reports none, and nothing should
        # invent one for it.
        self.assertFalse(routing.near_a_limit(self.outcome(tool_calls=99), {}))
        self.assertFalse(
            routing.near_a_limit(self.outcome(tool_calls=99), {"max_tool_calls": 0})
        )

    def test_the_totals_carry_the_count(self) -> None:
        outs = [self.outcome(tool_calls=4), self.outcome(tool_calls=0)]
        totals = routing.summarize(outs, ["s"], {"skills": ["s"], **self.CAPS})["totals"]
        self.assertEqual(totals["near_limit"], 1)

    def test_the_report_warns_before_the_reader_sees_the_score(self) -> None:
        summary = routing.summarize(
            [self.outcome(tool_calls=4)],
            ["s"],
            {"skills": ["s"], "model": "m", "effort": "e", **self.CAPS},
        )
        report = routing.render_markdown(summary)
        self.assertIn("within one call of a budget", report)
        self.assertLess(report.index("within one call"), report.index("| Verdict |"))


class TestProviderFailuresAreMarked(unittest.TestCase):
    """A gateway error is not a routing result, and must not read as one.

    A 504 lands in a report as a lower score with nothing in the verdict
    saying why, so a reader cannot tell it from the skill failing. At the rate
    observed on one catalogue -- roughly a third of runs -- that makes an
    unmarked provider error the most likely reason two runs of the same engine
    disagree, which has to be ruled out before a difference between engines
    means anything.

    Matching is deliberately generous: a false positive makes a reader look
    twice at a run that was fine, a false negative lets an outage score as a
    routing miss. The costs are not symmetric.
    """

    def test_the_shapes_actually_observed_are_recognised(self) -> None:
        for message in (
            "API Error: 504 Exception trying to (AnthropicVertex) Chat Completions",
            "API preflight timed out after 60s (is the network reachable?)",
            "APIConnectionError: Connection error.",
            "429 rate limit exceeded",
            "502 Bad Gateway",
            "upstream connect error",
        ):
            with self.subTest(message=message):
                self.assertTrue(routing.is_provider_error(message))

    def test_a_real_skill_failure_is_not_marked(self) -> None:
        for message in (
            "the skill produced no plan.md",
            "run ended without a routing decision (stopped after: tool_budget)",
            None,
            "",
        ):
            with self.subTest(message=message):
                self.assertFalse(routing.is_provider_error(message))

    def test_the_reason_a_result_event_gave_is_kept(self) -> None:
        # It was being discarded. A 504 arrived as "result event reported an
        # error", which no classifier and no reader can do anything with.
        self.assertIn(
            "504",
            routing._result_error({"is_error": True, "result": "API Error: 504 upstream"}),
        )

    def test_the_subtype_is_kept_when_there_is_no_body(self) -> None:
        # The CLI puts the reason in one field or the other depending on how
        # it failed, and only one of them is ever populated.
        self.assertIn(
            "error_max_turns",
            routing._result_error({"is_error": True, "subtype": "error_max_turns"}),
        )

    def test_an_agent_failure_is_not_blamed_on_the_provider(self) -> None:
        # `error_max_turns` is the agent running out of road, not the gateway.
        # Marking it degraded would excuse a real failure.
        self.assertFalse(routing.is_provider_error(
            routing._result_error({"is_error": True, "subtype": "error_max_turns"})
        ))

    def test_an_error_with_no_reason_is_not_guessed_at(self) -> None:
        # Neither marked nor excused: we do not know, and saying so is the
        # only honest option.
        self.assertFalse(routing.is_provider_error(
            routing._result_error({"is_error": True})
        ))

    def test_routing_totals_report_how_many_were_degraded(self) -> None:
        # Beside the score, because it decides whether the score can be read.
        outcomes = [
            routing.Outcome(
                id=str(i), category="c", skill="s", prompt="p", expect=None,
                observed=None, verdict="error", passed=False, stop_reason="result",
                elapsed_s=0.0, tool_calls=0, error=err,
                degraded=routing.is_provider_error(err),
            )
            for i, err in enumerate(["API Error: 504 upstream", "the skill did nothing"])
        ]
        totals = routing.summarize(outcomes, ["s"], {"skills": ["s"]})["totals"]
        self.assertEqual(totals["errors"], 2)
        self.assertEqual(totals["degraded"], 1)

    def test_behavioral_totals_report_it_too(self) -> None:
        # Same vocabulary on both commands, or a degraded behavioral run reads
        # as a failing skill.
        outcomes = [
            behavior.BehaviorOutcome(
                id="a", skill="s", prompt="p", passed=False, elapsed_s=0.0,
                error="API Error: 504", degraded=True,
            )
        ]
        totals = behavior.summarize(outcomes, {"model": "m", "effort": "e"})["totals"]
        self.assertEqual(totals["degraded"], 1)

    def test_the_report_warns_before_the_reader_sees_the_score(self) -> None:
        # Underneath the table is too late: a reader has already formed a view.
        outcomes = [
            routing.Outcome(
                id="a", category="c", skill="s", prompt="p", expect=None,
                observed=None, verdict="error", passed=False, stop_reason="result",
                elapsed_s=0.0, tool_calls=0, error="API Error: 504", degraded=True,
            )
        ]
        summary = routing.summarize(
            outcomes, ["s"], {"skills": ["s"], "model": "m", "effort": "e"}
        )
        report = routing.render_markdown(summary)
        self.assertIn("failed at the model provider", report)
        self.assertLess(report.index("model provider"), report.index("| Verdict |"))


class TestRoutingStopsAtTheDecision(unittest.TestCase):
    """The sandboxed leg stops when the decision is made, as legacy does.

    Legacy reads the CLI's stream and kills the process the moment a skill
    activates, because everything after that is work the routing question does
    not ask for and does pay for. The rule here runs earlier: it sees each tool
    call the bridged CLI *proposes*, so the case stops without the work
    happening at all.

    The skill's name goes into the reason on purpose. Approval runs before the
    bridge adopts the assistant message, so the tool call that revealed the
    decision may be absent from the transcript afterwards -- and a suppressed
    activation reads exactly like an agent that correctly declined to route.

    Exercised through `routing_decision`, which is pure, rather than through
    the approver that wraps it: the unit suite runs without the inspect extra
    on purpose, and a rule that can only be tested with it would not be tested
    at all in CI.
    """

    ROOM = ["alpha", "beta", "gamma"]

    def decide(self, function, arguments, tally=None, tools=None, inspections=None):
        return engine_routing.routing_decision(
            function, arguments, self.ROOM,
            tally or engine_routing._Tally(), tools, inspections,
        )

    def test_activating_a_skill_stops_the_case(self) -> None:
        decision, _ = self.decide("Skill", {"command": "alpha"})
        self.assertEqual(decision, "terminate")

    def test_the_skill_that_fired_survives_in_the_reason(self) -> None:
        # The half that cannot go missing when the message does.
        _, reason = self.decide("Skill", {"command": "alpha"})
        self.assertEqual(
            engine_routing.activation_from_limit(reason, self.ROOM), "alpha"
        )

    def test_a_skill_nobody_installed_is_still_a_decision(self) -> None:
        # A contaminated room is a routing result, not a non-event: the run
        # has to be able to say a stranger fired.
        decision, reason = self.decide("Skill", {"command": "dataviz"})
        self.assertEqual(decision, "terminate")
        self.assertEqual(
            engine_routing.activation_from_limit(reason, self.ROOM), "other:dataviz"
        )

    def test_ordinary_work_is_allowed_through(self) -> None:
        self.assertEqual(self.decide("Bash", {"command": "ls"})[0], "approve")

    def test_a_skills_survey_spends_the_inspection_budget_not_the_work_one(self) -> None:
        # Legacy's rule, and it matters: counting a survey against the work
        # budget too ends a run mid-deliberation and scores it as a missed
        # trigger. Observed doing exactly that on a real sandboxed run.
        tally = engine_routing._Tally()
        self.decide("Read", {"file_path": "/w/.claude/skills/alpha/SKILL.md"}, tally, 4, 8)
        self.assertEqual((tally.tools, tally.inspections), (0, 1))

    def test_unrelated_work_spends_the_work_budget(self) -> None:
        tally = engine_routing._Tally()
        self.decide("Bash", {"command": "pip install torch"}, tally, 4, 8)
        self.assertEqual((tally.tools, tally.inspections), (1, 0))

    def test_bookkeeping_spends_neither(self) -> None:
        tally = engine_routing._Tally()
        for name in sorted(routing.BOOKKEEPING_TOOLS):
            self.decide(name, {}, tally, 4, 8)
        self.assertEqual((tally.tools, tally.inspections), (0, 0))

    def test_the_budget_is_counted_per_case_not_per_run(self) -> None:
        # An approver is built once per task. A counter captured there counts
        # every case in the run, creeping up until it crosses the budget and
        # then terminating any case that makes a counted call before choosing.
        # Seen as four cases terminated at tallies of 10, 11, 12 and 13 --
        # consecutive across different prompts.
        self.assertIn("store()", inspect.getsource(engine_routing.routing_approver))
        self.assertNotIn(
            "tally = _Tally()\n\n    @approver",
            inspect.getsource(engine_routing.routing_approver),
        )

    def test_the_tool_call_budget_stops_a_case_that_is_rummaging(self) -> None:
        tally = engine_routing._Tally()
        decisions = [
            self.decide("Bash", {"command": f"ls {i}"}, tally, tools=2)[0]
            for i in range(4)
        ]
        self.assertEqual(decisions, ["approve", "approve", "terminate", "terminate"])

    def test_bookkeeping_calls_do_not_count_against_the_budget(self) -> None:
        # Same rule as legacy: the budget is about work, not housekeeping.
        tally = engine_routing._Tally()
        for name in sorted(routing.BOOKKEEPING_TOOLS):
            self.decide(name, {}, tally, tools=1)
        self.assertEqual(tally.tools, 0)

    def test_a_limit_reason_from_elsewhere_is_not_read_as_an_activation(self) -> None:
        self.assertIsNone(
            engine_routing.activation_from_limit("operator cancelled", self.ROOM)
        )

    def test_a_named_skill_outside_the_room_is_not_read_as_an_activation(self) -> None:
        # The room is the authority. Otherwise a stray string becomes a verdict.
        forged = engine_routing.ACTIVATION_MARK + "delta"
        self.assertIsNone(engine_routing.activation_from_limit(forged, self.ROOM))

    def test_the_approver_delegates_to_the_rule_rather_than_repeating_it(self) -> None:
        self.assertIn(
            "routing_decision(", inspect.getsource(engine_routing.routing_approver)
        )

    def test_the_host_leg_gets_no_approver(self) -> None:
        # Its CLI buffers until exit, so there is nothing to approve in time;
        # attaching one would suggest a bound that does not exist.
        self.assertIn(
            "if engine == CLAUDE_CODE", inspect.getsource(engine_routing.build_task)
        )


class TestRoutingCaseTimeoutBinds(unittest.TestCase):
    """`--case-timeout` has to reach the new legs, or it is a cap in name only.

    The flag exists so one hung prompt cannot spend the whole run -- `deadline`
    says so in its own docstring. The inspect legs originally bounded a sample
    only by the command's remaining budget, which is precisely the thing the
    flag guards against, while the report recorded `case_timeout` as though it
    had applied. A cap that is reported and not enforced is worse than none.
    """

    def tearDown(self) -> None:
        deadline.use(None)

    def test_the_flag_is_the_bound_when_the_command_has_room(self) -> None:
        deadline.use(deadline.Deadline(3000.0, command="routing"))
        self.assertEqual(engine_routing.case_time_limit(90), 90)

    def test_the_command_deadline_clips_a_longer_case_cap(self) -> None:
        # The command deadline ends the process outright, taking the report
        # with it, so a per-case cap must not outlive it.
        deadline.use(deadline.Deadline(200.0, command="routing"))
        self.assertLess(engine_routing.case_time_limit(9999), 200)

    def test_no_case_cap_falls_back_to_the_command_budget(self) -> None:
        deadline.use(deadline.Deadline(3000.0, command="routing"))
        self.assertEqual(
            engine_routing.case_time_limit(None),
            engine_behavioral.task_time_limit(deadline.active()),
        )

    def test_an_unbounded_command_still_honours_the_case_cap(self) -> None:
        deadline.use(None)
        self.assertEqual(engine_routing.case_time_limit(45), 45)

    def test_the_cli_hands_the_flag_to_the_leg(self) -> None:
        # Asserted on the source: the failure mode is the argument silently
        # not being passed, which no unit of the leg can notice.
        self.assertIn("case_timeout=args.case_timeout", inspect.getsource(cli.cmd_routing))


class TestRoutingRoomIsTheRoomThatWasAskedFor(unittest.TestCase):
    """The host routing leg isolates the config dir, or it does not run.

    The legacy engine warns and carries on, because it reads the CLI's `init`
    event and can name a user-level skill that gate-crashed the room. Neither
    inspect leg gets that event, so the same contamination would be invisible
    -- and a stray skill is offered for every prompt, so it changes every
    decision at once while the run still reports a clean accuracy.

    Observed rather than feared: a probe of this leg on a developer machine put
    roughly forty user-level skills in the room and none of the three staged.
    """

    def setUp(self) -> None:
        self.addCleanup(os.environ.pop, "ANTHROPIC_API_KEY", None)

    def test_the_host_leg_refuses_without_the_credential_that_isolates_it(self) -> None:
        os.environ.pop("ANTHROPIC_API_KEY", None)
        with self.assertRaises(SystemExit) as caught:
            engine_routing.require_isolated_room("claude-code-no-sandbox")
        message = str(caught.exception)
        self.assertIn("ANTHROPIC_API_KEY", message)
        self.assertIn("--engine claude-code", message)

    def test_the_host_leg_runs_when_it_can_isolate(self) -> None:
        os.environ["ANTHROPIC_API_KEY"] = "sk-test"
        self.assertIsNone(engine_routing.require_isolated_room("claude-code-no-sandbox"))

    def test_the_sandboxed_leg_needs_no_credential_to_have_a_clean_room(self) -> None:
        # The guest has no `~/.claude` to keep out, which is the whole reason
        # this leg is the one to prefer for routing.
        os.environ.pop("ANTHROPIC_API_KEY", None)
        self.assertIsNone(engine_routing.require_isolated_room("claude-code"))

    def test_the_host_solver_can_be_pointed_at_a_config_dir(self) -> None:
        # Without this parameter the leg reads the runner's own config dir and
        # nothing downstream can tell.
        self.assertIn(
            "config_dir", inspect.signature(engine_no_sandbox.claude_code_no_sandbox).parameters
        )


class TestRoutingReportsWhereItRan(unittest.TestCase):
    """Each leg states what contained it; none may overstate it.

    Deriving this from the engine's name is what let a run that started no
    container report `sandbox: docker, sandbox_isolated: true`. Hardcoding
    `host` was right only while no routing leg had a sandbox. Now one does, so
    the value is passed in by the leg that knows.
    """

    def setUp(self) -> None:
        self.repo = Repo(self)
        self.repo.skill("alpha", dataset=tier0_dataset("alpha"))
        self.repo.activate()
        self.written: dict = {}
        patch = mock.patch.object(
            cli, "_write_report", lambda summary, *a, **k: self.written.update(summary)
        )
        patch.start()
        self.addCleanup(patch.stop)

    def meta(self, engine: str, **containment) -> dict:
        args = cli.build_parser().parse_args(
            ["routing", "--engine", engine, "--skip-preflight"]
        )
        cli._finish_routing(
            args, [], {"alpha": None}, time.time(), isolated=True, **containment
        )
        return self.written["meta"]

    def test_a_host_leg_says_host(self) -> None:
        meta = self.meta("legacy", sandbox="host", sandbox_isolated=False)
        self.assertEqual(meta["sandbox"], "host")
        self.assertIs(meta["sandbox_isolated"], False)

    def test_the_sandboxed_leg_names_the_provider_it_used(self) -> None:
        meta = self.meta("claude-code", sandbox="docker", sandbox_isolated=True)
        self.assertEqual(meta["sandbox"], "docker")
        self.assertIs(meta["sandbox_isolated"], True)

    def test_a_leg_that_does_not_say_what_contained_it_cannot_report(self) -> None:
        # Required keywords, so a leg added later fails at the call site
        # instead of quietly reporting a containment it never had.
        args = cli.build_parser().parse_args(["routing", "--skip-preflight"])
        with self.assertRaises(TypeError):
            cli._finish_routing(args, [], {"alpha": None}, time.time(), isolated=True)

    def test_a_legs_own_meta_cannot_overwrite_what_contained_it(self) -> None:
        meta = self.meta(
            "legacy", sandbox="host", sandbox_isolated=False,
        )
        self.assertEqual(meta["sandbox"], "host")

    def test_the_report_does_not_ask_the_provider_what_contained_a_run(self) -> None:
        # The defect in one line: `_sandbox_meta` returns `provider()`, which
        # examines nothing and returns the default string.
        self.assertNotIn("_sandbox_meta", inspect.getsource(cli._finish_routing))


class TestClaudeCliPreflightChecksBothCredentials(unittest.TestCase):
    """`claude-code-no-sandbox` uses two, so probing one proves nothing about the other.

    The real CLI is the agent and the inspect provider grades it, so a run dies
    on whichever is missing. The preflight tested the provider alone -- and for
    routing, which uses neither, it tested the provider and demanded the
    inspect extra for a leg that runs no inspect code.
    """

    def setUp(self) -> None:
        self.repo = Repo(self)
        self.repo.skill("alpha", dataset=tier0_dataset("alpha"))
        self.repo.activate()
        self.probed: list[str] = []

    def args(self, engine: str) -> argparse.Namespace:
        return cli.build_parser().parse_args(
            ["behavioral", "--engine", engine, "--model", "opus"]
        )

    def run_preflight(self, engine: str, cli_ok: bool = True) -> list[str]:
        patches = [
            mock.patch.object(cli.engine, "require", lambda *a, **k: None),
            mock.patch.object(
                cli, "check_api_reachable",
                lambda *a, **k: (self.probed.append("cli"), (cli_ok, "ok"))[1],
            ),
        ]
        probe = mock.patch(
            "skillscope.engine.models.check_reachable",
            lambda *a, **k: (self.probed.append("provider"), (True, "ok"))[1],
        )
        for patch in [*patches, probe]:
            patch.start()
            self.addCleanup(patch.stop)
        cli._prepare_graded_run(self.args(engine))
        return self.probed

    def test_it_probes_the_cli_as_well_as_the_provider(self) -> None:
        self.assertEqual(sorted(self.run_preflight("claude-code-no-sandbox")), ["cli", "provider"])

    def test_a_sandboxed_engine_probes_only_the_provider(self) -> None:
        # `claude-code` reaches the CLI inside the container through inspect's
        # own bridge, so the host's CLI credential is not what it uses.
        self.assertEqual(self.run_preflight("claude-code"), ["provider"])

    def test_an_unreachable_cli_stops_the_run(self) -> None:
        with self.assertRaises(SystemExit) as caught:
            self.run_preflight("claude-code-no-sandbox", cli_ok=False)
        self.assertIn("claude API not reachable", str(caught.exception))


class TestTheModelProbeIsBounded(unittest.TestCase):
    """The preflight must not become the hang it exists to prevent.

    Pointed at a closed port it was still running after four minutes: inspect's
    `GenerateConfig.max_retries` defaults to `None`, which retries a connection
    error without a limit, and nothing clipped the call to `--timeout`. The
    legacy probe has had both guards from the start, and says why in its own
    docstring.
    """

    def tearDown(self) -> None:
        deadline.use(None)

    def test_retries_are_off(self) -> None:
        self.assertEqual(engine_models.PROBE_RETRIES, 0)

    def test_the_probe_asks_for_that_and_a_per_attempt_timeout(self) -> None:
        source = inspect.getsource(engine_models._probe)
        self.assertIn("max_retries=PROBE_RETRIES", source)
        self.assertIn("timeout=", source)
        # Belt and braces: neither of the above covers a connect that stalls
        # before the provider's own clock starts.
        self.assertIn("fail_after", source)

    def test_the_probe_config_is_not_memoized_into_the_graded_run(self) -> None:
        # Retries off is right for a one-shot check and wrong for the run it
        # precedes, and inspect memoizes models by default.
        self.assertIn("memoize=False", inspect.getsource(engine_models._probe))

    def test_an_unbounded_command_gets_the_default(self) -> None:
        deadline.use(None)
        self.assertEqual(
            engine_models.probe_bound(), (engine_models.PROBE_TIMEOUT_S, "")
        )

    def test_a_tighter_command_timeout_wins(self) -> None:
        deadline.use(deadline.Deadline(5.0, command="behavioral"))
        seconds, _ = engine_models.probe_bound()
        self.assertLessEqual(seconds, 5.0)

    def test_a_looser_command_timeout_does_not_extend_it(self) -> None:
        deadline.use(deadline.Deadline(10_000.0, command="behavioral"))
        seconds, _ = engine_models.probe_bound()
        self.assertEqual(seconds, engine_models.PROBE_TIMEOUT_S)

    def test_an_expired_command_probes_nothing_at_all(self) -> None:
        deadline.use(deadline.Deadline(0.0, command="behavioral"))
        seconds, why = engine_models.probe_bound()
        self.assertIsNone(seconds)
        self.assertIn("--timeout", why)

    def test_an_expired_command_is_reported_rather_than_dialled(self) -> None:
        deadline.use(deadline.Deadline(0.0, command="behavioral"))
        ok, detail = engine_models.check_reachable("anthropic/claude-x")
        self.assertFalse(ok)
        self.assertIn("--timeout", detail)

    def test_mockllm_still_costs_nothing(self) -> None:
        self.assertEqual(
            engine_models.check_reachable("mockllm/model"),
            (True, "mockllm (no provider)"),
        )


class TestTheProbeSaysWhatWentWrong(unittest.TestCase):
    """A preflight whose message is unreadable has not done its job.

    Turning retries off makes tenacity raise `RetryError`, whose string form is
    `RetryError[<Future at 0x7f...>]` -- it names neither the host nor the
    failure. The point of probing early is a message on the first line.
    """

    class Future:
        def __init__(self, error: BaseException | None) -> None:
            self.error = error

        def exception(self) -> BaseException | None:
            return self.error

    def wrapper(self, error: BaseException | None) -> Exception:
        wrapped = RuntimeError("RetryError[<Future at 0x0>]")
        wrapped.last_attempt = self.Future(error)
        return wrapped

    def test_the_wrapped_error_is_what_gets_reported(self) -> None:
        cause = ConnectionError("Connection error.")
        self.assertIs(engine_models._underlying(self.wrapper(cause)), cause)

    def test_a_plain_error_is_left_alone(self) -> None:
        plain = ValueError("401 unauthorized")
        self.assertIs(engine_models._underlying(plain), plain)

    def test_an_empty_wrapper_falls_back_to_itself(self) -> None:
        empty = self.wrapper(None)
        self.assertIs(engine_models._underlying(empty), empty)

    def test_unwrapping_never_raises_on_its_own(self) -> None:
        # This runs on the failure path. An exception here would replace a
        # useful message with a traceback from the reporting code.
        class Exploding:
            def exception(self):
                raise RuntimeError("boom")

        hostile = RuntimeError("wrapped")
        hostile.last_attempt = Exploding()
        self.assertIs(engine_models._underlying(hostile), hostile)


class TestTheShippedSandboxExample(unittest.TestCase):
    """The worked example has to keep working.

    `sandbox:` was documented with no example of what the file it names looks
    like, which left the one thing a reader actually needs -- a device bound in
    and egress granted -- as an exercise. An example that drifts from the
    schema, or from what inspect requires of a compose file, is worse than
    none, so it is checked here rather than trusted.
    """

    EXAMPLE = REPO_ROOT / "examples" / "skill-with-a-device" / "evals"

    def setUp(self) -> None:
        import yaml

        self.machine = yaml.safe_load(
            (self.EXAMPLE / "machine.yml").read_text(encoding="utf-8")
        )
        self.compose = yaml.safe_load(
            (self.EXAMPLE / "compose.yaml").read_text(encoding="utf-8")
        )

    def test_the_machine_file_uses_only_keys_the_parser_knows(self) -> None:
        self.assertEqual(set(self.machine) - datasets.MACHINE_KEYS, set())

    def test_it_names_the_compose_file_that_sits_beside_it(self) -> None:
        # Resolved beside machine.yml, not at the skill root. An example that
        # got this wrong would teach the one mistake the layout invites.
        self.assertTrue((self.EXAMPLE / self.machine["sandbox"]).is_file())

    def test_the_compose_file_offers_a_service_inspect_will_use(self) -> None:
        services = self.compose["services"]
        default = "default" in services or any(
            spec.get("x-default") for spec in services.values()
        )
        self.assertTrue(default, f"no default service among {sorted(services)}")

    def test_the_container_is_told_to_stay_up(self) -> None:
        # inspect execs into a container that is already running. One that
        # exits on start fails every case on a sandbox that is not there.
        self.assertIn("command", self.compose["services"]["default"])

    def test_it_demonstrates_the_two_things_the_default_withholds(self) -> None:
        default = self.compose["services"]["default"]
        self.assertIn("devices", default)
        # Egress is granted by *not* setting this, which is worth asserting:
        # an example that carried it would grant nothing and say it did.
        self.assertNotIn("network_mode", default)


if __name__ == "__main__":
    unittest.main(verbosity=2)
