# Copyright Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: MIT

"""`--harness` and `--engine`, from the command line down to the subprocess.

The central test runs a real routing command against a stub `copilot` that
replays the recorded session, with no Anthropic credential anywhere in the
environment. It is the check that a run on another harness really is that
harness end to end: its command line, its model name, its isolated home, the
skills staged where it looks, and a report that names it.
"""

from __future__ import annotations

import contextlib
import io
import json
import os
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from skillscope import cli, harness, usage
from skillscope.engine import sandbox as engine_sandbox
from tests.test_harness_hooks import stub_cli
from tests.test_skillscope import Repo, tier0_dataset

RECORDING = Path(__file__).parent / "fixtures" / "harness" / "copilot" / "trigger.jsonl"

# A `copilot` that records how it was started and replays the recorded session.
# Nothing here uses a backslash or a quote that would need escaping twice.
STUB = """
import json, os, sys, time

if "--version" in sys.argv:
    print("GitHub Copilot CLI 9.9.9.")
    sys.exit(0)

prompt = sys.stdin.read()
skills = os.path.join(os.getcwd(), ".claude", "skills")
# One file per process, named so they sort in the order they started. Cases run
# concurrently, and appends from two processes to one file are not atomic on
# Windows: a record can be overwritten or cut in half.
name = "%020d-%d.json" % (time.time_ns(), os.getpid())
with open(os.path.join(os.environ["STUB_RECORD"], name), "w", encoding="utf-8") as handle:
    handle.write(json.dumps({
        "argv": sys.argv[1:],
        "prompt": prompt,
        "home": os.environ.get("COPILOT_HOME"),
        "staged": sorted(os.listdir(skills)) if os.path.isdir(skills) else None,
        "token_present": bool(os.environ.get("GH_TOKEN")),
    }))

if prompt == "Reply with the single word: ok":
    print(json.dumps({"type": "result", "exitCode": 0}), flush=True)
    sys.exit(0)

with open(os.environ["STUB_REPLAY"], encoding="utf-8") as recording:
    for line in recording:
        print(line.rstrip(chr(10)), flush=True)
"""

SKILL = "rocm-doctor"  # the skill the recording activates


class _Copilot(unittest.TestCase):
    """A repo with one skill, a stub `copilot` on PATH, and no Anthropic key."""

    def setUp(self) -> None:
        self.repo = Repo(self)
        self.repo.skill(SKILL, dataset=tier0_dataset(SKILL))
        self.repo.activate(routing_room=SKILL)
        self.repo.chdir()

        stub_cli(self, "copilot", STUB)
        scratch = Path(tempfile.mkdtemp(prefix="ss-cli-"))
        self.addCleanup(shutil.rmtree, scratch, True)
        self.record = scratch / "starts"
        self.record.mkdir()
        self.report = scratch / "report.json"

        env = mock.patch.dict(
            os.environ,
            {
                engine_sandbox.SANDBOX_ENV: "local",
                "GH_TOKEN": "a-token-for-the-test",
                "STUB_RECORD": str(self.record),
                "STUB_REPLAY": str(RECORDING),
            },
        )
        env.start()
        self.addCleanup(env.stop)
        for variable in (
            "ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "COPILOT_GITHUB_TOKEN",
            "GITHUB_TOKEN", "CI", "GITHUB_ACTIONS", "SKILLSCOPE_HARNESS",
            "SKILLSCOPE_ENGINE",
        ):
            os.environ.pop(variable, None)
        usage.reset()
        self.addCleanup(usage.reset)

    def parse(self, *argv: str):
        return cli.build_parser().parse_args(["routing", *argv])

    def run_routing(self, *argv: str, whole_room: bool = False) -> int:
        # One case that should activate the skill and one that should not: enough
        # to exercise everything but the totals, at a fraction of the time.
        only = () if whole_room else ("--only", f"{SKILL}-yes-0,{SKILL}-no-0")
        args = self.parse(
            "--output", str(self.report), "--min-accuracy", "0", *only, *argv
        )
        with contextlib.redirect_stdout(io.StringIO()):
            return cli.cmd_routing(args)

    def starts(self) -> list[dict]:
        """How each CLI process was started, in the order they started."""
        return [
            json.loads(path.read_text(encoding="utf-8"))
            for path in sorted(self.record.glob("*.json"))
        ]

    def meta(self) -> dict:
        return json.loads(self.report.read_text(encoding="utf-8"))["meta"]

    def totals(self) -> dict:
        return json.loads(self.report.read_text(encoding="utf-8"))["totals"]


class TestARoutingRunOnCopilot(_Copilot):
    def test_it_runs_with_no_anthropic_credential(self) -> None:
        self.assertEqual(self.run_routing("--harness", "copilot", whole_room=True), 0)
        # The five cases of the smallest dataset that clears the coverage bar,
        # and the five negatives skillscope adds to every room. The recording
        # activates the skill every time, so the three positives are right and
        # the seven negatives are false triggers.
        self.assertEqual((self.totals()["graded"], self.totals()["passed"]), (10, 3))

    def test_the_report_names_the_harness_and_where_it_ran(self) -> None:
        self.run_routing("--harness", "copilot")
        meta = self.meta()
        self.assertEqual(meta["harness"], "copilot")
        # The older engine names say Claude Code, which did not run.
        self.assertEqual(meta["engine"], "host")
        self.assertEqual(meta["harness_version"], "GitHub Copilot CLI 9.9.9.")
        self.assertTrue(meta["isolated_config_dir"])

    def test_the_cli_is_started_the_way_the_harness_says(self) -> None:
        self.run_routing("--harness", "copilot")
        cases = [s for s in self.starts() if s["prompt"] != "Reply with the single word: ok"]
        self.assertEqual(len(cases), 2)
        for start in cases:
            argv = start["argv"]
            self.assertEqual(argv[argv.index("--model") + 1], "claude-sonnet-5.5")
            self.assertEqual(argv[argv.index("--reasoning-effort") + 1], "high")
            self.assertEqual(start["staged"], [SKILL])
            self.assertTrue(start["home"], "no isolated home was given to the CLI")
            self.assertTrue(start["token_present"])
        self.assertEqual(
            {s["prompt"] for s in cases},
            {f"please do {SKILL} work 0", f"nothing to do with {SKILL} 0"},
        )

    def test_the_preflight_proves_the_cli_can_answer_first(self) -> None:
        self.run_routing("--harness", "copilot")
        first = self.starts()[0]
        self.assertEqual(first["prompt"], "Reply with the single word: ok")
        self.assertEqual(first["argv"][first["argv"].index("--model") + 1], "claude-sonnet-5.5")

    def test_skipping_the_preflight_skips_only_the_preflight(self) -> None:
        self.run_routing("--harness", "copilot", "--skip-preflight")
        self.assertNotIn("Reply with the single word: ok", [s["prompt"] for s in self.starts()])

    def test_an_alias_is_understood_in_its_own_names(self) -> None:
        self.run_routing("--harness", "copilot", "--model", "haiku", "--skip-preflight")
        argv = self.starts()[0]["argv"]
        self.assertEqual(argv[argv.index("--model") + 1], "claude-haiku-5.5")
        self.assertEqual(self.meta()["model"], "claude-haiku-5.5")

    def test_ci_holds_the_run_to_the_pinned_model(self) -> None:
        with mock.patch.dict(os.environ, {"CI": "true"}):
            self.run_routing("--harness", "copilot", "--model", "opus", "--skip-preflight")
        argv = self.starts()[0]["argv"]
        self.assertEqual(argv[argv.index("--model") + 1], "claude-sonnet-5.5")
        self.assertEqual(self.meta()["model"], "claude-sonnet-5.5")

    def test_the_harness_can_come_from_the_environment(self) -> None:
        with mock.patch.dict(os.environ, {"SKILLSCOPE_HARNESS": "copilot"}):
            self.run_routing("--skip-preflight")
        self.assertEqual(self.meta()["harness"], "copilot")

    def test_a_dollar_cap_that_cannot_bind_is_reported_as_such(self) -> None:
        self.run_routing("--harness", "copilot", "--max-budget-usd", "2", "--skip-preflight")
        meta = self.meta()
        self.assertEqual(meta["max_budget_usd"], 2.0)
        self.assertFalse(meta["max_budget_can_bind"])
        self.assertIsNone(meta["max_budget_enforced_by"])

    def test_the_default_cap_is_reported_as_unable_to_bind(self) -> None:
        # A cap is in force unless one is given, so every run on a harness with
        # no dollar control says so rather than implying it stayed under a limit.
        self.run_routing("--harness", "copilot", "--skip-preflight")
        meta = self.meta()
        self.assertEqual(meta["max_budget_usd"], 0.75)
        self.assertFalse(meta["max_budget_can_bind"])

    def test_the_tokens_it_spent_are_counted(self) -> None:
        self.run_routing("--harness", "copilot", "--skip-preflight")
        # The recording stops at the skill call, before usage is reported, so
        # what this proves is that the counters are the harness's own and exist.
        self.assertIn("input_tokens", self.meta())


class TestWhatIsRefusedBeforeAnythingRuns(_Copilot):
    def refused(self, *argv: str) -> str:
        with self.assertRaises(SystemExit) as raised, contextlib.redirect_stdout(io.StringIO()):
            self.run_routing(*argv)
        self.assertEqual(self.starts(), [], "the CLI was started before the refusal")
        return str(raised.exception)

    def test_a_room_that_cannot_be_isolated(self) -> None:
        os.environ.pop("GH_TOKEN")
        message = self.refused("--harness", "copilot", "--skip-preflight")
        self.assertIn("--harness copilot", message)
        for variable in harness.get("copilot").credentials:
            self.assertIn(variable, message)
        # There is no container leg to suggest instead.
        self.assertNotIn("--engine claude-code,", message)

    def test_an_unknown_harness(self) -> None:
        message = self.refused("--harness", "no-such-agent")
        self.assertIn("no-such-agent", message)
        self.assertIn("copilot", message)

    def test_an_older_engine_name_that_contradicts_the_harness(self) -> None:
        message = self.refused("--harness", "copilot", "--engine", "claude-code-no-sandbox")
        self.assertIn("contradicts", message)
        self.assertIn("--engine host", message)

    def test_a_harness_with_no_sandboxed_leg_asked_for_one(self) -> None:
        message = self.refused("--harness", "copilot", "--engine", "sandbox")
        self.assertIn("no sandboxed leg", message)

    def test_behavioral_while_its_judge_is_not_available(self) -> None:
        args = cli.build_parser().parse_args(["behavioral", "--harness", "copilot"])
        with self.assertRaises(SystemExit) as raised:
            cli._prepare_graded_run(args)
        self.assertIn("cannot run behavioral yet", str(raised.exception))
        self.assertEqual(self.starts(), [])

    def test_a_preflight_that_fails_says_why(self) -> None:
        with mock.patch.object(harness.get("copilot"), "preflight",
                               lambda model, timeout=60.0: (False, "Bad credentials")):
            message = self.refused("--harness", "copilot")
        self.assertIn("copilot not reachable", message)
        self.assertIn("Bad credentials", message)


class TestResolvingTheLeg(unittest.TestCase):
    def setUp(self) -> None:
        env = mock.patch.dict(os.environ, clear=False)
        env.start()
        self.addCleanup(env.stop)
        for variable in ("SKILLSCOPE_HARNESS", "SKILLSCOPE_ENGINE"):
            os.environ.pop(variable, None)

    def leg(self, engine, harness_name):
        leg, subject = cli.resolve_leg(engine, harness_name)
        return leg, subject.name

    def test_nothing_said_is_claude_code_on_the_host(self) -> None:
        self.assertEqual(self.leg(None, None), ("claude-code-no-sandbox", "claude-code"))

    def test_the_neutral_engine_names(self) -> None:
        self.assertEqual(self.leg("host", None), ("claude-code-no-sandbox", "claude-code"))
        self.assertEqual(self.leg("sandbox", None), ("claude-code", "claude-code"))

    def test_the_older_engine_names_keep_meaning_claude_code(self) -> None:
        self.assertEqual(self.leg("claude-code", None), ("claude-code", "claude-code"))
        self.assertEqual(
            self.leg("claude-code-no-sandbox", "claude-code"),
            ("claude-code-no-sandbox", "claude-code"),
        )

    def test_another_harness_defaults_to_the_host(self) -> None:
        self.assertEqual(self.leg(None, "copilot"), ("claude-code-no-sandbox", "copilot"))
        self.assertEqual(self.leg("host", "copilot"), ("claude-code-no-sandbox", "copilot"))

    def test_the_environment_supplies_what_the_command_line_did_not(self) -> None:
        os.environ["SKILLSCOPE_HARNESS"] = "copilot"
        self.assertEqual(self.leg(None, None), ("claude-code-no-sandbox", "copilot"))
        os.environ["SKILLSCOPE_ENGINE"] = "claude-code"
        # An environment that contradicts itself is refused like a command line.
        with self.assertRaises(SystemExit):
            cli.resolve_leg(None, None)

    def test_an_unknown_engine_is_refused_by_name(self) -> None:
        with self.assertRaises(SystemExit) as raised:
            cli.resolve_leg("no-such-engine", None)
        self.assertIn("host", str(raised.exception))

    def test_claude_code_is_reported_in_the_names_it_always_was(self) -> None:
        args = cli.build_parser().parse_args(["routing", "--engine", "host"])
        args.engine, args.harness = cli.resolve_leg(args.engine, args.harness)[0], "claude-code"
        with mock.patch("shutil.which", lambda _: None):
            self.assertEqual(
                cli._leg_meta(args),
                {"engine": "claude-code-no-sandbox", "harness": "claude-code",
                 "harness_version": None},
            )

    def test_the_model_default_is_the_harnesses_own(self) -> None:
        for name, expected in (("claude-code", "opus"), ("copilot", "claude-sonnet-5.5")):
            with self.subTest(harness=name):
                self.assertEqual(harness.get(name).resolve_model(None), expected)


if __name__ == "__main__":
    unittest.main()
