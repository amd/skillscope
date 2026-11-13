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

# The real CLI prints UTF-8. A Python child on Windows defaults to the legacy code
# page, which turns the recording's non-ASCII text into bytes no UTF-8 reader accepts.
sys.stdout.reconfigure(encoding="utf-8")

if "--version" in sys.argv:
    print("GitHub Copilot CLI 9.9.9.")
    sys.exit(0)

if sys.argv[1:3] == ["skill", "list"]:
    # Asked once per case for the room check. Recorded apart from case starts.
    name = "room-%020d-%d.json" % (time.time_ns(), os.getpid())
    with open(os.path.join(os.environ["STUB_RECORD"], name), "w", encoding="utf-8") as handle:
        handle.write(json.dumps({"cwd": os.getcwd(), "home": os.environ.get("COPILOT_HOME")}))
    if os.environ.get("STUB_SKILLS_FAIL"):
        sys.exit(1)
    print(os.environ.get("STUB_SKILLS") or json.dumps([
        {"name": "rocm-doctor", "source": "project"},
        {"name": "customize-cloud-agent", "source": "builtin"},
    ]))
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

if prompt.startswith("You are grading whether a coding agent"):
    # The judge: one question, no tools. Answer as told.
    if os.environ.get("STUB_JUDGE_FAIL"):
        sys.stderr.write(os.environ["STUB_JUDGE_FAIL"])
        sys.exit(1)
    answer = os.environ.get("STUB_VERDICT") or json.dumps({"pass": True, "reason": "ok"})
    print(json.dumps({"type": "assistant.message", "data": {"content": answer}}), flush=True)
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
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            code = cli.cmd_routing(args)
        self.output = out.getvalue()
        return code

    def starts(self) -> list[dict]:
        """How each CLI process was started, in the order they started."""
        return [
            json.loads(path.read_text(encoding="utf-8"))
            for path in sorted(self.record.glob("*.json"))
            if not path.name.startswith("room-")
        ]

    def rooms(self) -> list[dict]:
        """How each `skill list` the engine asked for was run."""
        return [
            json.loads(path.read_text(encoding="utf-8"))
            for path in sorted(self.record.glob("room-*.json"))
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

    def test_the_console_names_the_harness_that_ran(self) -> None:
        # A run on Copilot must not print the name of an engine that means Claude
        # Code, in the summary line or in the line that starts the cases.
        self.run_routing("--harness", "copilot", "--skip-preflight")
        self.assertIn("harness=copilot, engine=host", self.output)
        self.assertIn("[copilot] routing:", self.output)
        # The progress lines, not the whole output: the closing note about the room
        # check still names the engine that can perform it.
        progress = [line for line in self.output.splitlines() if line.startswith("[")]
        self.assertTrue(progress)
        self.assertFalse(
            [line for line in progress if "claude-code" in line],
            "a progress line names an engine that means Claude Code",
        )

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


class TestTheAdviceForAContaminatedRoom(unittest.TestCase):
    """The report tells the reader what would clean the room, for the agent that ran."""

    def advice(self, name: str) -> str:
        from skillscope import routing

        return routing._isolate_advice({"meta": {"harness": name}})

    def test_claude_code_names_its_variable(self) -> None:
        self.assertIn("ANTHROPIC_API_KEY", self.advice("claude-code"))

    def test_another_harness_names_every_variable_it_can_authenticate_from(self) -> None:
        advice = self.advice("copilot")
        for variable in harness.get("copilot").credentials:
            self.assertIn(variable, advice)
        self.assertNotIn("ANTHROPIC", advice)

    def test_setting_what_the_advice_names_really_isolates_the_room(self) -> None:
        # The advice is only true if each variable it names clears the condition
        # that printed it. Read back out of the sentence the reader gets, so a
        # name in the text that is not one the harness reads would fail here.
        import re

        from skillscope import harness as harnesses

        subject = harnesses.get("copilot")
        named = re.findall(r"`([A-Z][A-Z0-9_]+)`", self.advice("copilot"))
        self.assertEqual(sorted(named), sorted(subject.credentials))
        self.assertFalse(subject.can_isolate_room({}))
        for variable in named:
            self.assertTrue(subject.can_isolate_room({variable: "a-token"}), variable)

    def test_a_harness_with_no_credential_path_is_not_told_to_set_one(self) -> None:
        advice = self.advice("an-agent-that-is-not-registered")
        self.assertNotIn("Set one of", advice)
        self.assertIn("Remove", advice)


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

    def test_a_sandboxed_run_of_such_a_harness_is_refused_below_the_cli_too(self) -> None:
        # `routing.run` is a public entry point; `cli.resolve_leg` is not the
        # only thing standing between a caller and the Claude Code container.
        from skillscope.engine import routing

        with self.assertRaises(SystemExit) as raised:
            routing.run(
                [], {"a-skill": Path(".")}, "claude-sonnet-5.5", "low",
                engine="claude-code", harness=harness.get("copilot"),
            )
        self.assertIn("no sandboxed routing leg", str(raised.exception))
        self.assertEqual(self.starts(), [])

    def test_a_preflight_that_fails_says_why(self) -> None:
        with mock.patch.object(harness.get("copilot"), "preflight",
                               lambda model, timeout=60.0: (False, "Bad credentials")):
            message = self.refused("--harness", "copilot")
        self.assertIn("copilot not reachable", message)
        self.assertIn("Bad credentials", message)


class TestTheRoomOnCopilot(_Copilot):
    """Copilot does not announce its skills, so the engine asks for them."""

    def markdown(self) -> str:
        return self.summary.read_text(encoding="utf-8")

    def run_routing(self, *argv: str, **kwargs) -> int:
        self.summary = self.report.parent / "summary.md"
        return super().run_routing("--summary", str(self.summary), *argv, **kwargs)

    def cases(self) -> list[dict]:
        return json.loads(self.report.read_text(encoding="utf-8"))["cases"]

    def test_the_cli_is_asked_once_per_case_in_the_cases_own_isolation(self) -> None:
        self.run_routing("--harness", "copilot", "--skip-preflight")
        rooms = self.rooms()
        self.assertEqual(len(rooms), 2)
        self.assertTrue(all(room["home"] for room in rooms))

    def test_the_warnings_do_not_send_a_copilot_reader_to_claude_codes_folders(self) -> None:
        listed = json.dumps([{"name": "a-stray-skill", "source": "personal"}])
        with mock.patch.dict(os.environ, {"STUB_SKILLS": listed}):
            self.run_routing("--harness", "copilot", "--skip-preflight")
        warning = self.markdown()
        self.assertIn("a-stray-skill", warning)
        self.assertNotIn(".claude/skills", warning)
        # The CLI answered, and listed only the stray skill. That reads the same as a
        # CLI that could not answer, so the note must not claim the second.
        self.assertIn("listed none of the staged skills", warning)
        # The extras warning above can still fire, so the note may only speak for the
        # warning that cannot.
        self.assertIn("missing-skills warning above cannot fire", warning)
        self.assertNotIn("the two warnings above cannot fire", warning)

        # The other warning needs a room with two skills and one of them missing,
        # which this fixture does not have, so render the report it wrote with that
        # one field set. Claude Code's report keeps its own wording.
        from skillscope import routing

        summary = json.loads(self.report.read_text(encoding="utf-8"))
        summary["skills_missing_from_session"] = ["a-staged-skill"]
        rendered = routing.render_markdown(summary)
        self.assertIn("did not list these installed skills for the session", rendered)
        self.assertNotIn("session init", rendered)
        summary["meta"]["harness"] = "claude-code"
        claude = routing.render_markdown(summary)
        self.assertIn("did not report these installed skills at session init", claude)
        self.assertIn("usually `~/.claude/skills`", claude)

    def test_the_check_is_counted_against_the_subprocess_limit(self) -> None:
        # It starts a real process beside the case's own, so it must be inside the
        # limiter that bounds those, not a second unaccounted pool next to it.
        from skillscope.engine import no_sandbox

        inside: list[bool] = []
        held: list[int] = []

        @contextlib.asynccontextmanager
        async def limiter():
            held.append(1)
            try:
                yield
            finally:
                held.pop()

        subject = harness.get("copilot")
        real = subject.room

        def watching(workspace, env):
            inside.append(bool(held))
            return real(workspace, env)

        with mock.patch.object(no_sandbox, "_subprocess_concurrency", limiter), \
                mock.patch.object(subject, "room", watching):
            self.run_routing("--harness", "copilot", "--skip-preflight")
        self.assertEqual(inside, [True, True])

    def test_a_name_the_cli_spells_in_another_case_is_still_the_staged_skill(self) -> None:
        listed = json.dumps([{"name": SKILL.upper(), "source": "project"}])
        with mock.patch.dict(os.environ, {"STUB_SKILLS": listed}):
            self.run_routing("--harness", "copilot", "--skip-preflight")
        for case in self.cases():
            self.assertEqual(case["visible_skills"], [SKILL])
            self.assertEqual(case["extra_skills"], [])

    def test_the_staged_skill_is_seen_and_the_clis_own_are_not_extras(self) -> None:
        self.run_routing("--harness", "copilot", "--skip-preflight")
        for case in self.cases():
            self.assertEqual(case["visible_skills"], [SKILL])
            self.assertEqual(case["extra_skills"], [])
        self.assertNotIn("room was not verified", self.markdown())

    def test_a_skill_nobody_staged_is_named_as_an_extra(self) -> None:
        listed = json.dumps([
            {"name": SKILL, "source": "project"},
            {"name": "a-stray-skill", "source": "personal"},
        ])
        with mock.patch.dict(os.environ, {"STUB_SKILLS": listed}):
            self.run_routing("--harness", "copilot", "--skip-preflight")
        self.assertEqual(self.cases()[0]["extra_skills"], ["a-stray-skill"])
        warning = self.markdown()
        self.assertIn("a-stray-skill", warning)
        # The remedy must not tell a Copilot user to set a Claude credential.
        self.assertNotIn("ANTHROPIC_API_KEY", warning)
        # It names what would clear the warning. Whether those variables really do
        # is `test_those_variables_are_the_ones_that_isolate_the_room`.
        for variable in harness.get("copilot").credentials:
            self.assertIn(variable, warning)

    def test_a_cli_that_cannot_say_leaves_the_room_unverified_and_says_so(self) -> None:
        with mock.patch.dict(os.environ, {"STUB_SKILLS_FAIL": "1"}):
            self.run_routing("--harness", "copilot", "--skip-preflight")
        note = self.markdown()
        self.assertIn("room was not verified", note)
        self.assertIn("harness could not say", note)
        self.assertNotIn("claude-code-no-sandbox", note)


# One behavioral case that the recording satisfies: it ran `rocm --version`, and its
# closing answer says it ran nothing. Judged expectations are answered by the stub.
def behavioral_dataset() -> dict:
    dataset = tier0_dataset(SKILL)
    first = dataset["evaluations"][0]
    first["logs_contain"] = ["rocm --version"]
    first["expected_behavior"] = ["The agent checks whether the rocm CLI is present"]
    first["unexpected_behavior"] = ["The agent changes anything on the machine"]
    return dataset


class TestABehavioralRunOnCopilot(unittest.TestCase):
    """Its agent and its judge are both the harness's CLI: no model credential."""

    def setUp(self) -> None:
        self.repo = Repo(self)
        self.repo.skill(SKILL, dataset=behavioral_dataset())
        self.repo.activate()
        self.repo.chdir()
        stub_cli(self, "copilot", STUB)
        scratch = Path(tempfile.mkdtemp(prefix="ss-behavioral-"))
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
            "GITHUB_TOKEN", "CI", "GITHUB_ACTIONS", "SKILLSCOPE_HARNESS", "SKILLSCOPE_ENGINE",
        ):
            os.environ.pop(variable, None)
        usage.reset()
        self.addCleanup(usage.reset)

    def run_behavioral(self, *argv: str) -> int:
        args = cli.build_parser().parse_args(
            ["behavioral", "--harness", "copilot", "--output", str(self.report),
             "--skip-preflight", *argv]
        )
        with contextlib.redirect_stdout(io.StringIO()):
            return cli.cmd_behavioral(args)

    def starts(self) -> list[dict]:
        return [
            json.loads(path.read_text(encoding="utf-8"))
            for path in sorted(self.record.glob("*.json"))
            if not path.name.startswith("room-")
        ]

    def checks(self) -> dict[str, dict]:
        report = json.loads(self.report.read_text(encoding="utf-8"))
        [case] = report["cases"]
        return {check["kind"] + ":" + check["expectation"]: check for check in case["checks"]}

    def test_it_runs_and_passes_with_no_anthropic_credential(self) -> None:
        self.assertEqual(self.run_behavioral(), 0)
        totals = json.loads(self.report.read_text(encoding="utf-8"))["totals"]
        self.assertEqual((totals["cases"], totals["passed"], totals["errors"]), (1, 1, 0))

    def test_each_judge_question_is_counted_against_the_subprocess_limit(self) -> None:
        # The same limiter the agent's own process is under. A regression that took the
        # judge out of it would not fail anything else here: the run would still pass.
        from skillscope.engine import no_sandbox

        held: list[int] = []
        inside: list[bool] = []

        @contextlib.asynccontextmanager
        async def limiter():
            held.append(1)
            try:
                yield
            finally:
                held.pop()

        subject = harness.get("copilot")
        real = subject.complete

        def watching(prompt, model, *args, **kwargs):
            inside.append(bool(held))
            return real(prompt, model, *args, **kwargs)

        with mock.patch.object(no_sandbox, "_subprocess_concurrency", limiter),                 mock.patch.object(subject, "complete", watching):
            self.assertEqual(self.run_behavioral(), 0)
        self.assertTrue(inside, "no judge question was asked")
        self.assertTrue(all(inside), inside)

    def test_the_room_is_not_asked_for_on_this_command(self) -> None:
        # `routing` asks the CLI what a session would load; `behavioral` does not, and
        # says so. The stub records every `skill list`, so none means none was asked.
        self.assertEqual(self.run_behavioral(), 0)
        self.assertEqual(list(self.record.glob("room-*.json")), [])

    def test_the_run_is_held_to_the_limit_a_real_model_gets(self) -> None:
        # inspect is handed a model that reaches no provider, whose own limit is a
        # test value a single real run exceeds and loses its transcript to.
        from skillscope.engine import behavioral

        real = behavioral.build_task
        limits: list[int] = []
        calls: list[tuple] = []

        def spy(*args, **kwargs):
            task = real(*args, **kwargs)
            limits.append(task.message_limit)
            calls.append((args, kwargs))
            return task

        with mock.patch.object(behavioral, "build_task", spy):
            self.assertEqual(self.run_behavioral(), 0)
        self.assertEqual(limits, [behavioral.MESSAGE_LIMIT])
        self.assertNotEqual(limits[0], behavioral.MOCK_MESSAGE_LIMIT)

        # An override is whatever was given, including a limit that is falsy.
        args, kwargs = calls[0]
        self.assertEqual(real(*args, **{**kwargs, "message_limit": 0}).message_limit, 0)

    def test_every_kind_of_expectation_was_graded(self) -> None:
        self.run_behavioral()
        checks = self.checks()
        self.assertEqual(len(checks), 3)
        self.assertTrue(all(check["passed"] for check in checks.values()), checks)
        judged = [c for c in checks.values() if c["kind"] != "logs_contain"]
        self.assertEqual({c["detail"] for c in judged}, {"judge: ok"})

    def test_the_judge_is_the_cli_asked_with_nothing_allowed(self) -> None:
        self.run_behavioral()
        starts = self.starts()
        judging = [s for s in starts if s["prompt"].startswith("You are grading")]
        running = [s for s in starts if not s["prompt"].startswith("You are grading")]
        self.assertEqual(len(judging), 2)
        self.assertEqual(len(running), 1)
        for start in judging:
            self.assertNotIn("--allow-all", start["argv"])
            self.assertTrue([a for a in start["argv"] if a.startswith("--available-tools=")])
            self.assertEqual(start["argv"][start["argv"].index("--model") + 1], "claude-sonnet-5.5")
        self.assertIn("--allow-all", running[0]["argv"])

    def test_the_judge_is_shown_what_the_agent_did(self) -> None:
        self.run_behavioral()
        question = next(s["prompt"] for s in self.starts() if s["prompt"].startswith("You are grading"))
        self.assertIn("REQUIREMENT", question)
        # The denied shell call in the recording is evidence of what it tried.
        self.assertIn("rocm --version", question)

    def test_a_negative_verdict_fails_that_expectation_and_the_case(self) -> None:
        verdict = json.dumps({"pass": False, "reason": "it did not"})
        with mock.patch.dict(os.environ, {"STUB_VERDICT": verdict}):
            code = self.run_behavioral()
        self.assertEqual(code, 1)
        self.assertIn("judge: it did not", {c["detail"] for c in self.checks().values()})

    def test_a_judge_that_fails_is_reported_against_the_expectation(self) -> None:
        with mock.patch.dict(os.environ, {"STUB_JUDGE_FAIL": "Error: Bad credentials"}):
            code = self.run_behavioral()
        self.assertEqual(code, 1)
        details = [c["detail"] for c in self.checks().values()]
        self.assertTrue(any(d.startswith("judge failed:") and "Bad credentials" in d for d in details))
        # The deterministic expectation is still graded: one judge failing does
        # not end the case.
        self.assertTrue(self.checks()["logs_contain:rocm --version"]["passed"])

    def test_the_report_names_the_harness(self) -> None:
        self.run_behavioral()
        meta = json.loads(self.report.read_text(encoding="utf-8"))["meta"]
        self.assertEqual((meta["harness"], meta["engine"]), ("copilot", "host"))


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
