# Copyright Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: MIT

"""What the engines ask of a harness beyond translating its events.

The model it defaults to and how it names models, the pin a CI run is held to,
what it can authenticate from, its spend controls, its version, and the
preflight that proves it can reach its model before any case runs. The shared
parts run for every registered harness; what is particular to one is tested for
that one, against a stub executable so the real subprocess code runs.
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from skillscope import deadline, harness
from skillscope.harness import base as base_module
from skillscope.harness import copilot as copilot_module


def stub_cli(test: unittest.TestCase, name: str, script: str) -> None:
    """Put an executable called `name` that runs `script` first on PATH.

    A script rather than a mock, so `shutil.which`, the subprocess and its
    environment all run. Windows resolves a bare name through PATHEXT, so there
    the shim is a `.cmd`; elsewhere it is a `/bin/sh` that starts the interpreter,
    which survives an interpreter path with a space in it. It does not `exec` it:
    an npm-installed CLI is a shim that starts the real program as a child, and
    that child is what a timeout has to kill along with the shim.
    """
    bin_dir = Path(tempfile.mkdtemp(prefix=f"ss-stub-{name}-"))
    test.addCleanup(shutil.rmtree, bin_dir, True)
    program = bin_dir / f"{name}_stub.py"
    program.write_text(script, encoding="utf-8")
    if os.name == "nt":
        (bin_dir / f"{name}.cmd").write_text(
            f'@"{sys.executable}" "{program}" %*\r\n', encoding="utf-8"
        )
    else:
        shim = bin_dir / name
        shim.write_text(
            f'#!/bin/sh\n"{sys.executable}" "{program}" "$@"\n', encoding="utf-8"
        )
        shim.chmod(0o755)
    patch = mock.patch.dict(
        os.environ, {"PATH": f"{bin_dir}{os.pathsep}{os.environ.get('PATH', '')}"}
    )
    patch.start()
    test.addCleanup(patch.stop)


# A `copilot` that records how it was started, then answers or fails as told.
STUB_COPILOT = """
import json, os, sys

# The real CLI prints UTF-8. A Python child on Windows defaults to the legacy code
# page, which turns the recording's non-ASCII text into bytes no UTF-8 reader accepts.
sys.stdout.reconfigure(encoding="utf-8")

if "--version" in sys.argv:
    print("GitHub Copilot CLI 9.9.9.")
    print("Run 'copilot update' to check for updates.")
    sys.exit(0)

if sys.argv[1:3] == ["skill", "list"]:
    # Said out loud, so a test can tell a CLI that was asked and could not answer
    # from a CLI that was never asked.
    if os.environ.get("STUB_SKILLS_LOG"):
        with open(os.environ["STUB_SKILLS_LOG"], "a", encoding="utf-8") as handle:
            handle.write("asked" + chr(10))
    if os.environ.get("STUB_SKILLS_HANG"):
        import time
        if os.environ.get("STUB_SKILLS_PID"):
            with open(os.environ["STUB_SKILLS_PID"], "w", encoding="utf-8") as handle:
                handle.write(str(os.getpid()))
        time.sleep(30)
    if os.environ.get("STUB_SKILLS_FAIL"):
        sys.exit(1)
    print(os.environ.get("STUB_SKILLS", "[]"))
    sys.exit(int(os.environ.get("STUB_SKILLS_EXIT", "0")))

stdin = sys.stdin.read()
if os.environ.get("STUB_RAW_LINE"):
    print(os.environ["STUB_RAW_LINE"], flush=True)
record = os.environ.get("STUB_RECORD")
if record:
    with open(record, "w", encoding="utf-8") as handle:
        json.dump({
            "argv": sys.argv[1:],
            "stdin": stdin,
            "cwd": os.getcwd(),
            "home": os.environ.get("COPILOT_HOME"),
            "auto_update": os.environ.get("COPILOT_AUTO_UPDATE"),
        }, handle)
if os.environ.get("STUB_FAIL"):
    sys.stderr.write(os.environ["STUB_FAIL"])
    sys.exit(1)
if os.environ.get("STUB_HANG"):
    import time
    time.sleep(60)
if os.environ.get("STUB_ANSWER"):
    message = {"type": "assistant.message", "data": {"content": os.environ["STUB_ANSWER"]}}
    print(json.dumps(message), flush=True)
print(json.dumps({"type": "result", "exitCode": 0}), flush=True)
"""


class TestEveryHarnessHasTheEngineHooks(unittest.TestCase):
    def test_a_model_is_named_when_the_caller_names_none(self) -> None:
        for name in harness.names():
            with self.subTest(harness=name):
                subject = harness.get(name)
                self.assertTrue(subject.default_model)
                self.assertEqual(subject.resolve_model(None), subject.default_model)

    def test_a_model_the_harness_does_not_know_passes_through(self) -> None:
        for name in harness.names():
            with self.subTest(harness=name):
                self.assertEqual(harness.get(name).resolve_model("mockllm/model"), "mockllm/model")

    def test_a_room_can_only_be_isolated_with_a_credential_in_the_environment(self) -> None:
        for name in harness.names():
            with self.subTest(harness=name):
                subject = harness.get(name)
                self.assertTrue(subject.credentials, "no credential variable is named")
                self.assertFalse(subject.can_isolate_room({}))
                self.assertFalse(subject.can_isolate_room({subject.credentials[0]: "  "}))
                for variable in subject.credentials:
                    self.assertTrue(subject.can_isolate_room({variable: "a-token"}), variable)

    def test_nothing_is_pinned_off_ci(self) -> None:
        for name in harness.names():
            with self.subTest(harness=name):
                subject = harness.get(name)
                with mock.patch.dict(os.environ, clear=False):
                    for variable in ("CI", "GITHUB_ACTIONS"):
                        os.environ.pop(variable, None)
                    self.assertEqual(subject.pin_model("some-model"), "some-model")

    def test_a_no_provider_model_is_never_pinned(self) -> None:
        # It reaches no cloud service, so there is no cost or drift to pin.
        for name in harness.names():
            with self.subTest(harness=name), mock.patch.dict(os.environ, {"CI": "true"}):
                self.assertEqual(harness.get(name).pin_model("mockllm/model"), "mockllm/model")


class TestClaudeCodeKeepsItsBehaviour(unittest.TestCase):
    """The hooks delegate to what the engine did inline; pin that they still do."""

    def test_ci_forces_opus_as_it_always_did(self) -> None:
        with mock.patch.dict(os.environ, {"CI": "true"}):
            self.assertEqual(harness.get("claude-code").pin_model("sonnet"), "opus")
            self.assertEqual(harness.get("claude-code").pin_model("opus"), "opus")

    def test_the_default_model_is_the_alias_it_always_was(self) -> None:
        self.assertEqual(harness.get("claude-code").resolve_model(None), "opus")

    def test_the_credential_check_follows_the_environment(self) -> None:
        with mock.patch.dict(os.environ, clear=False):
            for variable in harness.get("claude-code").credentials:
                os.environ.pop(variable, None)
            self.assertFalse(harness.get("claude-code").can_isolate_room())
            os.environ["ANTHROPIC_AUTH_TOKEN"] = "a-token"
            self.assertTrue(harness.get("claude-code").can_isolate_room())


class TestCopilotNamesModelsItsOwnWay(unittest.TestCase):
    def test_the_generic_aliases_become_its_names(self) -> None:
        copilot = harness.get("copilot")
        self.assertEqual(copilot.resolve_model("opus"), "claude-opus-5.5")
        self.assertEqual(copilot.resolve_model("Sonnet"), "claude-sonnet-5.5")
        self.assertEqual(copilot.resolve_model("haiku"), "claude-haiku-5.5")

    def test_an_exact_name_is_left_alone(self) -> None:
        self.assertEqual(harness.get("copilot").resolve_model("gpt-6-astra"), "gpt-6-astra")

    def test_the_default_is_the_cheap_model_not_the_largest(self) -> None:
        # Measured: one premium request per prompt on the default, fifteen on
        # the largest. A caller who names nothing must not get the latter.
        self.assertEqual(harness.get("copilot").resolve_model(None), "claude-sonnet-5.5")

    def test_ci_holds_every_run_to_the_one_model(self) -> None:
        copilot = harness.get("copilot")
        with mock.patch.dict(os.environ, {"CI": "true"}):
            self.assertEqual(copilot.pin_model("claude-opus-5.5"), copilot_module.CI_MODEL)
            self.assertEqual(copilot.pin_model(copilot_module.CI_MODEL), copilot_module.CI_MODEL)

    def test_a_coercion_is_said_out_loud(self) -> None:
        import contextlib
        import io

        out = io.StringIO()
        with mock.patch.dict(os.environ, {"CI": "true"}), contextlib.redirect_stdout(out):
            harness.get("copilot").pin_model("claude-opus-5.5")
        self.assertIn("coercing model 'claude-opus-5.5'", out.getvalue())

    def test_a_dollar_cap_cannot_be_given_to_a_cli_billed_in_requests(self) -> None:
        self.assertEqual(harness.get("copilot").cost_flags(5.0), [])


class TestTheVersionTheReportNames(unittest.TestCase):
    def test_it_is_the_first_line_the_cli_prints(self) -> None:
        stub_cli(self, "copilot", STUB_COPILOT)
        self.assertEqual(harness.get("copilot").version(), "GitHub Copilot CLI 9.9.9.")

    def test_it_is_none_when_the_cli_is_not_installed(self) -> None:
        with mock.patch("shutil.which", lambda _: None):
            self.assertIsNone(harness.get("copilot").version())


class TestAskingTheModelOneQuestion(unittest.TestCase):
    """`complete`: what the judge uses when the harness has no API of its own."""

    def setUp(self) -> None:
        stub_cli(self, "copilot", STUB_COPILOT)
        self.record = Path(tempfile.mkdtemp(prefix="ss-record-")) / "run.json"
        self.addCleanup(shutil.rmtree, self.record.parent, True)
        env = mock.patch.dict(
            os.environ,
            {"STUB_RECORD": str(self.record), "STUB_ANSWER": '{"pass": true, "reason": "ok"}'},
        )
        env.start()
        self.addCleanup(env.stop)

    def ask(self, **kwargs) -> str:
        return harness.get("copilot").complete("Is it so?", "claude-sonnet-5.5", **kwargs)

    def test_the_answer_is_what_the_cli_said(self) -> None:
        self.assertEqual(self.ask(), '{"pass": true, "reason": "ok"}')

    def test_the_question_goes_in_on_stdin(self) -> None:
        self.ask()
        self.assertEqual(json.loads(self.record.read_text(encoding="utf-8"))["stdin"], "Is it so?")

    def test_nothing_is_allowed_and_no_tool_is_offered(self) -> None:
        self.ask()
        argv = json.loads(self.record.read_text(encoding="utf-8"))["argv"]
        # The agent's command grants every permission; the judge's must not.
        self.assertNotIn("--allow-all", argv)
        # An allowlist naming nothing the CLI has, not a list of tools to take away:
        # a denylist depends on spelling the CLI's own tool names, and a name that is
        # wrong removes nothing. Nothing in this repository can check a name against
        # the real CLI, which is why the flag is an allowlist and the tool is not one.
        self.assertEqual(
            [a for a in argv if a.startswith("--available-tools=")],
            [f"--available-tools={copilot_module.JUDGE_TOOLS}"],
        )
        self.assertFalse([a for a in argv if a.startswith("--excluded-tools")])

    def test_the_one_tool_it_names_is_not_a_tool_the_cli_has(self) -> None:
        # The recordings are real CLI runs, so every name they carry is one the CLI
        # has. If the judge's allowlist named one of them, the judge would be offered it.
        recorded = set()
        for path in (Path(__file__).parent / "fixtures" / "harness" / "copilot").glob("*.jsonl"):
            for line in path.read_text(encoding="utf-8").splitlines():
                event = json.loads(line)
                data = event.get("data") if isinstance(event.get("data"), dict) else {}
                if event.get("type") == "tool.execution_start":
                    recorded.add(data.get("toolName"))
        self.assertTrue(recorded, "the recordings name no tool at all")
        self.assertNotIn(copilot_module.JUDGE_TOOLS, recorded)

    def test_the_one_tool_it_names_is_not_one_of_the_tools_seen_in_a_real_run(self) -> None:
        # The recordings only exercise a few. These are what a real session of CLI
        # 1.0.105 reported when asked which tools it could see, so a rename of the
        # judge's name onto any of them is caught whatever the recordings hold.
        seen = {
            "powershell", "read_powershell", "stop_powershell", "list_powershell",
            "view", "create", "edit", "grep", "glob", "web_fetch", "skill", "task",
            "fetch_copilot_cli_documentation", "semantic_code_search", "sql",
            "session_store_sql", "read_agent", "list_agents", "write_agent",
            "run_dynamic_workflow", "dynamic_workflows_manage",
        }
        self.assertNotIn(copilot_module.JUDGE_TOOLS, seen)

    def test_it_runs_in_a_throwaway_home_and_workspace(self) -> None:
        self.ask()
        ran = json.loads(self.record.read_text(encoding="utf-8"))
        self.assertTrue(ran["home"])
        self.assertNotEqual(ran["cwd"], os.getcwd())

    def test_a_line_that_is_not_an_object_does_not_reach_the_translator(self) -> None:
        # A translator written like the default one takes whatever it is given, and any
        # JSON value is a legal line. `7` is one.
        class Unguarded(copilot_module.Copilot):
            def normalizer(self):
                return lambda event: [event]

        with mock.patch.dict(os.environ, {"STUB_RAW_LINE": "7"}):
            with self.assertRaises(harness.CompletionError) as raised:
                Unguarded().complete("Is it so?", "claude-sonnet-5.5")
        self.assertIn("no answer", str(raised.exception))

    def test_a_command_with_no_time_left_is_not_asked_at_all(self) -> None:
        bound = deadline.Deadline(600.0, command="behavioral")
        previous = deadline.use(bound)
        self.addCleanup(deadline.use, previous)
        starting = mock.Mock(side_effect=AssertionError("the CLI was started with no time left"))
        with mock.patch.object(bound, "remaining", lambda: 0.0), \
                mock.patch.object(base_module.subprocess, "Popen", starting):
            with self.assertRaises(harness.CompletionError):
                self.ask()
        starting.assert_not_called()

    def test_a_refusal_carries_what_the_cli_said(self) -> None:
        with mock.patch.dict(os.environ, {"STUB_FAIL": "Error: Bad credentials"}):
            with self.assertRaises(harness.CompletionError) as raised:
                self.ask()
        self.assertIn("Bad credentials", str(raised.exception))

    def test_a_cli_that_prints_no_answer_is_an_error_not_an_empty_one(self) -> None:
        with mock.patch.dict(os.environ, {"STUB_ANSWER": ""}):
            with self.assertRaises(harness.CompletionError) as raised:
                self.ask()
        self.assertIn("no answer", str(raised.exception))

    def test_a_cli_that_never_answers_is_stopped(self) -> None:
        import time

        started = time.monotonic()
        with mock.patch.dict(os.environ, {"STUB_HANG": "1"}):
            with self.assertRaises(harness.CompletionError) as raised:
                self.ask(timeout=1)
        self.assertIn("no answer within", str(raised.exception))
        self.assertLess(time.monotonic() - started, 20)

    def test_a_missing_cli_is_named(self) -> None:
        with mock.patch("shutil.which", lambda _: None), self.assertRaises(harness.CompletionError):
            self.ask()

    def test_claude_code_cannot_be_asked_this_way(self) -> None:
        # Its judge is its inspect model. Building the command with permissions
        # skipped would run a question with the run of the machine.
        claude = harness.get("claude-code")
        spec = harness.RunSpec(workspace="ws", prompt="q", tool_free=True)
        with self.assertRaises(NotImplementedError):
            claude.command(spec)


class TestWhichSkillsASessionWouldLoad(unittest.TestCase):
    def setUp(self) -> None:
        stub_cli(self, "copilot", STUB_COPILOT)

    def room(self) -> list[str] | None:
        return harness.get("copilot").room(tempfile.gettempdir(), dict(os.environ))

    def test_what_the_cli_lists_minus_its_own_skills(self) -> None:
        listed = json.dumps([
            {"name": "rocm-doctor", "source": "project"},
            {"name": "a-personal-skill", "source": "personal"},
            {"name": "customize-cloud-agent", "source": "builtin"},
        ])
        with mock.patch.dict(os.environ, {"STUB_SKILLS": listed}):
            self.assertEqual(self.room(), ["rocm-doctor", "a-personal-skill"])

    def asked(self, **env: str) -> tuple[list[str] | None, int]:
        """What `room()` answered, and how many times the CLI was really asked."""
        log = Path(tempfile.mkdtemp(prefix="ss-skills-")) / "asked"
        self.addCleanup(shutil.rmtree, log.parent, True)
        with mock.patch.dict(os.environ, {"STUB_SKILLS_LOG": str(log), **env}):
            answer = self.room()
        times = len(log.read_text(encoding="utf-8").split()) if log.exists() else 0
        return answer, times

    def test_the_real_output_of_the_cli(self) -> None:
        # A recording of `copilot skill list --json`, not output written to fit the
        # parser: the staged skill is `project`, the CLI's own are `builtin`.
        folder = Path(__file__).parent / "fixtures" / "harness" / "copilot"
        recorded = (folder / "skill_list.json").read_text(encoding="utf-8")
        expected = json.loads((folder / "expected.json").read_text(encoding="utf-8"))
        self.assertGreater(len(json.loads(recorded)), len(expected["skill_list"]["staged"]))
        with mock.patch.dict(os.environ, {"STUB_SKILLS": recorded}):
            self.assertEqual(self.room(), expected["skill_list"]["staged"])

    def test_a_skill_the_cli_reports_as_switched_off_is_not_in_the_room(self) -> None:
        listed = json.dumps([
            {"name": "rocm-doctor", "source": "project", "enabled": True},
            {"name": "switched-off", "source": "project", "enabled": False},
            {"name": "no-field", "source": "project"},
        ])
        with mock.patch.dict(os.environ, {"STUB_SKILLS": listed}):
            self.assertEqual(self.room(), ["rocm-doctor", "no-field"])

    def test_an_entry_whose_name_is_not_a_name_is_dropped(self) -> None:
        listed = json.dumps([
            {"name": "rocm-doctor", "source": "project"},
            {"name": 7, "source": "project"},
            {"name": ["a"], "source": "project"},
            {"name": "", "source": "project"},
            {"source": "project"},
        ])
        with mock.patch.dict(os.environ, {"STUB_SKILLS": listed}):
            self.assertEqual(self.room(), ["rocm-doctor"])

    def test_a_cli_that_cannot_say_leaves_the_room_unverified(self) -> None:
        answer, times = self.asked(STUB_SKILLS_FAIL="1")
        self.assertEqual(times, 1, "the CLI was never asked, so this proves nothing")
        self.assertIsNone(answer)

    def test_output_that_is_not_a_list_leaves_the_room_unverified(self) -> None:
        answer, times = self.asked(STUB_SKILLS='{"not": "a list"}')
        self.assertEqual(times, 1)
        self.assertIsNone(answer)

    def test_a_list_from_a_cli_that_failed_is_not_believed(self) -> None:
        # Valid JSON, but the exit status says the command did not succeed.
        listed = json.dumps([{"name": "rocm-doctor", "source": "project"}])
        answer, times = self.asked(STUB_SKILLS=listed, STUB_SKILLS_EXIT="2")
        self.assertEqual(times, 1)
        self.assertIsNone(answer)

    def test_a_cli_that_hangs_is_stopped_and_leaves_nothing_running(self) -> None:
        import time

        # Through the shim `stub_cli` puts on PATH, which is what an npm-installed
        # CLI is: killing only the process that was started leaves the program
        # behind with the pipe open, and the wait lasts until it ends by itself.
        pidfile = Path(tempfile.mkdtemp(prefix="ss-pid-")) / "pid"
        self.addCleanup(shutil.rmtree, pidfile.parent, True)
        started = time.monotonic()
        with mock.patch.object(copilot_module, "ROOM_TIMEOUT_S", 1.5):
            answer, times = self.asked(STUB_SKILLS_HANG="1", STUB_SKILLS_PID=str(pidfile))
        elapsed = time.monotonic() - started
        self.assertEqual(times, 1)
        self.assertIsNone(answer)
        # Windows waits for the program to close the pipe, which is what takes long
        # there. POSIX gives the pipe up at once, so what shows a program left
        # running is the program itself.
        self.assertLess(elapsed, 15, "the stop waited for the program instead of killing it")
        if sys.platform.startswith("linux"):
            program = int(pidfile.read_text(encoding="utf-8"))
            for _ in range(50):
                try:
                    state = Path(f"/proc/{program}/stat").read_text(encoding="utf-8").rsplit(")", 1)[1].split()[0]
                except FileNotFoundError:
                    break
                if state == "Z":  # dead, only not yet collected
                    break
                time.sleep(0.1)
            else:
                self.fail("the program the CLI started is still running after the stop")

    def test_a_command_with_no_time_left_does_not_ask(self) -> None:
        bound = deadline.Deadline(600.0, command="routing")
        previous = deadline.use(bound)
        self.addCleanup(deadline.use, previous)
        # Not merely killed early: the CLI must not be started at all. A process
        # given no time can die before it prints anything, which looks the same.
        # The process is started through `_run_bounded`, which uses `Popen`: patching
        # a call the code no longer makes would pass whether or not the guard exists.
        starting = mock.Mock(side_effect=AssertionError("the CLI was started with no time left"))
        with mock.patch.object(bound, "remaining", lambda: 0.0), \
                mock.patch.object(base_module.subprocess, "Popen", starting):
            answer = self.room()
        starting.assert_not_called()
        self.assertIsNone(answer)

    def test_a_harness_that_announces_its_skills_itself_is_not_asked(self) -> None:
        self.assertIsNone(harness.get("claude-code").room(tempfile.gettempdir(), {}))


class TestAToolFreeCommandTakesNoExtraFlags(unittest.TestCase):
    def test_a_flag_that_could_grant_a_tool_back_is_refused(self) -> None:
        spec = harness.RunSpec(
            workspace="ws", prompt="p", tool_free=True, extra_flags=("--allow-all",)
        )
        with self.assertRaises(NotImplementedError):
            harness.get("copilot").command(spec)

    def test_the_agents_own_command_still_takes_them(self) -> None:
        spec = harness.RunSpec(workspace="ws", prompt="p", extra_flags=("--a-flag",))
        self.assertIn("--a-flag", harness.get("copilot").command(spec))


class TestTheAgentsOwnCommand(unittest.TestCase):
    def test_it_is_not_cut_down_to_the_judges_no_tools(self) -> None:
        # The allowlist belongs to the judge. On the agent's command it would leave a
        # run with no tool at all, which no other test would notice.
        command = harness.get("copilot").command(harness.RunSpec(workspace="ws", prompt="p"))
        self.assertIn("--allow-all", command)
        self.assertFalse([flag for flag in command if flag.startswith("--available-tools")])


class TestAHarnessThatIsJudgedAnotherWay(unittest.TestCase):
    def test_asking_it_to_complete_is_a_failed_completion_not_a_crash(self) -> None:
        # `claude` has to be found for the refusal to be reached at all; without it
        # the missing executable would answer first and this would prove nothing.
        stub_cli(self, "claude", "pass")
        with self.assertRaises(harness.CompletionError) as raised:
            harness.get("claude-code").complete("a question", None)
        self.assertIn("inspect model", str(raised.exception))


class TestThePreflight(unittest.TestCase):
    def setUp(self) -> None:
        stub_cli(self, "copilot", STUB_COPILOT)
        self.record = Path(tempfile.mkdtemp(prefix="ss-record-")) / "run.json"
        self.addCleanup(shutil.rmtree, self.record.parent, True)
        env = mock.patch.dict(os.environ, {"STUB_RECORD": str(self.record)})
        env.start()
        self.addCleanup(env.stop)

    def test_a_cli_that_answers_passes(self) -> None:
        self.assertEqual(harness.get("copilot").preflight("claude-sonnet-5.5"), (True, "ok"))

    def test_it_uses_the_same_command_a_case_will(self) -> None:
        harness.get("copilot").preflight("claude-sonnet-5.5")
        ran = json.loads(self.record.read_text(encoding="utf-8"))
        self.assertIn("--output-format", ran["argv"])
        self.assertEqual(ran["argv"][ran["argv"].index("--model") + 1], "claude-sonnet-5.5")
        self.assertEqual(ran["stdin"], "Reply with the single word: ok")

    def test_it_runs_isolated_from_the_runners_own_config(self) -> None:
        harness.get("copilot").preflight("claude-sonnet-5.5")
        ran = json.loads(self.record.read_text(encoding="utf-8"))
        self.assertTrue(ran["home"], "no throwaway home was given to the CLI")
        self.assertEqual(ran["auto_update"], "false")

    def test_output_that_is_not_utf8_is_read_not_fatal(self) -> None:
        from skillscope.harness import base

        proc = base._run_bounded(
            [sys.executable, "-c", "import sys; sys.stdout.buffer.write(bytes([111, 107, 32, 255, 254]))"],
            stdin="", cwd=os.getcwd(), env=dict(os.environ), timeout=30,
        )
        self.assertEqual(proc.returncode, 0)
        self.assertTrue(proc.stdout.startswith("ok "), proc.stdout)

    def test_a_refusal_says_why(self) -> None:
        with mock.patch.dict(os.environ, {"STUB_FAIL": "Error: Bad credentials"}):
            ok, detail = harness.get("copilot").preflight("claude-sonnet-5.5")
        self.assertFalse(ok)
        self.assertIn("Bad credentials", detail)

    def test_a_cli_that_never_answers_times_out(self) -> None:
        import time

        started = time.monotonic()
        with mock.patch.dict(os.environ, {"STUB_HANG": "1"}):
            ok, detail = harness.get("copilot").preflight("claude-sonnet-5.5", timeout=1)
        self.assertFalse(ok)
        self.assertIn("timed out", detail)
        # The stub sleeps for a minute. A CLI installed with npm is a shim that
        # starts the real program, and killing only the shim left the program
        # holding the output pipe, so the wait after the kill lasted that long.
        self.assertLess(time.monotonic() - started, 20)

    def test_a_missing_cli_is_named(self) -> None:
        with mock.patch("shutil.which", lambda _: None):
            ok, detail = harness.get("copilot").preflight("claude-sonnet-5.5")
        self.assertEqual((ok, detail), (False, "'copilot' CLI not found on PATH"))

    def test_it_does_not_outlive_the_commands_own_deadline(self) -> None:
        bound = deadline.Deadline(600.0, command="routing")
        previous = deadline.use(bound)
        self.addCleanup(deadline.use, previous)
        # Nothing left of it: a preflight must not start a call it cannot finish.
        with mock.patch.object(bound, "remaining", return_value=0.0):
            ok, detail = harness.get("copilot").preflight("claude-sonnet-5.5")
        self.assertFalse(ok)
        self.assertIn("--timeout", detail)


class TestReadingWhatTheAgentLeftBehind(unittest.IsolatedAsyncioTestCase):
    """`judge.artifacts` itself, against a stub sandbox: the part a mock hides."""

    async def read(self, read_images: bool) -> tuple[list[str], list[tuple[str, bytes]], list[str]]:
        from skillscope.engine import judge

        read: list[str] = []

        async def read_file(path: str, text: bool = True):
            read.append(path)
            return "notes" if text else bytes([137, 80, 78, 71])

        box = mock.Mock()
        box.read_file = read_file
        resolve = mock.AsyncMock(side_effect=lambda path: path)
        with mock.patch("inspect_ai.util.sandbox", lambda *a, **k: box), \
                mock.patch("skillscope.engine.tools.resolve", resolve):
            described, images = await judge.artifacts(
                ["out/cat.png", "notes.md"], read_images=read_images
            )
        return described, images, read

    async def test_an_image_the_judge_cannot_be_shown_is_named_and_never_read(self) -> None:
        described, images, read = await self.read(read_images=False)
        self.assertEqual(images, [("out/cat.png", b"")])
        self.assertEqual(read, ["notes.md"])
        self.assertTrue(any("notes" in entry for entry in described))

    async def test_a_judge_that_can_be_shown_it_gets_the_bytes(self) -> None:
        _, images, read = await self.read(read_images=True)
        self.assertEqual(images, [("out/cat.png", bytes([137, 80, 78, 71]))])
        self.assertIn("out/cat.png", read)


class TestTheJudgeCountsAgainstTheSubprocessLimit(unittest.IsolatedAsyncioTestCase):
    async def test_each_question_is_asked_inside_the_limiter(self) -> None:
        import contextlib

        from skillscope.engine import judge, no_sandbox

        held: list[int] = []
        inside: list[bool] = []

        @contextlib.asynccontextmanager
        async def limiter():
            held.append(1)
            try:
                yield
            finally:
                held.pop()

        class Recording:
            def complete(self, prompt: str, model: str | None) -> str:
                inside.append(bool(held))
                return "an answer"

        ask = judge.harness_completer(Recording(), "a-model")
        with mock.patch.object(no_sandbox, "_subprocess_concurrency", limiter):
            self.assertEqual(await ask("a question"), "an answer")
            self.assertEqual(await ask("another"), "an answer")
        self.assertEqual(inside, [True, True])


class TestTheJudgeNamesWhatItCannotShow(unittest.IsolatedAsyncioTestCase):
    """A harness judge sees text only; an image artifact is named in the question."""

    async def grade(self, images: list[tuple[str, bytes]]) -> tuple[tuple[bool, str], list[str]]:
        from skillscope.engine import judge

        asked: list[str] = []

        async def complete(question: str) -> str:
            asked.append(question)
            return '{"pass": true, "reason": "it did"}'

        listing = mock.AsyncMock(return_value=[path for path, _ in images])
        reading = mock.AsyncMock(return_value=(["--- notes.md ---\nhello"], images))
        with mock.patch.object(judge, "tools_list_paths", listing), \
                mock.patch.object(judge, "artifacts", reading), \
                mock.patch.object(judge, "transcript_of", lambda state: "the transcript"), \
                mock.patch.object(judge, "final_message_of", lambda state: "the answer"):
            verdict = await judge.grade(
                "it made a picture", object(), must_happen=True, complete=complete
            )
        self.read_with = reading.await_args
        return verdict, asked

    async def test_the_images_are_named_and_not_embedded(self) -> None:
        verdict, [question] = await self.grade([("out/cat.png", bytes([137, 80, 78, 71]))])
        self.assertEqual(verdict, (True, "judge: it did"))
        self.assertIn("1 image artifact(s) were produced and cannot be shown", question)
        self.assertIn("out/cat.png", question)
        self.assertNotIn("data:image", question)
        # Never read, since it cannot be shown: only its name is wanted.
        self.assertEqual(self.read_with.kwargs, {"read_images": False})

    async def test_a_grader_and_a_complete_together_are_refused(self) -> None:
        from skillscope.engine import judge

        async def complete(question: str) -> str:
            return "{}"

        with self.assertRaises(ValueError):
            await judge.grade("x", object(), must_happen=True, grader="a-model", complete=complete)

    async def test_with_no_image_nothing_is_said_about_one(self) -> None:
        _, [question] = await self.grade([])
        self.assertNotIn("image artifact", question)


if __name__ == "__main__":
    unittest.main()
