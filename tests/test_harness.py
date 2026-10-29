# Copyright Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: MIT

"""The harness contract: what every agent harness has to provide.

Three kinds of test, in the order a new harness meets them:

* the registry, which is how `--harness` will find one;
* the contract, run once per registered harness against the recorded output in
  `tests/fixtures/harness/<name>/`, so a new harness is held to the same bar as
  the first without anyone writing its tests from scratch;
* the seam, which drives the engine with a harness whose events look nothing
  like Claude Code's, because a contract only the one real harness has ever
  satisfied has not been shown to be a contract.
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

from skillscope import harness, routing
from skillscope.engine import no_sandbox

FIXTURES = Path(__file__).parent / "fixtures" / "harness"


class TestTheRegistry(unittest.TestCase):
    def test_claude_code_is_the_default(self) -> None:
        self.assertEqual(harness.DEFAULT, "claude-code")
        self.assertEqual(harness.get().name, "claude-code")
        self.assertIs(harness.get(), harness.get("claude-code"))

    def test_names_lists_what_is_registered(self) -> None:
        self.assertIn("claude-code", harness.names())

    def test_an_unknown_harness_is_refused_with_the_known_ones(self) -> None:
        with self.assertRaises(SystemExit) as raised:
            harness.get("no-such-agent")
        message = str(raised.exception)
        self.assertIn("no-such-agent", message)
        self.assertIn("claude-code", message)

    def test_a_name_cannot_be_registered_twice(self) -> None:
        with self.assertRaises(ValueError):
            harness.register(harness.ClaudeCode())


class TestClaudeCodeIsWhatTheEngineAlwaysRan(unittest.TestCase):
    """The adapter replaced inline code, so pin the output it replaced.

    The command line and environment are the whole of what changed hands, and
    nothing else in the suite compares them to the earlier values: the
    end-to-end test runs a stub that ignores its flags.
    """

    def spec(self, **overrides) -> harness.RunSpec:
        values = {"workspace": "/ws", "prompt": "do it", **overrides}
        return harness.RunSpec(**values)

    def command(self, **overrides) -> list[str]:
        with mock.patch("shutil.which", lambda _: "/usr/bin/claude"):
            return harness.get("claude-code").command(self.spec(**overrides))

    def test_the_command_is_the_one_the_engine_built_inline(self) -> None:
        self.assertEqual(
            self.command(model="opus", effort="high"),
            [
                "/usr/bin/claude", "-p",
                "--output-format", "stream-json", "--verbose",
                "--dangerously-skip-permissions",
                "--add-dir", "/ws",
                "--model", "opus",
                "--effort", "high",
            ],
        )

    def test_model_and_effort_are_left_out_when_not_given(self) -> None:
        command = self.command()
        self.assertNotIn("--model", command)
        self.assertNotIn("--effort", command)

    def test_the_callers_flags_come_last_in_order(self) -> None:
        command = self.command(extra_flags=("--max-budget-usd", "1"))
        self.assertEqual(command[-2:], ["--max-budget-usd", "1"])

    def test_a_config_dir_is_only_exported_when_one_was_given(self) -> None:
        claude = harness.get("claude-code")
        self.assertEqual(
            claude.env(self.spec(config_dir=Path("/cfg")))["CLAUDE_CONFIG_DIR"],
            str(Path("/cfg")),
        )
        with mock.patch.dict(os.environ, clear=False):
            os.environ.pop("CLAUDE_CONFIG_DIR", None)
            self.assertNotIn("CLAUDE_CONFIG_DIR", claude.env(self.spec()))

    def test_the_cli_is_told_not_to_retry_unless_the_caller_says_otherwise(self) -> None:
        claude = harness.get("claude-code")
        with mock.patch.dict(os.environ, clear=False):
            os.environ.pop("CLAUDE_CODE_MAX_RETRIES", None)
            self.assertEqual(claude.env(self.spec())["CLAUDE_CODE_MAX_RETRIES"], "0")
            os.environ["CLAUDE_CODE_MAX_RETRIES"] = "3"
            self.assertEqual(claude.env(self.spec())["CLAUDE_CODE_MAX_RETRIES"], "3")

    def test_a_missing_binary_is_refused_by_name(self) -> None:
        with mock.patch("shutil.which", lambda _: None), self.assertRaises(SystemExit) as raised:
            harness.get("claude-code").require()
        self.assertEqual(str(raised.exception), "error: 'claude' CLI not found on PATH")


class TestEveryRegisteredHarnessMeetsTheContract(unittest.TestCase):
    """Run for each harness that ships fixtures; a harness without them fails.

    A fixture directory holds `trigger.jsonl` (the harness activated the skill
    and finished), `no_trigger.jsonl` (it finished without activating one) and
    `expected.json` with what the engine should read out of each. Real
    recordings, token-scrubbed, with the CLI version noted in `expected.json`;
    a harness whose recordings change shape gets new fixtures, not a looser
    test.
    """

    def fixtures(self, name: str) -> tuple[dict, dict[str, list[str]]]:
        folder = FIXTURES / name
        self.assertTrue(
            folder.is_dir(),
            f"harness {name!r} is registered but has no fixtures in {folder}",
        )
        expected = json.loads((folder / "expected.json").read_text(encoding="utf-8"))
        runs = {
            kind: (folder / f"{kind}.jsonl").read_text(encoding="utf-8").splitlines()
            for kind in ("trigger", "no_trigger")
        }
        return expected, runs

    def canonical(self, subject: harness.Harness, lines: list[str]) -> list[dict]:
        events: list[dict] = []
        for line in lines:
            if line.strip():
                events.extend(subject.normalize(json.loads(line)))
        return events

    def test_what_the_engine_reads_out_of_a_recorded_run(self) -> None:
        for name in harness.names():
            with self.subTest(harness=name):
                subject = harness.get(name)
                expected, runs = self.fixtures(name)
                for kind, lines in runs.items():
                    events = self.canonical(subject, lines)
                    want = expected[kind]

                    activation = next(
                        (
                            hit
                            for event in events
                            if (hit := routing.detect_activation(
                                event, expected["skills"],
                                allow_body_path=not subject.has_skill_tool,
                            ))
                        ),
                        None,
                    )
                    self.assertEqual(activation, want["activation"], kind)

                    completed, final = no_sandbox.run_completed(events)
                    self.assertEqual(completed, want["completed"], kind)
                    self.assertEqual(final, want["final"], kind)

    def test_the_room_is_read_off_the_init_event_when_the_harness_has_one(self) -> None:
        for name in harness.names():
            with self.subTest(harness=name):
                subject = harness.get(name)
                expected, runs = self.fixtures(name)
                events = self.canonical(subject, runs["trigger"])
                seen = [
                    visible
                    for event in events
                    if (visible := routing.init_skills(event, expected["skills"])) is not None
                ]
                # A harness with no init event reports nothing, which the engine
                # words as "the check did not run". It must never report a room
                # it did not read.
                for visible in seen:
                    self.assertEqual(visible, expected["skills"])

    def test_the_command_carries_the_model_the_effort_and_no_secret(self) -> None:
        for name in harness.names():
            with self.subTest(harness=name):
                subject = harness.get(name)
                spec = harness.RunSpec(
                    workspace=str(Path(tempfile.gettempdir()) / "ws"),
                    prompt="the prompt",
                    model="a-model",
                    effort="high",
                )
                with mock.patch.dict(os.environ, {"SOME_API_KEY": "sk-secret-value"}):
                    with mock.patch("shutil.which", lambda b: f"/bin/{b}"):
                        command = subject.command(spec)
                self.assertIn("a-model", command)
                self.assertIn("high", command)
                self.assertFalse(
                    any("sk-secret-value" in part for part in command),
                    "a credential on the command line shows up in `ps` and in logs",
                )

    def test_the_config_is_isolated_from_the_runners_own(self) -> None:
        for name in harness.names():
            with self.subTest(harness=name):
                subject = harness.get(name)
                spec = harness.RunSpec(
                    workspace="ws", prompt="p", config_dir=Path("throwaway-config"),
                )
                env = subject.env(spec)
                self.assertIn(
                    str(Path("throwaway-config")), env.values(),
                    "no environment variable points the harness at the throwaway "
                    "config dir, so the runner's own skills would join the room",
                )

    def test_a_skill_is_staged_where_the_harness_looks(self) -> None:
        for name in harness.names():
            with self.subTest(harness=name):
                subject = harness.get(name)
                with tempfile.TemporaryDirectory() as tmp:
                    skill = Path(tmp) / "alpha"
                    (skill / "evals").mkdir(parents=True)
                    (skill / "SKILL.md").write_text("# alpha", encoding="utf-8")
                    (skill / "evals" / "evals.json").write_text("{}", encoding="utf-8")
                    workspace = Path(tmp) / "ws"
                    workspace.mkdir()

                    no_sandbox.install_skill(skill, str(workspace), subject.skills_dir)

                    staged = workspace / subject.skills_dir / "alpha"
                    self.assertTrue((staged / "SKILL.md").is_file())
                    # The answer key stays out of the room for every harness.
                    self.assertFalse((staged / "evals").exists())


# What the foreign harness below prints. Deliberately nothing like stream-json:
# no `type` keys, different names for everything.
_FOREIGN_AGENT = r"""
import json, os, sys

def emit(**event):
    print(json.dumps(event), flush=True)

staged = os.path.isfile(os.path.join(".foreign", "skills", "alpha", "SKILL.md"))
# What the harness was handed: its prompt as an argument, and a stdin that was
# closed without ever being written to.
stdin_seen = sys.stdin.read()
emit(evt="started")
emit(evt="skill", name="alpha")
emit(
    evt="command",
    command="echo foreign-marker staged=%s stdin=%r arg=%r" % (staged, stdin_seen, sys.argv[1]),
)
emit(evt="finished", text="all done")
"""


class ForeignHarness(harness.Harness):
    """A harness whose events share nothing with Claude Code's.

    Not registered: it exists to show the engine only needs what `Harness`
    promises. It runs a script, takes its prompt as an argument, and keeps its
    skills somewhere other than `.claude/skills`.
    """

    name = "foreign"
    binary = Path(sys.executable).name
    skills_dir = ".foreign/skills"
    prompt_on_stdin = False

    def __init__(self, script: Path) -> None:
        self.script = script

    def require(self) -> None:
        return None

    def command(self, spec: harness.RunSpec) -> list[str]:
        return [sys.executable, "-u", str(self.script), spec.prompt]

    def env(self, spec: harness.RunSpec) -> dict[str, str]:
        return dict(os.environ)

    def normalize(self, event: dict) -> list[dict]:
        kind = event.get("evt")
        if kind == "skill":
            return [_tool_use("Skill", {"skill": event["name"]})]
        if kind == "command":
            return [_tool_use("Bash", {"command": event["command"]})]
        if kind == "finished":
            return [{"type": "result", "subtype": "success", "result": event["text"]}]
        return []


def _tool_use(name: str, arguments: dict) -> dict:
    return {
        "type": "assistant",
        "message": {"content": [{"type": "tool_use", "name": name, "input": arguments}]},
    }


class TestTheSeamTranslatesAnotherHarnessesEvents(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.mkdtemp(prefix="ss-foreign-")
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.script = Path(self.tmp) / "foreign_agent.py"
        self.script.write_text(_FOREIGN_AGENT, encoding="utf-8")
        self.foreign = ForeignHarness(self.script)

    async def stream(self, stop_when) -> tuple[list[dict], str | None]:
        spec = harness.RunSpec(workspace=self.tmp, prompt="go")
        events, reason, _, _ = await no_sandbox._stream_until(
            self.foreign.command(spec), "", self.tmp, self.foreign.env(spec),
            stop_when, self.foreign.normalize,
        )
        return events, reason

    async def test_the_engine_reads_canonical_events_only(self) -> None:
        events, reason = await self.stream(lambda event: None)
        self.assertIsNone(reason)
        # `started` carries nothing the engine reads, so it is dropped rather
        # than passed through for something downstream to trip on.
        calls = [call for event in events for call in routing._iter_tool_uses(event)]
        self.assertEqual([name for name, _ in calls], ["Skill", "Bash"])
        self.assertEqual(calls[0][1], '{"skill": "alpha"}')
        self.assertIn("foreign-marker", calls[1][1])
        completed, final = no_sandbox.run_completed(events)
        self.assertEqual((completed, final), (True, "all done"))

    async def test_the_stop_rule_sees_canonical_events(self) -> None:
        seen: list[dict] = []

        def stop_when(event: dict) -> str | None:
            seen.append(event)
            return "activated" if routing.detect_activation(event, ["alpha"]) else None

        events, reason = await self.stream(stop_when)
        self.assertEqual(reason, "activated")
        self.assertEqual(len(events), 1)
        self.assertTrue(all("evt" not in event for event in seen))

    async def test_an_unreadable_line_is_skipped_not_fatal(self) -> None:
        self.script.write_text(
            "import json\nprint('not json', flush=True)\n"
            "print(json.dumps({'evt': 'finished', 'text': 'ok'}), flush=True)\n",
            encoding="utf-8",
        )
        events, _ = await self.stream(lambda event: None)
        self.assertEqual(no_sandbox.run_completed(events), (True, "ok"))


class TestTheEngineRunsACaseWithAnotherHarness(unittest.TestCase):
    """The whole solver, under inspect, with a harness that is not Claude Code."""

    def setUp(self) -> None:
        try:
            import inspect_ai  # noqa: F401
        except ImportError:
            self.skipTest("needs inspect_ai")
        from skillscope.engine import sandbox as engine_sandbox

        env = mock.patch.dict(os.environ, {engine_sandbox.SANDBOX_ENV: "local"})
        env.start()
        self.addCleanup(env.stop)
        self.tmp = tempfile.mkdtemp(prefix="ss-foreign-e2e-")
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.script = Path(self.tmp) / "foreign_agent.py"
        self.script.write_text(_FOREIGN_AGENT, encoding="utf-8")
        self.skill = Path(self.tmp) / "alpha"
        self.skill.mkdir()
        (self.skill / "SKILL.md").write_text(
            "---\nname: alpha\ndescription: does alpha things\n---\n", encoding="utf-8"
        )

    def run_case(self, *, stop_when_factory=None):
        import contextlib
        import io

        from inspect_ai import Task
        from inspect_ai import eval as inspect_eval
        from inspect_ai.dataset import Sample

        task = Task(
            dataset=[Sample(id="c1", input="do the alpha thing")],
            solver=no_sandbox.harness_solver(
                ForeignHarness(self.script), "mockllm/model", None, self.skill,
                stop_when_factory=stop_when_factory,
            ),
            sandbox="local",
        )
        with contextlib.redirect_stdout(io.StringIO()):
            [log] = inspect_eval(
                task, model="mockllm/model", log_dir=str(Path(self.tmp) / "logs"),
                display="plain",
            )
        [sample] = log.samples
        self.assertIsNone(sample.error, sample.error)
        return sample

    def calls(self, sample) -> list[tuple[str, dict]]:
        return [
            (call.function, call.arguments)
            for message in sample.messages
            for call in getattr(message, "tool_calls", None) or []
        ]

    def test_the_foreign_events_become_tool_calls_the_scorers_read(self) -> None:
        sample = self.run_case()
        functions = [function for function, _ in self.calls(sample)]
        self.assertEqual(functions, ["Skill", "Bash"])
        self.assertIn("all done", sample.output.completion)

    def test_the_skill_is_staged_where_that_harness_looks(self) -> None:
        # The agent script checks for the file itself, in the workspace it was
        # started in, and reports what it found.
        sample = self.run_case()
        [(_, command)] = [call for call in self.calls(sample) if call[0] == "Bash"]
        self.assertIn("staged=True", command["command"])

    def test_a_prompt_taken_as_an_argument_is_not_also_written_to_stdin(self) -> None:
        sample = self.run_case()
        [(_, command)] = [call for call in self.calls(sample) if call[0] == "Bash"]
        self.assertIn("arg='do the alpha thing'", command["command"])
        # Closed and empty, not left open: a CLI that waits on stdin would hang.
        self.assertIn("stdin=''", command["command"])

    def test_the_stop_rule_ends_the_run_on_a_foreign_activation(self) -> None:
        def factory():
            return lambda event: (
                "activated" if routing.detect_activation(event, ["alpha"]) else None
            )

        sample = self.run_case(stop_when_factory=factory)
        self.assertEqual([function for function, _ in self.calls(sample)], ["Skill"])


if __name__ == "__main__":
    unittest.main()
