# Copyright Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: MIT

"""GitHub Copilot CLI as a harness: what is particular to it.

The shared contract (`tests/test_harness.py`) already runs against the
recordings in `tests/fixtures/harness/copilot/`. These tests cover what only
this harness does: the command it is started with, the closing answer it has to
remember, how its events become tool calls, and where its token counts come
from. The recordings are real output from CLI 1.0.105; the synthetic events
below exist only for shapes the recordings do not contain, and say so.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from skillscope import harness, routing, usage
from skillscope.engine import no_sandbox

FIXTURES = Path(__file__).parent / "fixtures" / "harness" / "copilot"


def recorded(name: str) -> list[dict]:
    lines = (FIXTURES / f"{name}.jsonl").read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines if line.strip()]


def translate(events: list[dict]) -> list[dict]:
    normalize = harness.get("copilot").normalizer()
    return [canonical for event in events for canonical in normalize(event)]


def tool_calls(events: list[dict]) -> list[tuple[str, dict]]:
    return [
        (name, json.loads(arguments))
        for event in events
        for name, arguments in routing._iter_tool_uses(event)
    ]


class TestTheCommand(unittest.TestCase):
    def command(self, **overrides) -> list[str]:
        spec = harness.RunSpec(workspace="/ws", prompt="do it", **overrides)
        return harness.get("copilot").command(spec)

    def test_the_command_is_headless_json_with_everything_allowed(self) -> None:
        self.assertEqual(
            self.command(model="claude-sonnet-5.5", effort="high"),
            [
                "copilot",
                "--output-format", "json",
                "--allow-all",
                "--no-ask-user",
                "--disable-builtin-mcps",
                "--model", "claude-sonnet-5.5",
                "--reasoning-effort", "high",
            ],
        )

    def test_the_prompt_is_not_on_the_command_line(self) -> None:
        # It goes in on stdin: no quoting to get wrong on Windows, and no limit
        # on its length.
        self.assertTrue(harness.get("copilot").prompt_on_stdin)
        self.assertNotIn("do it", self.command())

    def test_model_and_effort_are_left_out_when_not_given(self) -> None:
        command = self.command()
        self.assertNotIn("--model", command)
        self.assertNotIn("--reasoning-effort", command)

    def test_the_callers_flags_come_last(self) -> None:
        self.assertEqual(self.command(extra_flags=("--log-level", "none"))[-2:],
                         ["--log-level", "none"])


class TestTheEnvironment(unittest.TestCase):
    def env(self, **overrides) -> dict[str, str]:
        return harness.get("copilot").env(
            harness.RunSpec(workspace="ws", prompt="p", **overrides)
        )

    def test_a_config_dir_becomes_the_cli_home(self) -> None:
        self.assertEqual(self.env(config_dir="cfg")["COPILOT_HOME"], "cfg")

    def test_no_config_dir_leaves_the_runners_own_home_alone(self) -> None:
        with mock.patch.dict(os.environ, clear=False):
            os.environ.pop("COPILOT_HOME", None)
            self.assertNotIn("COPILOT_HOME", self.env())

    def test_the_cli_does_not_update_itself_mid_run_unless_told_to(self) -> None:
        with mock.patch.dict(os.environ, clear=False):
            os.environ.pop("COPILOT_AUTO_UPDATE", None)
            self.assertEqual(self.env()["COPILOT_AUTO_UPDATE"], "false")
            os.environ["COPILOT_AUTO_UPDATE"] = "true"
            self.assertEqual(self.env()["COPILOT_AUTO_UPDATE"], "true")

    def test_the_credential_is_the_runners_to_supply_not_the_harnesss_to_carry(self) -> None:
        # Not in what the harness sets, so a container is not handed the host's
        # token by default; on the host the runner's environment goes underneath.
        spec = harness.RunSpec(workspace="ws", prompt="p")
        with mock.patch.dict(os.environ, {"GH_TOKEN": "token-for-the-test"}):
            self.assertNotIn("GH_TOKEN", self.env())
            self.assertEqual(
                harness.get("copilot").host_env(spec)["GH_TOKEN"], "token-for-the-test"
            )


class TestWhatTheRecordingBecomes(unittest.TestCase):
    def setUp(self) -> None:
        self.events = translate(recorded("trigger"))

    def test_the_skill_call_is_the_call_the_engines_read(self) -> None:
        [first, *_] = tool_calls(self.events)
        self.assertEqual(first, ("skill", {"skill": "rocm-doctor"}))
        self.assertEqual(
            routing.detect_activation(self.events[0], ["rocm-doctor"]), "rocm-doctor"
        )

    def test_every_call_is_kept_in_order_including_a_denied_one(self) -> None:
        # The shell call was denied by a permission rule and is still a call:
        # what the agent tried is what an expectation about it is graded on.
        self.assertEqual(
            [name for name, _ in tool_calls(self.events)], ["skill", "powershell", "grep"]
        )
        self.assertEqual(tool_calls(self.events)[1][1]["command"], "rocm --version")

    def test_a_denied_call_reports_why_as_its_result(self) -> None:
        results = [
            block["content"]
            for event in self.events
            if event["type"] == "user"
            for block in event["message"]["content"]
        ]
        self.assertEqual(len(results), 3)
        self.assertIn("denied", results[1])

    def test_streaming_deltas_and_bookkeeping_are_dropped(self) -> None:
        kept = {event["type"] for event in self.events}
        self.assertLessEqual(kept, {"assistant", "user", "result"})

    def test_the_run_ends_with_the_answer_it_remembered(self) -> None:
        completed, final = no_sandbox.run_completed(self.events)
        self.assertTrue(completed)
        self.assertEqual(self.events[-1]["subtype"], "success")
        # The closing record in the recording carries no answer at all.
        closing = recorded("trigger")[-1]
        self.assertEqual(closing["type"], "result")
        self.assertNotIn("result", closing)
        self.assertTrue(final.startswith("I couldn't confirm the cause"))

    def test_the_tokens_reach_the_usage_recorder(self) -> None:
        usage.reset()
        self.addCleanup(usage.reset)
        for event in self.events:
            usage.record_stream_event(event)
        # From `session.usage_checkpoint`, which reports them once, per model.
        self.assertEqual(usage.snapshot().input_tokens, 111919)
        self.assertEqual(usage.snapshot().output_tokens, 932)

    def test_a_near_miss_calls_no_skill(self) -> None:
        events = translate(recorded("no_trigger"))
        self.assertNotIn("skill", [name for name, _ in tool_calls(events)])


class TestShapesTheRecordingsDoNotHave(unittest.TestCase):
    """Synthetic: hand-written to the shapes seen elsewhere in the same stream."""

    def run_events(self, *events: dict) -> list[dict]:
        return translate(list(events))

    def test_a_failed_run_is_reported_as_an_error_result(self) -> None:
        [closing] = self.run_events({"type": "result", "exitCode": 1})
        self.assertEqual((closing["subtype"], closing["is_error"]), ("error", True))

    def test_the_answer_is_the_last_message_with_text(self) -> None:
        events = self.run_events(
            {"type": "assistant.message", "data": {"content": "first"}},
            {"type": "assistant.message", "data": {"content": "second"}},
            # A message that only requested tools has no text and is not an answer.
            {"type": "assistant.message", "data": {"content": ""}},
            {"type": "result", "exitCode": 0},
        )
        self.assertEqual(events[-1]["result"], "second")

    def test_a_sub_agents_message_is_not_the_answer(self) -> None:
        events = self.run_events(
            {"type": "assistant.message", "data": {"content": "the answer"}},
            {"type": "assistant.message", "agentId": "sub-1", "data": {"content": "side note"}},
            {"type": "result", "exitCode": 0},
        )
        self.assertEqual(events[-1]["result"], "the answer")

    def test_a_stream_that_never_finished_is_not_a_completed_run(self) -> None:
        events = self.run_events(
            {"type": "tool.execution_start",
             "data": {"toolCallId": "c1", "toolName": "skill", "arguments": {"skill": "a"}}},
        )
        self.assertFalse(no_sandbox.run_completed(events)[0])

    def test_malformed_events_are_ignored_not_fatal(self) -> None:
        events = self.run_events(
            {"type": "tool.execution_start", "data": "not an object"},
            {"type": "tool.execution_complete"},
            {"type": "session.usage_checkpoint", "data": {"accountingSnapshot": None}},
            {"type": "result", "exitCode": 0},
        )
        self.assertEqual(events[-1]["subtype"], "success")

    def test_tokens_are_summed_across_models(self) -> None:
        usage_event = {
            "type": "session.usage_checkpoint",
            "data": {"accountingSnapshot": {"modelMetrics": {
                "model-a": {"usage": {"inputTokens": 10, "outputTokens": 1}},
                "model-b": {"usage": {"inputTokens": 5, "outputTokens": 2}},
            }}},
        }
        [spent, _] = self.run_events(usage_event, {"type": "result", "exitCode": 0})
        self.assertEqual(spent["message"]["usage"], {"input_tokens": 15, "output_tokens": 3})

    def test_a_run_that_reported_no_tokens_adds_no_usage_event(self) -> None:
        events = self.run_events({"type": "result", "exitCode": 0})
        self.assertEqual([event["type"] for event in events], ["result"])


class TestTheRecordingThroughTheEnginesStream(unittest.IsolatedAsyncioTestCase):
    """The real recording, replayed by a process, read by the engine's own loop."""

    async def replay(self, stop_when):
        # A process that prints the recording, so the subprocess, the line
        # reader and the translator all run as they do against the real CLI.
        script = (
            "import sys\n"
            "for line in open(sys.argv[1], encoding='utf-8'):\n"
            "    print(line.rstrip('\\n'), flush=True)\n"
        )
        recording = str(FIXTURES / "trigger.jsonl")
        return await no_sandbox._stream_until(
            [sys.executable, "-u", "-c", script, recording],
            "",
            tempfile.gettempdir(),
            dict(os.environ),
            stop_when,
            harness.get("copilot").normalizer(),
        )

    async def test_the_whole_run_is_read_to_its_end(self) -> None:
        events, reason, _, _ = await self.replay(lambda event: None)
        self.assertIsNone(reason)
        completed, final = no_sandbox.run_completed(events)
        self.assertTrue(completed)
        self.assertTrue(final)

    async def test_the_routing_decision_stops_the_run_at_the_skill_call(self) -> None:
        def stop_when(event: dict) -> str | None:
            return "activated" if routing.detect_activation(event, ["rocm-doctor"]) else None

        events, reason, _, _ = await self.replay(stop_when)
        self.assertEqual(reason, "activated")
        # Stopped on the first canonical event, before the work that followed.
        self.assertEqual([name for name, _ in tool_calls(events)], ["skill"])


class TestTheMessagesTheScorersRead(unittest.TestCase):
    def setUp(self) -> None:
        try:
            import inspect_ai  # noqa: F401
        except ImportError:
            self.skipTest("needs inspect_ai")

    def test_calls_and_their_results_pair_up_with_the_answer_last(self) -> None:
        messages, final = no_sandbox.events_to_messages(
            translate(recorded("trigger")), "the prompt"
        )
        calls = [
            call.function for message in messages
            for call in getattr(message, "tool_calls", None) or []
        ]
        self.assertEqual(calls, ["skill", "powershell", "grep"])
        results = [m for m in messages if getattr(m, "role", "") == "tool"]
        self.assertEqual([m.function for m in results], ["skill", "powershell", "grep"])
        self.assertTrue(final.startswith("I couldn't confirm the cause"))
        self.assertEqual(messages[-1].content, final)


if __name__ == "__main__":
    unittest.main()
