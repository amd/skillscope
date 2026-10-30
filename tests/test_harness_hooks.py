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
from skillscope.harness import copilot as copilot_module


def stub_cli(test: unittest.TestCase, name: str, script: str) -> None:
    """Put an executable called `name` that runs `script` first on PATH.

    A script rather than a mock, so `shutil.which`, the subprocess and its
    environment all run. Windows resolves a bare name through PATHEXT, so there
    the shim is a `.cmd`; elsewhere it is a `/bin/sh` that execs the interpreter,
    which survives an interpreter path with a space in it.
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
            f'#!/bin/sh\nexec "{sys.executable}" "{program}" "$@"\n', encoding="utf-8"
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

if "--version" in sys.argv:
    print("GitHub Copilot CLI 9.9.9.")
    print("Run 'copilot update' to check for updates.")
    sys.exit(0)

stdin = sys.stdin.read()
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


if __name__ == "__main__":
    unittest.main()
