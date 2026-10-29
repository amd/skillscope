<!--
Copyright Advanced Micro Devices, Inc.

SPDX-License-Identifier: MIT
-->

# Harnesses

A **harness** is the agent that is handed a prompt and a workspace and does the
work: Claude Code, GitHub Copilot CLI, Codex, and so on. It is a different
question from where the agent runs. `--engine` says *where* (on the host, or in
a container); the harness says *what*. The engines drive a harness as a
subprocess, so the contract between them is small, and this page is that
contract.

Today one harness is registered, `claude-code`. The contract exists so that a
second one is an adapter and a set of tests rather than a change to the engines.

## What a harness provides

A harness is a subclass of `skillscope.harness.Harness`, registered in
`skillscope/harness/__init__.py`.

| Member | What it is |
| --- | --- |
| `name` | The name reports and `--harness` use. |
| `binary` | The executable looked up on `PATH`. |
| `skills_dir` | Where, relative to the workspace, the harness discovers skills (`.claude/skills` for Claude Code). The engine stages each skill there. |
| `has_skill_tool` | Whether activating a skill is a tool call in the stream. A harness that activates a skill by reading its `SKILL.md` sets this `False`, and routing reads activation from the file access instead. |
| `prompt_on_stdin` | Whether the prompt is written to stdin. A harness that takes it as an argument builds it into `command` and sets this `False`; stdin is closed either way, because some CLIs wait on it. |
| `require()` | Refuse to start when the harness is not installed. The default checks `binary` is on `PATH`. |
| `command(spec)` | The argv that runs one case, headless, streaming JSON events. |
| `env(spec)` | The environment of that process, including whatever points it at `spec.config_dir` so the runner's own settings and skills stay out of the session. |
| `normalize(event)` | Turns one event the harness printed into canonical events (below). |

`spec` is a `RunSpec`: the workspace, the prompt, the model and effort, an
optional throwaway config directory, and any spend or session flags the caller
built.

## The canonical event form

The engines read one event shape, and a harness translates into it. It is
Claude Code's stream-json, reduced to what is read:

| Event | Carries |
| --- | --- |
| `{"type": "system", "subtype": "init", "skills": [...]}` | The skills the session registered. Optional: a harness that cannot say reports nothing, and the room check says it did not run rather than reporting an empty room. |
| `{"type": "assistant", "message": {"content": [{"type": "tool_use", "name": ..., "input": {...}}]}}` | A tool call. Activating a skill is a `tool_use` named `Skill` with `input.skill` set. |
| `{"type": "user", "message": {"content": [{"type": "tool_result", "content": ...}]}}` | A tool result. |
| `{"type": "result", "result": "..."}` | The end of the run, and its closing answer. A stream without one is a run that did not finish. |

A harness whose stream already has this shape keeps the default `normalize`,
which passes events through. Any other maps its events onto these and returns an
empty list for events that carry nothing the engines read. Every stop rule,
budget, score and judge sees canonical events only, never the raw ones, so each
is written once for every harness.

## Adding a harness

1. Subclass `Harness` in `skillscope/harness/<name>.py`, fill in the members
   above, and register it in `skillscope/harness/__init__.py`.
2. Record two real runs of the CLI, headless, on a trivial skill: one where it
   activates the skill and finishes, one where it finishes without activating
   one. Scrub tokens and paths. Put them in
   `tests/fixtures/harness/<name>/trigger.jsonl` and `no_trigger.jsonl`, with an
   `expected.json` saying what the engine should read out of each (see the
   Claude Code one) and the CLI version they were recorded with.
3. Run `python -m unittest tests.test_harness`. The contract test runs once per
   registered harness and fails if there are no fixtures. It checks activation,
   completion, the closing answer, that the command carries the model and effort
   and no credential, that the config directory is isolated, and that a skill
   is staged where the harness looks and without its `evals/` directory.
4. Say in the pull request how the harness differs from the others in ways a
   reader of a report should know: how it activates skills, what it cannot
   report, and what it needs to authenticate.

A report from one harness is its own result. Harnesses send different system
prompts, offer different tools, and activate skills in different ways, so a score
from one is not comparable with a score from another.

## What is not yet behind the contract

The contract covers staging, the command, the environment and the events. These
still assume Claude Code, and a second harness will need to move them behind it:

* choosing a harness from the command line (there is no `--harness` flag yet);
* the reachability preflight and the credential check that decides whether the
  runner's own config can be kept out of a routing run (`ANTHROPIC_*`);
* the spend-limit flags, which are probed from `claude --help`;
* the CI model pin, which forces `opus`;
* the judge, which uses an Anthropic model through `inspect_ai`;
* `.claude/skills` as the path routing treats as a skills inspection, rather than
  a harness's own `skills_dir`.
