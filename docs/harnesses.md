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

**A harness defines the agent once, for both answers to *where*.** On the host the
engine starts `command(spec)` as a subprocess: the executable resolved on the host,
and the runner's environment underneath what `env(spec)` sets. A sandbox leg starts
the same command inside the container, rather than delegating to a separate agent
definition such as an `inspect_swe` one. Otherwise every agent would exist twice in
this repo, with its own command line, environment, skill staging and event handling,
free to drift apart.

Today one harness is registered, `claude-code`. The contract exists so that a
second one is an adapter and a set of tests rather than a change to the engines.

## What a harness provides

A harness is a subclass of `skillscope.harness.Harness`, registered in
`skillscope/harness/__init__.py`.

| Member | What it is |
| --- | --- |
| `name` | The name reports and `--harness` use. |
| `binary` | The executable's name. Whoever starts the process resolves it: on the host with `shutil.which`, in a container by the guest's `PATH`. |
| `skills_dir` | Where, relative to the workspace, the harness discovers skills (`.claude/skills` for Claude Code). The engine stages each skill there. |
| `has_skill_tool` | Whether activating a skill is a tool call in the stream. A harness that activates a skill by reading its `SKILL.md` sets this `False`, and routing reads activation from the file access instead. |
| `prompt_on_stdin` | Whether the prompt is written to stdin. A harness that takes it as an argument builds it into `command` and sets this `False`; stdin is closed either way, because some CLIs wait on it. |
| `require()` | Refuse to start **on the host** when the harness is not installed there. The default checks `binary` is on the runner's `PATH`. A leg that starts the agent in a container checks the image instead. |
| `command(spec)` | The argv that runs one case, headless, streaming JSON events. The executable is named (`binary`), never an absolute path on this machine, so the same command can be started on the host or in a container. |
| `env(spec)` | What the harness sets in the process's environment, and nothing else: its own settings, and whatever points it at `spec.config_dir` so the runner's own settings and skills stay out of the session. Not the runner's environment (`PATH`, `HOME`, every key on the machine), which a container must not inherit. Whoever starts the process puts this on top of the environment it has. |
| `host_command(spec)`, `host_env(spec)` | What a run started on the host executes: `command(spec)` with its executable found by `shutil.which` (on Windows an npm-installed CLI is a `.cmd` shim that only that finds), and `os.environ` with `env(spec)` on top. A sandbox leg uses `command` and `env` directly. |
| `normalize(event)` | Turns one event the harness printed into canonical events (below). Stateless. |
| `normalizer()` | A translator for one run's events, which may keep state. The engine asks for a fresh one per case. Override it when an event cannot be translated on its own, for instance when the closing event does not repeat the answer and it has to be remembered from an earlier one. The default returns `normalize`. |

`spec` is a `RunSpec`: the workspace, the prompt, the model and effort, an
optional throwaway config directory, and any spend or session flags the caller
built. The workspace and the config directory are strings naming a path wherever
the agent runs: a host path on the host, a path inside the container in a sandbox.

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
empty list for events that carry nothing the engines read. What a harness remembers about a stream belongs in the
translator `normalizer()` returns, never on the harness itself: one harness
instance serves every case in a run, and cases run concurrently. Every stop rule,
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
   and no credential, that the config directory is isolated, that two runs translated at the same time
   do not share memory, and that a skill is staged where the harness looks and
   without its `evals/` directory.
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
* **the sandbox leg**: `--engine claude-code` still runs Claude Code through
  `inspect_swe`, with its own command line, environment, skill staging and event
  handling. It should start the harness's `command(spec)` in the container, so that
  each agent is defined once. Until it does, a harness other than Claude Code has no
  sandboxed leg;
* `.claude/skills` as the path routing treats as a skills inspection, rather than
  a harness's own `skills_dir`.
