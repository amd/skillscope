<!--
Copyright Advanced Micro Devices, Inc.

SPDX-License-Identifier: MIT
-->

# Writing evals for a skill

For flags, workflow inputs, and what each check asserts, see
[usage.md](usage.md).

One file, `<your-skill>/evals/evals.json`, holding an `evaluations` array:

```json
{
  "evaluations": [
    {
      "id": "images-cost",
      "skill_should_trigger": true,
      "prompt": "I'm burning too much money on image generation APIs. Generate images on my own machine instead."
    },
    {
      "id": "generate-cat-image",
      "skill_should_trigger": true,
      "prompt": "Learn how to generate images locally, then save an image of a cat to out.png.",
      "expected_behavior": ["Install Lemonade Server if it is not already installed"],
      "unexpected_behavior": ["Reach for a cloud image path instead of local Lemonade"],
      "files_exist": ["AGENTS.md", "out.png"]
    },
    {
      "id": "finetune-on-laptop",
      "skill_should_trigger": false,
      "note": "Local, on-device, and model-shaped, but training is nobody's job here.",
      "prompt": "Fine-tune a small language model on my own dataset using my laptop GPU."
    }
  ]
}
```

Start from `skillscope template`, which prints a dataset with the shape of each
kind of evaluation and a comment on what it is for.

Every evaluation is a prompt plus `skill_should_trigger`: `true` if your skill
should fire for it, `false` if nothing should. No evaluation names a skill — the
folder does that, so `serving-llms-on-epyc/evals/evals.json` is about
`serving-llms-on-epyc` and the flag refers to it. A prompt that should trigger
a *different* skill belongs in that skill's dataset: routing installs those
skills in one workspace, so it is the same assertion either way, and filing it
under the neighbour keeps `false` meaning "nothing fires".

Your dataset's prompts are graded in a routing run only when your skill is in
the room — one the workflow lists in `routing_room`, or the only skill the
repo has, which needs no listing. How the room is chosen is in
[usage.md](usage.md#who-a-skill-competes-against-is-listed-not-inferred). A
prompt expecting a skill that is not in the room could only ever lose, and a
near miss says nothing about skills that were never installed.

**When a prompt is all you provide, the evaluation grades routing only** — did
your skill fire, and did nothing else? That is cheap, needs no hardware, and is
the failure most skills actually have.

**When you add expectations, the prompt also runs end to end** and what the
agent did is graded from the transcript and the workspace it left behind.

## The bar

Counted over `evals.json` alone:

- at least **3** evaluations with `skill_should_trigger: true`
- at least **2** with `skill_should_trigger: false`

Write the `false` ones yourself and write them close. A near miss is a prompt
that sounds like your skill's territory and is not: the wrong hardware, an
adjacent domain, a question rather than a task. Nobody else knows where your
skill's edges are, and a prompt about something entirely unrelated is already
covered — the harness ships a pool of those and runs them against every skill.

A repo may ask for more. Adding one graded expectation to a triggering
evaluation, so something beyond routing is checked, is a common one.

## Optional fields

All arrays, all valid only on a `skill_should_trigger: true` evaluation:

| Field | Graded by | Use it for |
| --- | --- | --- |
| `expected_behavior` | an LLM judge | a step the agent must take, in plain language |
| `unexpected_behavior` | an LLM judge | the mistake this skill exists to prevent |
| `logs_contain` | substring match | a literal that must appear: a script name, a flag, a pinned image tag |
| `files_exist` | the filesystem | an artifact the run must produce |

The bottom two are instant and free where a judged expectation costs a second
agent call, so reach for them when the thing you want is literal. Never assert
your own skill's name in `logs_contain`: routing already grades that properly,
and a substring match only proves the skill was staged.

A `files_exist` entry matches whole path segments anywhere in the workspace, so
`plan.md` is satisfied by `examples/plan.md` and `out/report.md` by
`run-1/out/report.md`. Name the artifact rather than the directory you hope the
agent picks — where a file lands is usually the agent's call, and a plan written
beside the fixture it describes should not fail the run. If the location
matters, ask for it in the prompt and grade it with `expected_behavior`.

Two more fields carry no expectation: `note`, which is where a comment goes
since JSON has none, and `workspace`, a directory (conventionally
`evals/files/<name>`) whose contents are copied into the agent's workspace
before the prompt runs. Use it to hand the agent a file to edit instead of
describing one in prose — the routing workspace holds nothing but the skills
tree, so a prompt that mentions a file it was never given sends the agent
hunting.

The full field reference is
[`skillscope/schema/evals.schema.json`](../skillscope/schema/evals.schema.json),
enforced by `skillscope structural`.

## A second dataset, for prompts consumers should not pay for

A skill may ship `evals/extended_evals.json` beside the required one, in the
same format. It carries no coverage bar and whether it runs is the caller's
decision: `--extended` (the default) includes it, `--no-extended` grades
`evals.json` alone. Either way it is structurally checked.

This is where a product repo keeps the prompts it wants graded in its own CI
without every consumer of its skills paying for them.

## When JSON is not enough

Two optional files sit beside the dataset.

### `evals/machine.yml`

Only for a skill whose behavioral cases need something the everyday runners
do not give them: a narrower set of platforms, particular hardware, a sandbox of
its own (`sandbox:`, see [usage.md](usage.md#where-a-sandboxed-run-is-sandboxed)),
or [prerequisites](#prerequisites). Every key is optional:

```yaml
os: [Linux]                 # defaults to whatever platforms the repo runs on
labels: [mi300x, gpu]       # hardware this skill's cases need
```

Say what the machine must have; what that *costs* is not yours to declare. The
base labels, the pull-request label rationing a scarce pool, and the
credentials that pay for it are set by the repo that owns the machines — see
[usage.md](usage.md#hardware-a-skill-needs). Asking for any label is what
moves the leg onto that pool.

Most skills that need this file need only `os: [Linux]`, to drop a Windows leg
that would just exercise the failure path of Linux-only tooling.

#### Prerequisites

A skill whose behavioral cases need something the runner may not have, such as
a library to build or a package to install, declares it here once. Every engine
applies the same declaration where its agent runs. Routing never does: it runs
no case to completion.

```yaml
os: [Linux]
prerequisites:
  - name: rocprof-trace-decoder
    check: [python3, -c, "import elftools, rocprof_trace_decoder"]
    install: [sh, install-decoder.sh]   # beside this file
    env:
      PYTHONPATH: "${SKILL_CACHE_DIR}/decoder/python"
      ROCPROF_TRACE_DECODER_LIB: "${SKILL_CACHE_DIR}/decoder/lib/librocprof-trace-decoder.so"
```

- **The order is check, then install, then check again.**
  - `check` runs first, without this prerequisite's `env`. If it exits 0, the
    prerequisite is already there: nothing is installed and its `env` is not
    applied, because whatever provides it already says where it is. Write
    `check` so it exits 0 only when the agent would work without this `env`.
    The log names which were satisfied by the environment and which were installed.
  - Otherwise `install` runs, then `check` runs again with `env` applied. If it
    still fails, the skill's cases are reported as failed, with the step's
    output.
  - Prerequisites run in order, and each one's steps see the `env` of the ones
    before it that had to be installed.
- **When it runs depends on the engine.**
  - The default engine runs it once per skill, on the host, before the first
    case.
  - `--engine claude-code` runs it in each case's container, before the agent
    starts, so the container needs whatever `install` uses: network access for
    a download, which the default sandbox does not have, and a user that can
    write to `/opt/skillscope`.
- **Commands are argv lists.** A command may also be a mapping per platform,
  for example `{linux: [sh, x.sh], windows: [pwsh, x.ps1]}`. A container
  always counts as `linux`, and the structural check wants a command for every
  platform in `os`. An `os` label that is not linux, windows, or macos is
  reported too, when this skill has prerequisites: a typo would otherwise skip
  that check. On the host, steps run in this directory. In a container
  they run in a scratch directory holding only the files they name.
- **What a container gets.** Under `claude-code`, an argv item of `check` or
  `install` that is a path to a file in this directory or below it, like
  `install-decoder.sh` above, is copied into the container for the step, at the
  same relative path, and removed before the agent starts. A file mentioned
  only inside a string, as in `sh -c "..."`, is not. A copied file is not
  executable there, so run a script through its interpreter (`[sh, install.sh]`,
  not `[./install.sh]`). A step that names `evals.json` or `extended_evals.json`
  is refused, and the skill's cases fail, rather than carry the answer key into
  the agent's room.
- **No step has a time limit of its own.** On the host only `--timeout` stops
  a hung install. It stops the processes the install started as well, whether
  it stops the run gracefully or kills it outright. In a container the case's
  own time limit does, which `--timeout` sets; without one, nothing does. A
  host step also waits for everything holding its output open, so a step that
  leaves a server running in the background does not finish.
- **Steps see two variables, and `env` values may refer to these two only.**
  - `SKILL_SOURCE_DIR` is the source tree the skill came from, resolved the
    way `sources.resolve` does (below). It is host only: a container has no
    copy of it, and on the host it is unset if no tree can be found. An `env`
    value naming it fails before anything is installed when it is missing.
  - `SKILL_CACHE_DIR` is scratch space for what `install` builds. On the host
    it is removed after the skill's last case. In a container it is
    `/opt/skillscope/cache`. That path is provisional until a skill's image
    is built against it.
- **An `env` value replaces the variable, it does not extend it.** No other
  `$` is expanded. `PATH` cannot be set here: it would replace the agent's
  search path, and everything the agent starts would see only that value.
  Variables the harness sets for the agent (`ANTHROPIC_*`, `CLAUDE_*`,
  `IS_SANDBOX`) cannot be set here either.
- **Installing once per image instead of once per case.** Under `claude-code`,
  build your `sandbox:` image so that `check` passes in it without this
  prerequisite's `env`: run the same `install` at `SKILL_CACHE_DIR=/opt/skillscope/cache`
  and set the same `env` in the image, or install wherever the image's own
  environment already finds it. A source tree the install needs has to be put
  into the image as well.
- **What it sets stays with its skill.** The `env` reaches only its own skill's
  agent, and `SKILL_CACHE_DIR` is removed after the skill. Anything `install`
  writes elsewhere on the host, such as a pip cache, is not cleaned up.
- **Steps are trusted like the rest of the skill.** On the host they run with
  the runner's whole environment, credentials included, as `evals/hooks.py`
  does; in a container they see only the container's. A failing step's output
  ends up in the report.

### `evals/hooks.py`

Setup a dataset cannot express: cloning a repo, tearing down a container,
running an external scoring script. Every function is optional:

```python
def setup(workspace, case, ctx): ...     # before each case
def teardown(workspace, case, ctx): ...  # after it, even if the agent blew up
```

`teardown` runs even when the agent itself blew up: it is wired to inspect's
`Task.cleanup`, which runs inside a `finally` under a shielded cancel scope, so
an exception or a cancelled sample does not skip it. The one exception is
`--timeout` expiring: that kills the process outright, and nothing shielded
survives it. Keep a case inside its own `--case-timeout` if a teardown is
load-bearing.

The three arguments:

| | |
|---|---|
| `workspace` | A fresh per-case directory **on the host**, removed after `teardown`. Yours to write in. |
| `case` | The `Case` being run — `case.id`, `case.prompt`, and the rest of the dataset row. |
| `ctx` | Always `{}`. It carried `setup_session`'s return value, and that entry point is gone. |

`workspace` is on the host under every engine, including `claude-code`, where
the agent itself works inside a container. It is scratch space for the hook's
own use — fetching a source tree, holding a scoring script — not the agent's
room. A hook that needs to put a file where the *agent* will see it should use
[inspect's sandbox API](https://inspect.aisi.org.uk/sandboxing.html), which can
address the guest; writing to `workspace` will not reach it.

**Two entry points are no longer supported**, and a skill that defines either
is refused rather than having it silently skipped:

```python
def setup_session(cache_dir): ...     # NOT supported
def check(run, case, ctx): ...        # NOT supported
```

`setup_session` is replaced by [prerequisites](#prerequisites). It ran once,
on the host, and anything it set reached the agent only through the host's
environment, so a skill that also ran under `claude-code` had to describe its
environment a second time for the container, and the two descriptions
drifted. Its `{placeholder}` return value has no replacement: nothing used it.
Put such values in the dataset instead.

`check` was handed the retired engine's `Run` object. The engines here have
inspect's `TaskState`, which is a different thing with different attributes.
Move what it asserted into the dataset. A `setup` that *returns* placeholders
is refused at runtime, because prompts are built before any hook runs.

Keep prompts and expectations in the dataset even when you use hooks, so what
is being asserted stays readable without opening Python.

A hook that needs the source tree of the repo the skill came from should ask
for it rather than cloning:

```python
from skillscope import sources

def setup(workspace, case, ctx):
    source = sources.resolve("my-skill", workspace)
    # Use source.path here; returning it as a {placeholder} is not supported.
```

`resolve` answers with the checkout that matches the skill under test: an
explicit `$SKILL_SOURCE_DIR`, else the commit a vendored skill records in its
`.federated.json`, else the git repository the skill folder sits in. That last
one is the reason not to clone by hand — on a pull request it is the merge
commit, so the eval grades the change under review instead of whatever `main`
holds.

## Running them

```bash
skillscope structural                               # your folder, prose, and dataset
skillscope behavioral --skill <your-skill>          # your skill, end to end
skillscope routing --routing-room <your-skill>,<a-neighbour>
skillscope routing --only <case-id> --keep-logs logs # one case, keeping the transcript
```

`structural` is instant and needs no agent. Everything else needs the `claude`
CLI authenticated, plus whatever your own cases need. Flags, what each check
asserts, the routing bar, and how to read a report are in
[usage.md](usage.md).
