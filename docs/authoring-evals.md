# Writing evals for a skill

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

Your dataset's prompts are graded in a routing run only when your skill is one
the workflow lists in `routing_skills`. A prompt expecting a skill that is not
in the room could only ever lose, and a near miss says nothing about skills
that were never installed.

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

## Pinning the harness

A dataset may name the build of skillscope that grades it:

```json
{
  "skillscope_version": "v1.2.0",
  "evaluations": ["..."]
}
```

Anything git can resolve — a tag, a branch, a commit. It governs **this skill's
behavior run**, and it exists so the version that runs your prompts is bumped
in the same file, and the same review, as the prompts themselves.

It does not govern routing, which installs several skills in one session and so
cannot honor several pins at once; that runs at the `version` the workflow asks
for. Leave the key out and your behavior run uses that version too, which is
the right answer for most skills.

## When JSON is not enough

Two optional files sit beside the dataset.

### `evals/machine.yml`

Only for a skill whose behavior cases cannot run on the everyday runners on
every platform. Both keys are optional:

```yaml
os: [Linux]                 # defaults to whatever platforms the repo runs on
labels: [mi300x, gpu]       # hardware this skill's cases need
```

Say what the machine must have; what that *costs* is not yours to declare. The
base labels, the pull-request label rationing a scarce pool, and the
credentials that pay for it are set by the repo that owns the machines, in the
workflow that calls skillscope. Asking for any label is what moves the leg onto
that pool.

Most skills that need this file need only `os: [Linux]`, to drop a Windows leg
that would just exercise the failure path of Linux-only tooling.

### `evals/hooks.py`

Setup a dataset cannot express: cloning a repo, tearing down a container,
running an external scoring script. Every function is optional:

```python
def setup_session(cache_dir): ...     # once per skill; returns {name: value} for {placeholders} in prompts
def setup(workspace, case, ctx): ...  # before each case; may return more placeholders
def teardown(workspace, case, ctx): ...
def check(run, case, ctx): ...        # after each case; raise AssertionError to fail it
```

`teardown` runs even when the agent itself blew up, and a `check` that raises
fails its case without killing the run.

Keep prompts and expectations in the dataset even when you use hooks, so what
is being asserted stays readable without opening Python.

A hook that needs the source tree of the repo the skill came from should ask
for it rather than cloning:

```python
from skillscope import sources

def setup_session(cache_dir):
    source = sources.resolve("my-skill", cache_dir)
    return {"repo": source.path}
```

`resolve` answers with the checkout that matches the skill under test: an
explicit `$SKILL_SOURCE_DIR`, else the commit a vendored skill records in its
`.federated.json`, else the git repository the skill folder sits in. That last
one is the reason not to clone by hand — on a pull request it is the merge
commit, so the eval grades the change under review instead of whatever `main`
holds.

## Running them

```bash
skillscope structural                               # structure only: no agent, no tokens, instant
skillscope run --mode behavior --skill <your-skill> # your skill, end to end
skillscope run --mode routing --routing-skills <your-skill>,<a-neighbour>
skillscope run --only <case-id> --keep-logs logs    # one case, keeping the transcript
```

Everything but `structural` needs the `claude` CLI authenticated, plus whatever
your own cases need.

Two failure modes are worth knowing before you read a report. A routing case
that ends without the agent either activating a skill or answering is reported
as an **error**, not a missed trigger: grading it would invent a result out of
an infrastructure problem. And if the runner has its own skills installed
(usually `~/.claude/skills`), they join the room for every case and the report
says so — set `ANTHROPIC_API_KEY` so the run can use an isolated config dir.
