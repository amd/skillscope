<!--
Copyright Advanced Micro Devices, Inc.

SPDX-License-Identifier: MIT
-->

# What a skill has to do to pass

Skillscope grades a skill three ways. This page lists what each one wants from
you, in order.

1. **Structural** — are the files the right shape? No agent, no cost, seconds.
2. **Routing** — when someone types a prompt, does the right skill switch on?
3. **Behavioral** — once it is on, does it do the job?

Run `skillscope structural` first. It checks everything in section 1 for free.
Routing and behavioral both run those same checks before they spend anything,
so while section 1 is red, nothing else will run at all.

Why each rule exists is in [usage.md](usage.md). How to write the dataset is in
[authoring-evals.md](authoring-evals.md).

## 1. Structural

Reads your files and checks their shape. It never starts an agent, so it
cannot tell you whether the skill is any good — only whether it is put
together correctly enough to be graded.

### `SKILL.md`

This file *is* the skill. The block at the top, between the `---` lines, is
how an agent learns the skill exists and when to use it. If that block is
broken, the skill is never loaded, and all you see is an agent that ignores it.

* [ ] Sits in a directory that `skill_globs` matches, one level down.
* [ ] Starts with a `---` line, and that block is closed by another `---` on
      its own line.
* [ ] What is between them is valid YAML, and a set of key/value pairs.

| Field | What it has to be |
| --- | --- |
| `name` | not empty, 64 characters or fewer, `lowercase-with-hyphens` |
| `name` | no `anthropic` and no `claude` in it; the runtime reserves both |
| `name` | the same as the folder it sits in |
| `description` | not empty, 1024 characters or fewer |
| body | 500 lines or fewer; put the rest in files the body links to |

The body limit is about cost: an agent reads the whole body every time the
skill loads, but only opens a linked file when it needs it.

### `evals/evals.json`

One file per skill, holding the prompts you are graded on. Each entry is a
prompt plus a yes/no answer: should *this* skill switch on here? You write
both kinds. The prompts where it should not fire are as important as the ones
where it should.

* [ ] The file exists. `skillscope template` prints a starting point.
* [ ] The top level is an object, and `evaluations` and `comment` are the only
      keys allowed in it.
* [ ] `evaluations` is not empty, and each entry has `id`, `prompt` and
      `skill_should_trigger`.
* [ ] Every `id` is different from every other id **in the whole repo**, not
      just in this file. Routing puts all the skills' prompts in one run, so
      the ids meet.
* [ ] At least 3 entries say `skill_should_trigger: true` and at least 2 say
      `false`. Only this file counts; `extended_evals.json` does not.
* [ ] If an entry has a `workspace`, that directory exists under the skill.

Which keys an entry may use depends on its own `skill_should_trigger`:

* **`true`** — `id`, `prompt`, `skill_should_trigger` and `note`, plus any of
  `expected_behavior`, `unexpected_behavior`, `logs_contain`, `files_exist`
  and `workspace`.
* **`false`** — those first four and nothing else. No skill is loaded for such
  a prompt, so there is nothing to make claims about. Adding one of the other
  keys is an error, not something quietly skipped.

### Links

Your skill is text an agent reads and then acts on. If it follows a link to a
file that was renamed, it finds nothing and makes something up instead. So
every link in every markdown file under the skill is resolved.

* [ ] Relative paths, paths from the repo root, and `#heading` anchors all
      point at something that exists.
* [ ] Web links answer when you pass `--external`. A normal 2xx answer counts,
      and so does 429, which just means the site is telling you to slow down.

Links inside code blocks, `code spans` and HTML comments are skipped. Those
are examples of links, not real ones.

### Extras your repo may ask for

Nothing here applies unless your repo turns it on.

* [ ] Every file listed in `skill_files` sits next to `SKILL.md` — a
      governance card, say.
* [ ] Every `##` heading listed in `skill_sections` is in those files, with
      something written under it.

### `evals/machine.yml`

Only for a skill that cannot run on an ordinary runner. Most skills have no
such file, and that is the normal case.

* [ ] It is a YAML mapping, and `os`, `labels` and `sandbox` are the only keys.
* [ ] `labels` says what hardware the work needs, like `mi300x`. Naming any
      label moves these runs onto the special pool your repo keeps for them.
      Keep the list short: a label no machine has means a job that waits
      forever instead of failing.
* [ ] `sandbox` names a compose file sitting beside the `machine.yml`. You
      need it as well as `labels` if the skill needs network access or a
      device. The labels choose the machine; the compose file decides what the
      container running on it can reach.

## 2. Routing

Installs the skills, sends each prompt, and watches which skill switches on,
if any. It stops right there. What the skill does afterwards is section 3's
question.

The pass mark is `min_accuracy`, and it defaults to `1`: every graded prompt
has to come out right. A skill that answers its neighbour's prompts makes that
neighbour worse, so this is treated as a bug rather than a statistic.

* [ ] Your skill switches on for all of its own `skill_should_trigger: true`
      prompts.
* [ ] It stays quiet for its own `false` prompts, for every other skill's
      prompts, and for the shared pool of prompts that belong to no skill.
* [ ] `routing_room` says which skills to install together. There is no good
      default: install everything on disk and one unfinished skill drags down
      everybody's score, install only your own and it wins every prompt
      because nothing else is there to win. A repo with one skill can leave it
      blank. `none` switches routing off.
* [ ] Something was actually graded, and at least one skill switched on
      somewhere, if any prompt expected one. These two checks fire at any pass
      mark, including `0`.
* [ ] `ANTHROPIC_API_KEY` or `ANTHROPIC_AUTH_TOKEN` is set. Without it the
      host engine refuses to run routing at all, because that key is what lets
      it point the CLI away from whatever skills are installed on the runner
      itself. Those would otherwise join the room and change every answer.

Two things to know when reading a red report. Routing gives the agent the
skills and nothing else: any `workspace` on the entry is ignored here, and the
prompt is sent word for word with no `{}` filled in. Both only work in section
3, so a prompt that relies on either is not testing what it looks like. And a
prompt that ends in an **error** (the agent neither switched a skill on nor
answered) or **degraded** (the model provider failed) is left out of the score
entirely, counting for neither side.

## 3. Behavioral

Installs one skill on its own, runs the prompt all the way through, then
checks what actually happened. Only `skill_should_trigger: true` entries that
claim something run here. An entry that claims nothing was only ever a routing
test.

An entry passes when **every** claim on it passes.

| You write | How it is checked | So write it as |
| --- | --- | --- |
| `expected_behavior` | a model reads what the agent did | something visible in the agent's actions |
| `unexpected_behavior` | the same, in reverse | one thing the agent must not do |
| `logs_contain` | plain text search, ignoring case, across everything in the run including tool calls and their arguments | exact text: a script name, a flag, a pinned tag. Never the skill's own name, which routing already covers |
| `files_exist` | whole path pieces, matched anywhere in the workspace | the file itself, not the folder you hope for. `plan.md` is matched by `examples/plan.md` |

The model doing the judging is shown what the agent did, not what the agent
said about it. An agent claiming it avoided something is not evidence.

* [ ] Anything that must exist before the prompt runs is put there by
      `workspace`, or by `evals/hooks.py` when a JSON file cannot express it.

## 4. In CI

Two workflows will run these graders for you. They are not interchangeable:
each accepts its own set of inputs, and passing one of the other's fails.

### `reusable.yml` — the one to start with

Grades the skills you name, on one runner each.

```yaml
jobs:
  evals:
    uses: amd/skillscope/.github/workflows/reusable.yml@v0.1.3
    secrets:
      api_key: ${{ secrets.ANTHROPIC_API_KEY }}
    with:
      skills: skills/*
```

It accepts these inputs and no others: `skills`, `structural`, `routing`,
`behavioral`, `runner`, `min_accuracy`, `api_key`, `api_key_secret`,
`api_base_url`, `api_custom_headers`.

* [ ] `skills` names what to grade: one directory, or a glob matching several.
* [ ] You know its routing runs install one skill at a time. That shows whether
      a skill fires on its own prompts and stays quiet on its own negatives. It
      cannot show whether the skill steals a neighbour's prompt, because the
      neighbour is never in the room.

### `skill-evals.yml` — when that is not enough

Reach for it when two skills in the repo could be mistaken for one another, or
when a skill's behavioral cases need particular hardware.

```yaml
jobs:
  skill-evals:
    uses: amd/skillscope/.github/workflows/skill-evals.yml@v0.1.3
    secrets: inherit
    with:
      routing_room: my-skill,its-neighbour
```

* [ ] `routing_room` lists the skills to install side by side, which is the
      only way to catch one stealing another's prompts. `reusable.yml` has no
      such input, and that is the usual reason to move here.
* [ ] `infra_paths` lists files that affect every skill rather than one — your
      own workflow file, say — so changing one re-grades the whole catalogue
      instead of guessing what it touched.
* [ ] `scoped_runner`, `scoped_gate` and `scoped_environment` are set if any
      skill's `machine.yml` asks for `labels`. They decide which machines those
      jobs land on, which pull-request label unlocks them, and which
      environment's credentials pay for them.

### Whichever you use

* [ ] The `uses:` line ends in a version tag (`@v0.1.3`). That tag *is* the
      harness — it decides which version grades your skills.
* [ ] The model key reaches the run. Either map one secret in as `api_key`,
      as above, or pass the whole vault with `secrets: inherit` and name the
      key you want with `api_key_secret`.
* [ ] A key issued by an API gateway rather than by the model provider also
      needs `api_base_url`, and usually `api_custom_headers` carrying the key
      as the literal text `$API_KEY`. GitHub masks only values it was handed
      as secrets, so writing the key itself there publishes it to the logs.
* [ ] A grader you are not ready to be judged on is set to `optional` rather
      than removed: it still runs and still reports, but a red result leaves
      the build green. `off` does not run it at all.

## Running them

```bash
skillscope structural                        # section 1, free
skillscope structural --external             # the same, plus checking web links
skillscope routing --routing-room my-skill,its-neighbour
skillscope behavioral --skill my-skill
```

Reports are printed, written to `$GITHUB_STEP_SUMMARY` under Actions, and
saved as JSON under `.skillscope/runs/`. `--help` lists every flag.
