<!--
Copyright Advanced Micro Devices, Inc.

SPDX-License-Identifier: MIT
-->

# Using skillscope

Everything the [README](../README.md) leaves out: what each check asserts, every
CLI flag, and every workflow input. For writing a skill's dataset, see
[authoring-evals.md](authoring-evals.md).

## Contents

* [Configuring the repo under test](#configuring-the-repo-under-test)
* [What the structural check asserts](#what-the-structural-check-asserts)
* [References](#references)
* [Routing, and the bar it is held to](#routing-and-the-bar-it-is-held-to)
* [Hardware a skill needs](#hardware-a-skill-needs)
* [In CI: one job](#in-ci-one-job)
* [In CI: the full pipeline](#in-ci-the-full-pipeline)
* [Versions and pinning](#versions-and-pinning)
* [Hand tools](#hand-tools)

## Configuring the repo under test

There is no config file. Everything about your repo is a CLI flag, or the same
thing as a workflow input. A repo that runs these evals already has a workflow
saying when to run them and with which credentials; putting the other half of
that decision in a file at the repo root means two places to read and two
places to disagree.

| Flag | Workflow input | Default | What it decides |
| --- | --- | --- | --- |
| `--skills` | `skill_globs` | `skills/*` | Globs naming the directories that hold skills. A skill is a directory with a `SKILL.md`, and its directory name is its identity. |
| `--routing-skills` | `routing_skills` | the only skill, if there is one | The skills a routing run installs side by side. `all` means every skill with a dataset, `none` means no routing run, and blank means a repo with one skill runs that skill while a repo with several has to choose. |
| `--infra-paths` | `infra_paths` | none | Paths that change the harness rather than one skill, so touching one re-runs every skill instead of guessing at the blast radius. Your own workflow file belongs here. |
| `--docs` | `doc_globs` | none | Markdown outside the skills whose references should be checked too: a README, a docs tree. The skills themselves are always checked. |
| `--exclude-url` | `excluded_urls` | none | Regexes matching URLs the external reference check leaves alone. For hosts that are auth-gated or that answer a runner's IP with a 403. |
| `--skill-files` | `skill_files` | none | Files every skill must ship beside its `SKILL.md`, such as a governance card. |
| `--skill-sections` | `skill_sections` | none | `##` headings each of those markdown files must have something under, such as `Description,Owner,License`. |
| `--behavior-runner` | `behavior_runner` | `["ubuntu-latest"]` | `runs-on` labels for a behavioral leg. |
| `--behavior-os` | `behavior_os` | `Linux` | Platforms a skill runs on when its `machine.yml` does not narrow them. |
| `--scoped-runner` | `scoped_runner` | reuses `behavior_runner` | Base labels for a leg whose skill asks for hardware labels. |
| `--scoped-gate` | `scoped_gate` | none | Pull-request label required before those legs run. |
| `--scoped-environment` | `scoped_environment` | none | GitHub environment holding their credentials. |

Beyond those, the graded commands take `--skill` and `--only` to narrow a run to
some skills or some case ids, `--no-extended` to skip a skill's optional
`extended_evals.json`, `--model` and `--effort` to choose what grades it,
`--output` and `--summary` to place the reports, and `--timeout` to bound the
whole command. Routing adds `--jobs`, `--case-timeout`, `--max-tool-calls`,
`--max-budget-usd`, `--keep-logs`, and `--min-accuracy`. `--help` is the
authority on all of them.

### Who a skill competes against is listed, not inferred

`routing_skills` is the one input with no useful default, because the answer is
what the score *means*. Install every skill on disk and a work-in-progress
directory drops everyone's number; install only the skill under review and it
wins every prompt by walkover. A skill you leave off the list still gets its
dataset checked and its behavioral cases run — it just does not move anybody's
routing score. Listing them also makes the change visible: a skill joining or
leaving the room moves every other skill's number, and that deserves a diff.

A repo with one skill has no such choice, so leave `routing_skills` blank and
its only skill is the room. The score is then the half of the question that can
be answered alone — does the skill fire on its own prompts, and does it stay
quiet on its near misses and the shared negatives — and it stops meaning that
the moment a second skill shows up, at which point the flag becomes required
again rather than quietly picking a room for you. To turn routing off instead,
say so: `routing_skills: none`.

## What the structural check asserts

The Agent Skills format is small — a folder, a `SKILL.md`, and a frontmatter
block naming the skill and saying when to use it — and every part of it fails
quietly. A `name` that disagrees with the folder makes the dataset, the routing
verdict, and the report about a skill that does not exist. A missing
`description` leaves an agent nothing to match a prompt against. Frontmatter
that is not valid YAML stops the file loading at all, and what you see is an
agent that simply never uses the skill. So every `SKILL.md` is read first:

| What | Bar |
| --- | --- |
| frontmatter | opens the file, is valid YAML, and is a mapping |
| `name` | non-empty, at most 64 characters, lowercase-with-hyphens, free of `anthropic` and `claude`, and equal to the folder name |
| `description` | non-empty, at most 1024 characters |
| body | at most 500 lines — past that it is reference material, and an agent reads it in full every time the skill loads |

A directory your skill globs match that holds no `SKILL.md` is reported too.
Either the file is missing or the glob is too wide, and both are worth one line.

Whatever else your repo asks of a skill is policy rather than format, so it is
configuration:

```yaml
skill_files: skill-card.md
skill_sections: Description,Owner,License
```

No manifest is read. Which skills a repo publishes, and where it lists them, is
that repo's business — a harness with an opinion about it would be a second
place to update every time a skill ships.

## References

A skill is prose an agent reads and then acts on. A link to a reference file
that was renamed, or an anchor into a section that was retitled, is not
cosmetic: the agent follows it, finds nothing, and improvises. So the structural
check reads every markdown file under every skill — plus whatever `doc_globs`
names — and resolves what it finds.

| Check | What it reads | When |
| --- | --- | --- |
| internal | relative paths, root-relative paths, and heading anchors, against the files on disk | always, and a paid run waits on it |
| external | every `http(s)` URL, fetched | only with `--external`, or the `external_references` job |

The split is by failure mode, not by effort. The internal half is deterministic
and offline, so it can gate a merge and a token spend without ever being wrong
for a reason of its own. The external half fetches other people's servers,
which fail for reasons that have nothing to do with the change under review: a
rate limit, a runner IP a host blocks, DNS having a bad minute. It runs as its
own job that the aggregate result ignores, so link rot is visible without a bad
minute somewhere else holding up a merge. Point a schedule at the workflow to
catch rot in markdown nobody is editing.

A 2xx answer is reachable, and so is a 429 — that is a host saying "you again",
which is a fact about the run rather than the link. `HEAD` is asked first and
asked again as a `GET` whenever it does not come back with one, since plenty of
hosts refuse the method or ignore it. Hosts are fetched from in parallel, one
request at a time each: a checker that opens twenty connections to the same
documentation site gets itself throttled, and a throttled request is
indistinguishable from link rot.

Links inside fenced code blocks, inline code spans, and HTML comments are left
alone — those are illustrations of links rather than promises.

## Routing, and the bar it is held to

`--min-accuracy` / `min_accuracy` defaults to `1`: every graded routing case has
to land on the right skill, the same way every behavioral expectation has to
hold. A routing miss is a defect and not a statistic — a description that fires
on its neighbour's prompt makes that neighbour worse — so it turns the run red
rather than sitting in a summary nobody reads.

Loosening it is a one-line diff. `0` reports the score without gating on it,
which is where a repo that has never measured its prompts should start; anything
in between holds a bar short of perfect. The bar is over the cases that were
*graded*: a case whose own run errored counts towards neither side of it,
because a timeout says nothing about routing. No value turns off the
infrastructure checks — a run where nothing was graded, or where no skill
activated anywhere, fails at any bar, since those numbers are an artifact
rather than a result.

Two failure modes are worth knowing before reading a report. A routing case that
ends without the agent either activating a skill or answering is reported as an
**error** rather than a missed trigger. And if the runner has its own skills
installed (usually `~/.claude/skills`), they join the room for every case and
the report says so — set `ANTHROPIC_API_KEY` so the run can use an isolated
config dir.

## Hardware a skill needs

A skill says what it needs in `evals/machine.yml`, beside the prompts that need
it:

```yaml
os: [Linux]
labels: [mi300x, gpu, rocm]
```

Those labels are added to `scoped_runner`, and asking for any of them is what
makes a leg *scoped*: it lands on the pool your workflow rations with
`scoped_gate` and pays for out of `scoped_environment`. The split is
deliberate. The person who knows a skill needs a GPU is its owner; the person
who knows which pool has one, who may spend it, and whose key pays for it is
whoever runs the repo. A central table mapping skills to runners drifts from
reality the first time a skill is added.

Legs with a scoped environment run as a separate job, because a job's
credentials are fixed before its matrix expands. A repo that declares no scoped
environment gets one matrix, labels and all.

## In CI: one job

[`reusable.yml`](../.github/workflows/reusable.yml) grades a repo's skills with
one runner per skill:

```yaml
jobs:
  evals:
    uses: danielholanda/skillscope/.github/workflows/reusable.yml@main
    secrets:
      api_key: ${{ secrets.ANTHROPIC_API_KEY }}
    with:
      skills: path/to/my-skill
```

Map your model key onto `api_key`; this workflow never sees the rest of the
vault, and never learns what you named the secret. `secrets: inherit` still
works if you would rather pass the vault and name the key with
`api_key_secret`.

All three graders run, all three can fail the run, and every skill named gets a
runner of its own: the structural checks first, then a routing leg and a
behavioral leg per skill, all at once. Ten skills is twenty paid legs running in
parallel, each with its own check and its own report.

Naming several, and holding them to different bars:

```yaml
    with:
      skills: |
        path/to/my-skill
        path/to/another-skill
      routing: optional     # graded and reported, but cannot fail the run
      behavioral: off       # not run at all
```

| Input | Default | What it decides |
| --- | --- | --- |
| `skills` | `skills/*` | The skills to grade: a directory, or a glob matching several, one per line or comma-separated. |
| `structural` | `required` | `required`, `optional`, or `off`. |
| `routing` | `required` | `required`, `optional`, or `off`. |
| `behavioral` | `required` | `required`, `optional`, or `off`. |
| `runner` | `ubuntu-latest` | `runs-on` for every job: one label, or a JSON array of them. |
| `min_accuracy` | `1` | The routing bar. `0` reports the score without gating on it. |
| `version` | the skills' own pins | The build of the harness that grades this repo. |
| `api_key` | (none) | The model API key, mapped from the caller's vault. One secret, not the whole set. |
| `api_key_secret` | `ANTHROPIC_API_KEY` | Name to look up under `secrets: inherit`, if you would rather pass the vault than map one key. |

`optional` is for a bar you have not met yet: the leg runs, the report lands in
the step summary, and a red leg leaves the run green. `off` does not run it at
all. Both are one-word diffs a reviewer can see, which is the point — a step
nobody can see being dropped is a step nobody notices is gone.

Each routing leg here installs one skill, which answers the half of the routing
question a skill can be asked alone. For the other half — whether a skill
answers a prompt that belongs to its neighbour — the neighbour has to be in the
room, and that is the pipeline below.

[`examples/amd-skills-checks.yml`](../examples/amd-skills-checks.yml) is a
caller with its triggers filled in.

## In CI: the full pipeline

[`skill-evals.yml`](../.github/workflows/skill-evals.yml) is the same three
graders with three more decisions on top: routing pools the listed skills'
datasets into one run, so each skill's prompts are the others' negatives; a pull
request grades only what it changed; and a skill that asks for particular
hardware in its `evals/machine.yml` lands on the pool your workflow rations and
pays for.

```yaml
jobs:
  skill-evals:
    uses: danielholanda/skillscope/.github/workflows/skill-evals.yml@main
    secrets: inherit
    with:
      routing_skills: my-skill,its-neighbour
      api_key_secret: MY_MODEL_API_KEY
```

Reach for it when skills in the repo could plausibly be confused for one
another, or when a behavioral run needs a GPU. Its inputs are the table in
[Configuring the repo under test](#configuring-the-repo-under-test), and the
workflow file documents every one.

To run a single command instead of a pipeline, use the action directly:

```yaml
- uses: danielholanda/skillscope@main
  with:
    command: structural
```

Deciding what to run by hand is also possible: `select` emits the plan for a
change as JSON.

```bash
git diff --name-only main HEAD | skillscope select --changed
```

## Versions and pinning

The action's ref is a contract, not a release: it only resolves a version and
execs it, and it imports nothing from the harness, so it cannot break on a
payload version it predates. The version that actually grades your skills is
data:

| Where | Scope |
| --- | --- |
| the `version` input on the action or workflow | the repo, and everything that is not one skill's behavioral run |
| `skillscope_version` in a skill's `evals/evals.json` | that skill's behavioral run, overriding the input |

Both are one-line diffs a reviewer can see, which a `uses:` ref spread across
every caller is not. Routing always runs at the workflow's version: it installs
several skills in one session and so cannot honor several pins at once.

## Hand tools

`tools/` holds tools that are no part of the graded pipeline:
`claude_eval.py` (what did this one prompt cost?), `compare_skill.py` (the same
prompt with and without a skill, side by side), and
`verify_selection_parity.py` (does `select` still plan what `amd/skills` planned
before the harness moved out of it?).
