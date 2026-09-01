<!--
Copyright Advanced Micro Devices, Inc.

SPDX-License-Identifier: MIT
-->

# skillscope

Structural, routing, and behavioral tests for agent skills, in whatever repo the
skills live in.

A skill is a description plus a body, and each half fails in a way the other
cannot catch. A description that never fires makes the body irrelevant; a
description that fires on its neighbour's work makes every skill around it
worse.
And a skill that routes perfectly can still do the job badly once it runs.
skillscope grades all three, from one dataset per skill and the skill's own
prose:

* **structural** — every skill folder, every dataset, and every reference the
  skill's markdown makes. No agent, no tokens, instant, and so it runs on every
  change and gates the two below.
* **routing** — installs several skills side by side and asks which one wins a
  prompt. You cannot test this alone: a skill tested by itself will happily
  answer prompts that belong to its neighbour.
* **behavioral** — installs one skill, runs the prompt to completion, and grades
  what the agent actually did.

The prompt is written once and both graded commands read it. The alternative — a
central routing prompt set plus a per-skill test file that re-asserts routing
with a substring match on the transcript — is two things to maintain and one of
them is a worse copy of the other.

## Quick start

```bash
# structure only: no agent, no tokens, instant
uvx --from git+https://github.com/amd/skillscope skillscope structural

# the same, plus fetching every external URL the skills link to
uvx --from git+https://github.com/amd/skillscope skillscope structural --external

# behavioral, for one skill (needs an authenticated `claude` CLI)
uvx --from git+https://github.com/amd/skillscope skillscope behavioral --skill my-skill

# routing, for the skills you want in the room together
uvx --from git+https://github.com/amd/skillscope skillscope routing \
  --routing-skills my-skill,its-neighbour

# a repo with one skill: the room is that skill, so nothing names it
uvx --from git+https://github.com/amd/skillscope skillscope routing

# what CI should run for a change
git diff --name-only main HEAD | uvx --from git+https://github.com/amd/skillscope skillscope select --changed
```

Run from the root of the repo you want tested. A repo that keeps its skills in
`skills/*` and has a `SKILL.md` in each passes no flags at all. A second skill
is what makes one necessary: which skills a routing run should install together
has no sensible default once there is a choice, and
[why](#who-a-skill-competes-against-is-listed-not-inferred).

Writing the datasets themselves: [docs/authoring-evals.md](docs/authoring-evals.md).

## In CI

One job, in the calling repo, naming the skills to grade:

```yaml
jobs:
  evals:
    uses: danielholanda/skillscope/.github/workflows/reusable.yml@main
    secrets:
      api_key: ${{ secrets.ANTHROPIC_API_KEY }}
    with:
      skills: path/to/my-skill
```

That is a whole caller. Map your model key onto `api_key`; this workflow never
sees the rest of the vault, and never learns what you named the secret.
`secrets: inherit` still works if you would rather pass the vault and name the
key with `api_key_secret`. All three graders run, all three can fail the run,
and every skill named gets a runner of its own: the structural checks first,
then a routing leg and a behavioral leg per skill, all at once. Ten skills is
twenty paid legs running in parallel, each with its own check and its own report.

Naming several, and holding them to different bars:

```yaml
    with:
      skills: |
        path/to/my-skill
        path/to/another-skill
      routing: optional     # graded and reported, but cannot fail the run
      behavioral: off       # not run at all
```

| Setting | Default | What it decides |
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

Each routing leg installs one skill, which answers the half of the routing
question a skill can be asked alone: does it fire on its own prompts, and does
it stay quiet on its near misses and on the shared negatives? For the other half
— whether a skill answers a prompt that belongs to its neighbour — the neighbour
has to be in the room, and that is the pipeline below.

[examples/evals.yml](examples/evals.yml) is a caller with its triggers filled
in; [.github/workflows/reusable.yml](.github/workflows/reusable.yml) is what it
runs.

### The full pipeline

```yaml
jobs:
  skill-evals:
    uses: amd/skillscope/.github/workflows/skill-evals.yml@bootstrap
    secrets: inherit
    with:
      routing_skills: my-skill,its-neighbour
      api_key_secret: MY_MODEL_API_KEY
```

Same three graders, three more decisions on top: routing pools the listed
skills' datasets into one run, so each skill's prompts are the others'
negatives; a pull request grades only what it changed; and a skill that asks for
particular hardware in its `evals/machine.yml` lands on the pool your workflow
rations and pays for. Reach for it when skills in the repo could plausibly be
confused for one another, or when a behavioral run needs a GPU. See
[examples/skill-evals.yml](examples/skill-evals.yml) for a fully configured
caller and
[.github/workflows/skill-evals.yml](.github/workflows/skill-evals.yml) for every
input.

To run a single command instead of a pipeline, use the action directly:

```yaml
- uses: amd/skillscope@bootstrap
  with:
    command: structural
```

### A wrong routing decision fails the run

`min_accuracy` (`--min-accuracy`) defaults to `1`: every graded routing case
has to land on the right skill, the same way every behavioral expectation has to
hold. A routing miss is a defect and not a statistic — a description that fires
on its neighbour's prompt makes that neighbour worse — so it turns the run red
rather than sitting in a summary nobody reads.

Loosening it is a one-line diff a reviewer can see. `min_accuracy: "0"` reports
the score without gating on it, which is where a repo that has never measured
its prompts should start; anything in between holds a bar short of perfect. The
bar is over the cases that were *graded*: a case whose own run errored counts
towards neither side of it, because a timeout says nothing about routing. No
value turns off the infrastructure checks — a run where nothing was graded, or
where no skill activated anywhere, fails at any bar, since those numbers are an
artifact rather than a result.

### Pin `@bootstrap` and leave it there

`bootstrap` is a contract, not a release. The action behind it only resolves a
version and execs it, and it imports nothing from the harness, so it cannot
break on a payload version it predates.

The version that actually grades your skills is data:

| Where | Scope |
| --- | --- |
| the `version` input on the action or workflow | the repo, and everything that is not one skill's behavioral run |
| `skillscope_version` in a skill's `evals/evals.json` | that skill's behavioral run, overriding the input |

Both are one-line diffs a reviewer can see, which a `uses:` ref spread across
every caller is not. Routing always runs at the workflow's version: it installs
several skills in one session and so cannot honor several pins at once.

## Configuring the repo under test

There is no config file. Everything about your repo is a workflow input, or a
flag if you are running the CLI by hand. The full pipeline reads all of them;
`reusable.yml` reads the short list above and leaves the rest at their defaults:

| Input | Flag | Default | What it decides |
| --- | --- | --- | --- |
| `skill_globs` | `--skills` | `skills/*` | Globs naming the directories that hold skills. A skill is a directory with a `SKILL.md`, and its directory name is its identity. |
| `routing_skills` | `--routing-skills` | the only skill, if there is one | The skills a routing run installs side by side. `all` means every skill with a dataset, `none` means no routing run, and blank means a repo with one skill runs that skill while a repo with several has to choose. |
| `infra_paths` | `--infra-paths` | none | Paths that change the harness rather than one skill, so touching one re-runs every skill instead of guessing at the blast radius. Your own workflow file belongs here. |
| `doc_globs` | `--docs` | none | Markdown outside the skills whose references should be checked too: a README, a docs tree. The skills themselves are always checked. |
| `excluded_urls` | `--exclude-url` | none | Regexes matching URLs the external reference check leaves alone. For hosts that are auth-gated or that answer a runner's IP with a 403. |
| `skill_files` | `--skill-files` | none | Files every skill must ship beside its `SKILL.md`, such as a governance card. What the format itself requires is checked either way. |
| `skill_sections` | `--skill-sections` | none | `##` headings each of those markdown files must have something under, such as `Description,Owner,License`. |
| `behavior_runner` | `--behavior-runner` | `["ubuntu-latest"]` | `runs-on` labels for a behavioral leg. |
| `behavior_os` | `--behavior-os` | `Linux` | Platforms a skill runs on when its `machine.yml` does not narrow them. |
| `scoped_runner` | `--scoped-runner` | reuses `behavior_runner` | Base labels for a leg whose skill asks for hardware labels. |
| `scoped_gate` | `--scoped-gate` | none | Pull-request label required before those legs run. |
| `scoped_environment` | `--scoped-environment` | none | GitHub environment holding their credentials. |

A repo that runs these evals already has a workflow saying when to run them, on
what, and with which credentials. Putting the other half of the same decision
in a second file at the repo root means two places to read, two places to
change, and a file whose only reader is the workflow next to it.

### Who a skill competes against is listed, not inferred

Wherever there is a choice, `routing_skills` is the one input with no useful
default, because the answer is what the score *means*. Install every skill on
disk and a work-in-progress directory drops everyone's number; install only the
skill under review and it wins every prompt by walkover. Neither is a guess a
test should make on your behalf. A skill you leave off the list still gets its
dataset checked and its behavioral cases run — it just does not move anybody's
routing score.

Listing them also makes the change visible: a skill joining or leaving the room
moves every other skill's number, and that deserves a diff a reviewer can see.

A repo with one skill has no such choice, so it does not have to make one:
leave `routing_skills` blank and its only skill is the room. The score is the
half of the question that can be answered alone — does the skill fire on its
own prompts, and does it stay quiet on its near misses and the shared
negatives — and it stops meaning that the moment a second skill shows up, at
which point the flag becomes required again rather than quietly picking a room
for you. To turn routing off instead, say so: `routing_skills: none`.

### A skill that never loads fails here, not in a paid run

The standardized Agent Skills format is small — a folder, a `SKILL.md`, and a
frontmatter block naming the skill and saying when to use it — and every part
of it fails quietly. A `name` that disagrees with the folder makes the dataset,
the routing verdict, and the report about a skill that does not exist. A
missing `description` leaves an agent nothing to match a prompt against. A
frontmatter block that is not valid YAML stops the file loading at all, and
what you see is an agent that simply never uses the skill. So every `SKILL.md`
is read first:

| What | Bar |
| --- | --- |
| frontmatter | opens the file, is valid YAML, and is a mapping |
| `name` | non-empty, at most 64 characters, lowercase-with-hyphens, free of `anthropic` and `claude`, and equal to the folder name |
| `description` | non-empty, at most 1024 characters |
| body | at most 500 lines — past that it is reference material, and an agent reads it in full every time the skill loads |

A directory your `skill_globs` match that holds no `SKILL.md` is reported too.
It is not a skill, so nothing grades it, routes it, or reports on it; either
the file is missing or the glob is too wide, and both are worth one line.

Whatever else your repo asks of a skill is policy rather than format, so it is
configuration. `skill_files` names the files every skill ships beside its
`SKILL.md`, and `skill_sections` names the `##` headings each of those markdown
files has to have something under:

```yaml
skill_files: skill-card.md
skill_sections: Description,Owner,License
```

No manifest is read. Which skills a repo publishes, and where it lists them, is
that repo's business — a harness with an opinion about it would be a second
place to update every time a skill ships.

### A reference that goes nowhere is a defect, not a typo

A skill is prose an agent reads and then acts on. A link to a reference file
that was renamed, or an anchor into a section that was retitled, is not
cosmetic: the agent follows it, finds nothing, and improvises. So the
structural check reads every markdown file under every skill — plus whatever
`doc_globs` names — and resolves what it finds.

| Check | What it reads | When |
| --- | --- | --- |
| internal | relative paths, root-relative paths, and heading anchors, against the files on disk | always, and a paid run waits on it |
| external | every `http(s)` URL, fetched | only with `--external`, or the `external_references` job |

The split is by failure mode, not by effort. The internal half is
deterministic and offline, so it can gate a merge and a token spend without
ever being wrong for a reason of its own. The external half fetches other
people's servers, which fail for reasons that have nothing to do with the
change under review: a rate limit, a runner IP a host blocks, DNS having a bad
minute. It runs as its own job that the aggregate result ignores, so link rot
is visible without a bad minute somewhere else holding up a merge. Point a
schedule at the workflow to catch rot in markdown nobody is editing.

A 2xx answer is reachable, and so is a 429 — that is a host saying "you again",
which is a fact about the run rather than the link. `HEAD` is asked first and
asked again as a `GET` whenever it does not come back with one, since plenty of
hosts refuse the method or simply ignore it. Hosts are fetched from in
parallel, one request at a time each: a checker that opens twenty connections
to the same documentation site gets itself throttled, and a throttled request
is indistinguishable from link rot.

Links inside fenced code blocks, inline code spans, and HTML comments are left
alone — those are illustrations of links rather than promises. All of it is
standard library, so none of this adds a dependency to a graded run.

### A skill says what hardware it needs; the repo says what that costs

In `evals/machine.yml`, beside the prompts that need it:

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

## Commands

| Command | Does |
| --- | --- |
| `skillscope structural` | Every skill folder, every dataset, and every reference the skills' markdown makes. No agent, no tokens. `--docs`, `--skill-files`, `--skill-sections`, `--external`, `--exclude-url`. |
| `skillscope routing` | Which skill fires, with several installed together. `--routing-skills`, `--skill`, `--only`, `--min-accuracy`, `--keep-logs`. |
| `skillscope behavioral` | What a skill does once it has fired. `--skill`, `--only`. |
| `skillscope select` | The CI plan for a change, as JSON: which skills, which runners, which harness version. |
| `skillscope list-skills` | The skills that have a dataset, as JSON. |
| `skillscope template` | The dataset template a new skill starts from. |

`--help` on any of them is the reference. Reports go to stdout as markdown, to
`$GITHUB_STEP_SUMMARY` under Actions, and to JSON under `.skillscope/runs/` in
the repo under test — worth adding to `.gitignore`.

## Development

```bash
python -m pip install -e .
python -m unittest discover -s tests -t .
```

Every test that needs a repo builds a throwaway one in a temp directory. That
is not fastidiousness: the harness is supposed to work against any repo, so a
test that reads whichever tree it happens to be running in tests that tree
instead of the harness.

The runner is standard library only, apart from PyYAML, which reads a skill's
frontmatter and the optional `machine.yml`, so a graded run installs nothing
beyond this package.

`tools/` holds hand tools that are no part of the graded pipeline:
`claude_eval.py` (what did this one prompt cost?), `compare_skill.py` (the same
prompt with and without a skill, side by side), and
`verify_selection_parity.py` (does `select` still plan what amd/skills planned
before the harness moved out of it?).

## License

MIT. See [LICENSE](LICENSE).
