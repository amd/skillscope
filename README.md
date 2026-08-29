# skillscope

Structural, routing, and behavior tests for agent skills, in whatever repo the
skills live in.

A skill is a description plus a body, and each half fails in a way the other
cannot catch. A description that never fires makes the body irrelevant; a
description that fires on its neighbour's work makes every skill around it
worse.
And a skill that routes perfectly can still do the job badly once it runs.
skillscope grades all three, from one dataset per skill and the skill's own
prose:

* **structural** — every dataset, and every reference the skill's markdown
  makes. No agent, no tokens, instant, and so it runs on every change and
  gates the two below.
* **routing** — installs several skills side by side and asks which one wins a
  prompt. You cannot test this alone: a skill tested by itself will happily
  answer prompts that belong to its neighbour.
* **behavior** — installs one skill, runs the prompt to completion, and grades
  what the agent actually did.

The prompt is written once and both graded modes read it. The alternative — a
central routing prompt set plus a per-skill test file that re-asserts routing
with a substring match on the transcript — is two things to maintain and one of
them is a worse copy of the other.

## Quick start

```bash
# structure only: no agent, no tokens, instant
uvx --from git+https://github.com/amd/skillscope skillscope structural

# the same, plus fetching every external URL the skills link to
uvx --from git+https://github.com/amd/skillscope skillscope structural --external

# behavior for one skill (needs an authenticated `claude` CLI)
uvx --from git+https://github.com/amd/skillscope skillscope run \
  --mode behavior --skill my-skill

# routing, for the skills you want in the room together
uvx --from git+https://github.com/amd/skillscope skillscope run \
  --mode routing --routing-skills my-skill,its-neighbour

# what CI should run for a change
git diff --name-only main HEAD | uvx --from git+https://github.com/amd/skillscope skillscope select --changed
```

Run from the root of the repo you want tested. A repo that keeps its skills in
`skills/*` and has a `SKILL.md` in each passes no flags at all, except to say
which skills a routing run should install together — that one has no sensible
default, and [why](#who-a-skill-competes-against-is-listed-not-inferred).

Writing the datasets themselves: [docs/authoring-evals.md](docs/authoring-evals.md).

## In CI

One job, in the calling repo:

```yaml
jobs:
  skill-evals:
    uses: amd/skillscope/.github/workflows/skill-evals.yml@bootstrap
    secrets: inherit
    with:
      routing_skills: my-skill,its-neighbour
      api_key_secret: MY_MODEL_API_KEY
```

That is the whole pipeline: check the structure, select what the change
affects, run routing once, run a behavior leg per affected skill on the
hardware that skill asks for, and report one aggregate result. See
[examples/skill-evals.yml](examples/skill-evals.yml) for a fully configured
caller and
[.github/workflows/skill-evals.yml](.github/workflows/skill-evals.yml) for every
input. Credentials are passed by secret **name**, so this repo never learns your
provider or your gateway.

To run a single command instead of the pipeline, use the action directly:

```yaml
- uses: amd/skillscope@bootstrap
  with:
    command: structural
```

### A wrong routing decision fails the run

`min_accuracy` (`--min-accuracy`) defaults to `1`: every graded routing case
has to land on the right skill, the same way every behavior expectation has to
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
| the `version` input on the action or workflow | the repo, and everything that is not one skill's behavior run |
| `skillscope_version` in a skill's `evals/evals.json` | that skill's behavior run, overriding the input |

Both are one-line diffs a reviewer can see, which a `uses:` ref spread across
every caller is not. Routing always runs at the workflow's version: it installs
several skills in one session and so cannot honor several pins at once.

## Configuring the repo under test

There is no config file. Everything about your repo is a workflow input, or a
flag if you are running the CLI by hand:

| Input | Flag | Default | What it decides |
| --- | --- | --- | --- |
| `skill_globs` | `--skills` | `skills/*` | Globs naming the directories that hold skills. A skill is a directory with a `SKILL.md`, and its directory name is its identity. |
| `routing_skills` | `--routing-skills` | none | The skills a routing run installs side by side. `all` means every skill with a dataset; blank means no routing run. |
| `infra_paths` | `--infra-paths` | none | Paths that change the harness rather than one skill, so touching one re-runs every skill instead of guessing at the blast radius. Your own workflow file belongs here. |
| `doc_globs` | `--docs` | none | Markdown outside the skills whose references should be checked too: a README, a docs tree. The skills themselves are always checked. |
| `excluded_urls` | `--exclude-url` | none | Regexes matching URLs the external reference check leaves alone. For hosts that are auth-gated or that answer a runner's IP with a 403. |
| `behavior_runner` | `--behavior-runner` | `["ubuntu-latest"]` | `runs-on` labels for a behavior leg. |
| `behavior_os` | `--behavior-os` | `Linux` | Platforms a skill runs on when its `machine.yml` does not narrow them. |
| `scoped_runner` | `--scoped-runner` | reuses `behavior_runner` | Base labels for a leg whose skill asks for hardware labels. |
| `scoped_gate` | `--scoped-gate` | none | Pull-request label required before those legs run. |
| `scoped_environment` | `--scoped-environment` | none | GitHub environment holding their credentials. |

A repo that runs these evals already has a workflow saying when to run them, on
what, and with which credentials. Putting the other half of the same decision
in a second file at the repo root means two places to read, two places to
change, and a file whose only reader is the workflow next to it.

### Who a skill competes against is listed, not inferred

`routing_skills` is the one input with no useful default, because the answer is
what the score *means*. Install every skill on disk and a work-in-progress
directory drops everyone's number; install only the skill under review and it
wins every prompt by walkover. Neither is a guess a test should make on your
behalf. A skill you leave off the list still gets its dataset checked and its
behavior cases run — it just does not move anybody's routing score.

Listing them also makes the change visible: a skill joining or leaving the room
moves every other skill's number, and that deserves a diff a reviewer can see.

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
| `skillscope structural` | Every dataset, and every reference the skills' markdown makes. No agent, no tokens. `--docs`, `--external`, `--exclude-url`. |
| `skillscope run` | Routing and/or behavior. `--mode`, `--skill`, `--routing-skills`, `--only`, `--min-accuracy`, `--keep-logs`. |
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

The runner is standard library only, apart from PyYAML for the optional
`machine.yml`, so a graded run installs nothing beyond this package.

`tools/` holds hand tools that are no part of the graded pipeline:
`claude_eval.py` (what did this one prompt cost?), `compare_skill.py` (the same
prompt with and without a skill, side by side), and
`verify_selection_parity.py` (does `select` still plan what amd/skills planned
before the harness moved out of it?).

## License

MIT. See [LICENSE](LICENSE).
