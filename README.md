<!--
Copyright Advanced Micro Devices, Inc.

SPDX-License-Identifier: MIT
-->

# skillscope

Testing Harness for AI Agent Skills

> [!IMPORTANT]
> **Early Days**: Flags, workflow inputs, and report formats may evolve quickly as this repo takes shape.

A test harness for AI agent skills. Simply pip install the harness or point a workflow at your repo.

Skillscope does three main types of tests:

* **structural**: Checks whether the skill is well structured (contains the expected files, a proper hierarchy, etc.).
* **routing**: Checks whether the skill triggers when it should and stays quiet when it shouldn't. You may also point to more than one skill here to explore how skills interfere with each other's routing.
* **behavioral**: Runs the agent end to end and uses an LLM judge to confirm the skill behaves correctly.

All three read one dataset per skill, `<skill>/evals/evals.json`. An example is
in [docs/usage.md](docs/usage.md#evalsjson); more on writing one is at
[docs/authoring-evals.md](docs/authoring-evals.md).

## Quick start

Run from the root of the repo you want tested.

```bash
alias skillscope='uvx --from git+https://github.com/danielholanda/skillscope skillscope'

skillscope structural                        # no agent, no tokens
skillscope structural --external             # the same, plus checking external URLs
skillscope behavioral --skill my-skill       # needs an authenticated `claude` CLI
skillscope routing --routing-skills my-skill,its-neighbour
```

## In CI

```yaml
jobs:
  evals:
    uses: danielholanda/skillscope/.github/workflows/reusable.yml@main
    secrets:
      api_key: ${{ secrets.ANTHROPIC_API_KEY }}
    with:
      skills: skills/*
```

That is a whole caller: all three graders run, each skill gets a runner of its
own, and any grader can fail the run. Turning one down to `optional` or `off`,
moving the routing bar, change-based selection, and skills that need particular
hardware are all in [docs/usage.md](docs/usage.md).

## Commands

| Command | Does |
| --- | --- |
| `skillscope structural` | Skill folders, datasets, and markdown references. |
| `skillscope routing` | Which skill fires, with several installed together. |
| `skillscope behavioral` | What a skill does once it has fired. |
| `skillscope select` | The CI plan for a change, as JSON. |
| `skillscope list-skills` | The skills that have a dataset, as JSON. |
| `skillscope template` | The dataset template a new skill starts from. |

`--help` on any of them is the reference. Reports go to stdout as markdown, to
`$GITHUB_STEP_SUMMARY` under Actions, and to JSON under `.skillscope/runs/` in
the repo under test — worth adding to `.gitignore`.

## Docs

* [docs/authoring-evals.md](docs/authoring-evals.md) — writing a skill's dataset.
* [docs/usage.md](docs/usage.md) — every flag, every workflow input, and what
  each check actually asserts.

## Development

```bash
python -m pip install -e .
python -m unittest discover -s tests -t .
```

Standard library only, apart from PyYAML for frontmatter, so a graded run
installs nothing beyond this package. Tests build a throwaway repo in a temp
directory rather than reading the tree they run in.

## License

MIT. See [LICENSE](LICENSE).
