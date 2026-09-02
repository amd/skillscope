<!--
Copyright Advanced Micro Devices, Inc.

SPDX-License-Identifier: MIT
-->

# Skillscope

![AMD](https://img.shields.io/badge/AMD-Skillscope-ED1C24?logo=amd&logoColor=white)
![Agent Skills](https://img.shields.io/badge/Agent_Skills-Standard-7B2D8E)
![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)
[![Claude Code](https://img.shields.io/badge/Claude_Code-Compatible-F07535?logo=claude&logoColor=white)](https://www.anthropic.com/claude-code)
[![GitHub Actions](https://img.shields.io/badge/GitHub_Actions-Reusable-2088FF?logo=githubactions&logoColor=white)](.github/workflows/reusable.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

![Skillscope](assets/banner.png)

A test harness for AI agent skills. Install the harness, or point a workflow at your repo. Skillscope enables three main types of tests:

* **Structural**: Checks whether the skill is well structured (contains the expected files, a proper hierarchy, etc.).
* **Routing**: Checks whether the skill triggers when it should and stays quiet when it shouldn't. You may also point to more than one skill here to explore how skills interfere with each other's routing.
* **Behavioral**: Runs the agent end to end and uses an LLM judge to confirm the skill behaves correctly.

All three read one dataset per skill, `<skill>/evals/evals.json`. How to write one is in [docs/authoring-evals.md](docs/authoring-evals.md).

> [!IMPORTANT]
> **Early Days**: Flags, workflow inputs, and report formats may evolve quickly as this repo takes shape.

## Quick start

Run from the root of the repo you want tested.

```bash
uv tool install git+https://github.com/danielholanda/skillscope

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

* [docs/authoring-evals.md](docs/authoring-evals.md): writing a skill's tests.
* [docs/usage.md](docs/usage.md): every flag, every workflow input, and what
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
