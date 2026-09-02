<!--
Copyright Advanced Micro Devices, Inc.

SPDX-License-Identifier: MIT
-->

# skillscope

> [!IMPORTANT]
> **Beta — not a formal release.** skillscope grew up grading the skills in
> `amd/skills` and is being generalized to work for skill authors anywhere.
> Flags, workflow inputs, and report formats can still change without a
> deprecation path, so pin a version if that matters to you.

A test harness for agent skills, in whatever repo the skills live in. Point it
at a repo and it grades three things that fail independently of each other:

* **structural** — every skill folder, every dataset, and every reference the
  skills' markdown makes. No agent, no tokens, instant.
* **routing** — installs skills side by side and grades which one a prompt
  wakes up.
* **behavioral** — installs one skill, runs the prompt to completion, and
  grades what the agent actually did.

All three read one dataset per skill, `<skill>/evals/evals.json` — a prompt,
and what should be true after it runs. A prompt on its own grades routing; add
an expectation and the same prompt also runs end to end. Writing one is
[docs/authoring-evals.md](docs/authoring-evals.md).

## Quick start

Run from the root of the repo you want tested. A repo that keeps its skills in
`skills/*`, each with a `SKILL.md`, passes no flags at all.

```bash
alias skillscope='uvx --from git+https://github.com/danielholanda/skillscope skillscope'

skillscope structural                        # no agent, no tokens
skillscope structural --external             # the same, plus fetching every URL
skillscope behavioral --skill my-skill       # needs an authenticated `claude` CLI
skillscope routing --routing-skills my-skill,its-neighbour
```

`--routing-skills` is the one thing without a sensible default once a repo has
more than one skill: who a skill competes against is what its score means. A
repo with a single skill can leave it off.

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
