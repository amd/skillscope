<!--
Copyright Advanced Micro Devices, Inc.

SPDX-License-Identifier: MIT
-->

# Contributing to Skillscope

Thank you for helping. This page covers how to report a problem, how to set up
a checkout, which checks a change has to pass, and what happens in review.

## Code of Conduct

By taking part in this project you agree to abide by the
[Code of Conduct](CODE_OF_CONDUCT.md).

## Issues

Problems and requests are tracked as
[GitHub Issues](https://github.com/amd/skillscope/issues). Search the existing
ones first. Pick the template that matches:

| Template | Use it when |
| --- | --- |
| Bug report | Skillscope itself breaks: a crash, a wrong error message, a flag or workflow input that does not do what [docs/usage.md](docs/usage.md) says. |
| Eval failure | A verdict looks wrong: a case passes or fails when it should not, a score changes between identical runs, or the judge disagrees with what the agent visibly did. |
| Skill support request | You have a skill Skillscope cannot grade yet: a layout it does not find, an expectation the dataset cannot express, hardware or prerequisites it cannot provide. |
| Feature request | You want something else Skillscope does not do yet: a check, a harness, an engine, a workflow input, a report field. |

An eval failure is about Skillscope's verdict, not about the skill. If the
grading is right and the skill is what misbehaves, report it to the repo that
owns the skill.

Skillscope ships no skills. To propose a new skill, or to report a problem with
one in the AMD catalog, use the
[amd/skills issue templates](https://github.com/amd/skills/issues/new/choose).

Do not open a public issue for a security problem. See [SECURITY.md](SECURITY.md).

## Development setup

You need Python 3.10 or newer, the floor [pyproject.toml](pyproject.toml)
declares. From a clone:

```bash
python -m pip install -e .
```

That is enough for the unit suite, the `structural`, `select`, `list-skills`
and `template` commands, and the `host` engine. The `sandbox` engine is Linux
only and also needs `pip install -e '.[verify]'` and Docker, or
`pip install -e '.[verify,podman]'` and podman, which needs more setup;
[docs/usage.md](docs/usage.md) has the details under "Which engine grades a
run" and "Where a sandboxed run is sandboxed". `routing`, `behavioral` and
`staleness` drive a real agent, so they need its CLI and a credential, as the
[README](README.md) describes.

The repo has no pre-commit hooks. The checks below are the ones CI runs.

## Checks

[.github/workflows/selftest.yml](.github/workflows/selftest.yml) runs on every
pull request and every push to `main`. It needs no API key, so it runs the same
way on a pull request from a fork. Run the first three locally before you open a
pull request:

| CI job | Run it locally with |
| --- | --- |
| Unused imports and locals | `ruff check --select F401,F841 skillscope tests tools` |
| Unit tests | `python -m unittest discover -s tests -t .` |
| Action against a throwaway repo (its step that checks this repo's own references) | `python -m skillscope structural --docs '*.md,docs/**/*.md'` |
| Sandbox machinery | `python tools/sandbox_smoketest.py <fixture> --skill <name> --expect-sandbox docker`, against a fixture repo the workflow builds inline |

The unit suite runs on Ubuntu and Windows with Python 3.10 and 3.12. It builds
throwaway repos in a temporary directory and never reads the tree it runs in,
and it calls no model. Keep Windows in mind when you change path handling,
subprocesses or the sandbox helpers: that is where those break.

Some tests run `git commit` in their throwaway repos, under your global git
config. If that config signs commits and signing cannot run unattended, run the
suite with `GIT_CONFIG_GLOBAL` pointing at a config that does not sign.

The references check is offline. Every relative link in the root markdown files
and under `docs/` has to resolve, so cite a file in this repo as a relative
link rather than a bare path, and the check will tell you when it moves.

`ruff` is not a project dependency, so install it yourself (`pip install ruff`,
or run it as `uvx ruff`). The job selects only those two rules on purpose;
there is no house formatter to run.

## Making changes

1. Fork the repository, or branch from `main` if you have write access.
2. Keep each pull request to one logical change, and add or update tests under
   [tests/](tests/) for any behavior you change.
3. If you change a flag, a workflow input, a report format or the dataset
   schema, update [README.md](README.md) and the matching page under
   [docs/](docs/) in the same pull request.
4. Give new files the same SPDX license header their neighbours carry.
5. Open the pull request against `main` and fill in the template.

### Commits

Write commit subjects as a plain sentence saying what the change does for its
user, the way the history does: "Let a repo set the dataset bar its skills are
held to", not `feat: dataset bar`. Explain why in the body.

Sign your commits (SSH or GPG): branch protection on `main` accepts signed
commits only. Please also add a Developer Certificate of Origin sign-off with
`git commit -s`; no CI check enforces it yet.

### Review and merge

Branch protection on `main` requires one approving review, every review
conversation resolved, a linear history and signed commits. Together those
leave squash as the way a pull request lands: a merge commit breaks the linear
history, and GitHub cannot sign the commits a rebase merge rewrites.
Reviewers are requested automatically from [CODEOWNERS](.github/CODEOWNERS).

Keep the `selftest` checks green. A red check reads as not ready, even when it
is unrelated to your change; say so in the pull request if you believe it is.

## License

By contributing you agree that your contributions are licensed under the
[MIT License](LICENSE).
