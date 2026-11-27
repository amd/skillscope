<!--
Copyright Advanced Micro Devices, Inc.

SPDX-License-Identifier: MIT
-->

## What and why

<!-- What changes for someone using Skillscope, and why. Link the issue it closes. -->

## Checklist

- [ ] Behavior this PR changes is covered by tests under `tests/`, and `python -m unittest discover -s tests -t .` passes locally.
- [ ] `ruff check --select F401,F841 skillscope tests tools` passes.
- [ ] A changed flag, workflow input, report format or dataset schema is updated in `README.md` and the matching page under `docs/`, and `python -m skillscope structural --docs '*.md,docs/**/*.md'` passes.
- [ ] New files carry the SPDX license header.
- [ ] Commits are signed, and signed off with `git commit -s` as `CONTRIBUTING.md` asks.
