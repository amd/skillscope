# Copyright Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: MIT

"""`evals/hooks.py` on the engines built on inspect_ai.

A hook module holds environment plumbing that the dataset format cannot
express -- tearing down a container, clearing a stale endpoint -- so that
prompts and expectations stay readable without opening Python. The legacy
engine ran four entry points; two of them survive the move to inspect_ai
unchanged, and two do not.

**Supported.** `setup(workspace, case, ctx)` before each case and
`teardown(workspace, case, ctx)` after it. inspect already has both shapes:
`Task.setup` takes a solver that runs ahead of the agent, and `Task.cleanup`
takes a per-state callable that inspect invokes inside a `finally` under a
shielded cancel scope -- so teardown still runs when the agent raises or the
sample is cancelled, which is the property the whole hook exists for.

**Not supported, and refused rather than approximated.**

`check(run, case, ctx)` is handed the legacy engine's `Run` object and raises
to fail a case. These engines have inspect's `TaskState`, which is a different
thing with different attributes; passing one where the other is expected would
fail inside somebody's skill with an error about their code.

`setup_session(cache_dir)` returns `{name: value}` pairs that `expand()`
substitutes into prompts and expectations. Those are baked into the `Sample`
before any solver runs, so honouring it means building the dataset after the
hook rather than before -- a change to how every case is constructed, for a
feature nothing in the catalogue uses.

A `setup` that *returns* template variables has the same problem, and is caught
at runtime rather than by inspection because a docstring cannot say what a
function returns.

Nothing in the catalogue this was written against uses either: one skill ships
a hook, it defines `setup` and `teardown` only, and neither returns anything.
"""

from __future__ import annotations

import importlib.util
from types import ModuleType

from .. import datasets

# Entry points the legacy engine honoured that these engines cannot. Named
# rather than inferred so the refusal can list them, and so adding support
# later is a matter of removing a name.
UNSUPPORTED = ("check", "setup_session")


def load(skill: str) -> ModuleType | None:
    """Import `<skill>/evals/hooks.py`, or `None` when the skill ships none."""
    path = datasets.hooks_path(skill)
    if not path.is_file():
        return None
    spec = importlib.util.spec_from_file_location(
        f"evalhooks_{skill.replace('-', '_')}", path
    )
    if spec is None or spec.loader is None:
        raise SystemExit(f"error: could not import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def unsupported_entry_points(module: ModuleType | None) -> list[str]:
    """Which of `UNSUPPORTED` this module actually defines.

    Pure, and the whole of the refusal rule: a hook that defines neither runs
    here unchanged, and one that defines either is refused by name rather than
    by the file existing. The old guard refused any skill that shipped a hook
    at all, which grounded a skill whose hook this engine could have run.
    """
    if module is None:
        return []
    return [name for name in UNSUPPORTED if callable(getattr(module, name, None))]


def _returned_template_vars(result: object, skill: str) -> None:
    """Refuse a `setup` whose return value we would have to silently drop.

    Returning `{name: value}` is a documented use of `setup`, and those values
    are substituted into prompts that were built before this ran. Dropping them
    would leave `{placeholder}` in the prompt the agent is graded on, which
    reads as a badly written case rather than a missing feature.
    """
    if isinstance(result, dict) and result:
        raise SystemExit(
            f"error: {skill}'s evals/hooks.py setup() returned template "
            f"variables ({', '.join(sorted(result))}), which the inspect "
            "engines cannot substitute: prompts and expectations are built "
            "before any hook runs.\n"
            "    Move the value into the dataset, or compute it in the prompt "
            "itself."
        )


def setup_solver(module: ModuleType | None, skill: str):
    """`Task.setup` for this skill's hook, or `None` if it has no `setup`."""
    if module is None or not callable(getattr(module, "setup", None)):
        return None

    from inspect_ai.solver import solver

    from . import tools

    @solver
    def _hook_setup():
        async def solve(state, generate):
            workspace = await tools.workdir()
            _returned_template_vars(
                module.setup(workspace, state.metadata, {}), skill
            )
            return state

        return solve

    return _hook_setup()


def cleanup_fn(module: ModuleType | None, skill: str):
    """`Task.cleanup` for this skill's hook, or `None` if it has no `teardown`.

    Returned rather than wrapped in a solver because inspect runs `cleanup` in
    a `finally` under a shielded cancel scope, and a solver would not run at
    all once the agent had raised.
    """
    if module is None or not callable(getattr(module, "teardown", None)):
        return None

    async def _cleanup(state) -> None:
        from . import tools

        workspace = await tools.workdir()
        module.teardown(workspace, state.metadata, {})

    return _cleanup
