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

The three arguments keep the meanings the legacy engine gave them, which is
not where this started: `workspace` was briefly `tools.workdir()` -- `None`
off-container, the string `/workspace` on it -- and `case` was
`state.metadata`, a dict holding neither `id` nor `prompt`. Both were silent
changes to a documented contract, and both broke the example in
`docs/authoring-evals.md`. Now `workspace` is a real per-case host directory
(`case_workspace`) and `case` is the `Case` object itself.

One meaning did have to change. Legacy's `workspace` *was* the agent's room,
because legacy staged the skill into a temp directory and ran the CLI there.
Here the agent's room is inspect's -- a container under `claude-code` -- so
this is a host-side scratch directory for the hook's own use, which is what
`sources.resolve(skill, cache_dir)` wants. A hook that needs to reach the
agent's room should use inspect's sandbox API.

`ctx` is always `{}`. It carried `setup_session`'s return value, and that
entry point is refused below, so there is nothing left to put in it.

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
import shutil
import tempfile
from pathlib import Path
from types import ModuleType

from .. import datasets, deadline
from ..datasets import Case

# Entry points the legacy engine honoured that these engines cannot. Named
# rather than inferred so the refusal can list them, and so adding support
# later is a matter of removing a name.
UNSUPPORTED = ("check", "setup_session")

# Where a case's hook scratch directory is remembered. inspect's store is
# scoped to the sample, which is the scope the directory has: `setup` and
# `teardown` for one case must be handed the same path, and two cases must not
# share one.
WORKSPACE_KEY = "skillscope_hook_workspace"

# Every scratch directory this process made, for the one caller that cannot
# reach inspect's store: the `--timeout` watchdog, which fires from a thread
# with no sample bound to it. Registered with `deadline` below so a hard exit
# removes them rather than leaving one per case on the runner.
_created_workspaces: set[str] = set()


def _discard_all_workspaces() -> None:
    """Remove every scratch directory this process made. For the watchdog."""
    for path in list(_created_workspaces):
        shutil.rmtree(path, ignore_errors=True)
    _created_workspaces.clear()


deadline.on_expire(_discard_all_workspaces)


def case_workspace() -> Path:
    """This case's scratch directory on the host, created once per sample.

    The legacy engine handed `setup`/`teardown` a real local directory and
    asserted it was not `None`. The first inspect version handed them
    `tools.workdir()`, which is `None` off-container and the *string*
    `/workspace` on it -- so the documented example,
    `sources.resolve(skill, workspace)`, raised `TypeError` on both engines,
    and any hook doing `workspace / "x"` did too.

    A host path on both engines, deliberately. `sources.resolve` names this
    parameter `cache_dir` and uses it to fetch a source tree the hook will read
    *in this process*; a path inside the guest would be useless for that, and
    `claude-code`'s guest is torn down with the case anyway. A hook that needs
    to put files where the agent will see them should use inspect's sandbox
    API, which can address the guest -- `docs/authoring-evals.md` says so.
    """
    from inspect_ai.util import store

    cached = store().get(WORKSPACE_KEY)
    if cached:
        return Path(cached)
    created = Path(tempfile.mkdtemp(prefix="skillscope-hook-"))
    store().set(WORKSPACE_KEY, str(created))
    # Also tracked outside inspect's store, because the store is scoped to a
    # sample and the wall-clock watchdog runs in a thread that has no sample.
    # `_discard_workspace` is the ordinary path; this is what lets a hard exit
    # take the directories with it rather than leaving one per case behind.
    _created_workspaces.add(str(created))
    return created


def _discard_workspace() -> None:
    """Remove this case's scratch directory, if one was ever made.

    After `teardown`, not before: the directory is what a hook was given to
    work in, so removing it first would pull the ground out from under the
    entry point most likely to need it.
    """
    from inspect_ai.util import store

    cached = store().get(WORKSPACE_KEY)
    if not cached:
        return
    shutil.rmtree(cached, ignore_errors=True)
    _created_workspaces.discard(str(cached))
    store().set(WORKSPACE_KEY, "")


def _case_for(state, cases: list[Case] | None) -> Case | dict:
    """The `Case` this sample came from, for the hook's second argument.

    The object, not `state.metadata`. Legacy passed the `Case` itself, so a
    hook reading `case.id` or `case.prompt` -- both documented -- got an
    `AttributeError` against the dict that replaced it, and the two fields it
    most likely wanted were not in that dict under any spelling.

    Falls back to `state.metadata` only when a caller built a task without
    handing the cases down, which no caller in skillscope does; a hook seeing
    a dict means a bug here rather than a skill to fix.
    """
    if not cases:
        return state.metadata
    sample_id = str(getattr(state, "sample_id", ""))
    for case in cases:
        if case.id == sample_id:
            return case
    return state.metadata


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


def setup_solver(
    module: ModuleType | None, skill: str, cases: list[Case] | None = None
):
    """`Task.setup` for this skill's hook, or `None` if it has no `setup`."""
    if module is None or not callable(getattr(module, "setup", None)):
        return None

    from inspect_ai.solver import solver

    @solver
    def _hook_setup():
        async def solve(state, generate):
            _returned_template_vars(
                module.setup(case_workspace(), _case_for(state, cases), {}), skill
            )
            return state

        return solve

    return _hook_setup()


def cleanup_fn(
    module: ModuleType | None, skill: str, cases: list[Case] | None = None
):
    """`Task.cleanup` for this skill's hook, or `None` if it has no `teardown`.

    Returned rather than wrapped in a solver because inspect runs `cleanup` in
    a `finally` under a shielded cancel scope, and a solver would not run at
    all once the agent had raised.

    Returned whenever the skill defines `teardown`, even with no `setup`: the
    scratch directory is created on demand, so `teardown` gets a real one
    either way, and it is removed afterwards rather than left behind.

    Also returned for a hook with `setup` and no `teardown`, where it only
    removes the directory. `setup` is handed a real path now, so something has
    to delete it; without this a run of N cases left N temp directories on the
    runner, which on a shared one is a leak that outlives the job.
    """
    if module is None:
        return None
    teardown = getattr(module, "teardown", None)
    if not callable(teardown) and not callable(getattr(module, "setup", None)):
        return None

    async def _cleanup(state) -> None:
        try:
            if callable(teardown):
                teardown(case_workspace(), _case_for(state, cases), {})
        finally:
            _discard_workspace()

    return _cleanup
