# Copyright Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: MIT

"""Reading a sandbox, on a guest that may not be POSIX.

Not tools an agent calls -- those were removed with the harness-independent
engine. What is left is how *skillscope itself* inspects a sandbox after the
agent has finished: where the work was supposed to land (`workdir`), what is
actually there (`list_paths`), and how to resolve a path a case named. The
scorers, the judge and the claude-code leg all read a sandbox this way.

inspect's own helpers assume a POSIX guest: `bash()` execs
`["bash", "--login", "-c", ...]` and `list_files()`/`grep()` shell out to
`find`/`grep`. On a Windows host with the `local` sandbox that is a scorer
which cannot see the files it is grading.

Everything here is built on `SandboxEnvironment.exec` / `read_file` /
`write_file`, which are provider-level and platform-neutral. Only the shell
invocation differs, and that is probed once per sample rather than assumed:
the same Docker sandbox is POSIX whichever host started it, so the host's own
platform is not the answer.
"""

from __future__ import annotations

SHELL_KEY = "skillscope_shell"
WORKDIR_KEY = "skillscope_workdir"

# A container sandbox starts at `/`, so a relative path lands beside `/proc` and
# `/etc` and a recursive listing walks the whole image. Everything the case does
# happens here instead: fixtures are seeded into it, tools resolve against it,
# and it is what gets listed. inspect_swe resolves the same problem the same way
# -- its agent cwd falls back to the home directory when the sandbox default is
# `/`.
WORKDIR = "/workspace"

POSIX_SHELL = ["bash", "-lc"]
WINDOWS_SHELL = ["powershell", "-NoProfile", "-Command"]

# Listing is the one thing `SandboxEnvironment` has no method for, so it stays a
# shell command -- but only one, defined here, used by both the tools and the
# scorers.
POSIX_LIST = "find . -type f"
WINDOWS_LIST = "Get-ChildItem -Recurse -File | Resolve-Path -Relative"


async def shell_prefix() -> list[str]:
    """The argv prefix that runs a shell command in this sample's sandbox.

    Probed once and remembered: a probe per tool call would double the round
    trips on the slowest part of a run.
    """
    from inspect_ai.util import sandbox, store

    cached = store().get(SHELL_KEY)
    if cached:
        return list(cached)

    # A guest without bash does not answer "that failed" -- there is nothing
    # to run, so the exec raises before any result exists. On a Windows host
    # under the `local` sandbox that surfaced as WinError 2 and took the whole
    # task down, which reads as the harness being broken rather than the probe
    # learning what it asked.
    try:
        probe = await sandbox().exec(["bash", "-lc", "exit 0"], concurrency=False)
        posix = probe.success
    except (FileNotFoundError, OSError):
        posix = False
    prefix = POSIX_SHELL if posix else WINDOWS_SHELL
    store().set(SHELL_KEY, prefix)
    return list(prefix)


def containerized() -> bool:
    """Whether this run has a sandbox of its own to work in."""
    from . import sandbox as sandbox_spec

    return sandbox_spec.provider() not in sandbox_spec.NOT_ISOLATED


def workdir_path() -> str | None:
    """The same answer as `workdir()`, without creating anything.

    A task is built before any sandbox exists, so a solver that needs to be
    *told* the working directory at construction time cannot await the version
    that makes it.
    """
    return WORKDIR if containerized() else None


async def workdir() -> str | None:
    """The directory a case works in, or None to use the sandbox's own.

    `local` needs none: the harness's working directory is already a sensible
    place and creating `/workspace` on someone's machine would not be.
    """
    if not containerized():
        return None

    from inspect_ai.util import sandbox, store

    cached = store().get(WORKDIR_KEY)
    if cached:
        return cached

    prefix = await shell_prefix()
    await sandbox().exec(prefix + [f"mkdir -p {WORKDIR}"], concurrency=False)
    store().set(WORKDIR_KEY, WORKDIR)
    return WORKDIR


async def resolve(path: str) -> str:
    """A case-relative path, as the sandbox should see it."""
    base = await workdir()
    if base is None or path.startswith("/"):
        return path
    return f"{base}/{path.lstrip('./')}"


async def run(command: str, timeout: int | None = None):
    """Run `command` through whichever shell the sandbox has, in the workdir."""
    from inspect_ai.util import sandbox

    prefix = await shell_prefix()
    return await sandbox().exec(
        prefix + [command], cwd=await workdir(), timeout=timeout
    )


def normalize_listing(stdout: str) -> list[str]:
    """Turn a directory listing into relative POSIX-style paths.

    `find` and `Get-ChildItem` disagree about separators and prefixes, so this
    normalises both: backslashes become slashes and a leading `./` or `.\\` is
    dropped. The harness's own furniture is filtered out -- an installed skill
    is not something the case produced, and `files_exist` must not be satisfied
    by one.
    """
    paths: list[str] = []
    for line in stdout.splitlines():
        rel = line.strip().replace("\\", "/")
        while rel.startswith("./"):
            rel = rel[2:]
        if not rel or rel.startswith(".claude/") or rel.startswith("skills/"):
            continue
        paths.append(rel)
    return sorted(paths)


class ListingFailed(RuntimeError):
    """The sandbox could not be listed, which is not the same as it being empty.

    Returning an empty list here would make a broken sandbox look exactly like
    an idle agent: `files_exist` fails, and the judge -- which builds its
    evidence from the same listing -- reports that nothing was produced. Both
    read as the skill's fault. Raising keeps the two apart.
    """


async def list_paths() -> list[str]:
    """Files in the sandbox working directory, as relative POSIX-style paths."""
    prefix = await shell_prefix()
    listing = WINDOWS_LIST if prefix == WINDOWS_SHELL else POSIX_LIST
    result = await run(listing)
    if not result.success:
        raise ListingFailed(
            f"`{listing}` failed in the sandbox (exit {result.returncode}). "
            f"stderr: {result.stderr.strip()[:200] or '(none)'}"
        )
    return normalize_listing(result.stdout)
