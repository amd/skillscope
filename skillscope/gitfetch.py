# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.

"""Fetching a tree out of a GitHub repo.

A behavior hook sometimes needs the product repo its skill drives -- a
vendored skill's tests may want the source tree it was written against. That
comes down to "put exactly this ref of that repo in this directory".

Stdlib only, and fetching relies on ambient git credentials -- the same as
every other tool in a CI checkout.
"""

from __future__ import annotations

import subprocess
from collections.abc import Sequence
from pathlib import Path


def _run(cmd: list[str]) -> str:
    proc = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", check=False)
    if proc.returncode != 0:
        raise RuntimeError(
            f"command failed ({proc.returncode}): {' '.join(cmd)}\n"
            f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
        )
    return (proc.stdout or "").strip()


def head_commit(repo_dir: Path) -> str:
    return _run(["git", "-C", str(repo_dir), "rev-parse", "HEAD"])


def fetch_ref(
    repo: str, ref: str, dest: Path, sparse: Sequence[str] = ()
) -> str:
    """Check out `ref` of GitHub `repo` into `dest`, returning its commit.

    `ref` is anything git can resolve: a branch, a tag, or a commit. `sparse`
    narrows the checkout to a few top-level paths, which is the difference
    between a few hundred kilobytes and a repo's whole history of binaries.

    A `dest` that already holds a checkout is left alone, so a caller can
    treat this as a cache.
    """
    if (dest / ".git").is_dir():
        return head_commit(dest)

    dest.mkdir(parents=True, exist_ok=True)
    # writeCommitGraph=false: commit-graph writing is handed to a detached
    # process that outlives the fetch and keeps creating files under
    # .git/objects/info/, which then races the removal of the cache directory
    # this checkout lives in.
    git = ["git", "-c", "fetch.writeCommitGraph=false", "-C", str(dest)]
    _run([*git, "init", "--quiet"])
    _run([*git, "remote", "add", "origin", f"https://github.com/{repo}.git"])
    if sparse:
        _run([*git, "sparse-checkout", "init", "--cone"])
        _run([*git, "sparse-checkout", "set", *sparse])

    failures: list[str] = []
    try:
        _run([*git, "fetch", "--quiet", "--depth", "1", "origin", ref])
    except RuntimeError as shallow_error:
        # Not every host serves a request for a bare SHA, so fall back to
        # fetching the refs and finding the commit among them.
        failures.append(f"shallow fetch of {ref!r} failed:\n{shallow_error}")
        try:
            _run([*git, "fetch", "--quiet", "origin"])
        except RuntimeError as full_error:
            failures.append(f"full fetch failed too:\n{full_error}")
            raise RuntimeError(
                f"could not fetch {repo}@{ref}.\n" + "\n".join(failures)
            ) from full_error

    # After fetching a named ref, FETCH_HEAD is it; after the full-fetch
    # fallback, the ref itself (or its remote-tracking branch) is what
    # resolves. Try each rather than guessing which path was taken.
    for candidate in ("FETCH_HEAD", ref, f"origin/{ref}"):
        try:
            _run([*git, "checkout", "--quiet", candidate])
            return head_commit(dest)
        except RuntimeError as exc:
            failures.append(f"checkout of {candidate!r} failed:\n{exc}")
    raise RuntimeError(f"could not check out {repo}@{ref}.\n" + "\n".join(failures))
