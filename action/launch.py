# Copyright Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: MIT

"""Run one skillscope command against the repository being tested.

The body of ``action.yml``, and there is no version to work out: the checkout
the action already has *is* the harness. `uses: amd/skillscope@v0.1.1` makes
Actions download this repository at ``v0.1.1`` into ``$GITHUB_ACTION_PATH``,
and this script installs the harness from there::

    uvx --from $GITHUB_ACTION_PATH skillscope <command>

So the ref a caller pins is the build that grades their skills, by construction
rather than by convention -- there is no second place holding a version that
could disagree with the ref, and nothing to keep in step at release time.

`version` is how to ask for a different build than the one pinned, and it is
the only case that reaches the network: the harness is then fetched as
``git+https://github.com/<owner>/<repo>@<ref>``.

Written in Python rather than shell because the same step runs on Linux,
Windows, and macOS runners, self-hosted and not. Standard library only, and it
imports nothing from skillscope, which has to keep working before the harness
is installed.

Configuration arrives as environment variables, set from the action's inputs:

    SKILLSCOPE_COMMAND      the subcommand, e.g. "structural"
    SKILLSCOPE_ARGS         further arguments, shell-quoted
    SKILLSCOPE_REPO         root of the repo under test (default ".")
    SKILLSCOPE_SKILLS       globs naming the directories that are skills
    SKILLSCOPE_STDIN        a file to feed the command on stdin
    SKILLSCOPE_VERSION      a harness ref to fetch instead of using the checkout
    SKILLSCOPE_REPOSITORY   the owner/repo to fetch that ref from
    SKILLSCOPE_ACTION_PATH  this action's checkout, which is the harness

Everything else a run needs is passed straight through in SKILLSCOPE_ARGS,
unread: the launcher knows nothing about the payload's flags.
"""

from __future__ import annotations

import os
import re
import shlex
import subprocess
import sys
from pathlib import Path

# `version` is interpolated into `uvx --from git+https://...@<ref>`, so it has
# to be a plausible git ref and nothing more. Anything with a shell
# metacharacter in it is refused rather than escaped: there is no legitimate
# ref that needs one.
REF_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._/+-]*$")

# Read out of the checkout rather than imported, because the launcher runs
# before anything is installed. Only for the log line and the step summary, so
# a build whose version is somewhere unexpected loses a label, not a run.
VERSION_PATTERN = re.compile(r"""^__version__\s*=\s*['"]([^'"]+)['"]""", re.MULTILINE)


def _env(name: str, default: str = "") -> str:
    return os.environ.get(name, default).strip()


def _emit(name: str, value: str) -> None:
    path = _env("GITHUB_OUTPUT")
    if path:
        with open(path, "a", encoding="utf-8") as handle:
            handle.write(f"{name}={value}\n")


def _summarize(text: str) -> None:
    path = _env("GITHUB_STEP_SUMMARY")
    if path:
        with open(path, "a", encoding="utf-8") as handle:
            handle.write(text + "\n")


def packaged_version(checkout: Path) -> str:
    """The version declared by the harness in `checkout`, or "" if unreadable."""
    try:
        text = (checkout / "skillscope" / "__init__.py").read_text(encoding="utf-8")
    except OSError:
        return ""
    found = VERSION_PATTERN.search(text)
    return found.group(1) if found else ""


def install_source() -> tuple[str, str]:
    """What to hand ``uvx --from``, and the build that is, for the report."""
    version = _env("SKILLSCOPE_VERSION")
    if not version:
        checkout = Path(_env("SKILLSCOPE_ACTION_PATH") or ".").resolve()
        if not (checkout / "pyproject.toml").is_file():
            raise SystemExit(
                f"error: {checkout} holds no pyproject.toml, so it is not a "
                "skillscope to install. This is the action's own checkout, so "
                "either the action is being run from somewhere unexpected or "
                "`version` should name the build to fetch instead."
            )
        return str(checkout), packaged_version(checkout) or "this action's checkout"

    if not REF_PATTERN.match(version):
        raise SystemExit(
            f"error: `version` is {version!r}, which is not a usable git ref. "
            "It is fetched as one, so it has to be a tag, a branch, or a commit."
        )
    repository = _env("SKILLSCOPE_REPOSITORY")
    if not repository:
        raise SystemExit(
            f"error: `version` asks for skillscope {version}, but there is no "
            "repository to fetch it from. Leave `version` empty to run the "
            "build this action's own ref points at."
        )
    return f"git+https://github.com/{repository}@{version}", f"{repository}@{version}"


def main() -> int:
    repo = Path(_env("SKILLSCOPE_REPO", ".") or ".").expanduser().resolve()
    command = _env("SKILLSCOPE_COMMAND")
    if not command:
        raise SystemExit("error: no skillscope command given.")

    source, version = install_source()
    cmd = [
        "uvx",
        "--from",
        source,
        "skillscope",
        *shlex.split(command),
        *shlex.split(_env("SKILLSCOPE_ARGS")),
    ]
    print(f"[skillscope] {version}: {' '.join(cmd)}", flush=True)

    stdin_path = _env("SKILLSCOPE_STDIN")
    stdin = open(stdin_path, "rb") if stdin_path else subprocess.DEVNULL
    # `SKILLSCOPE_REPO` because the input may be relative -- `repo: fixture` --
    # and the child runs from the repo it names, where resolving that same
    # relative path again lands a directory deeper.
    child_env = {**os.environ, "SKILLSCOPE_REPO": str(repo)}
    try:
        proc = subprocess.Popen(
            cmd,
            cwd=str(repo),
            env=child_env,
            stdin=stdin,
            stdout=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
        captured: list[str] = []
        assert proc.stdout is not None
        for line in proc.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            if line.strip():
                captured.append(line.rstrip("\n"))
        code = proc.wait()
    except FileNotFoundError as exc:
        raise SystemExit(
            f"error: {exc.filename} is not on PATH. The action installs uv "
            "before this step; if you are running it by hand, install uv first."
        ) from exc
    finally:
        if stdin is not subprocess.DEVNULL:
            stdin.close()

    _emit("version", version)
    # Commands that answer with data (`select`) print one line of JSON, so the
    # last line of output is that answer. A command that prints a report leaves
    # a harmless last line here and is read from the step summary instead.
    _emit("stdout", captured[-1] if captured else "")
    _summarize(f"<sub>skillscope <code>{command}</code> ran at <code>{version}</code>.</sub>")
    return code


if __name__ == "__main__":
    sys.exit(main())
