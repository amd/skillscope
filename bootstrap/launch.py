# Copyright Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: MIT

"""Fetch the pinned build of skillscope and run one command with it.

The body of the `amd/skillscope@bootstrap` composite action. It resolves the
version (see ``resolve_version``), runs

    uvx --from git+https://github.com/<repo>@<version> skillscope <command>

in the repo being tested, and reports back to Actions: the resolved version and
the command's last line of stdout as step outputs, and a one-line note in the
step summary saying which build did the grading.

Everything here is standard library and none of it imports skillscope, which is
what lets callers pin `@bootstrap` once and never touch it again: the launcher
cannot break on a payload version it has never seen. It is written in Python
rather than shell because the same step runs on Linux, Windows, and macOS
runners, self-hosted and not.

Configuration arrives as environment variables, set from the action's inputs:

    SKILLSCOPE_COMMAND   the subcommand, e.g. "structural"
    SKILLSCOPE_ARGS      further arguments, shell-quoted
    SKILLSCOPE_REPO      root of the repo under test (default ".")
    SKILLSCOPE_SKILLS    globs naming the directories that hold skills
    SKILLSCOPE_SOURCE    owner/repo (or a local path) to install from
    SKILLSCOPE_REQUESTED an explicit version, which wins outright
    SKILLSCOPE_VERSION   a version from the environment
    SKILLSCOPE_SKILL     the skill being run, whose dataset may pin a version
    SKILLSCOPE_DEFAULT   fallback version: the launcher's own ref
    SKILLSCOPE_STDIN     a file to feed the command on stdin

Everything else a run needs is passed straight through in SKILLSCOPE_ARGS,
unread. The launcher stays ignorant of the payload's flags so that pinning
`@bootstrap` really is forever; the two variables it does understand are the
two it needs before the harness exists -- where the repo is, and where in it
to look for a version pin.
"""

from __future__ import annotations

import os
import shlex
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from resolve_version import resolve  # noqa: E402

DEFAULT_SOURCE = "amd/skillscope"


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


def source_argument(source: str, version: str) -> str:
    """What to hand ``uvx --from``.

    A local path is supported so this repo can dogfood the launcher against its
    own checkout; anything else is a GitHub repo at the resolved ref.
    """
    candidate = Path(source)
    if candidate.exists() and (candidate / "pyproject.toml").is_file():
        return str(candidate.resolve())
    return f"git+https://github.com/{source}@{version}"


def main() -> int:
    repo = Path(_env("SKILLSCOPE_REPO", ".") or ".").expanduser().resolve()
    command = _env("SKILLSCOPE_COMMAND")
    if not command:
        raise SystemExit("error: no skillscope command given.")

    globs = [g.strip() for g in _env("SKILLSCOPE_SKILLS").split(",") if g.strip()]
    version = resolve(
        root=repo,
        requested=_env("SKILLSCOPE_REQUESTED"),
        env=_env("SKILLSCOPE_VERSION"),
        skill=_env("SKILLSCOPE_SKILL"),
        default=_env("SKILLSCOPE_DEFAULT"),
        globs=globs or None,
    )
    source = _env("SKILLSCOPE_SOURCE") or DEFAULT_SOURCE

    cmd = [
        "uvx",
        "--from",
        source_argument(source, version),
        "skillscope",
        *shlex.split(command),
        *shlex.split(_env("SKILLSCOPE_ARGS")),
    ]
    print(f"[skillscope] {source}@{version}: {' '.join(cmd)}", flush=True)

    stdin_path = _env("SKILLSCOPE_STDIN")
    stdin = open(stdin_path, "rb") if stdin_path else subprocess.DEVNULL
    # `SKILLSCOPE_VERSION` because the harness echoes it into a CI plan, so
    # every leg the plan schedules launches the build that planned it rather
    # than re-resolving from scratch. `SKILLSCOPE_REPO` because the input may
    # be relative -- `repo: fixture` -- and the child runs from the repo it
    # names, where resolving that same relative path again lands a directory
    # deeper.
    child_env = {
        **os.environ,
        "SKILLSCOPE_VERSION": version,
        "SKILLSCOPE_REPO": str(repo),
    }
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
