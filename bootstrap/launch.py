# Copyright Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: MIT

"""Run one skillscope command from the composite action's own checkout.

Callers pin a tag on the action or on a reusable workflow in this repo
(``amd/skillscope@v0.1.3``, ``.../reusable.yml@v0.1.3``). This script installs
*that* checkout with ``uvx`` and execs the command. It does not fetch some
other ref: the ``uses:`` pin is the harness.

It is written in Python rather than shell because the same step runs on
Linux, Windows, and macOS runners, self-hosted and not.

Configuration arrives as environment variables, set from the action's inputs:

    SKILLSCOPE_COMMAND      the subcommand, e.g. "structural"
    SKILLSCOPE_ARGS         further arguments, shell-quoted
    SKILLSCOPE_REPO         root of the repo under test (default ".")
    SKILLSCOPE_SKILLS       globs naming the directories that are skills
    SKILLSCOPE_STDIN        a file to feed the command on stdin
    SKILLSCOPE_ACTION_PATH  this action's checkout (always set under Actions)

Everything else a run needs is passed straight through in SKILLSCOPE_ARGS,
unread, so a new CLI flag does not require a matching action input.
"""

from __future__ import annotations

import os
import re
import shlex
import subprocess
import sys
from pathlib import Path

# Read out of the checkout rather than imported, because the launcher runs
# before anything is installed.
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


def action_root() -> Path:
    """The skillscope checkout this action is running from."""
    override = _env("SKILLSCOPE_ACTION_PATH")
    if override:
        return Path(override).expanduser().resolve()
    return Path(__file__).resolve().parent.parent


def packaged_version(checkout: Path) -> str:
    """The version declared by the harness in `checkout`, or "" if unreadable.

    The `uses:` pin decides which harness runs, but a reusable workflow reaches
    this action through a nested checkout, where `$GITHUB_ACTION_REF` is empty
    and the path says nothing. Reporting what the checkout calls itself is what
    lets a caller confirm from the log that the pin did what they meant.
    """
    try:
        text = (checkout / "skillscope" / "__init__.py").read_text(encoding="utf-8")
    except OSError:
        return ""
    found = VERSION_PATTERN.search(text)
    return found.group(1) if found else ""


def main() -> int:
    repo = Path(_env("SKILLSCOPE_REPO", ".") or ".").expanduser().resolve()
    command = _env("SKILLSCOPE_COMMAND")
    if not command:
        raise SystemExit("error: no skillscope command given.")

    source = action_root()
    if not (source / "pyproject.toml").is_file():
        raise SystemExit(
            f"error: {source} has no pyproject.toml. The action must run from "
            "a skillscope checkout (for example amd/skillscope@v0.1.3)."
        )
    version = packaged_version(source) or "unknown"

    cmd = [
        "uvx",
        "--from",
        str(source),
        "skillscope",
        *shlex.split(command),
        *shlex.split(_env("SKILLSCOPE_ARGS")),
    ]
    print(f"[skillscope] {version} from {source}: {' '.join(cmd)}", flush=True)

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
