# Copyright Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: MIT

"""A tool set that works on a non-POSIX guest.

inspect's own tools assume one: `bash()` execs `["bash", "--login", "-c", ...]`,
`text_editor()` needs a Linux-only helper binary, and `list_files()`/`grep()`
shell out to `find`/`grep`. On a Windows host with the `local` sandbox that
leaves an agent that can read and think but cannot write a file or run a
command -- not enough to grade a behavioral case with.

Everything here is built on `SandboxEnvironment.exec` / `read_file` /
`write_file`, which are provider-level and platform-neutral. Only the shell
invocation differs, and that is probed once per sample rather than assumed:
the same Docker sandbox is POSIX whichever host started it, so the host's own
platform is not the answer.
"""

from __future__ import annotations

SHELL_KEY = "skillscope_shell"

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

    probe = await sandbox().exec(["bash", "-lc", "exit 0"], concurrency=False)
    prefix = POSIX_SHELL if probe.success else WINDOWS_SHELL
    store().set(SHELL_KEY, prefix)
    return list(prefix)


async def run(command: str, timeout: int | None = None):
    """Run `command` through whichever shell the sandbox has."""
    from inspect_ai.util import sandbox

    prefix = await shell_prefix()
    return await sandbox().exec(prefix + [command], timeout=timeout)


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


def _text(result) -> str:
    """`stdout` plus `stderr`, which is where a failing command says why."""
    parts = [result.stdout.strip(), result.stderr.strip()]
    body = "\n".join(p for p in parts if p)
    if result.success:
        return body or "(no output)"
    return f"exit code {result.returncode}\n{body}".strip()


def shell(timeout: int = 300):
    """Run shell commands in the sandbox, on whichever platform it is."""
    from inspect_ai.tool import Tool, tool

    @tool(name="shell")
    def _shell() -> Tool:
        async def execute(command: str) -> str:
            """Run a command in the sandbox and return its output.

            Uses bash on Linux and macOS, and PowerShell on Windows, so write
            commands for the platform you find yourself on. Check with `uname`
            or `$PSVersionTable` if you are unsure.

            Args:
                command: The command line to run.

            Returns:
                The command's combined output, or its exit code and error output
                when it fails.
            """
            return _text(await run(command, timeout=timeout))

        return execute

    return _shell()


def write_file():
    """Create or overwrite a file, without going through a shell."""
    from inspect_ai.tool import Tool, tool

    @tool(name="write_file")
    def _write_file() -> Tool:
        async def execute(path: str, content: str) -> str:
            """Write text to a file in the sandbox, replacing it if it exists.

            Prefer this over shell redirection: it needs no quoting or escaping
            and behaves the same on every platform.

            Args:
                path: File to write, relative to the working directory.
                content: The full text the file should contain.

            Returns:
                Confirmation of what was written.
            """
            from inspect_ai.util import sandbox

            await sandbox().write_file(path, content)
            return f"wrote {len(content)} characters to {path}"

        return execute

    return _write_file()


def edit_file():
    """Replace one exact occurrence of a string in a file."""
    from inspect_ai.tool import Tool, tool

    @tool(name="edit_file")
    def _edit_file() -> Tool:
        async def execute(path: str, old_text: str, new_text: str) -> str:
            """Replace an exact snippet in a file.

            `old_text` must appear exactly once, so include enough surrounding
            context to make it unique. To create a file, use write_file.

            Args:
                path: File to edit, relative to the working directory.
                old_text: The exact text to replace.
                new_text: What to put in its place.

            Returns:
                Confirmation, or an explanation of why the edit was refused.
            """
            from inspect_ai.util import sandbox

            current = await sandbox().read_file(path, text=True)
            found = current.count(old_text)
            if found == 0:
                return f"no edit made: {path} does not contain that text"
            if found > 1:
                return (
                    f"no edit made: that text appears {found} times in {path}. "
                    "Include more surrounding context so it matches once."
                )
            await sandbox().write_file(path, current.replace(old_text, new_text, 1))
            return f"edited {path}"

        return execute

    return _edit_file()


def list_files():
    """List the files the sandbox working directory holds."""
    from inspect_ai.tool import Tool, tool

    @tool(name="list_files")
    def _list_files() -> Tool:
        async def execute() -> str:
            """List every file in the working directory, recursively.

            Returns:
                One relative path per line.
            """
            try:
                paths = await list_paths()
            except ListingFailed as exc:
                return f"could not list the directory: {exc}"
            return "\n".join(paths) if paths else "(no files)"

        return execute

    return _list_files()


def toolset() -> list:
    """The tools a behavioral run gives the agent.

    inspect's `think()` is reused as-is -- it never touches the sandbox, so it
    is already platform-neutral. Its `bash()`, `text_editor()`, `list_files()`
    and `grep()` are the ones replaced above.
    """
    from inspect_ai.tool import think

    return [shell(), write_file(), edit_file(), list_files(), think()]
