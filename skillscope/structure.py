# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.

"""Whether a skill's own folder is in shape, before anything is graded.

The standardized Agent Skills format is small: a folder, a ``SKILL.md``, and a
frontmatter block naming the skill and describing when to use it. Small does
not mean safe. A ``name`` that disagrees with the folder, a ``description``
that never got written, a frontmatter block that is not valid YAML -- each one
either stops the skill loading or stops it being found, and each one is
invisible until an agent quietly does not use it.

So these rules are checked where they cost nothing:

  * ``SKILL.md`` exists, opens with a ``---`` frontmatter block, and that block
    is a YAML mapping.
  * ``name`` is a non-empty lowercase-with-hyphens string of at most
    :data:`MAX_NAME_LENGTH` characters, carries neither reserved substring, and
    matches the folder it is in.
  * ``description`` is a non-empty string of at most
    :data:`MAX_DESCRIPTION_LENGTH` characters.
  * the body is at most :data:`MAX_BODY_LINES` lines.

Then whatever else the repo requires of every skill, which is a policy rather
than a format and so arrives as configuration: ``skill_files`` names files each
skill must ship beside its ``SKILL.md``, and ``skill_sections`` names the
``##`` headings each of those markdown files must have something under. A repo
with a governance card asks for one that way::

    --skill-files skill-card.md --skill-sections Description,Owner,License

Nothing here reads a repo-wide manifest. Which skills a repo publishes, and
where it lists them, is that repo's business; a harness that also had opinions
about it would be a second place to update every time a skill ships.
"""

from __future__ import annotations

import re
from pathlib import Path

from . import config, references

SKILL_FILE = "SKILL.md"

# Limits from the standardized Agent Skills format. A description is read for
# every prompt, in competition with every other skill's, so the ceiling is the
# format's rather than one repo's taste.
MAX_NAME_LENGTH = 64
MAX_DESCRIPTION_LENGTH = 1024

# Past this, a body is reference material rather than instructions, and an
# agent reads it in full every time the skill loads. Sibling files linked from
# SKILL.md are read only when they are needed.
MAX_BODY_LINES = 500

# Names an agent runtime reserves for itself.
RESERVED_NAME_SUBSTRINGS = ("anthropic", "claude")

_NAME = re.compile(r"^[a-z0-9]+(-[a-z0-9]+)*$")
_FRONTMATTER = re.compile(
    r"\A---\r?\n(?P<frontmatter>.*?)\r?\n---\r?\n?(?P<body>.*)\Z", re.DOTALL
)
_SECTION = re.compile(r"^##\s+(?P<title>.+?)\s*$")


def errors() -> list[str]:
    """Every structural problem with every skill folder in the repo."""
    cfg = config.active()
    found: list[str] = []
    for skill in sorted(cfg.skills):
        found.extend(skill_errors(skill))
    found.extend(_undeclared(cfg))
    return found


def skill_errors(skill: str) -> list[str]:
    """Every structural problem with one skill, as human-readable strings."""
    cfg = config.active()
    folder = cfg.skill_path(skill)
    where = f"{skill}/{SKILL_FILE}"

    try:
        text = (folder / SKILL_FILE).read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        return [f"{where}: cannot be read ({exc})."]

    declared, body, problem = _frontmatter(text)
    if declared is None:
        return [f"{where}: {problem}"]

    found = [
        f"{where}: {problem}"
        for problem in (
            *_name_errors(declared.get("name"), skill),
            *_description_errors(declared.get("description")),
            *_body_errors(body),
        )
    ]
    return found + _required_errors(skill, folder, cfg)


def _frontmatter(text: str) -> tuple[dict | None, str, str]:
    """``(frontmatter, body, why it could not be read)`` for one SKILL.md.

    PyYAML is imported here rather than at module scope, the way
    ``datasets._read_machine`` does it: a frontmatter block is a mapping of
    scalars, but plenty are written with folded or quoted strings, and a hand
    parser that got one of those subtly wrong would report a description the
    agent never sees.
    """
    match = _FRONTMATTER.match(text)
    if match is None:
        return None, "", (
            "must open with a `---` YAML frontmatter block, closed by `---` on "
            "a line of its own. Without it an agent has no name or description "
            "to match a prompt against, so the skill is never loaded."
        )

    import yaml  # noqa: PLC0415 -- keeps the run path stdlib-only

    try:
        declared = yaml.safe_load(match.group("frontmatter"))
    except yaml.YAMLError as exc:
        return None, "", f"frontmatter is not valid YAML: {exc}"
    if not isinstance(declared, dict):
        return None, "", (
            "frontmatter must be a mapping holding at least `name` and "
            "`description`."
        )
    return declared, match.group("body"), ""


def _name_errors(name: object, skill: str) -> list[str]:
    if not isinstance(name, str) or not name.strip():
        return ["frontmatter `name` is missing, or is not a non-empty string."]

    found: list[str] = []
    shown = _shortened(name)
    if len(name) > MAX_NAME_LENGTH:
        found.append(
            f"`name` is {len(name)} characters; the format allows "
            f"{MAX_NAME_LENGTH}."
        )
    if not _NAME.match(name):
        found.append(
            f"`name` is `{shown}`, which is not lowercase-with-hyphens "
            "(letters, digits, single hyphens between segments)."
        )
    found += [
        f"`name` may not contain `{reserved}`, which an agent runtime reserves."
        for reserved in RESERVED_NAME_SUBSTRINGS
        if reserved in name.lower()
    ]
    if name != skill:
        found.append(
            f"`name` is `{shown}` but the folder is `{skill}`. The folder name "
            "is the skill's identity everywhere else -- in its dataset, in a "
            "routing verdict, in the report -- so the two have to agree."
        )
    return found


def _shortened(name: str) -> str:
    """A name short enough that the sentence around it is still readable.

    A frontmatter block whose `description` is misindented folds the whole
    thing into `name`, and quoting several hundred characters back twice
    buries the two lines saying what is wrong.
    """
    return name if len(name) <= MAX_NAME_LENGTH else f"{name[:MAX_NAME_LENGTH]}..."


def _description_errors(description: object) -> list[str]:
    if not isinstance(description, str) or not description.strip():
        return [
            (
                "frontmatter `description` is missing, or is not a non-empty "
                "string. It is the whole of what an agent sees when it decides "
                "whether this skill answers a prompt."
            )
        ]
    if len(description) > MAX_DESCRIPTION_LENGTH:
        return [
            (
                f"`description` is {len(description)} characters; the format "
                f"allows {MAX_DESCRIPTION_LENGTH}."
            )
        ]
    return []


def _body_errors(body: str) -> list[str]:
    # Surrounding blank lines are not content: the blank line after the closing
    # `---` should not count against a skill.
    lines = body.splitlines()
    while lines and not lines[0].strip():
        lines.pop(0)
    while lines and not lines[-1].strip():
        lines.pop()
    if len(lines) <= MAX_BODY_LINES:
        return []
    return [
        (
            f"body is {len(lines)} lines; the format allows {MAX_BODY_LINES}. "
            "Move reference material into sibling files and link to them, so "
            "an agent reads it when it needs it rather than on every load."
        )
    ]


def _required_errors(skill: str, folder: Path, cfg: config.Config) -> list[str]:
    """Files this repo requires of every skill, and the sections in them."""
    found: list[str] = []
    for relative in cfg.skill_files:
        path = folder.joinpath(*relative.split("/"))
        if not path.is_file():
            found.append(
                f"{skill}: no `{relative}`, which every skill in this repo has "
                "to ship."
            )
            continue
        if not cfg.skill_sections:
            continue
        if path.suffix.lower() not in references.MARKDOWN_SUFFIXES:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError) as exc:
            found.append(f"{skill}/{relative}: cannot be read ({exc}).")
            continue
        found.extend(_section_errors(f"{skill}/{relative}", text, cfg.skill_sections))
    return found


def _section_errors(where: str, text: str, required: tuple[str, ...]) -> list[str]:
    present = sections(text)
    found: list[str] = []
    for title in required:
        body = present.get(title.lower())
        if body is None:
            found.append(f"{where}: no `## {title}` section.")
        elif not body.strip():
            found.append(f"{where}: the `## {title}` section is empty.")
    return found


def sections(text: str) -> dict[str, str]:
    """Each ``##`` heading, lowercased, mapped to the text under it."""
    found: dict[str, str] = {}
    title: str | None = None
    body: list[str] = []

    for line in text.splitlines():
        heading = _SECTION.match(line)
        if heading:
            if title is not None:
                found[title] = "\n".join(body).strip()
            title, body = heading.group("title").lower(), []
        elif title is not None:
            body.append(line)
    if title is not None:
        found[title] = "\n".join(body).strip()
    return found


def _undeclared(cfg: config.Config) -> list[str]:
    """Directories the skill globs match that hold no SKILL.md.

    Discovery ignores them, which is the problem: a skill whose SKILL.md was
    never added, or was added under another name, is not graded, not routed,
    and not reported -- it simply is not there. Saying so costs one line and
    the fix is either the missing file or a narrower glob.
    """
    found: set[str] = set()
    for pattern in cfg.skill_globs:
        for path in cfg.root.glob(pattern):
            if not path.is_dir() or path.name.startswith("."):
                continue
            if (path / SKILL_FILE).is_file():
                continue
            shown = path.relative_to(cfg.root).as_posix()
            found.add(
                f"{shown}: no {SKILL_FILE}, so nothing in this directory is "
                "graded. Add one, or narrow --skills so the directory is not "
                "taken for a skill."
            )
    return sorted(found)
