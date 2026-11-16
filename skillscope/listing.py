# Copyright Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: MIT

"""What a repo's skills cost in the listing every agent reads at startup.

Every enabled skill contributes one line to a listing the agent is given at
the start of a session: its name, and its description. That listing has a
budget, and the budget is a fraction of the context window rather than a fixed
number, because the listing is re-sent on every turn and so is paid for again
and again.

When the listing does not fit, nothing fails. Names are kept and descriptions
are dropped, ordered by how recently and how often each skill was used, until
the rest fits. A skill with no description is still callable by name and can no
longer be matched against a prompt, because there is no longer any text to
match. Nobody is told, and a skill that was just installed has no usage history
at all, so it is first to lose its description and then never gets used.

:mod:`structure` reads one skill at a time, which is the right shape for a
limit the format sets on each skill. This limit belongs to the whole set, so no
per-skill check can see it: every skill can be within the format's limits and
the listing still overflow.

For a repo whose skills are installed elsewhere, that makes this a measure of
something the repo itself cannot observe. A published catalog is a guest in a
budget it does not own. It does not get the whole listing, it gets whatever is
left after the skills its installers already had, so the number worth watching
is the share it consumes rather than whether it fits on its own.

The arithmetic mirrors the agent's, so the totals mean the same thing:

    budget      = context window in tokens * bytes per token * fraction
    entry cost  = len(name) + 4 + min(len(description), per-description cap)
    listing     = sum(entry costs) + one separator between entries

The defaults below are the shipped ones. Both the fraction and the
per-description cap are settable by whoever installs the skills, so an
overflowing listing has two honest answers: publish less, or ask installers to
spend more of every turn on it.
"""

from __future__ import annotations

from dataclasses import dataclass

from . import config, structure

# The shipped defaults. A reader is assumed to be on none of the settings that
# change them, because almost everyone is, and a report against a tuned budget
# would flatter a catalog that overflows for its actual audience.
BUDGET_FRACTION = 0.01
BYTES_PER_TOKEN = 4
DEFAULT_CONTEXT_TOKENS = 200_000

# A single description is truncated at this many characters in the listing,
# separately from the shared budget, so one very long one cannot take the room
# every other skill needs. Defensive, for a caller that costs a repo without
# running the structural gate first: that gate rejects anything past
# `structure.MAX_DESCRIPTION_LENGTH`, which is lower, so a description reaching
# here through the CLI is never clamped.
MAX_DESCRIPTION_IN_LISTING = 1536

# `- ` before the name and `: ` after it, in each line of the listing.
_ENTRY_OVERHEAD = 4


def budget(context_tokens: int = DEFAULT_CONTEXT_TOKENS) -> int:
    """Characters available to the whole listing at a given context size."""
    return max(1, int(context_tokens * BYTES_PER_TOKEN * BUDGET_FRACTION))


@dataclass(frozen=True)
class Cost:
    """What this repo's skills add to a listing, and what that is a share of."""

    skills: int
    characters: int
    budget: int
    unreadable: tuple[str, ...]

    @property
    def share(self) -> float:
        """The fraction of the default budget this repo alone consumes."""
        return self.characters / self.budget if self.budget else 0.0


def cost(skills: list[str] | None = None) -> Cost:
    """The listing cost of every skill in the repo.

    A skill whose frontmatter cannot be read is counted in ``unreadable``
    rather than as zero. :func:`structure.errors` is what reports it as a
    fault; leaving it out of the total silently would make a broken repo look
    cheaper than a working one. The CLI runs that gate first, so ``unreadable``
    is empty there, and populated only for a caller that skips it.
    """
    cfg = config.active()
    wanted = skills if skills is not None else sorted(cfg.skills)

    characters = 0
    counted = 0
    unreadable: list[str] = []
    for skill in wanted:
        try:
            text = (cfg.skill_path(skill) / structure.SKILL_FILE).read_text(
                encoding="utf-8"
            )
        except (OSError, UnicodeDecodeError):
            unreadable.append(skill)
            continue

        declared, _, _ = structure._frontmatter(text)  # noqa: SLF001
        if declared is None:
            unreadable.append(skill)
            continue

        name = declared.get("name")
        description = declared.get("description")
        if not isinstance(name, str) or not isinstance(description, str):
            unreadable.append(skill)
            continue

        characters += (
            len(name)
            + _ENTRY_OVERHEAD
            + min(len(description), MAX_DESCRIPTION_IN_LISTING)
        )
        counted += 1

    # One separator between entries, so N entries carry N-1 of them.
    characters += max(0, counted - 1)
    return Cost(
        skills=counted,
        characters=characters,
        budget=budget(),
        unreadable=tuple(unreadable),
    )


def summary(measured: Cost) -> str:
    """One line for a run that reports rather than gates.

    No threshold is applied. Where the line sits between "a catalog people can
    install alongside their own skills" and "a catalog that takes the room" is
    a judgement this harness has no standing to make for another repo, and a
    number invented here would be argued with rather than watched.
    """
    return (
        f"{measured.skills} skill(s) cost {measured.characters} character(s) "
        f"in the startup listing, {measured.share:.0%} of the "
        f"{measured.budget}-character budget at a "
        f"{DEFAULT_CONTEXT_TOKENS:,}-token context window, "
        "assuming the shipped defaults."
    )
