# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.

"""Whether the references a skill's markdown makes still resolve.

A skill is prose an agent reads and then acts on, so a reference that goes
nowhere is not cosmetic: the agent follows it, finds nothing, and improvises.
That is the same class of defect as a malformed dataset, which is why this
lives in the structural tier rather than in a linter somebody runs by hand.

Two checks, split by whether they need the network:

  * **internal** -- relative paths and heading anchors, resolved against the
    files on disk. Deterministic, offline, instant. Part of every structural
    run, and so part of the gate a paid run passes through first.
  * **external** -- ``http(s)`` URLs, fetched. Asked for with ``--external``
    and never part of that gate, because it fails for reasons that have
    nothing to do with the change under review: a rate limit, a host that
    blocks runner IPs, DNS having a bad minute. A check that cries wolf is a
    check people learn to ignore, so this one is kept where its noise cannot
    block a merge.

What is read: every markdown file under a declared skill, plus whatever
``--docs`` names. A skill's folder is the default because that is what this
harness grades; a repo whose README and docs should be held to the same bar
says so with the flag rather than having its whole tree scanned on a guess.

Markdown is scanned rather than parsed. Fenced code blocks, inline code spans,
and HTML comments are skipped, because a link in a code sample is an
illustration rather than a promise -- which is also where a deliberately
fictional one tends to live. The cost of scanning is that a link split across
two lines is not seen. That is rare in a skill, and missing one reference is a
better failure than a parser dependency in a package that is otherwise
standard library only.
"""

from __future__ import annotations

import re
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

from . import config

MARKDOWN_SUFFIXES = (".md", ".markdown")

# Reachable enough. 2xx is the answer we want; 429 means the host is there and
# rate-limiting us, which says nothing about the reference.
ACCEPTED_STATUS = frozenset(range(200, 207)) | {429}

# Some servers refuse HEAD outright. That is a server preference, not a broken
# reference, so the same URL is asked for with GET before it is believed.
_HEAD_REFUSED = frozenset({400, 403, 404, 405, 406, 501})

# Worth asking again: a transient server-side or transport failure. A 4xx is
# an answer, and asking a second time will get the same one.
_RETRYABLE_STATUS = frozenset({408, 425, 500, 502, 503, 504})

DEFAULT_TIMEOUT_S = 20.0
DEFAULT_RETRIES = 2
RETRY_WAIT_S = 2.0
DEFAULT_JOBS = 8

# Plain urllib announces itself as a Python script, which a fair number of
# hosts answer with a 403. The reference is fine; the greeting was the problem.
_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (compatible; skillscope/1.0; "
        "+https://github.com/amd/skillscope)"
    ),
    "Accept": "*/*",
}

_FENCE = re.compile(r"^\s{0,3}(?P<fence>`{3,}|~{3,})")
_INLINE_CODE = re.compile(r"`+[^`]*`+")
_HTML_COMMENT = re.compile(r"<!--.*?-->", re.DOTALL)

# `[text](target)`, `![alt](target)`, and `[text](<target with spaces>)`. One
# level of nested parentheses is allowed so a Wikipedia-shaped URL survives.
_INLINE_LINK = re.compile(
    r"!?\[(?:[^\]\\]|\\.)*\]\(\s*(?P<target><[^>\n]*>|(?:[^\s()]|\([^\s()]*\))+)"
)
# `[label]: target "title"`, the definition half of a reference-style link.
_LINK_DEFINITION = re.compile(
    r"^\s{0,3}\[(?:[^\]\\]|\\.)+\]:\s*(?P<target><[^>\n]*>|\S+)"
)
_AUTOLINK = re.compile(r"<(?P<target>[a-zA-Z][a-zA-Z0-9+.\-]*:[^<>\s]+)>")
_HTML_ATTRIBUTE = re.compile(
    r"(?:href|src)\s*=\s*[\"'](?P<target>[^\"']+)[\"']", re.IGNORECASE
)
_BARE_URL = re.compile(r"https?://[^\s<>\"'`\\)\]}]+")
# Trailing punctuation belongs to the sentence, not to the URL.
_URL_TAIL = ".,;:!?'\""

_HEADING = re.compile(r"^\s{0,3}(?P<hashes>#{1,6})\s+(?P<title>.*?)\s*#*\s*$")
_HEADING_LINK = re.compile(r"\[(?P<label>[^\]]*)\]\([^)]*\)")
_EXPLICIT_ANCHOR = re.compile(
    r"<[^>]*?\b(?:id|name)\s*=\s*[\"'](?P<anchor>[^\"']+)[\"']", re.IGNORECASE
)
# `#L12` and `#L12-L20`: GitHub renders line anchors for source files, and
# nothing in the file itself declares them.
_LINE_ANCHOR = re.compile(r"^L\d+(?:-L?\d+)?$", re.IGNORECASE)


@dataclass(frozen=True)
class Reference:
    """One link, as written, and where it was written."""

    source: Path
    line: int
    target: str

    @property
    def parts(self) -> urllib.parse.SplitResult:
        return urllib.parse.urlsplit(self.target)

    @property
    def is_external(self) -> bool:
        parts = self.parts
        # `//host/path` inherits the page's scheme, which off a web page means
        # https; it is still a reference out to somebody else's server.
        return parts.scheme in ("http", "https") or (not parts.scheme and bool(parts.netloc))

    @property
    def url(self) -> str:
        parts = self.parts
        if not parts.scheme:
            parts = parts._replace(scheme="https")
        return urllib.parse.urlunsplit(parts._replace(fragment=""))

    @property
    def is_local(self) -> bool:
        """A path (or a bare anchor) in this repo, rather than a URL."""
        parts = self.parts
        return not parts.scheme and not parts.netloc and bool(parts.path or parts.fragment)


def markdown_files() -> list[Path]:
    """Every markdown file the reference checks read, in a stable order."""
    cfg = config.active()
    found: list[Path] = []
    for _, folder in sorted(cfg.skills.items()):
        found.extend(
            path
            for path in sorted(folder.rglob("*"))
            if path.suffix.lower() in MARKDOWN_SUFFIXES and path.is_file()
        )
    for pattern in cfg.doc_globs:
        found.extend(
            path
            for path in sorted(cfg.root.glob(pattern))
            if path.suffix.lower() in MARKDOWN_SUFFIXES and path.is_file()
        )
    return list(dict.fromkeys(found))


def _prose_lines(text: str) -> list[tuple[int, str]]:
    """The lines a reader would read, as ``(line number, text)``.

    Code and comments are dropped rather than blanked so nothing is extracted
    from them, but the line numbers stay those of the original file: an error
    a reader cannot navigate to costs more than the check saved.
    """
    lines: list[tuple[int, str]] = []
    fence: str | None = None
    in_comment = False

    for number, raw in enumerate(text.splitlines(), start=1):
        opener = _FENCE.match(raw)
        if fence is not None:
            closer = opener.group("fence") if opener else ""
            if closer[:1] == fence[:1] and len(closer) >= len(fence):
                fence = None
            continue
        if opener and not in_comment:
            fence = opener.group("fence")
            continue

        line = raw
        if in_comment:
            end = line.find("-->")
            if end == -1:
                continue
            line, in_comment = line[end + 3 :], False
        line = _HTML_COMMENT.sub(" ", line)
        start = line.find("<!--")
        if start != -1:
            line, in_comment = line[:start], True

        lines.append((number, _INLINE_CODE.sub(" ", line)))
    return lines


def _targets(line: str) -> list[str]:
    """Every link target written on one line of prose."""
    found = [match.group("target") for match in _INLINE_LINK.finditer(line)]
    definition = _LINK_DEFINITION.match(line)
    if definition:
        found.append(definition.group("target"))
    found += [match.group("target") for match in _AUTOLINK.finditer(line)]
    found += [match.group("target") for match in _HTML_ATTRIBUTE.finditer(line)]
    found += [match.group(0) for match in _BARE_URL.finditer(line)]

    cleaned = [target.strip().strip("<>").rstrip(_URL_TAIL).strip() for target in found]
    return [target for target in cleaned if target]


def collect(files: list[Path] | None = None) -> list[Reference]:
    """Every reference in `files`, deduplicated per file, line, and target."""
    references: list[Reference] = []
    seen: set[tuple[Path, int, str]] = set()
    for path in markdown_files() if files is None else files:
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            # Unreadable markdown is not this check's story to tell, and
            # refusing to run over the rest of the repo because of one file
            # would hide every real broken reference behind it.
            continue
        for number, line in _prose_lines(text):
            for target in _targets(line):
                key = (path, number, target)
                if key not in seen:
                    seen.add(key)
                    references.append(Reference(path, number, target))
    return references


def anchors(text: str) -> set[str]:
    """Every ``#fragment`` a markdown document offers.

    Headings, slugged the way GitHub slugs them -- lowercased, punctuation
    dropped, spaces hyphenated, repeats suffixed -- plus any ``id`` or ``name``
    an embedded HTML tag declares, which is how a hand-written anchor is
    spelled in a file that also has to render as plain markdown.
    """
    found: set[str] = set()
    counts: dict[str, int] = {}
    fence: str | None = None

    for raw in text.splitlines():
        opener = _FENCE.match(raw)
        if fence is not None:
            closer = opener.group("fence") if opener else ""
            if closer[:1] == fence[:1] and len(closer) >= len(fence):
                fence = None
            continue
        if opener:
            fence = opener.group("fence")
            continue

        found.update(match.group("anchor") for match in _EXPLICIT_ANCHOR.finditer(raw))
        heading = _HEADING.match(raw)
        if not heading:
            continue
        slug = _slug(heading.group("title"))
        if not slug:
            continue
        seen = counts.get(slug, 0)
        counts[slug] = seen + 1
        found.add(slug if not seen else f"{slug}-{seen}")
    return found


def _slug(title: str) -> str:
    """A heading's anchor, by GitHub's rules."""
    text = _HEADING_LINK.sub(r"\g<label>", title)
    text = _INLINE_CODE.sub(lambda match: match.group(0).strip("`"), text)
    text = re.sub(r"[^\w\- ]", "", text.strip().lower(), flags=re.UNICODE)
    return text.replace(" ", "-")


def _where(path: Path, line: int) -> str:
    root = config.active().root
    try:
        shown = path.relative_to(root).as_posix()
    except ValueError:  # a skill outside the repo root
        shown = path.as_posix()
    return f"{shown}:{line}"


def internal_errors(references: list[Reference]) -> list[str]:
    """Local references that resolve to nothing, as human-readable strings."""
    root = config.active().root
    errors: list[str] = []
    cache: dict[Path, set[str] | None] = {}

    for reference in references:
        if not reference.is_local:
            continue
        parts = reference.parts
        path = urllib.parse.unquote(parts.path)
        fragment = urllib.parse.unquote(parts.fragment)

        if not path:  # `#section`, an anchor into the file it is written in
            target = reference.source
        elif path.startswith("/"):
            # Root-relative, the way it renders on a repository host: from the
            # top of the repo under test, not the top of the filesystem.
            target = root.joinpath(*path.strip("/").split("/"))
        else:
            target = reference.source.parent.joinpath(*path.split("/"))

        if not target.exists():
            errors.append(
                f"{_where(reference.source, reference.line)}: `{reference.target}` "
                f"points at `{_shown(target, root)}`, which does not exist."
            )
            continue

        if not fragment or _LINE_ANCHOR.match(fragment):
            continue
        if target.is_dir() or target.suffix.lower() not in MARKDOWN_SUFFIXES:
            # Only markdown declares anchors this check can see. A fragment on
            # anything else is the host's business.
            continue
        if target not in cache:
            try:
                cache[target] = anchors(target.read_text(encoding="utf-8"))
            except (OSError, UnicodeDecodeError):
                cache[target] = None
        available = cache[target]
        if available is not None and fragment not in available:
            errors.append(
                f"{_where(reference.source, reference.line)}: `{reference.target}` "
                f"points at a heading `#{fragment}` that "
                f"`{_shown(target, root)}` does not have."
            )
    return errors


def _shown(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def external_errors(
    references: list[Reference],
    *,
    exclude: tuple[str, ...] | list[str] = (),
    jobs: int = DEFAULT_JOBS,
    timeout: float = DEFAULT_TIMEOUT_S,
    retries: int = DEFAULT_RETRIES,
    probe: Callable[[str], str] | None = None,
) -> list[str]:
    """URLs nobody answered for, reported once each with where they came from.

    `probe` answers "" for a reachable URL and a reason otherwise. It is a
    parameter so the tests can grade this logic without a network, which is
    also the only way they can grade it the same way twice.
    """
    patterns = [re.compile(pattern) for pattern in exclude]
    grouped: dict[str, list[Reference]] = {}
    for reference in references:
        if not reference.is_external:
            continue
        url = reference.url
        if any(pattern.search(url) for pattern in patterns):
            continue
        grouped.setdefault(url, []).append(reference)

    if not grouped:
        return []

    check = probe or (lambda url: _probe(url, timeout=timeout, retries=retries))
    urls = sorted(grouped)
    with ThreadPoolExecutor(max_workers=max(1, min(jobs, len(urls)))) as pool:
        details = list(pool.map(check, urls))

    errors: list[str] = []
    for url, detail in zip(urls, details):
        if not detail:
            continue
        seen = ", ".join(
            _where(reference.source, reference.line) for reference in grouped[url]
        )
        errors.append(f"{url} is unreachable ({detail}). Referenced from: {seen}.")
    return errors


def external_urls(references: list[Reference]) -> list[str]:
    """The distinct URLs an external run would fetch. For reporting counts."""
    return sorted({reference.url for reference in references if reference.is_external})


def _probe(url: str, *, timeout: float, retries: int) -> str:
    """Empty when the URL answers, else why it did not."""
    detail = ""
    for attempt in range(retries + 1):
        if attempt:
            time.sleep(RETRY_WAIT_S * attempt)
        detail, retryable = _attempt(url, timeout)
        if not detail or not retryable:
            return detail
    return detail


def _attempt(url: str, timeout: float) -> tuple[str, bool]:
    """One round trip: ``(why it failed or "", whether asking again may help)``."""
    for method in ("HEAD", "GET"):
        request = urllib.request.Request(url, method=method, headers=_HEADERS)
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                status = getattr(response, "status", None) or response.getcode()
                if status in ACCEPTED_STATUS:
                    return "", False
                return f"HTTP {status}", status in _RETRYABLE_STATUS
        except urllib.error.HTTPError as exc:
            if exc.code in ACCEPTED_STATUS:
                return "", False
            if method == "HEAD" and exc.code in _HEAD_REFUSED:
                continue
            return f"HTTP {exc.code}", exc.code in _RETRYABLE_STATUS
        except urllib.error.URLError as exc:
            return f"{exc.reason}", True
        except ValueError as exc:  # a URL urllib will not even attempt
            return f"{exc}", False
        except OSError as exc:  # timeouts, resets, TLS failures
            return f"{exc}", True
    return "", False
