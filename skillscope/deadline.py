# Copyright Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: MIT

"""Wall-clock bound for one skillscope command.

``--timeout`` is the same flag on ``structural``, ``routing``, and
``behavioral``: it is the command's life, not one case's. Routing still has a
shorter per-case cap (``--case-timeout``) so a single hung prompt cannot spend
the whole budget; that cap is itself clipped to whatever time is left here.

Armed from the CLI. Engines read the active bound and stop starting work when
it has elapsed. A watchdog is the backstop for a hook or subprocess that will
not return on its own.
"""

from __future__ import annotations

import os
import sys
import threading
import time

DEFAULT_TIMEOUT_S = 900.0


class Deadline:
    """Seconds remaining on the command that is currently running."""

    def __init__(
        self, seconds: float, *, command: str = "", start: float | None = None
    ) -> None:
        self.seconds = seconds
        self.command = command
        self.start = time.perf_counter() if start is None else start
        self._timer: threading.Timer | None = None

    def remaining(self) -> float:
        return self.seconds - (time.perf_counter() - self.start)

    def expired(self) -> bool:
        return self.remaining() <= 0

    def cap(self, seconds: float) -> float:
        """The tighter of this bound and ``seconds``. Never negative."""
        return max(0.0, min(seconds, self.remaining()))

    def message(self) -> str:
        label = self.command or "command"
        return f"{label} exceeded --timeout of {self.seconds:g}s"

    def arm(self) -> None:
        """Kill the process when the bound elapses, even if something is hung."""
        if self.seconds <= 0 or self._timer is not None:
            return
        self._timer = threading.Timer(self.seconds, self._expire)
        self._timer.daemon = True
        self._timer.start()

    def disarm(self) -> None:
        if self._timer is not None:
            self._timer.cancel()
            self._timer = None

    def _expire(self) -> None:
        print(f"error: {self.message()}", file=sys.stderr)
        # `os._exit` is the point of the watchdog -- it is reached only when
        # something is hung, and a hung process is exactly the one that will
        # not unwind on `sys.exit`. But it also skips every `finally`,
        # including the shielded scope inspect runs `Task.cleanup` in. So a run
        # that hits the wall clock gets no `teardown`, which is the single case
        # most likely to have left a container behind -- and it got no say
        # about it either, which is worse than the leak.
        #
        # Emergency callbacks run first, each bounded, so a cleanup that hangs
        # cannot defeat the watchdog that called it. They are a last resort:
        # the designed path is `behavioral.TIMEOUT_RESERVE_S`, which stops the
        # sample early enough for inspect's own cleanup to run normally.
        skipped = _run_expire_callbacks()
        if skipped:
            print(
                f"error: {skipped} cleanup callback(s) did not finish before "
                "the process was killed; containers or other resources this "
                "run created may still exist.",
                file=sys.stderr,
            )
        sys.stderr.flush()
        os._exit(1)


# Last-resort cleanups, run by the watchdog before it kills the process.
_expire_callbacks: list = []

# How long all of them together may take. The watchdog has already decided the
# run is hung; spending its whole remaining credibility on a cleanup that is
# hung too would turn a bounded command into an unbounded one.
EXPIRE_CLEANUP_BUDGET_S = 10.0


def on_expire(callback) -> None:
    """Register a cleanup to attempt if the wall-clock watchdog fires.

    For resources that outlive the process -- a container, a background server
    -- where `finally` is not enough because `--timeout` exits hard.
    """
    _expire_callbacks.append(callback)


def _run_expire_callbacks() -> int:
    """Run every registered cleanup, bounded. Returns how many did not finish."""
    deadline_at = time.perf_counter() + EXPIRE_CLEANUP_BUDGET_S
    skipped = 0
    for callback in list(_expire_callbacks):
        if time.perf_counter() >= deadline_at:
            skipped += 1
            continue
        try:
            callback()
        except Exception:
            # A cleanup that raises is still a cleanup that did not happen,
            # and this is the last code to run before the process dies: there
            # is nobody left to handle an exception raised here.
            skipped += 1
    return skipped


_active: Deadline | None = None


def active() -> Deadline | None:
    """The bound for this process, or ``None`` when ``--timeout`` is off."""
    return _active


def use(bound: Deadline | None) -> Deadline | None:
    """Install ``bound`` as the active one and return the previous one."""
    global _active
    previous, _active = _active, bound
    return previous
