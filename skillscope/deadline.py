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

import _thread
import os
import sys
import threading
import time

DEFAULT_TIMEOUT_S = 900.0

# How long before the wall to ask the command to stop of its own accord, so
# inspect can cancel its samples and run `Task.cleanup` -- the hook that
# removes containers -- instead of being shot in the head. Matches
# `behavioral.TIMEOUT_RESERVE_S`, which holds the same amount back from a
# per-sample limit for the same reason; the two bound different things and
# should not disagree about how long unwinding takes.
GRACEFUL_RESERVE_S = 120.0


class Deadline:
    """Seconds remaining on the command that is currently running."""

    def __init__(
        self, seconds: float, *, command: str = "", start: float | None = None
    ) -> None:
        self.seconds = seconds
        self.command = command
        self.start = time.perf_counter() if start is None else start
        self._timer: threading.Timer | None = None
        self._graceful: threading.Timer | None = None
        # Set when the graceful stage fired, so `cli.main` can tell a
        # deadline interrupt from a user pressing Ctrl-C.
        self.interrupted = False

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
        """Bound the command, in two stages.

        The hard stage kills the process when the bound elapses, even if
        something is hung. That is what a watchdog is for, and it is also why
        it cannot be the only stage: `os._exit` skips every `finally`,
        including the shielded cancel scope inspect runs `Task.cleanup` in. So
        the run most likely to have left a container behind -- the one that ran
        long -- was the one that got no `teardown`.

        The graceful stage fixes that by arriving first. `GRACEFUL_RESERVE_S`
        before the wall, it raises `KeyboardInterrupt` in the main thread,
        which is the signal inspect already understands: it cancels the
        running samples and unwinds *through* its cleanup rather than around
        it. `cli.main` catches it and reports the same overrun message.

        The reserve is the same idea as `behavioral.TIMEOUT_RESERVE_S`, which
        holds a per-sample limit back from the command's budget for the same
        reason, and is sized to match. The hard stage remains as the backstop
        for a main thread too stuck to take an interrupt.
        """
        if self.seconds <= 0 or self._timer is not None:
            return
        graceful_at = self.seconds - GRACEFUL_RESERVE_S
        if graceful_at > 0:
            self._graceful = threading.Timer(graceful_at, self._request_stop)
            self._graceful.daemon = True
            self._graceful.start()
        self._timer = threading.Timer(self.seconds, self._expire)
        self._timer.daemon = True
        self._timer.start()

    def disarm(self) -> None:
        for name in ("_timer", "_graceful"):
            timer = getattr(self, name, None)
            if timer is not None:
                timer.cancel()
                setattr(self, name, None)

    def _request_stop(self) -> None:
        """Ask the main thread to stop, in time for cleanup to run."""
        print(
            f"error: {self.message()}; stopping so cleanup can run "
            f"({GRACEFUL_RESERVE_S:g}s before the hard kill)",
            file=sys.stderr,
        )
        sys.stderr.flush()
        self.interrupted = True
        # The one way to reach a main thread that is blocked inside an event
        # loop. inspect treats it as a cancel, which is exactly the path that
        # runs `Task.cleanup`.
        _thread.interrupt_main()

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
    """Run every registered cleanup, bounded. Returns how many did not finish.

    Each callback runs in its own daemon thread and is *joined* with whatever
    budget is left, because a check between callbacks is not a bound: the
    blocking case -- shelling out to remove a container -- is exactly the one
    that overruns, and a loop that only looks at the clock between iterations
    lets a single slow cleanup run as long as it likes. Measured: a 6s callback
    against a 2s budget took the full 6s, which made a bounded command
    unbounded, which is worse than the hard exit this is trying to soften.

    A thread that outlives its join is abandoned rather than killed -- Python
    cannot kill a thread, and `os._exit` moments later takes it regardless.
    Daemon so it cannot hold the process open in the meantime.
    """
    deadline_at = time.perf_counter() + EXPIRE_CLEANUP_BUDGET_S
    skipped = 0
    for callback in list(_expire_callbacks):
        remaining = deadline_at - time.perf_counter()
        if remaining <= 0:
            skipped += 1
            continue

        outcome: list = []

        def _run(fn=callback) -> None:
            try:
                fn()
                outcome.append(True)
            except Exception:
                # A cleanup that raises is still a cleanup that did not happen,
                # and this is the last code to run before the process dies:
                # there is nobody left to handle an exception raised here.
                outcome.append(False)

        worker = threading.Thread(target=_run, daemon=True)
        worker.start()
        worker.join(remaining)
        if not outcome or outcome[0] is False:
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
