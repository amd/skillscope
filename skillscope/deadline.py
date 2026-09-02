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
        sys.stderr.flush()
        os._exit(1)


_active: Deadline | None = None


def active() -> Deadline | None:
    """The bound for this process, or ``None`` when ``--timeout`` is off."""
    return _active


def use(bound: Deadline | None) -> Deadline | None:
    """Install ``bound`` as the active one and return the previous one."""
    global _active
    previous, _active = _active, bound
    return previous
