# Copyright Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: MIT

"""``python -m skillscope``, for running from a checkout without installing."""

from __future__ import annotations

from .cli import main

if __name__ == "__main__":
    raise SystemExit(main())
