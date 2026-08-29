# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.

"""``python -m skillscope``, for running from a checkout without installing."""

from __future__ import annotations

from .cli import main

if __name__ == "__main__":
    raise SystemExit(main())
