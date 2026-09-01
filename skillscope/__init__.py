# Copyright Advanced Micro Devices, Inc.
#
# SPDX-License-Identifier: MIT

"""skillscope: routing and behavioral tests for agent skills.

A skill is a description plus a body, and both can be wrong in ways nothing
else catches: a description that never fires, or fires on its neighbour's work,
and a body that fires correctly and then does the job badly. This package
grades both from one dataset per skill, in whatever repo the skill lives in.

Start at ``skillscope.cli`` for the command line, ``skillscope.config`` for how
a repo describes itself, and ``skillscope.datasets`` for the dataset format.
"""

__all__ = ["__version__"]

__version__ = "0.1.0"
