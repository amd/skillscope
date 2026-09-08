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

# Released as the tag `v` + this, and the workflows in this repo reference the
# action at that tag so a caller who pins one gets the other. The suite checks
# all three agree, because a release where they do not is a caller running a
# harness they did not ask for.
__version__ = "0.1.1"
