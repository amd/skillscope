# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.

"""Does `skillscope select` still plan what amd/skills planned before the split?

Porting `.github/scripts/select_evals.py` into `skillscope select` changed where
every input comes from: skills globs, runner labels, infra paths, and the
routing set are now flags the caller's workflow passes. The output is supposed
to be unchanged. This runs both planners over the same sample diffs and prints
any difference, ignoring the per-leg `version` that only the new one emits.

    git -C /path/to/amd-skills worktree add --detach /tmp/pre-split <commit>
    python tools/verify_selection_parity.py /tmp/pre-split /path/to/amd-skills

`SETTINGS` below is amd/skills' configuration, spelled the new way; it has to
stay in step with that repo's .github/workflows/evals.yml, or a difference here
means the two callers disagree rather than that the planner regressed.

Kept because the question outlives the split: the pre-split commit stays in
amd/skills history, so the parity claim can be re-checked rather than trusted.
Not part of the suite, which must pass with no checkout but its own.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

# What amd/skills' workflow passes. Mirrors .github/workflows/evals.yml there.
SETTINGS = [
    "--skills",
    "skills/*",
    "--routing-skills",
    "local-ai-use,local-ai-app-integration,serving-llms-on-instinct,"
    "tracelens-analysis-orchestrator,hyperloom-workload-optimizer",
    "--infra-paths",
    ".github/workflows/evals.yml",
    "--behavior-runner",
    '["self-hosted", "strix_halo"]',
    "--behavior-os",
    "Linux,Windows",
    "--scoped-runner",
    '["self-hosted", "Linux", "X64"]',
    "--scoped-gate",
    "enable_mi_ci",
    "--scoped-environment",
    "behavioral-instinct",
]

# Diffs worth asking about: one per branch of the selection logic.
SAMPLES: dict[str, list[str]] = {
    "one skill's body": ["skills/local-ai-use/SKILL.md"],
    "one skill's dataset": ["skills/local-ai-use/evals/evals.json"],
    "a gated skill": ["skills/serving-llms-on-instinct/SKILL.md"],
    "a reference file under a skill": ["skills/local-ai-use/reference.md"],
    "an unrelated file": ["README.md"],
    "several skills at once": [
        "skills/local-ai-use/SKILL.md",
        "skills/serving-llms-on-epyc/evals/evals.json",
        "skills/tracelens-analysis-orchestrator/evals/hooks.py",
    ],
    # The expected differences, kept in the sample set rather than left out.
    "the harness itself (gone from this repo)": ["eval/datasets.py"],
    "the marketplace bundle": [".claude-plugin/marketplace.json"],
    "the workflow": [".github/workflows/evals.yml"],
    "nothing": [],
}

# Cases where the plans are supposed to differ, and why.
EXPECTED_DIFFERENCES = {
    "the harness itself (gone from this repo)": (
        "eval/** used to be an infra path; the harness is a version pin now, "
        "so touching a file that no longer exists selects nothing."
    ),
    "the marketplace bundle": (
        "publishing a skill used to change what routing installed. The routing "
        "set is listed in the workflow now, so the manifest is not an input."
    ),
}

LABEL_SETS = ["", "enable_mi_ci"]


def normalize(plan: dict) -> dict:
    """Drop what only the new implementation emits."""
    plan = json.loads(json.dumps(plan))
    plan.pop("version", None)
    for key in ("default", "scoped"):
        for leg in plan.get(key, []):
            leg.pop("version", None)
    return plan


def run(cmd: list[str], cwd: Path, changed: list[str]) -> dict:
    proc = subprocess.run(
        cmd,
        cwd=str(cwd),
        input="\n".join(changed) + "\n",
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    if proc.returncode != 0:
        raise SystemExit(f"{' '.join(cmd)} failed in {cwd}:\n{proc.stderr}")
    return json.loads(proc.stdout.strip().splitlines()[-1])


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        raise SystemExit(__doc__)
    old_root, new_root = (Path(p).resolve() for p in argv)

    differences = 0
    for name, changed in SAMPLES.items():
        for labels in LABEL_SETS:
            old = normalize(
                run(
                    [
                        sys.executable,
                        ".github/scripts/select_evals.py",
                        "--changed",
                        "--labels",
                        labels,
                        "--no-extended",
                    ],
                    old_root,
                    changed,
                )
            )
            new = normalize(
                run(
                    [
                        sys.executable,
                        "-m",
                        "skillscope",
                        "--repo",
                        str(new_root),
                        "select",
                        "--changed",
                        "--labels",
                        labels,
                        "--no-extended",
                        *SETTINGS,
                    ],
                    new_root / "skillscope",
                    changed,
                )
            )
            if old == new:
                print(f"[same] {name} (labels: {labels or 'none'})")
                continue
            reason = EXPECTED_DIFFERENCES.get(name)
            state = "expected" if reason else "DIFFERENT"
            print(f"[{state}] {name} (labels: {labels or 'none'})")
            if reason:
                print(f"  {reason}")
                continue
            differences += 1
            print("  old:", json.dumps(old, sort_keys=True))
            print("  new:", json.dumps(new, sort_keys=True))

    print()
    print(
        "identical apart from the expected differences"
        if not differences
        else f"{differences} unexplained difference(s)"
    )
    return 1 if differences else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
