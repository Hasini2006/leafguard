"""CI step 1: style and lint checks, run as the first DVC stage.

Writes reports/lint.txt and exits non-zero when anything fails, which stops `dvc repro`
before any data or model stage runs.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

TARGETS = ["src", "services/backend/app", "tests", "scripts"]
CHECKS = [
    ("ruff", ["ruff", "check", *TARGETS]),
    ("black", ["black", "--check", *TARGETS]),
]


def main() -> None:
    Path("reports").mkdir(exist_ok=True)
    lines, failed = [], False

    for name, command in CHECKS:
        lines.append(f"===== {name} =====")
        done = subprocess.run(command, capture_output=True, text=True)
        output = (done.stdout + done.stderr).strip()
        lines.append(output or "(no output)")
        if done.returncode != 0:
            failed = True
        print(f"{name}: {'failed' if done.returncode else 'passed'}")

    report = "\n".join(lines) + "\n"
    Path("reports/lint.txt").write_text(report, encoding="utf-8")
    if failed:
        print(report)
        print(
            "Lint failed. Fix it with:  ruff check --fix src services tests scripts && "
            "black src services tests scripts"
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
