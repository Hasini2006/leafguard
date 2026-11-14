"""CI step 2: unit tests with coverage, then a small pass/fail summary.

Produces reports/junit.xml, reports/coverage.xml and reports/tests.html, and writes
reports/ci_gate.json only when every test passed. The data stages depend on that file, so
a failing test stops the pipeline before any model is trained.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import xml.etree.ElementTree as ElementTree
from pathlib import Path

JUNIT = Path("reports/junit.xml")
GATE = Path("reports/ci_gate.json")


def run_pytest() -> int:
    environment = dict(
        os.environ, LEAFGUARD_NO_PUSH="1", LEAFGUARD_RUNS_LOG="/tmp/leafguard_test_runs.jsonl"
    )
    command = [
        sys.executable,
        "-m",
        "pytest",
        "tests/unit",
        "-q",
        f"--junitxml={JUNIT}",
        "--cov=leafguard",
        "--cov=app",
        "--cov-report=term-missing",
        "--cov-report=xml:reports/coverage.xml",
        "--html=reports/tests.html",
        "--self-contained-html",
    ]
    code = subprocess.run(command, env=environment).returncode
    # Strip the hostname and absolute path pytest/coverage write into the reports.
    subprocess.run(
        [
            sys.executable,
            "scripts/scrub_reports.py",
            str(JUNIT),
            "reports/coverage.xml",
            "reports/tests.html",
        ],
        check=False,
    )
    return code


def summarise() -> dict[str, int | str]:
    """Count tests, failures and errors across every suite in the JUnit report."""
    root = ElementTree.parse(JUNIT).getroot()
    suites = [root] if root.tag == "testsuite" else list(root.iter("testsuite"))
    totals = {"tests": 0, "failures": 0, "errors": 0, "skipped": 0}
    for suite in suites:
        for key in totals:
            totals[key] += int(suite.get(key, 0))
    clean = totals["failures"] == 0 and totals["errors"] == 0
    return {
        "lint": "passed",
        "unit_tests": "passed" if clean else "failed",
        **totals,
        "passed": totals["tests"] - totals["failures"] - totals["errors"] - totals["skipped"],
    }


def main() -> None:
    Path("reports").mkdir(exist_ok=True)
    code = run_pytest()
    if not JUNIT.exists():
        print("pytest produced no report", file=sys.stderr)
        sys.exit(1)

    summary = summarise()
    print(json.dumps(summary))
    if code != 0 or summary["unit_tests"] != "passed":
        GATE.unlink(missing_ok=True)
        print("Unit tests failed, so the pipeline stops here.", file=sys.stderr)
        sys.exit(1)

    GATE.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
