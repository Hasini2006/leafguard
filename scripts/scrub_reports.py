"""Remove machine-specific details from generated test reports before they are committed.

pytest writes the computer's hostname into JUnit files and coverage.py writes the absolute
project path into coverage.xml. Neither is needed to read the results, and both would
publish details of the machine the tests ran on. The hostname becomes "localhost" and the
project path becomes ".".

Usage:  python scripts/scrub_reports.py reports/junit.xml reports/coverage.xml ...
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = str(Path(__file__).resolve().parents[1])
HOSTNAME = re.compile(r'hostname="[^"]*"')


def scrub(path: Path) -> None:
    if not path.exists():
        return
    text = path.read_text(encoding="utf-8")
    cleaned = HOSTNAME.sub('hostname="localhost"', text).replace(ROOT, ".")
    if cleaned != text:
        path.write_text(cleaned, encoding="utf-8")


def main() -> None:
    for name in sys.argv[1:]:
        scrub(Path(name))


if __name__ == "__main__":
    main()
