"""Compare the public chat API contract copies in this directory with their originals.

The JSON files next to this script are byte-for-byte copies of the authoritative
public contract in the rastinaghighi.dev repository. README.md records, for each
copy, the original it came from and its SHA-256. This script checks that every
copy still has its recorded SHA-256 and is identical, byte for byte, to its
original in a website repository checkout.

It only reads files: it writes nothing, runs no Git command and makes no network
access. The ordinary test suite does not need it or the website repository;
tests/test_public_response.py checks the copies against README.md alone.

Exit status, as for cmp and diff:
  0  every copy has its recorded SHA-256 and is identical to its original
  1  a copy differs from its recorded SHA-256 or from its original
  2  the check could not be completed: the record cannot be read, or the
     website repository, an original or a copy is missing (this wins over 1)
"""

import argparse
import hashlib
import re
import sys
from pathlib import Path
from typing import NamedTuple

CONTRACTS = Path(__file__).resolve().parent
DEFAULT_WEBSITE = CONTRACTS.parents[2] / "rastinaghighi.dev"  # the checkout next to Mascot-Lab
SAME, DRIFT, CANNOT_CHECK = 0, 1, 2

# A row of the record table in README.md: | `copy` | `original` | `sha256` |
_ROW = re.compile(r"^\| `(?P<copy>[^`]+)` \| `(?P<original>[^`]+)` \| `(?P<sha256>[0-9a-f]{64})` \|$", re.MULTILINE)


class Entry(NamedTuple):
    copy: str  # relative to this directory
    original: str  # relative to the website repository
    sha256: str


def read_record(contracts: Path = CONTRACTS) -> list[Entry]:
    """The copies that README.md in ``contracts`` records, in its order."""
    entries = [Entry(**row.groupdict()) for row in _ROW.finditer((contracts / "README.md").read_text(encoding="utf-8"))]
    if not entries:
        raise ValueError("README.md records no contract copies")
    if len({entry.copy for entry in entries}) != len(entries):
        raise ValueError("README.md records a copy more than once")
    return entries


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def compare(website: Path, contracts: Path = CONTRACTS) -> tuple[int, list[str]]:
    """The exit status and the report for the copies in ``contracts`` against ``website``."""
    try:
        entries = read_record(contracts)
    except (OSError, ValueError) as error:
        return CANNOT_CHECK, [f"cannot check: the record cannot be read ({error})"]
    if not website.is_dir():
        return CANNOT_CHECK, [f"cannot check: no website repository at {website}"]
    status, report = SAME, []
    for entry in entries:
        copy, original = contracts / entry.copy, website / entry.original
        missing = [str(path) for path in (copy, original) if not path.is_file()]
        if missing:
            status = CANNOT_CHECK
            report.append(f"MISSING {entry.copy}: not found: {', '.join(missing)}")
            continue
        copied, authoritative = copy.read_bytes(), original.read_bytes()
        copied_hash, original_hash = hashlib.sha256(copied).hexdigest(), hashlib.sha256(authoritative).hexdigest()
        problems = []
        if copied_hash != entry.sha256:
            problems.append(f"the copy's SHA-256 is {copied_hash}, but README.md records {entry.sha256}")
        if copied != authoritative:
            same_text = copied.replace(b"\r\n", b"\n") == authoritative.replace(b"\r\n", b"\n")
            note = " (only line endings differ)" if same_text else ""
            problems.append(f"the copy differs from {entry.original}, whose SHA-256 is {original_hash}{note}")
        if problems:
            status = max(status, DRIFT)
            report += [f"DRIFT   {entry.copy}: {problem}" for problem in problems]
        else:
            report.append(f"ok      {entry.copy}  {copied_hash}")
    summary = {
        SAME: f"all {len(entries)} copies match their originals in {website}",
        DRIFT: "contract drift: update the copies and README.md, or restore the originals",
        CANNOT_CHECK: "cannot check: a file is missing",
    }
    return status, [*report, summary[status]]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Compare the contract copies in tests/contracts with their originals.",
        epilog="Exit status: 0 identical, 1 drift, 2 could not check.",
    )
    parser.add_argument(
        "website",
        nargs="?",
        type=Path,
        default=DEFAULT_WEBSITE,
        help="the rastinaghighi.dev repository (default: the checkout next to Mascot-Lab)",
    )
    status, report = compare(parser.parse_args(argv).website)
    print("\n".join(report))
    return status


if __name__ == "__main__":
    sys.exit(main())
