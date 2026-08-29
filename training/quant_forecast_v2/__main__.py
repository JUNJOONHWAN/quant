"""Forecast v2 command router."""

from __future__ import annotations

import argparse
from typing import Sequence

from . import evaluate, finalize, panel, report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command", choices=("build-panel", "evaluate", "finalize", "report")
    )
    args, remaining = parser.parse_known_args(argv)
    if args.command == "build-panel":
        return panel.main(remaining)
    if args.command == "evaluate":
        return evaluate.main(remaining)
    if args.command == "finalize":
        return finalize.main(remaining)
    return report.main(remaining)


if __name__ == "__main__":
    raise SystemExit(main())
