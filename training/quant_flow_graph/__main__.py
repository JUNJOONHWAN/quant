"""Command router that keeps dataset building independent of PyTorch."""

from __future__ import annotations

import sys


def main() -> int:
    if len(sys.argv) < 2:
        raise SystemExit("usage: python -m training.quant_flow_graph <overlay-flow|build-dataset|build-smoke|train-smoke|walk-forward> ...")
    command, remainder = sys.argv[1], sys.argv[2:]
    if command == "overlay-flow":
        from .flow_overlay import main as overlay_main

        return overlay_main(remainder)
    if command == "build-dataset":
        from .data import main as data_main

        return data_main(remainder)
    if command == "build-smoke":
        from .contracts import DEFAULT_OUTPUT_ROOT
        from .data import main as data_main, smoke_argv

        replace = "--replace" in remainder
        custom = [item for item in remainder if item != "--replace"]
        if custom:
            raise SystemExit("build-smoke accepts only --replace")
        root = DEFAULT_OUTPUT_ROOT / "smoke_20260629_20260729_10stocks"
        return data_main(smoke_argv(root, replace=replace))
    if command in {"train-smoke", "walk-forward"}:
        from .train import main as train_main

        return train_main([command, *remainder])
    raise SystemExit(f"unknown command: {command}")


if __name__ == "__main__":
    raise SystemExit(main())
