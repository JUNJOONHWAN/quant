import sys


def dispatch() -> int:
    explicit_command = len(sys.argv) > 1 and not sys.argv[1].startswith("-")
    command = sys.argv[1] if explicit_command else "phase-a"
    arguments = sys.argv[2:] if explicit_command else sys.argv[1:]
    if command == "phase-a":
        from .phase_a import main

        return main(arguments)
    if command == "phase-b-market":
        from .phase_b_market import main

        return main(arguments)
    if command == "phase-b-cluster":
        from .phase_b_cluster import main

        return main(arguments)
    if command == "phase-b-stock":
        from .phase_b_stock import main

        return main(arguments)
    raise SystemExit(f"unknown v11-R2 command: {command}")


if __name__ == "__main__":
    raise SystemExit(dispatch())
