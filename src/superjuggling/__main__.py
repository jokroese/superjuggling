import sys

from .cli import main as _cli_main


def main() -> None:
    """Console-script entry point (declared in pyproject ``[project.scripts]``)."""
    raise SystemExit(_cli_main(sys.argv[1:]))


if __name__ == "__main__":
    main()
