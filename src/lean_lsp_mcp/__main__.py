import sys

from . import main


def _run() -> None:
    main()
    sys.exit(0)


if __name__ == "__main__":
    _run()
