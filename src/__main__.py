"""Allow `python -m src` / `python -m src.generator` entrypoints."""

from .generator import main

if __name__ == "__main__":
    raise SystemExit(main())
