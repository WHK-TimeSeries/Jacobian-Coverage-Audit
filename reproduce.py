"""Portable reviewer entry point; does not authorize a new formal experiment."""
from pathlib import Path
import sys

sys.dont_write_bytecode = True

from reproduction.cli import main

if __name__ == "__main__":
    raise SystemExit(main(Path(__file__).resolve().parent))
