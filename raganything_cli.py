"""Compatibility entry point for a source checkout."""
import sys
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from aquabio_raganything.cli import main

if __name__ == "__main__":
    os.chdir(Path(__file__).resolve().parent)
    raise SystemExit(main())
