#!/usr/bin/env python3
"""Entrypoint for the YOLO arrow detection baseline."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from arrow_detection.cli import main


if __name__ == "__main__":
    main()
