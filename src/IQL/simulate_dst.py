#!/usr/bin/env python3
"""
Wrapper script to run DST simulation.
"""

import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Import and run the actual simulation
from src.dst.simulate import main  # noqa: E402

if __name__ == "__main__":
    main()
