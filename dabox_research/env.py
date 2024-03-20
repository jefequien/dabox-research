"""Environment Variables"""

import os
from pathlib import Path

ROOT_DIR = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEMO_DIR = ROOT_DIR / "demo"
DEFAULT_OUTPUT_DIR = ROOT_DIR / ".output"
