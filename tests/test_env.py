import os
from pathlib import Path

from dabox_research.env import ROOT_DIR


def test_root_dir():
    assert ROOT_DIR == Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
