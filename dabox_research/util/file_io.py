"""File utilities"""

import shutil
from pathlib import Path


def move_file(src_path: Path, dst_path: Path) -> None:
    """Move file from src_path to dst_path

    Args:
        src_path (Path): source path
        dst_path (Path): destination path
    """
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(src_path, dst_path)
