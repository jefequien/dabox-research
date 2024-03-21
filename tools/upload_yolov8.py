"""Open a PR to upload a model to huggingface"""

import argparse
from pathlib import Path

from huggingface_hub import HfApi

from dabox_research.env import HUGGING_FACE_MODEL_ZOO_REPO


def main(args):
    """Open a PR to upload a model to huggingface"""
    src_path = Path(args.src_path)
    dst_path = Path(args.dst_path)

    api = HfApi()
    api.upload_file(
        path_or_fileobj=str(src_path),
        path_in_repo=str(dst_path),
        repo_id=HUGGING_FACE_MODEL_ZOO_REPO,
        repo_type="model",
        commit_message=f"Uploaded {src_path}!",
        create_pr=True,
    )
    print("Review PR at https://huggingface.co/jefequien/dabox/discussions")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-s", "--src-path", required=True, help="Path of file to upload"
    )
    parser.add_argument(
        "-d", "--dst-path", required=True, help="Path in repo to upload"
    )
    input_args = parser.parse_args()
    print(input_args)
    main(input_args)
