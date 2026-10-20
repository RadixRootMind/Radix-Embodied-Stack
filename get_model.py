import os
import argparse
import subprocess
from hmatc.utils.utils import get_file_from_jfrog

HOUMO_TARGET = os.getenv("HOUMO_TARGET", "xh2")


def get_args() -> argparse.Namespace:
    """Parse commandline."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--type",
        dest="model_type",
        type=str,
        default="raw",
        choices=["raw", "hmm"],
        help="which model type to get, choice in [raw, hmm]",
    )
    parser.add_argument(
        "--hmm_dir",
        dest="hmm_dir",
        type=str,
        default=".",
        help="where to extract hmm model",
    )
    parser.add_argument(
        "--model_dir",
        dest="model_dir",
        type=str,
        default="./models",
        help="where to save downloaded raw model",
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_args()
    hmm_dir = args.hmm_dir
    model_type = args.model_type
    model_dir = args.model_dir

    model_name = "pi05"
    target = HOUMO_TARGET.lower()
    raw_download_script = "./download.sh"
    hmm_path = f"http://10.10.1.53:8082/artifactory/toolchain/support/models/{model_name}/pi05_libero_{target}_20260420.tar.xz"

    if model_type == "raw":
        print("Downloading raw models...")
        subprocess.run(["bash", raw_download_script], check=True)

    if model_type == "hmm":
        get_file_from_jfrog(hmm_path, model_dir, hmm_dir)
