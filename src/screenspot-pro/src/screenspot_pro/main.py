"""Canonical screenspot-pro CLI."""

import argparse
from pathlib import Path

from .adapter import DATASET_REVISION, ScreenSpotProAdapter


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Harbor tasks from ScreenSpot-Pro.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--limit", type=int, help="Generate first N tasks in sorted upstream ID order")
    parser.add_argument("--overwrite", action="store_true", help="Replace selected existing tasks")
    parser.add_argument("--task-ids", nargs="+", help="Upstream IDs or generated directory names")
    parser.add_argument("--language", choices=("en", "cn"), default="en")
    parser.add_argument("--dataset-dir", type=Path, help="Offline dataset root with annotations/ and images/")
    parser.add_argument("--cache-dir", type=Path, help="Hugging Face cache directory")
    parser.add_argument("--revision", default=DATASET_REVISION, help="Hugging Face dataset revision (ignored for local data)")
    args = parser.parse_args()
    try:
        ScreenSpotProAdapter(**vars(args)).run()
    except (ValueError, OSError) as exc:
        parser.exit(1, f"screenspot-pro: {exc}\n")


if __name__ == "__main__":
    main()
