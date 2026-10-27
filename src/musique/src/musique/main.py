"""Command-line entry point for MuSiQue train/development task generation."""

import argparse
from pathlib import Path

from musique.adapter import MuSiQueAdapter, validate_split


def parse_split(value: str) -> str:
    try:
        return validate_split(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError(str(error)) from error


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate labeled MuSiQue-Ans and MuSiQue-Full v1.0 train/dev tasks for Harbor.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--variant", choices=["ans", "full", "both"], default="both")
    parser.add_argument("--split", type=parse_split, choices=["train", "dev"], default="dev",
                        help="Labeled split (default: dev); public test gold labels are withheld.")
    parser.add_argument("--limit", type=int, help="Maximum task directories in sorted order; Full pairs stay together.")
    parser.add_argument("--task-ids", nargs="+", help="Upstream IDs or generated task names; selects both records of a Full pair.")
    parser.add_argument("--overwrite", action="store_true", help="Replace selected task directories.")
    parser.add_argument("--data-dir", type=Path, help="Directory containing the selected split's official-format JSONL files.")
    parser.add_argument("--cache-dir", type=Path, help="Download cache (default: ~/.cache/harbor-musique).")
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be a positive integer")
    adapter = MuSiQueAdapter(
        output_dir=args.output_dir, variant=args.variant, split=args.split, data_dir=args.data_dir,
        cache_dir=args.cache_dir, limit=args.limit, task_ids=args.task_ids, overwrite=args.overwrite,
    )
    try:
        paths = adapter.run()
    except (OSError, ValueError, RuntimeError) as error:
        parser.exit(1, f"musique: {error}\n")
    print(f"Generated {len(paths)} MuSiQue {args.variant} {args.split} tasks in {adapter.output_dir}")


if __name__ == "__main__":
    main()
