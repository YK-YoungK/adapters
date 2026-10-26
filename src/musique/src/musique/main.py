"""Command-line entry point for generating MuSiQue Harbor tasks."""

import argparse
from pathlib import Path

from musique.adapter import MuSiQueAdapter


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate MuSiQue-Ans v1.0 dev tasks for Harbor.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split", choices=["dev"], default="dev")
    parser.add_argument("--limit", type=int, help="First N selected records in source order.")
    parser.add_argument("--task-ids", nargs="+", help="Exact upstream IDs, e.g. 2hop__460946_294723.")
    parser.add_argument("--overwrite", action="store_true", help="Replace selected task directories.")
    parser.add_argument("--data-path", type=Path, help="Local official-format labeled Ans dev JSONL.")
    parser.add_argument("--cache-dir", type=Path, help="Download cache (default: ~/.cache/harbor-musique).")
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be a positive integer")
    adapter = MuSiQueAdapter(
        output_dir=args.output_dir, data_path=args.data_path, cache_dir=args.cache_dir,
        limit=args.limit, task_ids=args.task_ids, overwrite=args.overwrite,
    )
    try:
        tasks = adapter.run()
    except (OSError, ValueError, RuntimeError) as error:
        parser.exit(1, f"musique: {error}\n")
    print(f"Generated {len(tasks)} MuSiQue-Ans dev tasks in {adapter.output_dir}")


if __name__ == "__main__":
    main()
