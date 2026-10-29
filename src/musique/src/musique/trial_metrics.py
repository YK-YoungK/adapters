"""Summarize sparse per-record scores for Harbor's live diagnostics.

Only records containing a metric contribute to its mean. These summaries may
pool Ans and Full records and cannot reconstruct Full pairs. Use musique.audit
for the official variant-specific and paired benchmark scores.
"""

import argparse
import json
import math
from pathlib import Path


ANSWER_KEYS = ("answer_f1", "answer_em", "support_f1")
METRIC_NAMES = {
    "answer_f1": "record_answer_f1",
    "answer_em": "record_answer_em",
    "support_f1": "record_support_f1",
    "sufficiency_correct": "record_sufficiency_accuracy",
}


def aggregate_trial_metrics(
    rewards: list[dict[str, float | int] | None], *, require_complete: bool = False
) -> dict[str, float | int]:
    """Validate public rewards and average only applicable record components."""
    totals = {name: 0.0 for name in METRIC_NAMES}
    counts = {name: 0 for name in METRIC_NAMES}
    null_count = 0
    scored_records = 0
    for reward in rewards:
        if reward is None:
            null_count += 1
            continue
        if not isinstance(reward, dict) or not reward:
            raise ValueError("Each public reward must be a nonempty object or null")
        unknown = reward.keys() - METRIC_NAMES.keys()
        if unknown:
            raise ValueError(f"Unsupported public reward fields: {', '.join(sorted(unknown))}")
        answer_keys = reward.keys() & set(ANSWER_KEYS)
        if answer_keys and answer_keys != set(ANSWER_KEYS):
            raise ValueError("Answerable records require answer_f1, answer_em and support_f1")
        for name, value in reward.items():
            if type(value) not in (int, float) or not 0 <= value <= 1 or not math.isfinite(value):
                raise ValueError(f"Public reward {name} must be a finite number in [0, 1]")
            if name == "sufficiency_correct" and value not in (0, 1):
                raise ValueError("Public reward sufficiency_correct must be 0 or 1")
            totals[name] += value
            counts[name] += 1
        scored_records += 1

    if require_complete and (null_count or not scored_records):
        raise ValueError(
            f"Incomplete record diagnostics: {null_count} null rewards and "
            f"{scored_records} scored records"
        )
    metrics = {
        output_name: round(totals[name] / counts[name], 3)
        for name, output_name in METRIC_NAMES.items() if counts[name]
    }
    # Harbor's live display needs a nonempty metric result. Report missing
    # rewards when present, or an empty-input fallback; neither certifies a run.
    if null_count or not metrics:
        metrics["null_rewards"] = null_count
    return metrics


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-i", "--input", required=True, type=Path, help="Harbor rewards JSONL")
    parser.add_argument("-o", "--output", required=True, type=Path)
    parser.add_argument("--require-complete", action="store_true", help="Reject null or absent rewards")
    args = parser.parse_args(argv)
    try:
        rewards = [
            json.loads(line) for line in args.input.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        metrics = aggregate_trial_metrics(rewards, require_complete=args.require_complete)
        args.output.write_text(json.dumps(metrics, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    except (ValueError, OSError) as error:
        parser.error(str(error))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
