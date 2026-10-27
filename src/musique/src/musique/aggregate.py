"""Aggregate independent MuSiQue trials with the official paired Full scores.

Harbor calls this script during jobs. Incomplete pairs are counted explicitly;
group scores are provisional until every pair is complete. Use --require-complete
for final audits and compare record counts with the generation manifest. Repeats
must run as separate Harbor jobs, each with n_attempts=1.
Train and development records must be evaluated as separate datasets.
"""

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path


def aggregate_rewards(
    rewards: list[dict[str, float | int] | None], *, require_complete: bool = False
) -> dict[str, float | int]:
    """Compute original metrics from raw components, rounding only at the end."""
    grouped: dict[tuple[int, int], dict[int, dict]] = defaultdict(dict)
    null_count = 0
    selected_split = None
    for reward in rewards:
        if reward is None:
            null_count += 1
            continue
        if not isinstance(reward, dict):
            raise ValueError("Each reward must be an object or null")
        # Legacy development tasks predate explicit split metadata.
        split = reward.get("split", 0)
        if type(split) is not int or split not in (0, 1):
            raise ValueError("Reward split must be 0 (dev) or 1 (train)")
        if selected_split is not None and split != selected_split:
            raise ValueError("Cannot mix train and dev rewards; evaluate them as separate datasets")
        selected_split = split
        for name in ("variant", "group_id", "answerable", "sufficiency_correct"):
            if type(reward.get(name)) is not int:
                raise ValueError(f"Reward {name} must be an integer")
        if reward["variant"] not in (0, 1) or reward["answerable"] not in (0, 1):
            raise ValueError("Reward variant and answerable must be 0 or 1")
        if reward["sufficiency_correct"] not in (0, 1):
            raise ValueError("Reward sufficiency_correct must be 0 or 1")
        if not 0 <= reward["group_id"] < 2**52:
            raise ValueError("Reward group_id must be a 52-bit nonnegative integer")
        for name in ("answer_f1", "answer_em", "support_f1"):
            value = reward.get(name)
            if type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f"Reward {name} must be a finite score in [0, 1]")
        if reward["variant"] == 0 and reward["answerable"] != 1:
            raise ValueError("MuSiQue-Ans cannot contain unanswerable rewards")
        key = reward["variant"], reward["group_id"]
        answerable = reward["answerable"]
        if answerable in grouped[key]:
            raise ValueError("Duplicate question/variant/answerability reward; use one attempt per job")
        grouped[key][answerable] = reward

    ans_rows = [group[1] for (variant, _), group in grouped.items() if variant == 0]
    full_groups = [group for (variant, _), group in grouped.items() if variant == 1]
    full_answerable_rows = [group[1] for group in full_groups if 1 in group]
    complete_groups = [group for group in full_groups if set(group) == {0, 1}]
    pending_pairs = len(full_groups) - len(complete_groups)
    if require_complete:
        if not grouped:
            raise ValueError("No scored MuSiQue records were supplied")
        if null_count or pending_pairs:
            raise ValueError(
                f"Incomplete evaluation: {null_count} null rewards and {pending_pairs} pending Full pairs"
            )

    metrics: dict[str, float | int] = {}
    for prefix, rows in (("ans", ans_rows), ("full", full_answerable_rows)):
        if rows:
            for name in ("answer_f1", "answer_em", "support_f1"):
                metrics[f"{prefix}_{name}"] = round(sum(row[name] for row in rows) / len(rows), 3)
    if complete_groups:
        for name in ("answer", "support"):
            score = sum(
                group[1][f"{name}_f1"]
                if group[1]["sufficiency_correct"] and group[0]["sufficiency_correct"] else 0.0
                for group in complete_groups
            ) / len(complete_groups)
            metrics[f"full_group_{name}_sufficiency_f1"] = round(score, 3)

    # The reducer cannot detect records never submitted to Harbor. Final audits
    # must also compare these counts to the generated task manifest.
    metrics.update(
        ans_records=len(ans_rows),
        full_records=sum(len(group) for group in full_groups),
        full_completed_pairs=len(complete_groups),
        full_pending_pairs=pending_pairs,
        null_rewards=null_count,
    )
    return metrics


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-i", "--input", required=True, type=Path, help="Harbor rewards JSONL")
    parser.add_argument("-o", "--output", required=True, type=Path)
    parser.add_argument("--require-complete", action="store_true", help="Reject missing pairs or rewards")
    args = parser.parse_args(argv)
    try:
        rewards = [json.loads(line) for line in args.input.read_text().splitlines() if line.strip()]
        metrics = aggregate_rewards(rewards, require_complete=args.require_complete)
    except (ValueError, OSError) as error:
        parser.error(str(error))
    args.output.write_text(json.dumps(metrics, indent=2, allow_nan=False) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
