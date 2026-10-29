"""Verify one independently answered MuSiQue record using upstream metrics."""

import argparse
import hashlib
import json
import sys
from pathlib import Path

# Load verifier-mounted code, never metrics from the agent's working directory.
sys.path.insert(0, str(Path(__file__).resolve().parent / "upstream"))
from metrics.answer import AnswerMetric  # noqa: E402
from metrics.support import SupportMetric  # noqa: E402


class InvalidPrediction(ValueError):
    """The agent did not produce a valid official-format prediction."""


def validate_gold(gold: object) -> dict:
    """Reject damaged task assets before interpreting any agent output."""
    if not isinstance(gold, dict) or not isinstance(gold.get("record"), dict):
        raise ValueError("Gold must contain an original MuSiQue record")
    record = gold["record"]
    if gold.get("variant") not in ("ans", "full"):
        raise ValueError("Gold variant must be ans or full")
    if gold.get("split", "dev") not in ("train", "dev"):
        raise ValueError("Gold split must be train or dev; public test labels are withheld")
    if not isinstance(record.get("id"), str) or not record["id"]:
        raise ValueError("Gold id must be a nonempty string")
    expected_group = int(hashlib.sha256(record["id"].encode()).hexdigest()[:13], 16)
    if type(gold.get("group_id")) is not int or gold["group_id"] != expected_group:
        raise ValueError("Gold group_id does not match the upstream question id")
    if type(record.get("answerable")) is not bool:
        raise ValueError("Gold answerable must be a boolean")
    if gold["variant"] == "ans" and not record["answerable"]:
        raise ValueError("MuSiQue-Ans cannot contain unanswerable records")
    if not isinstance(record.get("answer"), str):
        raise ValueError("Gold answer must be a string")
    if not isinstance(record.get("answer_aliases"), list) or not all(
        isinstance(alias, str) for alias in record["answer_aliases"]
    ):
        raise ValueError("Gold answer_aliases must be a list of strings")
    paragraphs = record.get("paragraphs")
    if not isinstance(paragraphs, list) or not paragraphs:
        raise ValueError("Gold paragraphs must be a nonempty list")
    indices = []
    for paragraph in paragraphs:
        if (
            not isinstance(paragraph, dict)
            or type(paragraph.get("idx")) is not int
            or type(paragraph.get("is_supporting")) is not bool
        ):
            raise ValueError("Gold paragraphs require integer idx and boolean is_supporting")
        indices.append(paragraph["idx"])
    if len(set(indices)) != len(indices):
        raise ValueError("Gold paragraph indices must be unique")
    return record


def validate_prediction(prediction: object, record: dict) -> dict:
    if not isinstance(prediction, dict):
        raise InvalidPrediction("prediction.json must contain a JSON object")
    required = {"id", "predicted_answer", "predicted_support_idxs", "predicted_answerable"}
    missing = sorted(required - prediction.keys())
    if missing:
        raise InvalidPrediction(f"Missing prediction fields: {', '.join(missing)}")
    if prediction["id"] != record["id"]:
        raise InvalidPrediction("Prediction id does not match this task")
    if not isinstance(prediction["predicted_answer"], str):
        raise InvalidPrediction("predicted_answer must be a string")
    if not isinstance(prediction["predicted_support_idxs"], list):
        raise InvalidPrediction("predicted_support_idxs must be a list")
    if type(prediction["predicted_answerable"]) is not bool:
        raise InvalidPrediction("predicted_answerable must be a boolean")
    try:
        # Preserve upstream integer conversion, duplicate handling, and penalties
        # for unknown indices instead of adding a different scoring policy.
        list(map(int, prediction["predicted_support_idxs"]))
    except (TypeError, ValueError, OverflowError) as error:
        raise InvalidPrediction("Support indices must be convertible to integers") from error
    return {key: prediction[key] for key in sorted(required)}


def empty_reward(gold: dict) -> dict[str, float | int]:
    reward = {
        "variant": int(gold["variant"] == "full"),
        "group_id": gold["group_id"],
        "answerable": int(gold["record"]["answerable"]),
        "sufficiency_correct": 0,
        "answer_f1": 0.0,
        "answer_em": 0.0,
        "support_f1": 0.0,
    }
    # Legacy development tasks predate explicit split metadata.
    if "split" in gold:
        reward["split"] = int(gold["split"] == "train")
    return reward


def score_prediction(prediction: object, gold: dict) -> dict[str, float | int]:
    """Keep raw scores: the official evaluator rounds only after averaging."""
    record = validate_gold(gold)
    prediction = validate_prediction(prediction, record)
    reward = empty_reward(gold)
    reward["sufficiency_correct"] = int(
        prediction["predicted_answerable"] == record["answerable"]
    )
    # Official answer/support scores use only answerable records. On MuSiQue-Ans
    # these metrics ignore predicted_answerable entirely.
    if record["answerable"]:
        answer_metric = AnswerMetric()
        support_metric = SupportMetric()
        answer_metric(
            prediction["predicted_answer"], [record["answer"], *record["answer_aliases"]]
        )
        support_metric(
            prediction["predicted_support_idxs"],
            [p["idx"] for p in record["paragraphs"] if p["is_supporting"]],
        )
        answer_em, answer_f1 = answer_metric.get_metric()
        _, support_f1 = support_metric.get_metric()
        reward.update(
            answer_f1=float(answer_f1), answer_em=float(answer_em), support_f1=float(support_f1)
        )
    return reward


def public_metrics(scores: dict) -> dict[str, float | int]:
    """Expose applicable scores, keeping labels and grouping IDs in artifacts."""
    metrics = {}
    if scores["answerable"]:
        metrics.update({name: scores[name] for name in ("answer_f1", "answer_em", "support_f1")})
    if scores["variant"] == 1:
        # This is prediction correctness, not the gold answerability label.
        metrics["sufficiency_correct"] = scores["sufficiency_correct"]
    return metrics


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prediction", type=Path, default=Path("/workspace/prediction.json"))
    parser.add_argument("--gold", type=Path, default=Path("/tests/gold.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("/logs/verifier"))
    args = parser.parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name in ("reward.txt", "reward.json", "diagnostics.json", "prediction.json"):
        (args.output_dir / name).unlink(missing_ok=True)

    gold = json.loads(args.gold.read_text(encoding="utf-8"))
    record = validate_gold(gold)
    error_message = None
    try:
        prediction = json.loads(args.prediction.read_text(encoding="utf-8"))
        prediction = validate_prediction(prediction, record)
    except (OSError, UnicodeError, json.JSONDecodeError, InvalidPrediction) as error:
        error_message = f"Invalid prediction: {error}"
        reward = empty_reward(gold)
        print(error_message, file=sys.stderr)
    else:
        reward = score_prediction(prediction, gold)
        (args.output_dir / "prediction.json").write_text(
            # JSON permits escaped surrogate code points. Escape them again
            # rather than failing UTF-8 encoding after a prediction was scored.
            json.dumps(prediction, ensure_ascii=True, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )

    # Required by the adapter contract; benchmark scores are computed by the
    # post-run audit, not this per-record compatibility value.
    diagnostic_reward = reward["answer_f1"]
    if gold["variant"] == "full":
        diagnostic_reward = (
            reward["answer_f1"] * reward["sufficiency_correct"]
            if record["answerable"] else reward["sufficiency_correct"]
        )
    (args.output_dir / "reward.txt").write_text(f"{diagnostic_reward}\n", encoding="utf-8")
    metrics = public_metrics(reward)
    (args.output_dir / "reward.json").write_text(
        json.dumps(metrics, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    (args.output_dir / "diagnostics.json").write_text(
        json.dumps(
            {"valid_prediction": error_message is None, "error": error_message, "scores": reward},
            indent=2,
        ) + "\n", encoding="utf-8",
    )
    print(json.dumps(metrics, sort_keys=True))
    # Malformed submissions are scored failures, not infrastructure errors.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
