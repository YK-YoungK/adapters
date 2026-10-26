"""Score one MuSiQue-Ans prediction using the unchanged upstream metrics."""

import argparse
import json
from pathlib import Path
import sys

# The evaluator is verifier-mounted code, never imported from the agent's CWD.
sys.path.insert(0, str(Path(__file__).resolve().parent / "upstream"))
from metrics.answer import AnswerMetric  # noqa: E402
from metrics.support import SupportMetric  # noqa: E402


class InvalidPrediction(ValueError):
    """An invalid agent submission rather than a grading infrastructure error."""


def validate_gold(gold: object) -> None:
    """Validate trusted task assets even when an agent submitted no answer."""
    if not isinstance(gold, dict):
        raise ValueError("Gold must be an original MuSiQue record")
    if not isinstance(gold.get("id"), str) or not gold["id"]:
        raise ValueError("Gold id must be a nonempty string")
    if gold.get("answerable") is not True:
        raise ValueError("This verifier supports answerable MuSiQue-Ans records only")
    if not isinstance(gold.get("answer"), str):
        raise ValueError("Gold answer must be a string")
    aliases = gold.get("answer_aliases")
    if not isinstance(aliases, list) or any(not isinstance(alias, str) for alias in aliases):
        raise ValueError("Gold answer_aliases must be a list of strings")
    paragraphs = gold.get("paragraphs")
    if not isinstance(paragraphs, list) or not paragraphs:
        raise ValueError("Gold paragraphs must be a nonempty list")
    indices = set()
    for paragraph in paragraphs:
        if not isinstance(paragraph, dict):
            raise ValueError("Gold paragraph must be an object")
        idx = paragraph.get("idx")
        if type(idx) is not int or idx < 0 or idx in indices:
            raise ValueError("Gold paragraph indices must be unique nonnegative integers")
        if type(paragraph.get("is_supporting")) is not bool:
            raise ValueError("Gold paragraphs must have boolean supporting labels")
        indices.add(idx)
    if not any(paragraph["is_supporting"] for paragraph in paragraphs):
        raise ValueError("Answerable gold must contain supporting paragraphs")


def load_prediction(path: Path, expected_id: str) -> dict:
    try:
        prediction = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise InvalidPrediction(f"Cannot read prediction: {exc}") from exc
    if not isinstance(prediction, dict):
        raise InvalidPrediction("Prediction must be a JSON object")
    if prediction.get("id") != expected_id:
        raise InvalidPrediction("Prediction id must match the original question id")
    if not isinstance(prediction.get("predicted_answer"), str):
        raise InvalidPrediction("predicted_answer must be a string")
    support = prediction.get("predicted_support_idxs")
    if not isinstance(support, list):
        raise InvalidPrediction("predicted_support_idxs must be a list")
    try:
        # Upstream SupportMetric applies int() before constructing the index
        # set. Preserve its acceptance of numeric strings/floats and duplicates.
        list(map(int, support))
    except (TypeError, ValueError, OverflowError) as exc:
        raise InvalidPrediction("predicted_support_idxs must contain integer-convertible values") from exc
    if type(prediction.get("predicted_answerable")) is not bool:
        raise InvalidPrediction("predicted_answerable must be a boolean")
    return prediction


def score_prediction(prediction: dict, gold: dict) -> dict[str, float]:
    # The official Ans evaluator does not gate answer/support on the predicted
    # answerability value. Its sufficiency metrics are specific to MuSiQue-Full.
    answer_metric = AnswerMetric()
    answer_metric(prediction["predicted_answer"], [gold["answer"]] + gold["answer_aliases"])
    answer_em, answer_f1 = answer_metric.get_metric()
    support_metric = SupportMetric()
    support_metric(
        prediction["predicted_support_idxs"],
        [paragraph["idx"] for paragraph in gold["paragraphs"] if paragraph["is_supporting"]],
    )
    _, support_f1 = support_metric.get_metric()
    # Upstream rounds only after aggregating the dataset. Preserve precision for
    # per-task rewards so dataset means retain those original semantics.
    return {
        "answer_f1": float(answer_f1),
        "answer_em": float(answer_em),
        "support_f1": float(support_f1),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prediction", type=Path, default=Path("/workspace/prediction.json"))
    parser.add_argument("--gold", type=Path, default=Path("/tests/gold.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("/logs/verifier"))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name in ("reward.json", "reward.txt", "prediction_error.txt", "prediction.json"):
        (args.output_dir / name).unlink(missing_ok=True)

    # Broken task assets and scoring errors must propagate. Only failures in the
    # agent's prediction become valid zero scores.
    gold = json.loads(args.gold.read_text(encoding="utf-8"))
    validate_gold(gold)
    try:
        prediction = load_prediction(args.prediction, gold["id"])
    except InvalidPrediction as exc:
        rewards = {"answer_f1": 0.0, "answer_em": 0.0, "support_f1": 0.0}
        (args.output_dir / "prediction_error.txt").write_text(f"{exc}\n", encoding="utf-8")
        print(f"Invalid prediction: {exc}", file=sys.stderr)
    else:
        rewards = score_prediction(prediction, gold)
        (args.output_dir / "prediction.json").write_text(
            json.dumps(prediction, ensure_ascii=False) + "\n", encoding="utf-8"
        )

    # Harbor prioritizes reward.json; the guide-required scalar mirrors answer
    # F1, without inventing a combined score or discarding the other metrics.
    (args.output_dir / "reward.json").write_text(json.dumps(rewards) + "\n", encoding="utf-8")
    (args.output_dir / "reward.txt").write_text(f"{rewards['answer_f1']}\n", encoding="utf-8")
    print(json.dumps(rewards))


if __name__ == "__main__":
    main()
