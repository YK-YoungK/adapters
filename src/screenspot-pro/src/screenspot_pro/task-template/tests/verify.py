"""ScreenSpot-Pro positive grounding verifier (standard library only).

Parsing and scoring preserve models/gpt4x.py and eval_screenspot_pro.py at
https://github.com/likaixin2000/ScreenSpot-Pro-GUI-Grounding/tree/
dbe00114bc53a32c61c1a267786da85967710da8
See UPSTREAM_LICENSE for the upstream MIT license.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import re


def extract_first_bounding_box(text: str) -> list[float] | None:
    # Deliberately preserve upstream's strict syntax (including no whitespace).
    pattern = r"\[\[(\d+\.\d+|\d+),(\d+\.\d+|\d+),(\d+\.\d+|\d+),(\d+\.\d+|\d+)\]\]"
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return [float(match.group(i)) for i in range(1, 5)]
    return None


def extract_first_point(text: str) -> list[float] | None:
    pattern = r"\[\[(\d+\.\d+|\d+),(\d+\.\d+|\d+)\]\]"
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return [float(match.group(1)), float(match.group(2))]
    return None


def parse_prediction(text: str) -> list[float] | None:
    """Match GPT4XModel.ground_only_positive's point/bbox precedence."""
    bbox = extract_first_bounding_box(text)
    click_point = extract_first_point(text)
    if not click_point and bbox:
        click_point = [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2]
    return click_point


def eval_sample_positive_gt(sample: dict, response: dict) -> str:
    """Upstream's inclusive point-in-normalized-box metric, without rounding."""
    bbox = sample["bbox"]
    img_size = sample["img_size"]
    bbox = [
        bbox[0] / img_size[0],
        bbox[1] / img_size[1],
        bbox[2] / img_size[0],
        bbox[3] / img_size[1],
    ]
    click_point = response["point"]
    if click_point is None:
        return "wrong_format"
    if (bbox[0] <= click_point[0] <= bbox[2]) and (
        bbox[1] <= click_point[1] <= bbox[3]
    ):
        return "correct"
    return "wrong"


def validate_expected(sample: dict) -> None:
    """Treat broken verifier assets as infrastructure errors, not model misses."""
    if not isinstance(sample, dict):
        raise ValueError("Expected a ground-truth JSON object")
    for key, length in (("bbox", 4), ("img_size", 2)):
        values = sample.get(key)
        if not isinstance(values, list) or len(values) != length:
            raise ValueError(f"Ground truth {key} must contain {length} numbers")
        if any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            for value in values
        ):
            raise ValueError(f"Ground truth {key} must contain finite numbers")
    width, height = sample["img_size"]
    x1, y1, x2, y2 = sample["bbox"]
    if width <= 0 or height <= 0:
        raise ValueError("Ground truth image dimensions must be positive")
    if x1 > x2 or y1 > y2:
        raise ValueError("Ground truth bbox must be ordered")
    # Official annotations can extend outside the screenshot (e.g. y1=-1).
    # Preserve them exactly rather than clipping or rejecting them.


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prediction", type=Path, default=Path("/workspace/prediction.txt"))
    parser.add_argument("--expected", type=Path, default=Path("/tests/expected.json"))
    parser.add_argument("--reward-dir", type=Path, default=Path("/logs/verifier"))
    args = parser.parse_args()

    args.reward_dir.mkdir(parents=True, exist_ok=True)
    reward_file = args.reward_dir / "reward.txt"
    reward_file.write_text("0\n", encoding="utf-8")
    result = {"correctness": "error", "point": None, "reward": 0}
    exit_code = 2
    try:
        sample = json.loads(args.expected.read_text(encoding="utf-8"))
        validate_expected(sample)
        try:
            prediction = args.prediction.read_text(encoding="utf-8")
        except (OSError, UnicodeError) as exc:
            prediction = ""
            result["error"] = f"Could not read prediction: {exc}"
        point = parse_prediction(prediction)
        correctness = eval_sample_positive_gt(sample, {"point": point})
        reward = int(correctness == "correct")
        result.update(correctness=correctness, point=point, reward=reward)
        reward_file.write_text(f"{reward}\n", encoding="utf-8")
        exit_code = 0 if reward else 1
    except (OSError, ValueError, TypeError, OverflowError) as exc:
        result["error"] = str(exc)
    (args.reward_dir / "result.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
