"""Regression checks for MuSiQue metric fidelity and invalid submissions."""

import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

VERIFIER = Path(__file__).resolve().parents[1] / "src/musique/task-template/tests/verify.py"
GOLD = {
    "id": "2hop__fixture",
    "question": "Which city?",
    "answer": "New York City",
    "answer_aliases": ["NYC"],
    "answerable": True,
    "paragraphs": [
        {"idx": 0, "is_supporting": False},
        {"idx": 1, "is_supporting": True},
        {"idx": 3, "is_supporting": True},
    ],
}
PREDICTION = {
    "id": "2hop__fixture",
    "predicted_answer": "New York City",
    "predicted_support_idxs": [1, 3],
    "predicted_answerable": True,
}


class VerifierTests(unittest.TestCase):
    def run_verifier(self, prediction=None, *, raw=None, gold=None, stale=False, hostile_import=False):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            gold_path = root / "gold.json"
            gold_path.write_text(json.dumps(GOLD if gold is None else gold))
            prediction_path = root / "prediction.json"
            if raw is not None:
                prediction_path.write_text(raw)
            elif prediction is not None:
                prediction_path.write_text(json.dumps(prediction))
            output = root / "logs"
            if stale:
                output.mkdir()
                (output / "reward.json").write_text('{"answer_f1": 1}')
                (output / "reward.txt").write_text("1")
            environment = os.environ.copy()
            if hostile_import:
                (root / "json.py").write_text("raise RuntimeError('untrusted import')")
                (root / "metrics").mkdir()
                (root / "metrics/__init__.py").write_text("raise RuntimeError('untrusted metrics')")
                environment["PYTHONPATH"] = str(root)
            result = subprocess.run(
                [sys.executable, "-I", "-B", str(VERIFIER), "--prediction", str(prediction_path),
                 "--gold", str(gold_path), "--output-dir", str(output)],
                capture_output=True, text=True, check=False, cwd=root, env=environment,
            )
            logs = {path.name: path.read_text() for path in output.iterdir()} if output.exists() else {}
            return result, logs

    def assert_scores(self, prediction, answer_f1, answer_em, support_f1, **kwargs):
        result, logs = self.run_verifier(prediction, **kwargs)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(logs["reward.json"]), {
            "answer_f1": answer_f1, "answer_em": answer_em, "support_f1": support_f1,
        })
        self.assertEqual(float(logs["reward.txt"]), answer_f1)
        self.assertEqual(json.loads(logs["prediction.json"]), prediction)
        self.assertNotIn("prediction_error.txt", logs)

    def test_oracle(self):
        self.assert_scores(PREDICTION, 1.0, 1.0, 1.0)

    def test_wrong_answer_and_support(self):
        self.assert_scores(PREDICTION | {"predicted_answer": "London", "predicted_support_idxs": []}, 0.0, 0.0, 0.0)

    def test_alias_and_original_normalization(self):
        self.assert_scores(PREDICTION | {"predicted_answer": "The N.Y.C.!"}, 1.0, 1.0, 1.0)

    def test_partial_answer_and_support_remain_unrounded(self):
        self.assert_scores(PREDICTION | {"predicted_answer": "New York", "predicted_support_idxs": [1]}, 0.8, 0.0, 2 / 3)

    def test_support_duplicates_are_ignored(self):
        self.assert_scores(PREDICTION | {"predicted_support_idxs": [1, 1, 3, 3]}, 1.0, 1.0, 1.0)

    def test_support_integer_coercion_matches_upstream(self):
        self.assert_scores(PREDICTION | {"predicted_support_idxs": ["1", "3", 3.9]}, 1.0, 1.0, 1.0)
        self.assert_scores(PREDICTION | {"predicted_support_idxs": ["1", 3.9, 4.9]}, 1.0, 1.0, 0.8)
        self.assert_scores(PREDICTION | {"predicted_support_idxs": [True, 3]}, 1.0, 1.0, 1.0)

    def test_extra_support_is_false_positive(self):
        self.assert_scores(PREDICTION | {"predicted_support_idxs": [1, 3, 999]}, 1.0, 1.0, 0.8)

    def test_answerability_does_not_gate_ans_scores(self):
        self.assert_scores(PREDICTION | {"predicted_answerable": False}, 1.0, 1.0, 1.0)

    def assert_invalid(self, prediction=None, **kwargs):
        result, logs = self.run_verifier(prediction, **kwargs)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(logs["reward.json"]), {
            "answer_f1": 0.0, "answer_em": 0.0, "support_f1": 0.0,
        })
        self.assertEqual(float(logs["reward.txt"]), 0.0)
        self.assertIn("prediction_error.txt", logs)
        self.assertNotIn("prediction.json", logs)

    def test_missing_submission(self):
        self.assert_invalid()

    def test_invalid_json(self):
        self.assert_invalid(raw="{not valid json}")

    def test_invalid_schema(self):
        for prediction in (
            [], {}, PREDICTION | {"id": "other"},
            PREDICTION | {"predicted_answer": None},
            PREDICTION | {"predicted_support_idxs": "1,3"},
            PREDICTION | {"predicted_support_idxs": [None]},
            PREDICTION | {"predicted_support_idxs": ["not-an-index"]},
            PREDICTION | {"predicted_support_idxs": [float("inf")]},
            PREDICTION | {"predicted_support_idxs": [float("nan")]},
            PREDICTION | {"predicted_answerable": "true"},
        ):
            with self.subTest(prediction=prediction):
                self.assert_invalid(prediction)

    def test_corrupt_gold_is_infrastructure_failure_even_without_prediction(self):
        gold = copy.deepcopy(GOLD)
        del gold["answer_aliases"]
        for prediction in (PREDICTION, None):
            with self.subTest(prediction=prediction):
                result, logs = self.run_verifier(prediction, gold=gold, stale=True)
                self.assertNotEqual(result.returncode, 0)
                self.assertNotIn("reward.json", logs)
                self.assertNotIn("reward.txt", logs)

    def test_unanswerable_gold_is_rejected(self):
        result, logs = self.run_verifier(PREDICTION, gold=GOLD | {"answerable": False})
        self.assertNotEqual(result.returncode, 0)
        self.assertNotIn("reward.json", logs)

    def test_stale_reward_is_replaced_for_missing_submission(self):
        self.assert_invalid(stale=True)

    def test_agent_cannot_shadow_scoring_imports(self):
        self.assert_scores(PREDICTION, 1.0, 1.0, 1.0, hostile_import=True)


if __name__ == "__main__":
    unittest.main()
