"""Metric-only Harbor rewards and post-run reproduction of MuSiQue scores."""

import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


ADAPTER_DIR = Path(__file__).resolve().parents[1]
PACKAGE_DIR = ADAPTER_DIR / "src"
VERIFIER_DIR = PACKAGE_DIR / "musique/task-template/tests"
sys.path.insert(0, str(PACKAGE_DIR))

from musique.audit import audit_job  # noqa: E402
from musique.trial_metrics import aggregate_trial_metrics  # noqa: E402


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


verify = load_module("_musique_reporting_verify", VERIFIER_DIR / "verify.py")
official = load_module("_musique_reporting_official", VERIFIER_DIR / "upstream/evaluate_v1.0.py")
ANSWER_KEYS = {"answer_f1", "answer_em", "support_f1"}
METADATA_KEYS = {"variant", "group_id", "answerable", "split"}
OFFICIAL_KEYS = {
    "ans_answer_f1", "ans_answer_em", "ans_support_f1",
    "full_answer_f1", "full_answer_em", "full_support_f1",
    "full_group_answer_sufficiency_f1", "full_group_support_sufficiency_f1",
}


def fixture(variant, answerable=True, split="dev"):
    question_id = "2hop__reporting_1"
    gold = {
        "variant": variant, "split": split,
        "group_id": int(hashlib.sha256(question_id.encode()).hexdigest()[:13], 16),
        "record": {
            "id": question_id, "question": "Which bridge appears in both sources?",
            "answer": "red bridge", "answer_aliases": ["the red bridge"],
            "answerable": answerable,
            "paragraphs": [
                {"idx": 1, "title": "First", "paragraph_text": "A red bridge.", "is_supporting": True},
                {"idx": 3, "title": "Second", "paragraph_text": "The same bridge.", "is_supporting": answerable},
            ],
        },
    }
    prediction = {
        "id": question_id, "predicted_answer": gold["record"]["answer"],
        "predicted_support_idxs": [p["idx"] for p in gold["record"]["paragraphs"] if p["is_supporting"]],
        "predicted_answerable": answerable,
    }
    return gold, prediction


def three_records(split="dev"):
    return [fixture("ans", split=split), fixture("full", split=split), fixture("full", False, split)]


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value) + "\n", encoding="utf-8")


def compatibility_reward(scores):
    if scores["variant"] == 0:
        return scores["answer_f1"]
    if scores["answerable"]:
        return scores["answer_f1"] * scores["sufficiency_correct"]
    return scores["sufficiency_correct"]


class RewardReportingTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(prefix=".reporting-regression-", dir=ADAPTER_DIR)
        self.root = Path(self.directory.name)

    def tearDown(self):
        self.directory.cleanup()

    def verifier_cli(self, gold, payload, name):
        root = self.root / name
        gold_path, prediction_path, output = root / "gold.json", root / "prediction.json", root / "verifier"
        write_json(gold_path, gold)
        prediction_path.write_text(payload, encoding="utf-8")
        process = subprocess.run(
            [sys.executable, str(VERIFIER_DIR / "verify.py"), "--gold", str(gold_path),
             "--prediction", str(prediction_path), "--output-dir", str(output)],
            text=True, capture_output=True,
        )
        self.assertEqual(process.returncode, 0, process.stderr)
        self.assertEqual(json.loads(process.stdout), json.loads((output / "reward.json").read_text()))
        return output

    def make_job(self, name, *, schema="public", rows=None):
        rows = three_records() if rows is None else rows
        root = self.root / name
        tasks, job, audit = root / "tasks", root / "job", root / "audit"
        write_json(job / "result.json", {"finished_at": "2026-10-28T00:00:00Z", "n_total_trials": len(rows)})
        for index, (gold, prediction) in enumerate(rows):
            slug = f"{gold['variant']}-dev-record-{index}"
            task = tasks / slug
            write_json(task / "tests/gold.json", gold)
            (task / "task.toml").write_text(
                f'[task]\nname = "musique/{slug}"\n[metadata]\nsplit = "dev"\n', encoding="utf-8"
            )
            scores = verify.score_prediction(prediction, gold)
            saved = verify.public_metrics(scores) if schema == "public" else scores.copy()
            if schema == "legacy-primary":
                saved["reward"] = compatibility_reward(scores)
            trial = job / f"trial-{index}"
            write_json(trial / "verifier/prediction.json", prediction)
            write_json(trial / "verifier/reward.json", saved)
            # Deliberately omit diagnostics: the audit must reconstruct grouping
            # and raw scores from the task gold and exported prediction.
            write_json(trial / "result.json", {
                "task_name": f"musique/{slug}", "finished_at": "2026-10-28T00:00:00Z",
                "verifier_result": {"rewards": saved},
            })
        return tasks, job, audit

    def assert_official_metrics_match(self, summary):
        expected = {}
        for variant, export in summary["exports"].items():
            upstream = official.evaluate(export["predictions"], export["gold"])
            self.assertEqual(summary["official_metrics"][variant], upstream)
            expected.update({f"{variant}_{key}": value for key, value in upstream.items()})
        self.assertEqual(set(expected), OFFICIAL_KEYS)
        self.assertEqual({key: summary["aggregate"][key] for key in OFFICIAL_KEYS}, expected)
        self.assertNotIn("reward", summary["aggregate"])
        self.assertFalse(any(key.startswith("task_reward_") for key in summary["aggregate"]))
        return expected

    def test_oracle_public_rewards_contain_only_applicable_metrics_all_equal_one(self):
        expected_keys = [ANSWER_KEYS, ANSWER_KEYS | {"sufficiency_correct"}, {"sufficiency_correct"}]
        for split in ("dev", "train"):
            for index, (gold, prediction) in enumerate(three_records(split)):
                with self.subTest(split=split, variant=gold["variant"], answerable=gold["record"]["answerable"]):
                    output = self.verifier_cli(gold, json.dumps(prediction), f"perfect-{split}-{index}")
                    public = json.loads((output / "reward.json").read_text())
                    self.assertEqual(set(public), expected_keys[index])
                    self.assertEqual(set(public.values()), {1.0})
                    self.assertFalse(METADATA_KEYS & public.keys())
                    self.assertNotIn("reward", public)
                    diagnostic = json.loads((output / "diagnostics.json").read_text())
                    scores = verify.score_prediction(prediction, gold)
                    self.assertEqual(diagnostic["scores"], scores)
                    self.assertTrue(METADATA_KEYS <= scores.keys())
                    self.assertEqual(scores["split"], int(split == "train"))
                    self.assertNotIn("reward", scores)
                    self.assertEqual(public, verify.public_metrics(scores))
                    self.assertEqual(float((output / "reward.txt").read_text()), 1)
                    if not gold["record"]["answerable"]:
                        self.assertEqual([scores[key] for key in ANSWER_KEYS], [0, 0, 0])

    def test_malformed_predictions_expose_zero_only_for_applicable_metrics(self):
        expected_keys = [ANSWER_KEYS, ANSWER_KEYS | {"sufficiency_correct"}, {"sufficiency_correct"}]
        for index, (gold, _) in enumerate(three_records()):
            for kind, payload in (("json", "{invalid json"), ("schema", "{}")):
                with self.subTest(index=index, kind=kind):
                    output = self.verifier_cli(gold, payload, f"malformed-{index}-{kind}")
                    public = json.loads((output / "reward.json").read_text())
                    self.assertEqual(set(public), expected_keys[index])
                    self.assertEqual(set(public.values()), {0})
                    self.assertEqual(float((output / "reward.txt").read_text()), 0)
                    diagnostic = json.loads((output / "diagnostics.json").read_text())
                    self.assertFalse(diagnostic["valid_prediction"])
                    self.assertEqual(diagnostic["scores"], verify.empty_reward(gold))
                    self.assertFalse((output / "prediction.json").exists())

    def test_partial_answers_and_wrong_sufficiency_keep_metrics_separate(self):
        for index, variant in enumerate(("ans", "full")):
            gold, prediction = fixture(variant)
            prediction.update(predicted_answer="red", predicted_support_idxs=[1], predicted_answerable=False)
            output = self.verifier_cli(gold, json.dumps(prediction), f"partial-{index}")
            public = json.loads((output / "reward.json").read_text())
            self.assertAlmostEqual(public["answer_f1"], 2 / 3)
            self.assertAlmostEqual(public["support_f1"], 2 / 3)
            self.assertEqual(public["answer_em"], 0)
            if variant == "ans":
                self.assertNotIn("sufficiency_correct", public)
                self.assertAlmostEqual(float((output / "reward.txt").read_text()), 2 / 3)
            else:
                self.assertEqual(public["sufficiency_correct"], 0)
                self.assertEqual(float((output / "reward.txt").read_text()), 0)
        gold, prediction = fixture("full", False)
        prediction["predicted_answerable"] = True
        self.assertEqual(verify.public_metrics(verify.score_prediction(prediction, gold)), {"sufficiency_correct": 0})

    def test_live_means_exclude_inapplicable_fields_instead_of_inserting_zeros(self):
        public = [verify.public_metrics(verify.score_prediction(prediction, gold)) for gold, prediction in three_records()]
        metrics = aggregate_trial_metrics(public)
        expected = {"record_answer_f1": 1, "record_answer_em": 1,
                    "record_support_f1": 1, "record_sufficiency_accuracy": 1}
        self.assertEqual({key: metrics[key] for key in expected}, expected)
        public[0].update(answer_f1=0.5, answer_em=0, support_f1=0.5)
        public[2]["sufficiency_correct"] = 0
        metrics = aggregate_trial_metrics(public)
        self.assertEqual(metrics["record_answer_f1"], 0.75)
        self.assertEqual(metrics["record_answer_em"], 0.5)
        self.assertEqual(metrics["record_support_f1"], 0.75)
        self.assertEqual(metrics["record_sufficiency_accuracy"], 0.5)
        self.assertFalse(METADATA_KEYS & metrics.keys())
        self.assertFalse(OFFICIAL_KEYS & metrics.keys())
        self.assertNotIn("reward", metrics)

    def test_live_reducer_rejects_unknown_fields_and_counts_nulls_without_scoring_them(self):
        row = {"answer_f1": 1, "answer_em": 1, "support_f1": 1}
        metrics = aggregate_trial_metrics([row, None])
        self.assertEqual(metrics["record_answer_f1"], 1)
        self.assertEqual(metrics["null_rewards"], 1)
        with self.assertRaises(ValueError):
            aggregate_trial_metrics([row, None], require_complete=True)
        for field in (*METADATA_KEYS, "reward", "unexpected_score"):
            with self.subTest(field=field), self.assertRaises(ValueError):
                aggregate_trial_metrics([row | {field: 1}])

    def test_audit_reconstructs_all_eight_official_metrics_for_public_and_legacy_artifacts(self):
        for schema in ("public", "legacy", "legacy-primary"):
            with self.subTest(schema=schema):
                tasks, job, output = self.make_job(schema, schema=schema)
                summary = audit_job(tasks, job, output, require_perfect=True)
                self.assertEqual(summary["status"], "passed")
                self.assertEqual(summary["audited_trials"], 3)
                expected = self.assert_official_metrics_match(summary)
                self.assertEqual(set(expected.values()), {1.0})

    def test_bad_full_pair_changes_official_group_scores_without_gating_raw_metrics(self):
        rows = three_records()
        rows[0][1].update(predicted_answer="red", predicted_support_idxs=[1], predicted_answerable=False)
        rows[2][1]["predicted_answerable"] = True
        tasks, job, output = self.make_job("bad-pair", rows=rows)
        scores = self.assert_official_metrics_match(audit_job(tasks, job, output))
        self.assertEqual(scores, {
            "ans_answer_f1": 0.667, "ans_answer_em": 0, "ans_support_f1": 0.667,
            "full_answer_f1": 1, "full_answer_em": 1, "full_support_f1": 1,
            "full_group_answer_sufficiency_f1": 0, "full_group_support_sufficiency_f1": 0,
        })

    def test_audit_rejects_tampered_public_raw_or_legacy_primary_scores(self):
        for schema, field, value in (("public", "support_f1", 0),
                                     ("public", "unexpected_score", 1),
                                     ("legacy", "group_id", 1),
                                     ("legacy-primary", "reward", 0)):
            with self.subTest(schema=schema):
                tasks, job, output = self.make_job(f"tampered-{schema}-{field}", schema=schema)
                path = job / "trial-0/verifier/reward.json"
                saved = json.loads(path.read_text())
                saved[field] = value
                write_json(path, saved)
                result_path = job / "trial-0/result.json"
                result = json.loads(result_path.read_text())
                result["verifier_result"]["rewards"] = saved
                write_json(result_path, result)
                with self.assertRaisesRegex(ValueError, "does not match|Unsupported public reward fields"):
                    audit_job(tasks, job, output)
                self.assertFalse((output / "audit.json").exists())

    def test_strict_audit_still_rejects_bad_support(self):
        rows = three_records()
        rows[0][1]["predicted_support_idxs"] = []
        tasks, job, output = self.make_job("bad-support", rows=rows)
        summary = audit_job(tasks, job, output)
        self.assertEqual(summary["aggregate"]["ans_answer_f1"], 1)
        self.assertEqual(summary["aggregate"]["ans_support_f1"], 0)
        with self.assertRaisesRegex(ValueError, "raw support_f1 = 0"):
            audit_job(tasks, job, output, require_perfect=True)
        self.assertFalse((output / "audit.json").exists())


if __name__ == "__main__":
    unittest.main()
