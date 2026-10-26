"""Generator regressions for provenance, isolation, and reproducible selection."""

import copy
import json
from pathlib import Path
import sys
import tempfile
import tomllib
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from musique.adapter import MuSiQueAdapter, file_sha256, task_slug  # noqa: E402


def record(question_id="2hop__123_456"):
    return {
        "id": question_id,
        "question": "Which café did the researcher visit?",
        "paragraphs": [
            {"idx": 5, "title": "Second in index order", "paragraph_text": "Original café text.\nTwo lines.", "is_supporting": True},
            {"idx": 1, "title": "First in index order", "paragraph_text": "Another paragraph.", "is_supporting": False},
            {"idx": 9, "title": "Supporting source", "paragraph_text": "Final paragraph.", "is_supporting": True},
        ],
        "answer": "ORACLE_SECRET",
        "answer_aliases": ["ALIAS_SECRET", "Un café"],
        "answerable": True,
        "question_decomposition": [{"question": "DECOMPOSITION_SECRET", "answer": "INTERMEDIATE_SECRET"}],
        "extra_annotation": "ANNOTATION_SECRET",
    }


def snapshot(root):
    return {path.relative_to(root).as_posix(): path.read_bytes()
            for path in sorted(root.rglob("*")) if path.is_file()}


class AdapterTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.root = Path(self.directory.name)
        self.source = self.root / "source.jsonl"
        self.output = self.root / "tasks"

    def tearDown(self):
        self.directory.cleanup()

    def write_source(self, records):
        self.source.write_text("".join(json.dumps(item, ensure_ascii=False) + "\n" for item in records), encoding="utf-8")

    def generate(self, **kwargs):
        return MuSiQueAdapter(self.output, data_path=self.source, **kwargs).run()

    def test_task_contract_and_gold_isolation(self):
        source = record()
        self.write_source([source])
        task, = self.generate()
        self.assertEqual(task.name, "ans-dev-2hop-123-456")
        config = tomllib.loads((task / "task.toml").read_text())
        self.assertEqual(config["task"]["name"], "musique/ans-dev-2hop-123-456")
        self.assertEqual(config["version"], "1.0")
        self.assertEqual(len(config["task"]["authors"]), 4)
        self.assertEqual(config["metadata"]["source_sha256"], file_sha256(self.source))
        for path in ("instruction.md", "environment/Dockerfile", "solution/solve.sh", "tests/test.sh"):
            self.assertTrue((task / path).is_file(), path)
        self.assertTrue((task / "tests/test.sh").stat().st_mode & 0o111)
        self.assertTrue((task / "solution/solve.sh").stat().st_mode & 0o111)
        visible = json.loads((task / "environment/question.json").read_text())
        self.assertEqual(set(visible), {"id", "question", "paragraphs"})
        self.assertEqual(visible["question"], source["question"])
        self.assertEqual(visible["paragraphs"], [
            {key: paragraph[key] for key in ("idx", "title", "paragraph_text")}
            for paragraph in source["paragraphs"]
        ])
        self.assertEqual({path.name for path in (task / "environment").iterdir()}, {"Dockerfile", "question.json"})
        exposed_text = "\n".join(path.read_text() for path in (task / "environment").iterdir())
        exposed_text += (task / "instruction.md").read_text()
        for secret in ("ORACLE_SECRET", "ALIAS_SECRET", "DECOMPOSITION_SECRET", "INTERMEDIATE_SECRET", "ANNOTATION_SECRET"):
            self.assertNotIn(secret, exposed_text)
        self.assertEqual(json.loads((task / "tests/gold.json").read_text()), source)
        self.assertEqual(json.loads((task / "solution/prediction.json").read_text()), {
            "id": source["id"], "predicted_answer": "ORACLE_SECRET",
            "predicted_support_idxs": [5, 9], "predicted_answerable": True,
        })

    def test_selection_uses_source_order_before_limit(self):
        rows = [record("2hop__30_40"), record("2hop__10_20"), record("3hop__1_2_3")]
        self.write_source(rows)
        tasks = self.generate(task_ids=[rows[2]["id"], rows[0]["id"]], limit=1)
        self.assertEqual([path.name for path in tasks], [task_slug(rows[0]["id"])])
        self.assertEqual(len(list(self.output.iterdir())), 1)

    def test_unknown_id_is_rejected_before_writes(self):
        self.write_source([record()])
        with self.assertRaisesRegex(ValueError, "Unknown task IDs"):
            self.generate(task_ids=["missing"])
        self.assertFalse(self.output.exists())

    def test_generation_is_byte_deterministic(self):
        self.write_source([record(), record("3hop__1_2_3")])
        self.generate()
        first = snapshot(self.output)
        other = self.root / "other-output"
        MuSiQueAdapter(other, data_path=self.source).run()
        self.assertEqual(snapshot(other), first)
        self.generate(overwrite=True)
        self.assertEqual(snapshot(self.output), first)
        self.assertFalse(any("__pycache__" in path for path in first))

    def test_slug_sanitizes_unsafe_characters_and_rejects_empty(self):
        self.assertEqual(task_slug(" /A B_C/ "), "ans-dev-a-b-c")
        with self.assertRaisesRegex(ValueError, "registry-safe"):
            task_slug("///!!!")
        self.write_source([record("///!!!")])
        with self.assertRaisesRegex(ValueError, "registry-safe"):
            self.generate()
        self.assertFalse(self.output.exists())

    def test_duplicate_ids_and_sanitized_names_are_rejected(self):
        for ids in (("2hop__1_2", "2hop__1_2"), ("Case_Name", "case-name")):
            with self.subTest(ids=ids):
                self.write_source([record(question_id) for question_id in ids])
                with self.assertRaisesRegex(ValueError, "collision"):
                    self.generate()
                self.assertFalse(self.output.exists())

    def test_invalid_records_are_rejected_before_writes(self):
        base = record()
        duplicate_paragraphs = copy.deepcopy(base["paragraphs"])
        duplicate_paragraphs[1]["idx"] = duplicate_paragraphs[0]["idx"]
        unlabeled = copy.deepcopy(base["paragraphs"])
        del unlabeled[0]["is_supporting"]
        candidates = [
            base | {"id": ""}, base | {"id": 12}, base | {"answerable": False},
            base | {"answer_aliases": None}, base | {"paragraphs": duplicate_paragraphs},
            base | {"paragraphs": unlabeled},
        ]
        for candidate in candidates:
            with self.subTest(candidate=candidate):
                self.write_source([record("valid_first"), candidate])
                with self.assertRaises(ValueError):
                    self.generate()
                self.assertFalse(self.output.exists())

    def test_existing_task_preflight_prevents_partial_generation(self):
        self.write_source([record("first"), record("second")])
        existing = self.output / task_slug("second")
        existing.mkdir(parents=True)
        (existing / "keep.txt").write_text("existing output")
        before = snapshot(self.output)
        with self.assertRaises(FileExistsError):
            self.generate()
        self.assertEqual(snapshot(self.output), before)

    def test_overwrite_replaces_selected_tasks_only(self):
        self.write_source([record("first"), record("second")])
        first, second = self.generate()
        (first / "stale.txt").write_text("remove this")
        (second / "keep.txt").write_text("retain this")
        unselected = snapshot(second)
        self.generate(task_ids=["first"], overwrite=True)
        self.assertFalse((first / "stale.txt").exists())
        self.assertEqual(snapshot(second), unselected)

    def test_failed_render_preserves_existing_task(self):
        self.write_source([record()])
        self.generate()
        before = snapshot(self.output)
        with patch.object(MuSiQueAdapter, "_write_task", side_effect=RuntimeError("render failed")):
            with self.assertRaisesRegex(RuntimeError, "render failed"):
                self.generate(overwrite=True)
        self.assertEqual(snapshot(self.output), before)

    def test_task_symlink_is_rejected_even_with_overwrite(self):
        self.write_source([record()])
        self.output.mkdir()
        target = self.root / "unrelated"
        target.mkdir()
        (target / "keep.txt").write_text("keep")
        (self.output / task_slug(record()["id"])).symlink_to(target, target_is_directory=True)
        with self.assertRaisesRegex(ValueError, "symlink"):
            self.generate(overwrite=True)
        self.assertEqual((target / "keep.txt").read_text(), "keep")


if __name__ == "__main__":
    unittest.main()
