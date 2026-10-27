"""Convert labeled MuSiQue v1.0 train or dev records into isolated Harbor tasks."""

from __future__ import annotations

from collections import defaultdict
import hashlib
import json
from pathlib import Path
import re
import shutil
import tempfile
import tomllib
import zipfile

UPSTREAM_REVISION = "922ac98f19a201998dbdae6d7f2887a5258dbdeb"
DATA_SHA256 = {
    "dev": {
        "ans": "15fa63794d18a94ce12411aca6e2327e65b6e83b0b1490efab3f1962e48abf3b",
        "full": "8cab31d56a3a1c4ef491b205a8dab3f1ac9c66e472098c6cf1de4e20294f7a4a",
    },
    "train": {
        "ans": "83a75b1e11e4e9bb8f8308e72ac40ca617ae4431b3a0d955b61cab259248490a",
        "full": "b1cd998f7e0e2838d6fda024e4ad1eb0e7fc3edefdadb0bd9b5b10b0907f2034",
    },
}
TEST_SPLIT_REASON = (
    "The public MuSiQue test split withholds gold answers, answer aliases, "
    "support labels and answerability labels; it cannot be graded locally or "
    "supply reference oracles. Use --split train or --split dev."
)
ARCHIVE_SHA256 = "98f839bf2fd5319f5c688aed77901a6d5c30b3b9f9f691ab9a8ecafb045ee0cd"
GOOGLE_DRIVE_ID = "1tGdADlNjWFaHLeZZGShh2IRcpO6Lv24h"
TEMPLATE_DIR = Path(__file__).parent / "task-template"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_split(split: str) -> str:
    if split == "test":
        raise ValueError(TEST_SPLIT_REASON)
    if split not in DATA_SHA256:
        raise ValueError("split must be train or dev")
    return split


def data_filename(variant: str, split: str = "dev") -> str:
    return f"musique_{variant}_v1.0_{split}.jsonl"


def download_data(
    cache_dir: Path, split: str = "dev", variants: tuple[str, ...] = ("ans", "full")
) -> Path:
    """Cache only the requested labeled split/variants, verifying release bytes."""
    validate_split(split)
    if not variants or any(v not in DATA_SHA256[split] for v in variants):
        raise ValueError("Requested variants must be ans and/or full")
    paths = {v: cache_dir / data_filename(v, split) for v in variants}
    for variant, path in paths.items():
        if path.exists() and file_sha256(path) != DATA_SHA256[split][variant]:
            raise ValueError(f"Checksum mismatch in {path}; remove the corrupt cache file.")
    if all(path.is_file() for path in paths.values()):
        return cache_dir
    import gdown

    cache_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="musique-download-", dir=cache_dir) as tmp:
        archive_path = Path(tmp) / "musique_v1.0.zip"
        result = gdown.download(id=GOOGLE_DRIVE_ID, output=str(archive_path), quiet=False)
        if result is None or not archive_path.is_file():
            raise RuntimeError(f"Official download failed; supply --data-dir with the {split} JSONL files.")
        if file_sha256(archive_path) != ARCHIVE_SHA256:
            raise ValueError("Official archive checksum mismatch; refusing changed data.")
        with zipfile.ZipFile(archive_path) as archive:
            for variant, path in paths.items():
                extracted = Path(tmp) / path.name
                with archive.open(f"data/{path.name}") as source, extracted.open("wb") as target:
                    shutil.copyfileobj(source, target)
                if file_sha256(extracted) != DATA_SHA256[split][variant]:
                    raise ValueError(f"Official {variant} {split} split checksum mismatch.")
                extracted.replace(path)
    return cache_dir


def visible_record(record: dict) -> dict:
    """Whitelist visible fields; exclude answers, support labels and decomposition."""
    return {
        "id": record["id"], "question": record["question"],
        "paragraphs": [
            {k: p[k] for k in ("idx", "title", "paragraph_text")}
            for p in record["paragraphs"]
        ],
    }


def task_slug(record: dict, variant: str, split: str = "dev") -> str:
    suffix = re.sub(r"[^a-z0-9]+", "-", record["id"].lower()).strip("-")
    if not suffix:
        raise ValueError(f"Question ID has no registry-safe characters: {record['id']!r}")
    slug = f"{variant}-{split}-{suffix}"
    if variant == "full":
        # Distinguish repeated IDs using public input, never answerability labels.
        payload = json.dumps(visible_record(record), sort_keys=True, ensure_ascii=False)
        slug += "-" + hashlib.sha256(payload.encode()).hexdigest()[:12]
    return slug


def group_id(question_id: str) -> int:
    # Exactly representable in JSON consumers using double precision.
    return int(hashlib.sha256(question_id.encode()).hexdigest()[:13], 16)


def validate_record(record: dict, variant: str) -> None:
    if not isinstance(record, dict):
        raise ValueError("Each JSONL record must be an object")
    for key in ("id", "question", "answer"):
        if not isinstance(record.get(key), str) or not record[key].strip():
            raise ValueError(f"Missing or invalid {key!r} in MuSiQue record")
    if type(record.get("answerable")) is not bool:
        raise ValueError(f"Missing answerability label for {record['id']}")
    if variant == "ans" and not record["answerable"]:
        raise ValueError("MuSiQue-Ans cannot contain unanswerable records")
    aliases = record.get("answer_aliases")
    if not isinstance(aliases, list) or any(not isinstance(a, str) for a in aliases):
        raise ValueError(f"Invalid answer_aliases for {record['id']}")
    paragraphs = record.get("paragraphs")
    if not isinstance(paragraphs, list) or not paragraphs:
        raise ValueError(f"Missing paragraphs for {record['id']}")
    indices = set()
    for paragraph in paragraphs:
        if not isinstance(paragraph, dict):
            raise ValueError("Paragraph must be an object")
        index = paragraph.get("idx")
        if type(index) is not int or index < 0 or index in indices:
            raise ValueError(f"Invalid or duplicate paragraph index for {record['id']}")
        indices.add(index)
        if type(paragraph.get("is_supporting")) is not bool:
            raise ValueError(f"Missing support labels for {record['id']}")
        for key in ("title", "paragraph_text"):
            if not isinstance(paragraph.get(key), str):
                raise ValueError(f"Missing paragraph {key} for {record['id']}")
    if record["answerable"] and not any(p["is_supporting"] for p in paragraphs):
        raise ValueError(f"No supporting paragraphs for {record['id']}")


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


class MuSiQueAdapter:
    """One independent context per record; Full pairs meet only in aggregation."""

    def __init__(self, output_dir: Path, *, variant: str = "both", split: str = "dev",
                 data_dir: Path | None = None, cache_dir: Path | None = None,
                 limit: int | None = None, task_ids: list[str] | None = None,
                 overwrite: bool = False) -> None:
        self.output_dir = Path(output_dir).expanduser().resolve()
        self.variant = variant
        self.split = split
        self.data_dir = Path(data_dir).expanduser().resolve() if data_dir else None
        self.cache_dir = Path(cache_dir or Path.home() / ".cache" / "harbor-musique").expanduser()
        self.limit, self.task_ids, self.overwrite = limit, task_ids, overwrite
        self.data_sha256: dict[str, str] = {}

    def _load_units(self) -> list[list[tuple[str, dict]]]:
        validate_split(self.split)
        if self.variant not in ("ans", "full", "both"):
            raise ValueError("variant must be ans, full or both")
        if self.limit is not None and self.limit < 1:
            raise ValueError("--limit must be a positive integer")
        variants = ("ans", "full") if self.variant == "both" else (self.variant,)
        source_dir = self.data_dir or download_data(self.cache_dir, self.split, variants)
        units, slugs, group_ids = [], set(), {}
        for variant in variants:
            source = source_dir / data_filename(variant, self.split)
            self.data_sha256[variant] = file_sha256(source)
            groups: dict[str, list[dict]] = defaultdict(list)
            with source.open(encoding="utf-8") as stream:
                for line_number, line in enumerate(stream, 1):
                    if not line.strip():
                        continue
                    try:
                        record = json.loads(line)
                        validate_record(record, variant)
                        slug = task_slug(record, variant, self.split)
                        if slug in slugs:
                            raise ValueError(f"Duplicate task or task-name collision: {slug}")
                        slugs.add(slug)
                        token = group_id(record["id"])
                        if token in group_ids and group_ids[token] != record["id"]:
                            raise ValueError("Question group hash collision")
                        group_ids[token] = record["id"]
                        groups[record["id"]].append(record)
                    except (ValueError, TypeError) as error:
                        raise ValueError(f"{source}:{line_number}: {error}") from error
            if not groups:
                raise ValueError(f"No records in {source}")
            for question_id in sorted(groups):
                records = groups[question_id]
                if variant == "ans" and len(records) != 1:
                    raise ValueError(f"Duplicate Ans ID: {question_id}")
                if variant == "full" and (
                    len(records) != 2 or sorted(r["answerable"] for r in records) != [False, True]
                    or len({r["question"] for r in records}) != 1
                ):
                    raise ValueError(f"Full ID {question_id} must have one answerable/unanswerable pair with the same question")
                units.append([(variant, r) for r in sorted(records, key=lambda r: task_slug(r, variant, self.split))])
        if self.task_ids is not None:
            requested = set(self.task_ids)
            known = slugs | {f"musique/{slug}" for slug in slugs} | set(group_ids.values())
            unknown = requested - known
            if unknown:
                raise ValueError(f"Unknown task IDs: {', '.join(sorted(unknown))}")
            units = [unit for unit in units if any(
                requested & {r["id"], task_slug(r, v, self.split), f"musique/{task_slug(r, v, self.split)}"}
                for v, r in unit
            )]
        if self.limit is not None:
            selected, count = [], 0
            for unit in units:
                if count + len(unit) <= self.limit:
                    selected.append(unit)
                    count += len(unit)
            units = selected
        if not units:
            raise ValueError("No complete tasks selected; Full requires room for at least two records")
        return units

    def _write_task(self, record: dict, variant: str, task_dir: Path) -> None:
        shutil.copytree(TEMPLATE_DIR, task_dir, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        config = (TEMPLATE_DIR / "task.toml").read_text(encoding="utf-8")
        for key, value in {"task_id": task_slug(record, variant, self.split), "upstream_revision": UPSTREAM_REVISION,
                           "data_sha256": self.data_sha256[variant], "variant": variant, "split": self.split}.items():
            config = config.replace("{" + key + "}", value)
        (task_dir / "task.toml").write_text(config, encoding="utf-8")
        instruction = (TEMPLATE_DIR / "instruction.md").read_text(encoding="utf-8")
        instruction = instruction.replace("{answerability_instruction}", (
            "The supplied paragraphs contain sufficient evidence to answer this question. Set `predicted_answerable` to `true`."
            if variant == "ans" else
            "Decide whether the supplied paragraphs contain sufficient evidence to answer the question. Set `predicted_answerable` to `true` if they do, and `false` otherwise."
        ))
        instruction = instruction.replace("{question_json}", json.dumps(visible_record(record), ensure_ascii=False, indent=2))
        (task_dir / "instruction.md").write_text(instruction, encoding="utf-8")
        write_json(task_dir / "tests" / "gold.json", {
            "record": record, "variant": variant, "split": self.split,
            "group_id": group_id(record["id"]),
        })
        write_json(task_dir / "solution" / "prediction.json", {
            "id": record["id"], "predicted_answer": record["answer"],
            "predicted_support_idxs": [p["idx"] for p in record["paragraphs"] if p["is_supporting"]],
            "predicted_answerable": record["answerable"],
        })
        for script in ("solution/solve.sh", "tests/test.sh"):
            (task_dir / script).chmod(0o755)

    def run(self) -> list[Path]:
        validate_split(self.split)
        # Keep official train and dev averages separate, including incremental runs.
        for config_path in self.output_dir.glob("*/task.toml"):
            config = tomllib.loads(config_path.read_text(encoding="utf-8"))
            existing_split = config.get("metadata", {}).get("split")
            if existing_split != self.split:
                raise ValueError(
                    f"Output contains split {existing_split!r} at {config_path}; "
                    f"use a separate output directory for {self.split}."
                )
        records = [entry for unit in self._load_units() for entry in unit]
        paths = [self.output_dir / task_slug(record, variant, self.split) for variant, record in records]
        for path in paths:
            if path.is_symlink():
                raise ValueError(f"Refusing to write through task symlink: {path}")
            if path.exists() and not self.overwrite:
                raise FileExistsError(f"Task already exists: {path}; use --overwrite to replace it")
            if path.exists() and not path.is_dir():
                raise ValueError(f"Task path is not a directory: {path}")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        for (variant, record), path in zip(records, paths):
            with tempfile.TemporaryDirectory(prefix=".musique-", dir=self.output_dir) as tmp:
                staged = Path(tmp) / path.name
                self._write_task(record, variant, staged)
                if path.exists():
                    shutil.rmtree(path)
                staged.replace(path)
        return paths
