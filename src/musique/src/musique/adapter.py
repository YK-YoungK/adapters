"""Convert the official MuSiQue-Ans v1.0 development split to Harbor tasks."""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import tempfile
import zipfile
from pathlib import Path

UPSTREAM_REVISION = "922ac98f19a201998dbdae6d7f2887a5258dbdeb"
DATA_FILE = "musique_ans_v1.0_dev.jsonl"
DATA_SHA256 = "15fa63794d18a94ce12411aca6e2327e65b6e83b0b1490efab3f1962e48abf3b"
ARCHIVE_SHA256 = "98f839bf2fd5319f5c688aed77901a6d5c30b3b9f9f691ab9a8ecafb045ee0cd"
GOOGLE_DRIVE_ID = "1tGdADlNjWFaHLeZZGShh2IRcpO6Lv24h"
TEMPLATE_DIR = Path(__file__).parent / "task-template"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_data(cache_dir: Path) -> Path:
    """Cache the official dev JSONL, verifying both archive and data bytes."""
    data_path = cache_dir / DATA_FILE
    if data_path.exists():
        if file_sha256(data_path) != DATA_SHA256:
            raise ValueError(f"Checksum mismatch in {data_path}; remove it and retry.")
        return data_path

    import gdown

    cache_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="musique-download-", dir=cache_dir) as tmp:
        archive_path = Path(tmp) / "musique_v1.0.zip"
        result = gdown.download(id=GOOGLE_DRIVE_ID, output=str(archive_path), quiet=False)
        if result is None or not archive_path.is_file():
            raise RuntimeError("Official download failed; supply --data-path with the dev JSONL.")
        if file_sha256(archive_path) != ARCHIVE_SHA256:
            raise ValueError("Official archive checksum mismatch; refusing changed data.")
        extracted = Path(tmp) / DATA_FILE
        with zipfile.ZipFile(archive_path) as archive:
            with archive.open(f"data/{DATA_FILE}") as source, extracted.open("wb") as target:
                shutil.copyfileobj(source, target)
        if file_sha256(extracted) != DATA_SHA256:
            raise ValueError("Official dev split checksum mismatch.")
        extracted.replace(data_path)
    return data_path


def task_slug(question_id: str) -> str:
    suffix = re.sub(r"[^a-z0-9]+", "-", question_id.lower()).strip("-")
    if not suffix:
        raise ValueError(f"Question ID has no registry-safe characters: {question_id!r}")
    return f"ans-dev-{suffix}"


def validate_record(record: dict) -> None:
    """Reject hidden labels and Full variants instead of silently changing scope."""
    for key in ("id", "question", "answer"):
        if not isinstance(record.get(key), str) or not record[key].strip():
            raise ValueError(f"Missing or invalid {key!r} in MuSiQue record")
    if record.get("answerable") is not True:
        raise ValueError("Only labeled MuSiQue-Ans records are supported, not MuSiQue-Full/test.")
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
    if not any(p["is_supporting"] for p in paragraphs):
        raise ValueError(f"No supporting paragraphs for {record['id']}")


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


class MuSiQueAdapter:
    """One task per question, with gold data outside the build context."""

    def __init__(
        self,
        output_dir: Path,
        *,
        data_path: Path | None = None,
        cache_dir: Path | None = None,
        limit: int | None = None,
        task_ids: list[str] | None = None,
        overwrite: bool = False,
    ) -> None:
        self.output_dir = Path(output_dir).expanduser().resolve()
        self.data_path = Path(data_path).expanduser().resolve() if data_path else None
        self.cache_dir = Path(cache_dir or Path.home() / ".cache" / "harbor-musique").expanduser()
        self.limit = limit
        self.task_ids = task_ids
        self.overwrite = overwrite

    def _load_records(self) -> list[dict]:
        if self.limit is not None and self.limit < 1:
            raise ValueError("--limit must be a positive integer")
        source = self.data_path or download_data(self.cache_dir)
        self.data_sha256 = file_sha256(source)
        records = []
        ids, slugs = set(), set()
        with source.open(encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, 1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                    if not isinstance(record, dict):
                        raise ValueError("Each JSONL record must be an object")
                    validate_record(record)
                    slug = task_slug(record["id"])
                    if record["id"] in ids or slug in slugs:
                        raise ValueError(f"Duplicate ID or task-name collision: {record['id']}")
                    ids.add(record["id"])
                    slugs.add(slug)
                    records.append(record)
                except (ValueError, TypeError) as error:
                    raise ValueError(f"{source}:{line_number}: {error}") from error
        if not records:
            raise ValueError(f"No records in {source}")
        if self.task_ids is not None:
            requested = set(self.task_ids)
            unknown = requested - ids
            if unknown:
                raise ValueError(f"Unknown task IDs: {', '.join(sorted(unknown))}")
            records = [record for record in records if record["id"] in requested]
        return records[: self.limit]

    def _write_task(self, record: dict, task_dir: Path) -> None:
        shutil.copytree(TEMPLATE_DIR, task_dir, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        slug = task_slug(record["id"])
        task_config = (task_dir / "task.toml").read_text(encoding="utf-8")
        task_config = task_config.replace("{task_id}", slug)
        task_config = task_config.replace("{upstream_revision}", UPSTREAM_REVISION)
        task_config = task_config.replace("{data_sha256}", self.data_sha256)
        (task_dir / "task.toml").write_text(task_config, encoding="utf-8")

        # Whitelist visible fields: never copy gold/decomposition into the image.
        question = {
            "id": record["id"],
            "question": record["question"],
            "paragraphs": [
                {key: paragraph[key] for key in ("idx", "title", "paragraph_text")}
                for paragraph in record["paragraphs"]
            ],
        }
        write_json(task_dir / "environment" / "question.json", question)
        write_json(task_dir / "tests" / "gold.json", record)
        prediction = {
            "id": record["id"],
            "predicted_answer": record["answer"],
            "predicted_support_idxs": [p["idx"] for p in record["paragraphs"] if p["is_supporting"]],
            "predicted_answerable": True,
        }
        write_json(task_dir / "solution" / "prediction.json", prediction)
        for script in ("solution/solve.sh", "tests/test.sh"):
            (task_dir / script).chmod(0o755)

    def run(self) -> list[Path]:
        records = self._load_records()
        paths = [self.output_dir / task_slug(record["id"]) for record in records]
        # Check all selected paths before writing any tasks.
        for path in paths:
            if path.is_symlink():
                raise ValueError(f"Refusing to write through task symlink: {path}")
            if path.exists() and not self.overwrite:
                raise FileExistsError(f"Task already exists: {path}; use --overwrite to replace it.")
            if path.exists() and not path.is_dir():
                raise ValueError(f"Task path is not a directory: {path}")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        for record, path in zip(records, paths):
            # Render before replacing existing output so rendering errors leave it intact.
            with tempfile.TemporaryDirectory(prefix=".musique-", dir=self.output_dir) as tmp:
                staged = Path(tmp) / path.name
                self._write_task(record, staged)
                if path.exists():
                    shutil.rmtree(path)
                staged.replace(path)
        return paths
