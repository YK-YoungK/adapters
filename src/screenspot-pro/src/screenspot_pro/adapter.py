"""Convert the official ScreenSpot-Pro annotations and PNGs into Harbor tasks."""

from __future__ import annotations

import json
import math
import re
import shutil
import tempfile
from pathlib import Path, PurePosixPath
from string import Template

from huggingface_hub import hf_hub_download, snapshot_download
from PIL import Image

DATASET_REPO = "likaixin/ScreenSpot-Pro"
DATASET_REVISION = "210e78d3844251110bff86c95835ebd37a6930fa"
UPSTREAM_REVISION = "dbe00114bc53a32c61c1a267786da85967710da8"
TEMPLATE_DIR = Path(__file__).parent / "task-template"


def task_id_for(upstream_id: str, language: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", upstream_id.lower()).strip("-")
    if not slug:
        raise ValueError(f"Empty task name after sanitizing {upstream_id!r}")
    return f"{language}-{slug}"


def validate_sample(sample: dict, source: Path) -> None:
    """Validate structure without correcting upstream annotations."""
    for field in ("id", "instruction", "instruction_cn", "img_filename", "application", "platform", "ui_type", "group"):
        if not isinstance(sample.get(field), str) or not sample[field]:
            raise ValueError(f"{source}: missing or invalid {field}")
    filename = PurePosixPath(sample["img_filename"])
    if filename.is_absolute() or ".." in filename.parts or "\\" in sample["img_filename"]:
        raise ValueError(f"Unsafe image path: {filename}")
    for field, count in (("bbox", 4), ("img_size", 2)):
        values = sample.get(field)
        if not isinstance(values, list) or len(values) != count or any(
            isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v)
            for v in values
        ):
            raise ValueError(f"{source}: invalid {field} for {sample['id']}")
    width, height = sample["img_size"]
    x1, y1, x2, y2 = sample["bbox"]
    if width <= 0 or height <= 0 or x1 > x2 or y1 > y2:
        raise ValueError(f"{source}: invalid dimensions or reversed bbox for {sample['id']}")
    # Do not clamp: inventor_windows_60 has y1=-1 in the official annotations.


class ScreenSpotProAdapter:
    def __init__(
        self,
        output_dir: Path,
        limit: int | None = None,
        overwrite: bool = False,
        task_ids: list[str] | None = None,
        dataset_dir: Path | None = None,
        cache_dir: Path | None = None,
        revision: str = DATASET_REVISION,
        language: str = "en",
    ):
        if language not in ("en", "cn"):
            raise ValueError("language must be en or cn")
        if limit is not None and limit < 1:
            raise ValueError("limit must be positive")
        self.output_dir = Path(output_dir).resolve()
        self.limit = limit
        self.overwrite = overwrite
        self.task_ids = task_ids
        self.dataset_dir = Path(dataset_dir).resolve() if dataset_dir else None
        self.cache_dir = cache_dir
        self.revision = revision
        self.language = language
        self.resolved_revision = "local-unverified" if dataset_dir else revision

    def load_samples(self) -> list[dict]:
        if self.dataset_dir is None:
            root = Path(snapshot_download(
                repo_id=DATASET_REPO, repo_type="dataset", revision=self.revision,
                allow_patterns=["annotations/*.json"], cache_dir=self.cache_dir,
            ))
            self.resolved_revision = root.name
        else:
            root = self.dataset_dir
        files = sorted((root / "annotations").glob("*.json"))
        if not files:
            raise ValueError(f"No annotations/*.json found under {root}")
        samples = []
        seen = set()
        for source in files:
            rows = json.loads(source.read_text(encoding="utf-8"))
            if not isinstance(rows, list):
                raise ValueError(f"{source}: expected a list of annotations")
            for sample in rows:
                if not isinstance(sample, dict):
                    raise ValueError(f"{source}: expected an annotation object")
                validate_sample(sample, source)
                task_id = task_id_for(sample["id"], self.language)
                if task_id in seen:
                    raise ValueError(f"Duplicate task ID after sanitizing: {task_id}")
                seen.add(task_id)
                samples.append({**sample, "task_id": task_id, "annotation_file": source.name})
        if not samples:
            raise ValueError("Annotations contain no tasks")
        samples.sort(key=lambda row: row["id"])
        if self.task_ids:
            requested = set(self.task_ids)
            known = {s["id"] for s in samples} | {s["task_id"] for s in samples}
            unknown = requested - known
            if unknown:
                raise ValueError(f"Unknown task IDs: {', '.join(sorted(unknown))}")
            samples = [s for s in samples if s["id"] in requested or s["task_id"] in requested]
        return samples[:self.limit]

    def image_path(self, sample: dict) -> Path:
        filename = f"images/{sample['img_filename']}"
        if self.dataset_dir:
            image = self.dataset_dir / filename
        else:
            image = Path(hf_hub_download(
                repo_id=DATASET_REPO, repo_type="dataset", revision=self.resolved_revision,
                filename=filename, cache_dir=self.cache_dir,
            ))
        with Image.open(image) as png:
            if png.format != "PNG" or list(png.size) != sample["img_size"]:
                raise ValueError(f"Image format/dimensions disagree with annotations: {image}")
            png.verify()
        return image

    def write_task(self, sample: dict, destination: Path, image: Path) -> None:
        shutil.copytree(
            TEMPLATE_DIR, destination, dirs_exist_ok=True,
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
        )
        shutil.copyfile(image, destination / "environment" / "screenshot.png")
        instruction = sample["instruction_cn" if self.language == "cn" else "instruction"]
        values = {
            "task_name": json.dumps(f"screenspot-pro/{sample['task_id']}")[1:-1],
            "upstream_id": json.dumps(sample["id"])[1:-1],
            "language": json.dumps(self.language)[1:-1],
            "dataset_revision": json.dumps(self.resolved_revision)[1:-1],
            "upstream_revision": json.dumps(UPSTREAM_REVISION)[1:-1],
            **{k: json.dumps(sample[k])[1:-1] for k in ("application", "platform", "ui_type", "group")},
        }
        for name in ("task.toml", "instruction.md"):
            path = destination / name
            path.write_text(Template(path.read_text(encoding="utf-8")).substitute(
                values, instruction=instruction,
            ), encoding="utf-8")
        # Ground truth is mounted by Harbor at verification time, never in the image.
        (destination / "tests" / "expected.json").write_text(
            json.dumps({k: sample[k] for k in ("id", "bbox", "img_size")}, indent=2) + "\n",
            encoding="utf-8",
        )
        width, height = sample["img_size"]
        x1, y1, x2, y2 = sample["bbox"]
        x, y = (x1 + x2) / 2 / width, (y1 + y2) / 2 / height
        (destination / "solution" / "answer.txt").write_text(f"[[{x:.17f},{y:.17f}]]\n")
        for relative in ("tests/test.sh", "solution/solve.sh"):
            (destination / relative).chmod(0o755)

    def run(self) -> None:
        samples = self.load_samples()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        written = skipped = 0
        for sample in samples:
            target = self.output_dir / sample["task_id"]
            if target.is_symlink() or (target.exists() and not target.is_dir()):
                raise ValueError(f"Refusing to replace non-directory task path: {target}")
            if target.exists() and not self.overwrite:
                skipped += 1
                continue
            image = self.image_path(sample)
            # Build fully before replacing a task so download/render errors keep old output.
            temporary = Path(tempfile.mkdtemp(prefix=f".{sample['task_id']}-", dir=self.output_dir))
            try:
                self.write_task(sample, temporary, image)
                if target.exists():
                    shutil.rmtree(target)
                temporary.rename(target)
            finally:
                if temporary.exists():
                    shutil.rmtree(temporary)
            written += 1
        print(f"Generated {written} tasks; skipped {skipped} existing tasks in {self.output_dir}")
