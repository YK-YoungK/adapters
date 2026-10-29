"""Strictly audit completed Harbor MuSiQue jobs against the official evaluator."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tomllib

from musique.aggregate import aggregate_rewards
from musique.trial_metrics import aggregate_trial_metrics


VERIFIER_DIR = Path(__file__).parent / "task-template/tests"


def _read_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def _load_verifier():
    spec = importlib.util.spec_from_file_location("_musique_audit_verifier", VERIFIER_DIR / "verify.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_tasks(tasks_dir: Path, verifier) -> tuple[dict, dict]:
    configs = ([tasks_dir / "task.toml"] if (tasks_dir / "task.toml").is_file()
               else sorted(tasks_dir.glob("*/task.toml")))
    if not configs:
        raise ValueError(f"No generated task.toml files found in {tasks_dir}")
    tasks, aliases = {}, {}
    selected_split = None
    for config_path in configs:
        path = config_path.parent
        config = tomllib.loads(config_path.read_text(encoding="utf-8"))
        name = config.get("task", {}).get("name")
        if not isinstance(name, str) or not name:
            raise ValueError(f"Task has no [task].name: {config_path}")
        gold = _read_json(path / "tests/gold.json")
        verifier.validate_gold(gold)
        split = gold.get("split", "dev")
        metadata_split = config.get("metadata", {}).get("split", split)
        if metadata_split != split:
            raise ValueError(f"Task metadata split and gold split do not match: {config_path}")
        if selected_split is not None and split != selected_split:
            raise ValueError("Cannot mix train and dev tasks; audit them as separate datasets")
        selected_split = split
        tasks[path.name] = {"path": path, "name": name, "gold": gold, "split": split}
        for alias in (path.name, name):
            if alias in aliases and aliases[alias] != path.name:
                raise ValueError(f"Duplicate task name: {alias}")
            aliases[alias] = path.name
    return tasks, aliases


def _trial_task(result: dict, aliases: dict[str, str], result_path: Path) -> str:
    candidates = set()
    name = result.get("task_name")
    if isinstance(name, str) and name in aliases:
        candidates.add(aliases[name])
    path = result.get("config", {}).get("task", {}).get("path")
    if isinstance(path, str) and Path(path).name in aliases:
        candidates.add(aliases[Path(path).name])
    if len(candidates) != 1:
        raise ValueError(f"Unknown or conflicting task identity in {result_path}")
    return candidates.pop()


def _require_perfect_reward(reward: dict, task_name: str) -> None:
    """Require perfect raw task scores; rounded dataset means can hide failures."""
    required = []
    if reward["answerable"]:
        required.extend(("answer_em", "answer_f1", "support_f1"))
    if reward["variant"] == 1:
        required.append("sufficiency_correct")
    for name in required:
        if reward[name] != 1.0:
            raise ValueError(
                f"Expected a perfect oracle score for {task_name}: raw {name} = {reward[name]}"
            )


def audit_job(
    tasks_dir: Path, job_dir: Path, output_dir: Path, *, require_perfect: bool = False
) -> dict:
    """Verify coverage and saved scores; export inputs for official reproduction.

    This consumes completed Harbor trial artifacts. It does not run tasks and
    cannot substitute for a Harbor oracle job. Invalid/missing prediction files
    are rejected because they cannot be replayed by the official evaluator.
    """
    tasks_dir, job_dir, output_dir = (Path(p).resolve() for p in (tasks_dir, job_dir, output_dir))
    output_dir.mkdir(parents=True, exist_ok=True)
    # A failed audit must not leave a previous success certificate in place.
    (output_dir / "audit.json").unlink(missing_ok=True)
    verifier = _load_verifier()
    tasks, aliases = _load_tasks(tasks_dir, verifier)
    split = next(iter(tasks.values()))["split"]
    job_result = _read_json(job_dir / "result.json")
    if not job_result.get("finished_at"):
        raise ValueError("Harbor job has not finished")
    if job_result.get("exception_info") is not None:
        raise ValueError("Harbor job records an exception")
    if job_result.get("n_total_trials", len(tasks)) != len(tasks):
        raise ValueError("Harbor job trial count differs from the generated task count")

    trials = {}
    for result_path in sorted(job_dir.glob("*/result.json")):
        result = _read_json(result_path)
        task_id = _trial_task(result, aliases, result_path)
        if task_id in trials:
            raise ValueError(f"Duplicate Harbor trial for task {task_id}")
        if result.get("exception_info") is not None:
            raise ValueError(f"Harbor trial has an exception: {result_path}")
        if not result.get("finished_at"):
            raise ValueError(f"Harbor trial has not finished: {result_path}")
        trials[task_id] = (result_path.parent, result)
    missing = set(tasks) - trials.keys()
    if missing:
        sample = ", ".join(sorted(missing)[:10])
        raise ValueError(f"Missing {len(missing)} Harbor trials: {sample}")

    rewards, grouped_rows, agents, environments = [], {}, {}, set()
    for task_id, task in tasks.items():
        trial_dir, result = trials[task_id]
        prediction_path = trial_dir / "verifier/prediction.json"
        if not prediction_path.is_file():
            raise ValueError(f"Missing validated prediction export: {prediction_path}")
        prediction = _read_json(prediction_path)
        regenerated = verifier.score_prediction(prediction, task["gold"])
        saved = _read_json(trial_dir / "verifier/reward.json")
        if "variant" in saved or "group_id" in saved:
            # Legacy jobs carried private grouping fields in reward.json.
            # Validate their complete schema, including an optional old scalar.
            aggregate_rewards([saved])
            expected = regenerated.copy()
            if "reward" in saved:
                value = regenerated["answer_f1"]
                if regenerated["variant"] == 1:
                    value = (
                        regenerated["answer_f1"] * regenerated["sufficiency_correct"]
                        if regenerated["answerable"] else regenerated["sufficiency_correct"]
                    )
                expected["reward"] = value
        else:
            aggregate_trial_metrics([saved])
            expected = verifier.public_metrics(regenerated)
        if expected != saved:
            raise ValueError(f"Saved reward does not match the exported prediction: {trial_dir}")
        recorded_verifier = result.get("verifier_result")
        if not isinstance(recorded_verifier, dict) or recorded_verifier.get("rewards") != saved:
            raise ValueError(f"Trial result rewards differ from verifier/reward.json: {trial_dir}")
        if require_perfect:
            _require_perfect_reward(regenerated, task["name"])
        # Pairing identity and answerability come from task gold, independently
        # of the sparse public metrics that Harbor receives from each trial.
        rewards.append(regenerated)
        variant = task["gold"]["variant"]
        grouped_rows.setdefault(variant, []).append((task["gold"]["record"], prediction))
        agent_info = result.get("agent_info", {})
        agents[json.dumps(agent_info, sort_keys=True)] = agent_info
        environment = result.get("config", {}).get("environment", {}).get("type")
        if isinstance(environment, str):
            environments.add(environment)

    aggregate = aggregate_rewards(rewards, require_complete=True)
    official_scores, exports = {}, {}
    for variant, rows in sorted(grouped_rows.items()):
        gold_path = output_dir / f"{variant}_gold.jsonl"
        prediction_path = output_dir / f"{variant}_predictions.jsonl"
        metrics_path = output_dir / f"{variant}_official_metrics.json"
        with gold_path.open("w", encoding="utf-8") as gold_file, prediction_path.open("w", encoding="utf-8") as prediction_file:
            for record, prediction in rows:
                gold_file.write(json.dumps(record, ensure_ascii=True) + "\n")
                prediction_file.write(json.dumps(prediction, ensure_ascii=True) + "\n")
        command = [
            sys.executable, str(VERIFIER_DIR / "upstream/evaluate_v1.0.py"),
            str(prediction_path), str(gold_path), "--output_filepath", str(metrics_path),
        ]
        evaluated = subprocess.run(command, capture_output=True, text=True)
        if evaluated.returncode:
            raise ValueError(f"Official {variant} evaluation failed: {evaluated.stderr}")
        official_scores[variant] = _read_json(metrics_path)
        for name, score in official_scores[variant].items():
            if aggregate.get(f"{variant}_{name}") != score:
                raise ValueError(f"Official/reducer mismatch for {variant}_{name}")
            if require_perfect and score != 1.0:
                raise ValueError(f"Expected a perfect oracle score: {variant}_{name} = {score}")
        exports[variant] = {
            "gold": str(gold_path), "predictions": str(prediction_path),
            "official_metrics": str(metrics_path), "records": len(rows),
        }

    summary = {
        "status": "passed",
        "evidence": "completed Harbor trial artifacts",
        "audited_at": datetime.now(timezone.utc).isoformat(),
        "tasks_dir": str(tasks_dir), "job_dir": str(job_dir),
        "job_started_at": job_result.get("started_at"),
        "job_finished_at": job_result["finished_at"],
        "expected_tasks": len(tasks), "audited_trials": len(trials),
        "split": split,
        "require_perfect": require_perfect,
        "agents": list(agents.values()), "environments": sorted(environments),
        "aggregate": aggregate, "official_metrics": official_scores, "exports": exports,
    }
    (output_dir / "audit.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", type=Path, required=True, help="Generated task directory")
    parser.add_argument("--job", type=Path, required=True, help="Completed Harbor job directory")
    parser.add_argument("--output", type=Path, required=True, help="Audit/export output directory")
    parser.add_argument("--require-perfect", action="store_true", help="Require every official score to equal 1")
    args = parser.parse_args(argv)
    try:
        summary = audit_job(args.tasks, args.job, args.output, require_perfect=args.require_perfect)
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.exit(1, f"MuSiQue audit failed: {error}\n")
    print(json.dumps({
        "audited_trials": summary["audited_trials"],
        "metrics": summary["official_metrics"],
    }, indent=2))
    print(f"Audit saved to {args.output / 'audit.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
