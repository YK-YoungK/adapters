# MuSiQue → Harbor Adapter

## Overview

This adapter converts all **2,417 examples from MuSiQue-Ans v1.0's development split** into Harbor tasks for English multi-hop question answering. Each task includes the original question and all its candidate Wikipedia paragraphs. The agent submits an answer and the indices of the supporting paragraphs.

The source is [StonyBrookNLP/MuSiQue](https://github.com/StonyBrookNLP/MuSiQue), released under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). This initial implementation covers guide Steps 1–3; model parity and dataset publication are pending. Excluded splits and adaptations are described in Notes & Caveats.

## What is MuSiQue?

[MuSiQue: Multihop Questions via Single-hop Question Composition](https://arxiv.org/abs/2108.00573) composes connected single-hop questions into questions requiring multiple reasoning steps. MuSiQue-Ans contains answerable questions; MuSiQue-Full additionally includes paired unanswerable variants. This adapter targets MuSiQue-Ans.

The pinned upstream evaluator reports `answer_em`, `answer_f1`, and `support_f1`. Answer scores use the best match against the canonical answer and its aliases after upstream normalization. Support F1 compares sets of paragraph indices. All three scores are in [0, 1].

## Adapter Features

- Downloads and caches the official v1.0 archive with SHA-256 verification; accepts a local JSONL file through `--data-path`.
- Preserves question text, paragraph titles, paragraph text, original indices, and paragraph order. No supporting paragraphs are selected for the agent in advance.
- Keeps answers, aliases, support labels, and question decompositions out of the agent's input and container build context.
- Uses stable task names derived from upstream question IDs, with filtering and overwrite controls for reproducible generation.
- Runs the original metric implementations in a Python container without model inference, API credentials, AllenNLP, or an LLM judge.
- Writes all three metrics to `/logs/verifier/reward.json` and mirrors `answer_f1` to the guide-required `/logs/verifier/reward.txt`. Harbor prioritizes the JSON rewards.
- Supplies a reference oracle using the upstream answer and supporting indices. No custom agent or parity runner has been added at this stage.

## Generated Task Structure

```text
musique/
└── <task-id>/
    ├── task.toml
    ├── instruction.md
    ├── environment/
    │   ├── Dockerfile
    │   └── question.json          # Only public question/context fields
    ├── solution/
    │   ├── solve.sh
    │   └── prediction.json        # Reference prediction, mounted for oracle only
    └── tests/
        ├── test.sh
        ├── verify.py
        ├── gold.json              # Original labeled record, verifier only
        └── upstream/              # Pinned evaluator, metrics, license, provenance
```

The implementation lives in `src/musique/src/musique/`, and its reusable task files are under `task-template/`. Generated datasets and run logs are separate from the adapter package. Task names use `musique/ans-dev-<sanitized-upstream-id>`; for example, `2hop__460946_294723` becomes directory `ans-dev-2hop-460946-294723` and task name `musique/ans-dev-2hop-460946-294723`.

## Run Evaluation / Harness

### Registry

This dataset is **not published to the Harbor registry**. The intended dataset ID is `musique/musique-ans-dev`. After Step 8 publication, run `harbor run -d musique/musique-ans-dev -a oracle`; this command is unavailable before publication. Use the local commands below during development.

### Local dataset

After generating tasks, run these commands from the repository root:

```bash
# Complete oracle verification, using the supplied reference configuration.
harbor run -c src/musique/run_musique.yaml

# Equivalent local dataset invocation.
harbor run -p datasets/musique -a oracle

# Inspect one task before running the complete set.
harbor trial start -p datasets/musique/<task-id> -a oracle
```

The configuration defaults to Docker, four concurrent trials, and the oracle agent. Results are written under `jobs/`. The oracle needs no model or API key. Agent/model selection for parity belongs to Step 4 and is not configured here.

The agent reads `/workspace/question.json` and writes one JSON object to `/workspace/prediction.json` with the official prediction fields:

```json
{
  "id": "<original question ID from question.json>",
  "predicted_answer": "<short answer>",
  "predicted_support_idxs": [0, 1],
  "predicted_answerable": true
}
```

The indices above illustrate the format only; the agent must identify the actual supporting paragraphs. Every task in this split is answerable. Upstream MuSiQue-Ans scoring does not gate answer or support scores on `predicted_answerable`.

## Usage: Create Task Directories

Run the adapter inside its independent uv project:

```bash
cd src/musique
uv sync
uv run musique --output-dir ../../datasets/musique

# A small generation sample, stored separately from the complete dataset.
uv run musique --output-dir ../../datasets/musique-smoke --limit 5

# Select by an original upstream ID.
uv run musique --output-dir ../../datasets/musique-selected \
  --task-ids 2hop__460946_294723

# Use an already downloaded official JSONL file.
uv run musique --output-dir ../../datasets/musique \
  --data-path /path/to/musique_ans_v1.0_dev.jsonl --overwrite
```

| Flag | Purpose |
|---|---|
| `--output-dir` | Destination for generated task directories. |
| `--limit` | Generate at most N selected tasks. |
| `--task-ids` | Select specific original question IDs. |
| `--overwrite` | Replace existing selected task directories. |
| `--data-path` | Read a local labeled JSONL file, including a dev subset for debugging. |
| `--cache-dir` | Choose the official-download cache location; default `~/.cache/harbor-musique`. |
| `--split dev` | Explicitly select the supported split. |

The default download checks the official archive and dev-file hashes. An explicit `--data-path` permits fixtures or subsets without requiring the official file hash; each task records the actual source-file SHA-256. Use the pinned complete dev file for the 2,417-task verification claim. Do not mix sample and full-dataset output directories when reporting oracle coverage.

## Comparison with Original Benchmark (Parity)

Adapter PR: [WIP #53](https://github.com/harbor-framework/adapters/pull/53).

**Model parity has not been run.** `parity_experiment.json` is an empty array. Agents, models, repetitions, and an original-side agent wrapper will be agreed in Step 4. No parity fork, parity-results PR, or dataset PR exists yet. The [original repository](https://github.com/StonyBrookNLP/MuSiQue) remains the source of benchmark definitions.

| Agent | Model | Metric | Runs | Dataset Size | Original (mean ± SEM) | Harbor (mean ± SEM) |
|---|---|---|---|---|---|---|
| Pending | Pending | answer_em / answer_f1 / support_f1 | 0 | 2,417 | Not run | Not run |

Oracle verification completed on **2026-10-26: all 2,417 tasks received 1.0 for `answer_em`, `answer_f1`, and `support_f1`, with zero exceptions in the completed Harbor job**. The unchanged upstream evaluator independently returned the same three perfect scores for all 2,417 reference predictions. This is an implementation check, not a model parity experiment.

Evidence: [Harbor result](evidence/oracle-result.json), [independent per-task verification](evidence/verification.json), [actual terminal transcript](evidence/oracle-terminal.log), [terminal screenshot](evidence/oracle-terminal.png), [runtime provenance](evidence/runtime.json), and [saved run configuration](evidence/config.yaml). The screenshot captures a real terminal displaying the unedited final resume transcript.

The initial run encountered five Podman timeouts before verifier execution and was then interrupted, cancelling eight in-flight trials. Harbor resumed the unchanged job, retrying those 13 tasks and completing the 393 pending tasks. The final result includes exactly one successful result for each of the 2,417 original question IDs. [Initial terminal output](evidence/oracle-initial-terminal.log) and [the retried trial list](evidence/oracle-initial-exceptions.json) are retained; runtime provenance records the archive hashes and resume command.

To reproduce the original-side oracle check with the exact scoring revision:

```bash
git clone https://github.com/StonyBrookNLP/MuSiQue.git musique-upstream
cd musique-upstream
git checkout 922ac98f19a201998dbdae6d7f2887a5258dbdeb
uvx --from gdown gdown 1tGdADlNjWFaHLeZZGShh2IRcpO6Lv24h -O musique_v1.0.zip
unzip musique_v1.0.zip data/musique_ans_v1.0_dev.jsonl
python3 - <<'PY'
import json
from pathlib import Path
source = Path("data/musique_ans_v1.0_dev.jsonl")
with source.open(encoding="utf-8") as records, Path("oracle_predictions.jsonl").open("w", encoding="utf-8") as predictions:
    for line in records:
        if not line.strip():
            continue
        record = json.loads(line)
        prediction = {
            "id": record["id"],
            "predicted_answer": record["answer"],
            "predicted_support_idxs": [p["idx"] for p in record["paragraphs"] if p["is_supporting"]],
            "predicted_answerable": record["answerable"],
        }
        predictions.write(json.dumps(prediction, ensure_ascii=False) + "\n")
PY
python3 evaluate_v1.0.py oracle_predictions.jsonl data/musique_ans_v1.0_dev.jsonl
```

Reproduce the Harbor oracle check by generating the complete split and running `harbor run -c src/musique/run_musique.yaml` from the adapters repository root. Individual task scores retain full precision; round dataset means to three decimal places only when comparing with upstream CLI output. Oracle success does not establish model parity or replace the repeated runs required by Step 5.

## Notes & Caveats

- **Scope:** all 2,417 MuSiQue-Ans development examples are included. The 19,938 training examples are outside this evaluation split. The 2,459 test examples have hidden answers and support labels and cannot be verified locally with the oracle.
- **MuSiQue-Full is excluded:** its answerable and unanswerable variants share IDs and require grouped sufficiency metrics. Showing both variants in a single task would reveal the removed context; faithful support requires a separate aggregation design.
- **Prompt adaptation:** a file-based submission instruction wraps the original question and paragraphs. This changes only the interaction format. A future original-side agent must use the same wrapper for parity.
- **Context length:** the official dev records contain 17–20 paragraphs. The adapter preserves each record's complete context instead of assuming a fixed count or truncating it.
- **Data pin:** the official archive SHA-256 is `98f839bf2fd5319f5c688aed77901a6d5c30b3b9f9f691ab9a8ecafb045ee0cd`; the dev JSONL SHA-256 is `15fa63794d18a94ce12411aca6e2327e65b6e83b0b1490efab3f1962e48abf3b`. The archive download is approximately 272 MB.
- **Scoring pin:** evaluator and metric sources are copied unchanged from upstream commit `922ac98f19a201998dbdae6d7f2887a5258dbdeb`. Three separate metrics are preserved, with no combined reward or pass/fail threshold. Missing or malformed submissions receive zero rewards with a diagnostic.
- **Ground-truth handling:** gold records and oracle predictions are outside the container build context. They are supplied separately by Harbor at their respective execution stages.
- **Environment:** the Dockerfile uses `python:3.11.14-slim-bookworm` and copies only the public question/context file into `/workspace`. This replaces upstream's model-training environment with the minimal runtime needed for file-based inference and its standard-library evaluator.
- **Oracle runtime:** the recorded run used Harbor 0.23.0 at the guide's compatibility commit `99218a4611e3bd76e49ce3b5f977e8c8134ad3ac`, Podman 4.9.4-rhel, and podman-compose 1.6.0. On this shared host, the run used local VFS storage, ignored unavailable CPU/memory enforcement, disabled container networking and build-layer caching, and multiplied the 60-second verifier timeout by five. These are oracle-run settings; the generated tasks and Docker reference configuration remain unchanged. Exact image digests, commands, interrupted-attempt details, and hashes are in `evidence/runtime.json`.
- **Validation provenance:** the oracle used adapter source commit `621386e1aedec60a984da2371337c74fb6f51861` and contract commit `cfe10f29e1cd8dfeac70381f6547c006dcf8437b`. All 45,923 generated task files matched the frozen manifest after the run. The adapter regression suite passed 26 tests, and the built wheel includes all task assets.
- **Workflow status:** this is a WIP adapter. Parity, result uploads, dataset registration, and final review are outside the requested Steps 1–3 scope. No upstream scoring bug or oracle correction was needed.
- **Structural validation:** `python scripts/validate_adapter.py musique` currently reports the intentionally empty `parity_experiment.json` as an error and a related adapter-name cross-check warning. That checker expects final-submission parity evidence from Step 6; these findings remain pending rather than filling the file with unrun experiments.
- **Contamination:** upstream cautions that its source single-hop questions may overlap with training data from SQuAD, T-REx, Natural Questions, MLQA, and Zero Shot RE. Consult the upstream release's dev/test single-hop IDs before using those sources for training.

## Installation / Prerequisites

Use Python 3.11 or newer, uv, an external Harbor installation, and a working Docker engine with Compose support:

```bash
uv tool install harbor
cd src/musique
uv sync
```

Generation needs network access for the first official download unless `--data-path` points to a local file. Container creation needs the Python base image. The verifier uses only Python's standard library. No benchmark model weights, GPU, or API key is required for oracle verification. Each adapter is an independent uv project; do not create a repository-wide virtual environment.

## Troubleshooting

- **Google Drive download fails:** download the official archive using upstream's instructions, extract `data/musique_ans_v1.0_dev.jsonl`, and pass it with `--data-path`. Check the checksum above.
- **Checksum mismatch:** do not silently substitute another release or a reformatted mirror. Obtain the pinned official file again.
- **No tasks selected:** check that requested IDs are original MuSiQue IDs, and that `--limit` and the input file match the intended split.
- **Existing task directories:** use a fresh output directory or pass `--overwrite` for the selected tasks.
- **Container startup fails:** confirm the container runtime works before interpreting scores. Environment errors do not count as benchmark failures.
- **Zero rewards:** inspect the verifier diagnostic and `/workspace/prediction.json`. The submission must contain the matching ID, a string answer, integer support indices, and a Boolean answerability flag.
- **Different aggregate scores:** use the same examples and original aliases; retain per-task precision and round only after aggregation.

## Citation

```bibtex
@article{trivedi2021musique,
  title={{M}u{S}i{Q}ue: Multihop Questions via Single-hop Question Composition},
  author={Trivedi, Harsh and Balasubramanian, Niranjan and Khot, Tushar and Sabharwal, Ashish},
  journal={Transactions of the Association for Computational Linguistics},
  year={2022},
  publisher={MIT Press}
}
```

## Authors & Contributions

MuSiQue was created by Harsh Trivedi, Niranjan Balasubramanian, Tushar Khot, and Ashish Sabharwal. Their work is credited in every generated task and in the vendored evaluator's attribution.

Harbor adapter: [yk](mailto:yangkai@umich.edu). Submit adapter issues and contributions to [harbor-framework/adapters](https://github.com/harbor-framework/adapters).
