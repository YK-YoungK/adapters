# MuSiQue → Harbor Adapter

## Overview

This adapter converts the labeled **train and dev splits** of both **MuSiQue-Ans
v1.0** and **MuSiQue-Full v1.0** into independent Harbor tasks. It preserves the
original questions, candidate paragraphs and grading code.
Each Full record has its own agent context; its answerable/unanswerable partner is
joined only when computing the official dataset metrics.

| Split | MuSiQue-Ans records | MuSiQue-Full records | Direct local adaptation |
|-------|--------------------:|---------------------:|-------------------------|
| Train | 19,938 | 39,876 | Yes: public gold labels |
| Dev | 2,417 | 4,834 | Yes: public gold labels |
| Test | 2,459 | 4,918 | No: gold labels withheld |

That is **67,065 adaptable records**: 59,814 train and 7,251 dev. The default split
remains `dev`; select training data with `--split train`. Evaluate the two splits
in separate dataset directories. Test exclusions are explained in Notes & Caveats.

From the repository root, enter the adapter once with `cd src/musique`. All commands
below run from that directory. Downloaded inputs, generated tasks, and results stay
in its `data/`, `datasets/`, and `jobs/` directories, which are excluded from version
control.

## What is MuSiQue?

[MuSiQue](https://github.com/StonyBrookNLP/MuSiQue) is a multi-hop reading
comprehension benchmark introduced by Harsh Trivedi, Niranjan Balasubramanian,
Tushar Khot and Ashish Sabharwal at TACL 2022. Questions require combining evidence
across two to four supporting paragraphs among supplied candidates.
MuSiQue-Ans contains answerable questions. MuSiQue-Full includes an answerable
record and an insufficient-evidence record for each question ID. The authors
release gold answers, aliases, support annotations and answerability labels for
the train and development splits, enabling deterministic reference solutions.

## Adapter Features

- Both variants on train or dev: `--split train|dev` (default: `dev`) and
  `--variant ans|full|both` (default: `both`). Public test is explicitly rejected.
- Official Google Drive archive with pinned SHA-256 checks, reusable download cache,
  and offline generation from `--data-dir`.
- Stable names under `musique/`; Full record names use a hash of public input rather
  than an answerability label. Duplicate names and group hash collisions fail early.
- Exact upstream question text, paragraph titles/text/order and indices. Gold labels,
  aliases and decompositions are excluded from the agent's instruction and image.
- A reference oracle writes the upstream gold prediction. No model or API key is needed.
- Vendored upstream answer and support metrics with no runtime dependencies beyond Python.
- A Harbor `uv-script` metric computes answer F1, answer EM, support F1 and Full's
  grouped answer/support sufficiency F1. Raw scores are averaged before upstream's
  three-decimal rounding. The Full answer/support denominator includes only its
  answerable records; both sufficiency predictions must be correct for group credit.
- `--task-ids` and `--limit` keep Full pairs intact. One attempt per Harbor job;
  repeat experiments as separate jobs.
- Split-specific task names, source checksums, and metadata. Generation, metric
  aggregation, and auditing reject mixed train/dev datasets rather than pooling scores.

No custom agent is implemented at this stage. Upstream uses trained AllenNLP models;
any compatible LLM agent and parity protocol require Step 4 coordination.

## Generated Task Structure

```text
<output-dir>/
└── <ans|full>-<train|dev>-<question-id>[-<public-input-hash>]/
    ├── task.toml
    ├── instruction.md             # question and all candidate paragraphs
    ├── environment/Dockerfile    # common Python environment; no gold data
    ├── solution/
    │   ├── solve.sh
    │   └── prediction.json       # oracle only
    └── tests/
        ├── test.sh
        ├── verify.py
        ├── gold.json             # verifier only
        └── upstream/             # unmodified evaluator, metrics, license
```

The agent writes the official prediction object to `/workspace/prediction.json`:
`id`, `predicted_answer`, `predicted_support_idxs`, `predicted_answerable`.
The verifier exports that prediction, raw score components and diagnostics under
`/logs/verifier`. It writes both `reward.txt` and `reward.json`; Harbor reads the
latter. The scalar text reward is a per-record diagnostic, not the Full group score.
Use the configured reducer for benchmark results.

## Run Evaluation / Harness

Registry execution will be available after a later publishing step. No registry
ID or release is claimed by this WIP adapter.

Run all commands from `src/musique`, after generating the selected split as shown
in Usage below. The reference configs select the oracle agent, Docker, and one
attempt per task. No model, API key, or GPU is needed.

```bash
harbor run -c run_musique.yaml --job-name musique-dev-oracle
harbor run -c run_musique_train.yaml --job-name musique-train-oracle
```


Use the reference configs to enable the required metric reducer. **Bare
`harbor run -p ...` uses Harbor's default mean, which is not valid MuSiQue
aggregation**, because raw rewards carry grouping metadata and Full scores require
pairing. Run train and dev as separate jobs.

For a small smoke test, select one upstream question ID (three tasks across both
variants), then point a copy of the dev config's dataset path at that directory:

```bash
uv run musique --cache-dir data --output-dir datasets/musique-smoke \
  --task-ids 2hop__460946_294723
```

For full oracle verification, generate tasks without `--limit` or `--task-ids`.
Successful verification requires 7,251 dev tasks or 59,814 train tasks when using
both variants, no trial errors, perfect applicable raw scores, and complete Full
pairs.

Audit each finished job against its generated tasks and replay the exported
predictions with the unmodified original evaluator:

```bash
uv run python -m musique.audit --tasks datasets/musique \
  --job jobs/musique-dev-oracle --output jobs/musique-dev-oracle-audit \
  --require-perfect
uv run python -m musique.audit --tasks datasets/musique-train \
  --job jobs/musique-train-oracle --output jobs/musique-train-oracle-audit \
  --require-perfect
```

This checks every applicable raw per-task score, exact task coverage, saved rewards,
and original evaluation results. It exports prediction/gold JSONL files and an
`audit.json` report with `status: passed` on success. Confirm `expected_tasks` and
`audited_trials` match the full split counts above; the audit uses the supplied task
directory to determine expected coverage. Results are preserved under `jobs/`.
Use a new job name for a fresh run. If adapter code changes, regenerate tasks before
starting the new run.

## Usage: Create Task Directories

```bash
uv sync --locked
uv run musique --split dev --cache-dir data --output-dir datasets/musique
uv run musique --split train --cache-dir data --output-dir datasets/musique-train
```

Each command downloads the official archive if necessary and caches only the
requested split/variants in `data/`. The default remains both variants of dev. For
predownloaded official files, the data directory must contain
`musique_<variant>_v1.0_<split>.jsonl` for each selected variant:

```bash
uv run musique --data-dir data \
  --split train --output-dir datasets/musique-train
```

Variant selection and reproducible debugging:

```bash
uv run musique --split train --variant ans --cache-dir data --limit 10 \
  --output-dir datasets/musique-ans-train
uv run musique --split train --variant full --cache-dir data --limit 10 \
  --output-dir datasets/musique-full-train
uv run musique --cache-dir data --task-ids 2hop__460946_294723 \
  --output-dir datasets/musique-selected --overwrite
```

`--limit` caps the total number of task directories after filtering. Selection is
sorted by variant (Ans first), then original question ID. Full pairs consume two
slots; an unmatched last slot is unused. `--variant full --limit 1` fails.
`--task-ids` accepts original IDs, generated directory names or `musique/<name>`;
selecting either Full record includes its partner. Unknown IDs fail before writing.
Existing selected tasks require `--overwrite`; unrelated output is retained.
Train and dev require separate output directories, including with `--overwrite`.
Use distinct directory basenames (such as `musique` and `musique-train`) because
Harbor groups local-dataset metrics by directory basename. Each dataset uses the
same official `ans_*` and `full_*` metric names, scoped to its selected split.

`--data-dir` accepts labeled official-format local files for debugging. Their actual
checksums are recorded in task metadata, so modified local inputs are distinguishable
from the pinned release. Downloaded data always undergoes the pinned checksum checks.

## Comparison with Original Benchmark (Parity)

Not run: Step 4 coordination and Step 5 model experiments are outside this request.
`parity_experiment.json` is an empty array, not an oracle result or a parity claim.
There are no dataset, parity-upload or adapter PR links yet.

| Agent | Model | Metric | Runs | Dataset Size | Original (mean ± SEM) | Harbor (mean ± SEM) |
|-------|-------|--------|------|--------------|-----------------------|---------------------|

The original evaluator is pinned at
[`922ac98f19a201998dbdae6d7f2887a5258dbdeb`](https://github.com/StonyBrookNLP/MuSiQue/tree/922ac98f19a201998dbdae6d7f2887a5258dbdeb).
Given exported predictions in the same order as gold data, reproduce upstream scores:

```bash
uv run python src/musique/task-template/tests/upstream/evaluate_v1.0.py \
  jobs/musique-dev-oracle-audit/ans_predictions.jsonl \
  jobs/musique-dev-oracle-audit/ans_gold.jsonl
uv run python src/musique/task-template/tests/upstream/evaluate_v1.0.py \
  jobs/musique-dev-oracle-audit/full_predictions.jsonl \
  jobs/musique-dev-oracle-audit/full_gold.jsonl
```

Both Full records must be present for every included ID. Future parity must use
identical per-record context, output instructions and agent/model settings on both
sides. A multi-record agent context would change the benchmark and is not supported.

## Notes & Caveats

- **Test is not directly adaptable as a locally scored Harbor dataset.** The public
  test records contain only `id`, `question`, and `paragraphs`. They omit `answer`,
  `answer_aliases`, `answerable`, `question_decomposition`, and paragraph-level
  `is_supporting`. The [upstream label-hiding code](https://github.com/StonyBrookNLP/MuSiQue/blob/922ac98f19a201998dbdae6d7f2887a5258dbdeb/raw_data_to_official_format.py#L143-L151)
  explicitly removes these fields. Without the gold labels, the official answer,
  support, and Full sufficiency metrics and reference oracles cannot be reproduced
  locally. Input-only conversion and prediction submission to the upstream test
  leaderboard are possible, but would require a separate external evaluation
  integration or author-provided gold labels. This adapter rejects `--split test`
  instead of inventing answers or replacing the benchmark's grader.
- Ans and Full answerable records sharing an ID have different candidate sets and
  sometimes different answer aliases. Each variant uses its own original records.
- Paragraph counts vary; the adapter does not pad, truncate or assume exactly 20.
- MuSiQue provides structured inputs rather than a terminal-agent prompt. The
  wrapper renders those inputs as JSON and specifies an output file. No question,
  paragraph or gold label is rewritten. Future original-side parity must apply
  this same wrapper.
- Full group scores shown during a job are provisional. `full_pending_pairs` and
  `null_rewards` must be zero at completion; final certification must also check
  exact generated-task coverage. Multiple attempts in one job are rejected as
  duplicate records.
- Malformed/missing agent predictions receive zero scores with diagnostics.
  Corrupt gold data is an infrastructure error. Numeric support indices follow
  upstream's integer conversion and set semantics.
- The official archive SHA-256 is
  `98f839bf2fd5319f5c688aed77901a6d5c30b3b9f9f691ab9a8ecafb045ee0cd`.
  Ans dev SHA-256: `15fa63794d18a94ce12411aca6e2327e65b6e83b0b1490efab3f1962e48abf3b`.
  Full dev SHA-256: `8cab31d56a3a1c4ef491b205a8dab3f1ac9c66e472098c6cf1de4e20294f7a4a`.
  Ans train SHA-256: `83a75b1e11e4e9bb8f8308e72ac40ca617ae4431b3a0d955b61cab259248490a`.
  Full train SHA-256: `b1cd998f7e0e2838d6fda024e4ad1eb0e7fc3edefdadb0bd9b5b10b0907f2034`.
- The guide contract used is adapters commit
  `cfe10f29e1cd8dfeac70381f6547c006dcf8437b` (contract v1).
- The repository's final-submission validator reports the deliberately empty parity
  array and its adapter-name cross-check. These Step 6 requirements remain pending;
  unrun parity experiments are not populated to satisfy the validator.
- Benchmark data and upstream code are distributed under CC BY 4.0. Attribution,
  license and the pinned source revision accompany the vendored evaluator.

## Installation / Prerequisites

- Python 3.11+ and `uv` for this independent adapter project.
- External Harbor CLI (`uv tool install harbor`) and working Docker.
- Network access for the first dataset/image download; oracle evaluation needs no
  model credentials. Scoring is standard-library-only.

```bash
uv sync --locked
```

Harbor versions used for actual experiments must be recorded with the oracle
results. Installed package metadata and `harbor --version` can differ; record both.

## Troubleshooting

- Google Drive rate limit/download failure: download the archive from the upstream
  repository's link, extract it, and pass the resulting `data` directory.
- Cache checksum mismatch: remove the corrupt cached JSONL and retry; do not update
  the pinned hash without investigating the upstream release.
- Unknown task ID or incomplete Full pair: use the original ID to select a complete
  pair, and provide both labeled records in local fixture data.
- `--split test`: public test gold labels are deliberately withheld; use `train` or
  `dev` for locally verified tasks. Test requires official external grading.
- Mixed split error: generate and evaluate train/dev in separate dataset directories;
  `--overwrite` does not permit changing an existing directory's split.
- Scores missing or unexpectedly halved: run through `run_musique.yaml`, which uses
  the pair-aware reducer rather than Harbor's default mean.
- Container startup errors: inspect the trial exception/logs. An environment failure
  is not a benchmark score and must be fixed before interpreting oracle results.

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

- Benchmark: Harsh Trivedi, Niranjan Balasubramanian, Tushar Khot and Ashish Sabharwal.
- Harbor adapter: yk (yangkai@umich.edu).
