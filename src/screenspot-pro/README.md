# ScreenSpot-Pro → Harbor Adapter

## Overview

This adapter converts the official [ScreenSpot-Pro dataset](https://huggingface.co/datasets/likaixin/ScreenSpot-Pro) into one Harbor task per screenshot and grounding instruction. The pinned dataset contains **1,581 tasks** across 26 annotation files: 23 professional applications and three operating-system interfaces. English is the default; Chinese instructions are available with `--language cn`, generating the same 1,581 examples as a separate language configuration. No examples are excluded.

The adapter preserves the original screenshots, selected instruction text, bounding boxes, and point-in-box scoring. It supplies a Linux container, a file-based response interface, and annotation-derived oracle answers. Implementation is prepared for step 3 of the [adapter guide](../../docs/adapters.mdx); full Docker oracle verification remains pending. No parity, registry publication, or PR is claimed.

## What is ScreenSpot-Pro?

[ScreenSpot-Pro](https://gui-agent.github.io/grounding-leaderboard/) evaluates whether a vision model can locate a UI element described by an instruction in a high-resolution screenshot. It covers development, creative tools, CAD, scientific applications, office software, and operating-system interfaces. It is a static screenshot benchmark, so executing the applications depicted in the screenshots is unnecessary. See the [paper](https://arxiv.org/abs/2504.07981) and [original evaluation repository](https://github.com/likaixin2000/ScreenSpot-Pro-GUI-Grounding).

Each prediction resolves to a normalized click point. A task earns reward `1` when the point lies inside the annotated bounding box, including its boundary, and `0` otherwise. The mean task reward is the upstream overall `action_acc` (micro-average); multiply by 100 for percentage accuracy. The dataset contains 977 text targets and 604 icon targets.

## Adapter Features

- Downloads from the official Hugging Face repository at a pinned revision, with caching and an offline local-directory option.
- Generates stable language-prefixed task IDs from original annotation IDs, such as `en-android-studio-macos-0`.
- Preserves the positive-grounding user prompt and response parsing used by upstream `models/gpt4x.py`; the agent writes its raw response to `/workspace/prediction.txt`.
- Copies the screenshot at its original resolution into the container. Ground-truth annotations and oracle answers remain outside the environment build context.
- Uses the upstream inclusive, normalized point-in-box criterion, with no IoU threshold, padding, rounding of target bounds, or coordinate clamping.
- Creates a deterministic oracle from the center of each reference bounding box. No model calls or GPU are needed for generation or oracle verification.
- Includes an oracle-default Docker run configuration. Custom model agents and parity work belong to the later workflow steps.

## Generated Task Structure

```text
screenspot-pro/
└── en-android-studio-macos-0/
    ├── task.toml
    ├── instruction.md
    ├── environment/
    │   ├── Dockerfile
    │   └── screenshot.png
    ├── solution/
    │   ├── solve.sh
    │   └── answer.txt
    └── tests/
        ├── test.sh
        ├── verify.py
        ├── expected.json       # Private reference annotation
        └── UPSTREAM_LICENSE
```

Every `task.toml` uses schema version `"1.0"`, a namespaced task name, and attribution to the original benchmark authors. The agent receives the screenshot at `/workspace/screenshot.png`. The oracle copies its prepared response into `/workspace/prediction.txt`; the verifier writes `/logs/verifier/reward.txt`.

## Run Evaluation / Harness

Registry evaluation is unavailable until the dataset is published in the later registration step. After publication, the expected namespace is `screenspot-pro/screenspot-pro`; do not use registry commands to validate this unpublished implementation.

Generate tasks using the next section, then run the following **from the adapters repository root on a Docker-capable machine**:

```bash
# Default agent is oracle.
harbor run -c src/screenspot-pro/run_screenspot-pro.yaml

# Equivalent full-dataset invocation.
harbor run -p datasets/screenspot-pro -a oracle

# Single-task smoke check before the full run.
harbor trial start -p datasets/screenspot-pro/en-android-studio-macos-0 -a oracle
```

The reference configuration writes results under `jobs/`. Confirm 100% oracle reward and inspect any infrastructure failures before proceeding to step 4. Docker builds and the full oracle run have not been executed here; they are left for manual verification. Record the Harbor version/revision used with those results. The repository guide's compatibility baseline is Harbor commit `99218a4611e3bd76e49ce3b5f977e8c8134ad3ac`.

## Usage: Create Task Directories

```bash
cd src/screenspot-pro
uv sync --locked
uv run screenspot-pro --output-dir ../../datasets/screenspot-pro

# Small generation smoke check.
uv run screenspot-pro --output-dir /tmp/screenspot-pro-smoke --limit 3

# Select original IDs or generated directory IDs.
uv run screenspot-pro --output-dir /tmp/screenspot-pro-selected \
  --task-ids android_studio_macos_0 en-inventor-windows-60

# Use an already downloaded official dataset; no network download is needed.
uv run screenspot-pro --dataset-dir /path/to/ScreenSpot-Pro \
  --output-dir ../../datasets/screenspot-pro

# Chinese configuration; use a separate output directory for separate reporting.
uv run screenspot-pro --language cn \
  --output-dir ../../datasets/screenspot-pro-cn
```

The local dataset root must contain `annotations/*.json` and `images/`, retaining upstream relative image paths. It is the raw official dataset layout, not a third-party parquet conversion.

| Flag | Purpose |
|------|---------|
| `--output-dir` | Destination for generated task directories. |
| `--limit` | Generate the first N tasks in deterministic order after task selection. |
| `--overwrite` | Replace existing selected task directories. |
| `--task-ids` | Select original annotation IDs or generated directory IDs. |
| `--language {en,cn}` | Select the upstream English or Chinese instruction; default `en`. |
| `--dataset-dir` | Use a local official dataset root instead of downloading. |
| `--cache-dir` | Set the Hugging Face download cache location. |
| `--revision` | Override the pinned Hugging Face dataset revision. |

## Comparison with Original Benchmark (Parity)

Parity experiments have **not** been run. `parity_experiment.json` is intentionally absent until step 6 records actual experiments. There is no original-benchmark fork, adapter PR, dataset PR, or parity-results PR yet.

| Agent | Model | Metric | Runs | Dataset Size | Original (mean ± SEM) | Harbor (mean ± SEM) |
|-------|-------|--------|------|--------------|-----------------------|---------------------|
| Pending | Pending | action_acc | 0 | 1,581 per language | Not measured | Not measured |

The implementation targets upstream repository commit [`dbe00114bc53a32c61c1a267786da85967710da8`](https://github.com/likaixin2000/ScreenSpot-Pro-GUI-Grounding/tree/dbe00114bc53a32c61c1a267786da85967710da8), specifically `models/gpt4x.py` for the positive prompt and parsing, and `eval_screenspot_pro.py::eval_sample_positive_gt` for scoring. Dataset revision: [`210e78d3844251110bff86c95835ebd37a6930fa`](https://huggingface.co/datasets/likaixin/ScreenSpot-Pro/tree/210e78d3844251110bff86c95835ebd37a6930fa).

For reference, the original English evaluation entry point, run inside a checkout at that commit with its model dependencies installed, is:

```bash
python eval_screenspot_pro.py \
  --model_type gpt4o \
  --screenspot_imgs /path/to/ScreenSpot-Pro/images \
  --screenspot_test /path/to/ScreenSpot-Pro/annotations \
  --task all --language en --gt_type positive --inst_style instruction \
  --log_path ./results/screenspot-pro-en.json
```

The upstream GPT wrapper requires `OPENAI_API_KEY`. This command documents the original harness; it is **not yet a matched parity recipe** for a Harbor CLI agent. Step 4 must agree the agent/model and implement matching prompts, image handling, parser, tools, and generation settings on both sides. After that work, use `harbor run -p datasets/screenspot-pro -a <agent> -m <model>` for the Harbor side and document the corresponding upstream implementation. Follow the guide's paired sanity-check and full-run order, with at least two runs per side and mean ± sample SEM calculated from raw run scores.

## Notes & Caveats

- The official dataset is approximately 3.38 GB. Generated tasks retain original images, so allow additional disk space for task copies and Docker images. `--limit` is useful for checking setup before generating the full dataset.
- The 1,581-task count comes from the pinned annotation JSON files. The dataset's `eval.yaml` description says 1,585; that description is inconsistent with the actual data. English and Chinese are alternative language configurations of the same examples, not independent benchmark splits.
- The selected upstream instruction is inserted verbatim. The GPT positive-grounding user prompt is preserved, with additional screenshot/output-file directions required by Harbor. Upstream's model API provides the image directly and uses a system instruction prohibiting external tools; Harbor CLI agents receive a screenshot file and use their own tools/system instructions. These interaction differences must be matched explicitly for later parity. This adapter does not claim equivalence to published model scores before that work.
- The upstream parser expects compact numeric lists such as `[[0.25,0.50,0.30,0.55]]` or `[[0.275,0.525]]`. Spaces inside the brackets, signed numbers, and scientific notation do not match its regular expressions. A parsed point takes precedence over a parsed bounding box; otherwise the predicted box center is used. Malformed or missing responses receive zero reward.
- `inventor_windows_60` has the upstream bounding box `[135,-1,151,19]`. It is retained unchanged, including the negative top edge; the oracle center is a valid point in the image. Clamping would change the benchmark's grading rule.
- Source-platform metadata describes the screenshot, not the execution container. All tasks use Linux containers; Windows/macOS applications do not need to be installed. Annotation values are retained even where upstream prose or tables differ.
- Only positive grounding with the upstream `instruction` style is adapted. No synthetic negative examples or alternative instruction variants are generated. No official positive examples are excluded.
- Dataset downloads use an immutable revision by default. Supplying `--revision` or `--dataset-dir` changes the input provenance; record the chosen revision/source when reporting results. Remote revisions resolve to a commit recorded in task metadata. Local inputs are marked `local-unverified` because the adapter cannot attest their origin.
- Ground-truth files belong only in `tests/` and `solution/`; the Docker context contains only the screenshot and runtime setup. Do not copy the complete dataset annotations into the agent environment.
- The upstream code and dataset are MIT licensed. Adapted prompt/parser code retains upstream attribution; see the upstream repository's [license](https://github.com/likaixin2000/ScreenSpot-Pro-GUI-Grounding/blob/dbe00114bc53a32c61c1a267786da85967710da8/LICENSE).

## Installation / Prerequisites

Generation requires Python 3.11+ and `uv`. Each adapter is an independent project:

```bash
cd src/screenspot-pro
uv sync --locked
uv run screenspot-pro --help
```

Harbor is an external dependency: install it with `uv tool install harbor`. Docker must be installed and running on the machine performing step 3 verification. Dataset generation and local unit checks do not require Docker. The official dataset is public; model API credentials are unnecessary for generation and oracle verification. Later non-oracle evaluation requires an agent capable of viewing the screenshot and any credentials required by that agent.

Local checks (no Docker or model calls):

```bash
# From src/screenspot-pro:
uv run python -m unittest discover -s tests -v

# From the repository root:
python scripts/validate_adapter.py screenspot-pro
```

The repository validator covers the full submission workflow. At this WIP stage it
reports a missing `parity_experiment.json`, which belongs to step 6. Do not invent
parity results to satisfy that later-stage check.

## Troubleshooting

- **Docker unavailable:** generate tasks locally, then regenerate or transfer them to a Docker-capable machine and run the oracle commands above. Local structural checks do not establish Docker runtime success.
- **Download failure:** retry with a writable `--cache-dir`, or download the pinned official dataset on a connected machine and supply `--dataset-dir`.
- **Missing images:** ensure `--dataset-dir` points to the root containing both `annotations/` and `images/`, with original subdirectories intact.
- **Existing tasks:** use a clean output directory or `--overwrite` when regenerating after code/data changes. Keep languages and subsets in separate directories to avoid accidentally evaluating a mixed set.
- **Zero reward:** inspect `/logs/verifier/` in the Harbor trial output and the agent's `/workspace/prediction.txt`. Check exact upstream numeric formatting and normalized coordinates, including whether an image-viewing tool resized the screenshot.
- **Task selection fails:** use the original ID from annotation JSON, or the generated language-prefixed directory ID. Verify that the selected language matches the prefix.

## Citation

```bibtex
@article{li2025screenspotpro,
  title={ScreenSpot-Pro: GUI Grounding for Professional High-Resolution Computer Use},
  author={Li, Kaixin and Meng, Ziyang and Lin, Hongzhan and Luo, Ziyang and
          Tian, Yuchen and Ma, Jing and Huang, Zhiyong and Chua, Tat-Seng},
  year={2025},
  eprint={2504.07981},
  archivePrefix={arXiv},
  url={https://arxiv.org/abs/2504.07981}
}
```

## Authors & Contributions

The original benchmark is by Kaixin Li, Ziyang Meng, Hongzhan Lin, Ziyang Luo, Yuchen Tian, Jing Ma, Zhiyong Huang, and Tat-Seng Chua. This Harbor adapter is maintained by [Kai Yang](mailto:yangkai@umich.edu). Report adapter issues and contributions in [harbor-framework/adapters](https://github.com/harbor-framework/adapters).
