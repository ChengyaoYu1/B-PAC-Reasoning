<div align="center">

# Anytime Safe PAC Efficient Reasoning

**Official implementation for the ICML 2026 paper**

Chengyao Yu, Hao Zeng, Youxin Zhu, Jianguo Huang, Huajun Zeng, Bingyi Jing

[![arXiv](https://img.shields.io/badge/arXiv-2601.22446-b31b1b.svg)](https://arxiv.org/abs/2601.22446)
[![Conference](https://img.shields.io/badge/ICML-2026-4b8bbe.svg)](https://icml.cc/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](pyproject.toml)
[![vLLM](https://img.shields.io/badge/Inference-vLLM-green.svg)](https://github.com/vllm-project/vllm)

</div>

## Overview

Large Reasoning Models are accurate but expensive. B-PAC reasoning dynamically
routes each query between a non-thinking model and a thinking model, while
controlling anytime performance loss under partial feedback.

This repository provides:

- local vLLM generation scripts for thinking and non-thinking model outputs;
- benchmark evaluation utilities for MATH, MMLU-Pro, BBH, and Magpie;
- B-PAC, IPS+Hoeff, and O-Naive simulation code;
- scripts for stationary and non-stationary experiment reproduction.

## News

- `2026-01-30`: Paper released on arXiv: [2601.22446](https://arxiv.org/abs/2601.22446).
- `2026`: Accepted by ICML 2026.

## Project Structure

```text
B-PAC-Reasoning/
|-- bpac/                         # B-PAC, IPS+Hoeff, O-Naive, simulation API
|-- scripts/
|   |-- build_simulation_table.py  # Convert evaluated outputs to B-PAC tables
|   |-- run_bpac_experiment.py     # Stationary experiments
|   |-- run_nonstationary_experiment.py
|   |-- llm_as_judge.py            # Magpie LLM-as-judge scoring
|   `-- generate_baseline_responses.py
|-- zeroeval/                      # Local vLLM inference and answer evaluation
|-- requirements.txt
`-- README.md
```

Generated outputs, parsed result directories, datasets, caches, and API keys are
intentionally excluded from version control.

## Environment

The paper uses local vLLM inference, not an API server, for model generation.
Install the environment on a CUDA Linux machine.

```bash
git clone <repo-url>
cd B-PAC-Reasoning
conda create -n bpac python=3.10 -y
conda activate bpac
pip install -e .
pip install -r requirements.txt
```

If your cluster has a fixed CUDA/PyTorch stack, install the matching `torch`
wheel first and then run `pip install -r requirements.txt`.

## Paper Settings

### B-PAC

The default arguments in `scripts/run_bpac_experiment.py` and
`scripts/run_nonstationary_experiment.py` match Appendix C.3.

| Parameter | Argument | Default |
| --- | --- | --- |
| Error tolerance | `--epsilon` | `0.08` |
| Failure probability | `--alpha` | `0.1` |
| Warm-up duration | `--warmup-steps` | `200` |
| Warm-up exploration | `--rho-warm` | `0.7` |
| Deployment exploration | `--rho-deploy` | `0.05` |
| Betting clipping constant | `--c-clip` | `0.9` |
| Threshold grid size | `--num-thresholds` | `1001` |

The threshold grid is `U={0,0.001,0.002,...,1}`.

### Decoding

The local vLLM scripts follow Appendix C.2.

| Model variant | Script | Temperature | Top-p | Top-k | Min-p |
| --- | --- | ---: | ---: | ---: | ---: |
| Qwen3-4B-Instruct-2507 | `zero_eval_local.sh` | `0.7` | `0.8` | `20` | `0` |
| Qwen3-4B-Thinking-2507 | `zero_eval_local_thinking*.sh` | `0.6` | `0.95` | `20` | `0` |

## Reproduction Pipeline

### 1. Configure Dataset Paths

For local parquet datasets, set the corresponding environment variables before
running `zeroeval`:

```bash
export ZEROEVAL_MATH_PATH=/path/to/math.parquet
export ZEROEVAL_MMLUPRO_PATH=/path/to/mmlupro_reconstructed.parquet
export ZEROEVAL_BBH_PATH=/path/to/bbh.parquet
export ZEROEVAL_MAGPIE_PATH=/path/to/magpie_reconstructed.parquet
```

### 2. Generate Model Outputs With Local vLLM

Run the thinking model:

```bash
cd zeroeval

bash zero_eval_local_thinking.sh \
  -d math \
  -m /path/to/Qwen3-4B-Thinking-2507 \
  -p qwen3-think \
  -s 1 \
  -g 0
```

Run the non-thinking model:

```bash
bash zero_eval_local.sh \
  -d math \
  -m /path/to/Qwen3-4B-Instruct-2507 \
  -p qwen3-ins \
  -s 1 \
  -g 0
```

Use `zero_eval_local_thinking_3.sh` for runs that start from index 3000.
The scripts invoke `zeroeval/src/unified_infer.py`, which loads vLLM directly.

### 3. Evaluate Generated Answers

For MATH, MMLU-Pro, and BBH, use the evaluation scripts under
`zeroeval/src/evaluation/` to produce parsed result files with correctness labels.

For Magpie, score both thinking and non-thinking outputs with LLM-as-judge:

```bash
export OPENAI_API_KEY=...
export OPENAI_BASE_URL=https://your-compatible-api/v1  # optional

python scripts/llm_as_judge.py \
  --json zeroeval/result_dirs/magpie/qwen3-think.json \
         zeroeval/result_dirs/magpie/qwen3-ins.json
```

### 4. Build B-PAC Simulation Tables

For verifiable tasks:

```bash
python scripts/build_simulation_table.py \
  --task verifiable \
  --dataset mmlupro \
  --expert-json zeroeval/result_dirs_parsed/mmlupro/qwen3-think.json \
  --instant-json zeroeval/result_dirs_parsed/mmlupro/qwen3-ins.json \
  --output outputs/mmlupro_table.csv
```

For Magpie:

```bash
python scripts/build_simulation_table.py \
  --task magpie \
  --dataset magpie \
  --expert-json zeroeval/result_dirs/magpie/qwen3-think.json \
  --instant-json zeroeval/result_dirs/magpie/qwen3-ins.json \
  --output outputs/magpie_table.csv
```

The required table columns are:

| Column | Description |
| --- | --- |
| `uncertainty` | Routing uncertainty score in `[0,1]`; larger means more uncertain. |
| `instant_correct` | Correctness of the non-thinking model for verifiable tasks. |
| `expert_correct` | Correctness of the thinking model for verifiable tasks. |
| `loss` | Optional bounded loss in `[0,1]`; used directly if present. |
| `instant_token` | Token cost of the non-thinking model output. |
| `expert_token` | Token cost of the thinking model output. |

For Magpie, the script constructs the paper loss:

```text
sqrt(max(0, score_expert - score_instant) / score_range)
```

The default `score_range` is `9`, matching scores from 1 to 10.

### 5. Run Stationary Experiments

```bash
python scripts/run_bpac_experiment.py \
  --input outputs/mmlupro_table.csv \
  --output-dir outputs/mmlupro_bpac \
  --method bpac \
  --num-runs 100 \
  --epsilon 0.08 \
  --alpha 0.1
```

Online baselines:

```bash
python scripts/run_bpac_experiment.py --input outputs/mmlupro_table.csv --method ips
python scripts/run_bpac_experiment.py --input outputs/mmlupro_table.csv --method naive
```

### 6. Run The Non-Stationary Experiment

Figure 2 uses a distribution-shift stream with 1,500 MMLU-Pro samples followed
by 3,000 BBH samples, with `epsilon=0.05` and `alpha=0.1`.

Build the MMLU-Pro and BBH tables first, then run:

```bash
python scripts/run_nonstationary_experiment.py \
  --source-a outputs/mmlupro_table.csv \
  --source-b outputs/bbh_table.csv \
  --name-a mmlupro \
  --name-b bbh \
  --a-size 1500 \
  --b-size 3000 \
  --epsilon 0.05 \
  --alpha 0.1 \
  --num-runs 100 \
  --methods bpac \
  --output-dir outputs/nonstationary_mmlupro_bbh
```

The script samples within each segment for every seed and concatenates the two
segments in order, preserving the distribution shift.

## Baselines

Generate CoD responses:

```bash
python scripts/generate_baseline_responses.py \
  --mode cod \
  --dataset math \
  --dataset-path /path/to/math.parquet \
  --model-path /path/to/Qwen3-4B-Thinking-2507 \
  --output outputs/baselines/math_cod.jsonl \
  --gpu-ids 0
```

Generate NoThinking responses by replacing `--mode cod` with
`--mode no-thinking`.

## Citation

```bibtex
@misc{yu2026anytime,
  title={Anytime Safe PAC Efficient Reasoning},
  author={Chengyao Yu and Hao Zeng and Youxin Zhu and Jianguo Huang and Huajun Zeng and Bingyi Jing},
  year={2026},
  eprint={2601.22446},
  archivePrefix={arXiv},
  primaryClass={cs.AI}
}
```

Proceedings metadata can replace this entry after the official ICML publication
entry is available.

## Acknowledgements

This repository includes adapted components from ZeroEval. We also rely on vLLM,
Hugging Face Transformers, and Qwen3 for local generation.

## License

This repository is released under the MIT License. The `zeroeval/` directory
contains adapted ZeroEval utilities and keeps its original license file.
