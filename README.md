# Anytime Safe PAC Efficient Reasoning

Official code for the ICML 2026 paper:

**Anytime Safe PAC Efficient Reasoning**  
Chengyao Yu, Hao Zeng, Youxin Zhu, Jianguo Huang, Huajun Zeng, and Bingyi Jing.

Paper: https://arxiv.org/abs/2601.22446

This repository contains the B-PAC reasoning simulator, local vLLM inference
scripts, benchmark evaluation utilities, and command-line experiment runners.

## Structure

```text
bpac/                 Python package for B-PAC, IPS+Hoeff, and O-Naive
examples/             Minimal runnable examples
scripts/              Data construction, LLM-as-judge, and experiment runners
zeroeval/             Local vLLM inference and benchmark evaluation utilities
```

Generated outputs, parsed result directories, datasets, caches, and API keys are
intentionally not committed.

## Installation

The full reproduction environment uses local vLLM inference.

```bash
git clone <repo-url>
cd B-PAC-Reasoning
python -m venv .venv
source .venv/bin/activate
pip install -e .
pip install -r requirements.txt
```

Install on a CUDA Linux machine for vLLM runs. If your cluster uses a fixed
CUDA/PyTorch stack, install the matching `torch` wheel first, then install the
remaining requirements.

## Quick Start

Run a toy B-PAC simulation:

```bash
python examples/minimal_simulation.py
```

Use the Python API:

```python
import pandas as pd
from bpac import BPACConfig, run_simulation

data = pd.DataFrame(
    {
        "uncertainty": [0.1, 0.8],
        "instant_correct": [1, 0],
        "expert_correct": [1, 1],
        "instant_token": [100, 120],
        "expert_token": [1200, 1400],
    }
)

logs, model = run_simulation(data, BPACConfig(epsilon=0.08, alpha=0.1, seed=0))
print(logs.tail())
```

## Paper Settings

The B-PAC defaults match Appendix C.3:

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

Figure 2 uses `--epsilon 0.05 --alpha 0.1`. The main MATH/MMLU-Pro table and
the online-method comparisons use `--epsilon 0.08 --alpha 0.1`.

Appendix C.2 reports the local decoding settings:

| Model variant | Temperature | Top-p | Top-k | Min-p |
| --- | ---: | ---: | ---: | ---: |
| Qwen3-4B-Instruct-2507 | `0.7` | `0.8` | `20` | `0` |
| Qwen3-4B-Thinking-2507 | `0.6` | `0.95` | `20` | `0` |

`zeroeval/zero_eval_local.sh` is configured for the instruct model defaults.
`zeroeval/zero_eval_local_thinking*.sh` is configured for the thinking model
defaults. These scripts start vLLM inside `zeroeval/src/unified_infer.py`; no
external API server is required for model generation.

## Data Format

B-PAC simulation consumes a CSV/JSON/JSONL table with:

| Column | Meaning |
| --- | --- |
| `uncertainty` | Routing uncertainty score in `[0, 1]`; larger means more uncertain. |
| `instant_correct` | `1` if the non-thinking model answer is correct; verifiable tasks. |
| `expert_correct` | `1` if the thinking model answer is correct; verifiable tasks. |
| `loss` | Optional bounded loss in `[0, 1]`. If present, the simulator uses this directly. |
| `instant_token` | Token cost of the non-thinking model answer. |
| `expert_token` | Token cost of the thinking model answer. |

For MATH, MMLU-Pro, and BBH, the paper keeps instances where the thinking model
is correct. For Magpie, it keeps instances where the thinking model has judge
score at least as high as the non-thinking model.

For Magpie, `scripts/build_simulation_table.py --task magpie` constructs the
paper loss:

```text
sqrt(max(0, score_expert - score_instant) / score_range)
```

The default `score_range` is `9`, matching judge scores from 1 to 10.

## Reproduction

### 1. Generate Responses With Local vLLM

Run thinking and non-thinking models with the ZeroEval scripts:

```bash
cd zeroeval

bash zero_eval_local_thinking.sh \
  -d math \
  -m /path/to/Qwen3-4B-Thinking-2507 \
  -p qwen3-think \
  -s 1 \
  -g 0

bash zero_eval_local.sh \
  -d math \
  -m /path/to/Qwen3-4B-Instruct-2507 \
  -p qwen3-ins \
  -s 1 \
  -g 0
```

Use `zero_eval_local_thinking_3.sh` for runs that start from index 3000.

The local dataset parquet paths are configured through environment variables in
`zeroeval/src/task_configs.py`, for example:

```bash
export ZEROEVAL_MATH_PATH=/path/to/math.parquet
export ZEROEVAL_MMLUPRO_PATH=/path/to/mmlupro_reconstructed.parquet
export ZEROEVAL_BBH_PATH=/path/to/bbh.parquet
export ZEROEVAL_MAGPIE_PATH=/path/to/magpie_reconstructed.parquet
```

### 2. Evaluate Answers

For verifiable benchmarks, use the scripts in `zeroeval/src/evaluation`.

For Magpie, run LLM-as-judge scoring after generation:

```bash
export OPENAI_API_KEY=...
export OPENAI_BASE_URL=https://your-compatible-api/v1  # optional
python scripts/llm_as_judge.py --json zeroeval/result_dirs/magpie/qwen3-ins.json
```

### 3. Build Simulation Tables

For MATH, MMLU-Pro, or BBH:

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

### 4. Run Stationary Experiments

```bash
python scripts/run_bpac_experiment.py \
  --input outputs/mmlupro_table.csv \
  --output-dir outputs/mmlupro_bpac \
  --method bpac \
  --num-runs 100 \
  --epsilon 0.08 \
  --alpha 0.1
```

Online baselines are available with `--method ips` and `--method naive`.

### 5. Run The Non-Stationary Experiment

Figure 2 uses a distribution-shift stream with 1,500 MMLU-Pro samples followed
by 3,000 BBH samples, with `epsilon=0.05` and `alpha=0.1`.

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
segments in order, preserving the non-stationary distribution shift.

## Baseline Response Generation

CoD and NoThinking responses can be generated with:

```bash
python scripts/generate_baseline_responses.py \
  --mode cod \
  --dataset math \
  --dataset-path /path/to/math.parquet \
  --model-path /path/to/Qwen3-4B-Thinking-2507 \
  --output outputs/baselines/math_cod.jsonl \
  --gpu-ids 0
```

Use `--mode no-thinking` for the NoThinking baseline.

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

## License

This repository is released under the MIT License. The `zeroeval/` directory
contains adapted ZeroEval utilities and keeps its original license file.
