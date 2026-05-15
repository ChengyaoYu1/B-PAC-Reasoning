# Anytime Safe PAC Efficient Reasoning

Official code repository for the ICML 2026 paper:

**Anytime Safe PAC Efficient Reasoning**  
Chengyao Yu, Hao Zeng, Youxin Zhu, Jianguo Huang, Huajun Zeng, and Bingyi Jing

## Overview

Large Reasoning Models (LRMs) achieve strong performance on complex reasoning tasks, but they often incur substantial computational cost and latency. This paper studies how to safely improve reasoning efficiency by adaptively routing queries between a thinking model and a non-thinking model.

We propose **Betting PAC (B-PAC) reasoning**, an anytime-valid and model-agnostic framework for efficient online reasoning under partial feedback. B-PAC dynamically updates the routing threshold through a betting-based procedure and provides rigorous performance-loss control relative to the thinking model.

## Status

The repository is currently being organized.

The following materials will be released soon:

- source code for B-PAC reasoning;
- scripts for reproducing the main experiments;
- benchmark preprocessing instructions;
- configuration files for different datasets and settings;
- instructions for reproducing the figures and tables in the paper.

## Experiments

The paper evaluates B-PAC reasoning on reasoning benchmarks including:

- MATH;
- MMLU-Pro;
- BIG-Bench Hard (BBH);
- Magpie.

The main evaluation metrics include empirical risk, expert call percentage, and token percentage.

## Citation

The BibTeX entry will be added after the official proceedings information becomes available.

## Contact

For questions, please open an issue in this repository.
