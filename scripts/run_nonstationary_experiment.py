#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
from tqdm import trange

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from bpac import run_simulation
from run_bpac_experiment import make_model, summarize_run


def read_table(path: Path) -> pd.DataFrame:
    if path.suffix == ".jsonl":
        return pd.read_json(path, lines=True)
    if path.suffix == ".json":
        return pd.read_json(path)
    if path.suffix == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"Unsupported input format: {path}")


def take_segment(
    data: pd.DataFrame,
    *,
    size: int,
    seed: int,
    shuffle: bool,
    name: str,
) -> pd.DataFrame:
    if size > len(data):
        raise ValueError(f"Requested {size} samples from {name}, but only {len(data)} rows are available.")
    segment = data.sample(n=size, random_state=seed).reset_index(drop=True)
    if not shuffle:
        segment = data.iloc[:size].reset_index(drop=True)
    segment["stream_segment"] = name
    return segment


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the non-stationary B-PAC experiment. By default this follows "
            "the paper's MMLU-Pro -> BBH stream: 1500 MMLU-Pro samples followed "
            "by 3000 BBH samples, epsilon=0.05, alpha=0.1."
        )
    )
    parser.add_argument("--source-a", type=Path, required=True, help="First distribution table, e.g. MMLU-Pro.")
    parser.add_argument("--source-b", type=Path, required=True, help="Second distribution table, e.g. BBH.")
    parser.add_argument("--name-a", default="mmlupro")
    parser.add_argument("--name-b", default="bbh")
    parser.add_argument("--a-size", type=int, default=1500)
    parser.add_argument("--b-size", type=int, default=3000)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/nonstationary"))
    parser.add_argument("--methods", nargs="+", choices=["bpac", "ips", "naive"], default=["bpac"])
    parser.add_argument("--num-runs", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--shuffle-within-segment", action=argparse.BooleanOptionalAction, default=True)

    parser.add_argument("--epsilon", type=float, default=0.05)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--rho-warm", type=float, default=0.7)
    parser.add_argument("--rho-deploy", type=float, default=0.05)
    parser.add_argument("--warmup-steps", type=int, default=200)
    parser.add_argument("--c-clip", type=float, default=0.9)
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--num-thresholds", type=int, default=1001)
    parser.add_argument("--save-logs", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--save-first-stream", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_a = read_table(args.source_a)
    data_b = read_table(args.source_b)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    all_summaries = []
    for run in trange(args.num_runs, desc="non-stationary streams"):
        run_seed = args.seed + run
        segment_a = take_segment(
            data_a,
            size=args.a_size,
            seed=run_seed,
            shuffle=args.shuffle_within_segment,
            name=args.name_a,
        )
        segment_b = take_segment(
            data_b,
            size=args.b_size,
            seed=run_seed + 10_000,
            shuffle=args.shuffle_within_segment,
            name=args.name_b,
        )
        stream = pd.concat([segment_a, segment_b], ignore_index=True)
        if run == 0 and args.save_first_stream:
            stream.to_csv(args.output_dir / "stream_run000.csv", index=False)

        for method in args.methods:
            cfg, model = make_model(method, args, run_seed)
            logs, _ = run_simulation(stream, cfg, model=model)
            summary = summarize_run(logs, run, method)
            summary["source_a"] = args.name_a
            summary["source_b"] = args.name_b
            summary["a_size"] = args.a_size
            summary["b_size"] = args.b_size
            all_summaries.append(summary)
            if args.save_logs:
                logs.to_csv(args.output_dir / f"{method}_run{run:03d}.csv", index=False)

    summary_df = pd.DataFrame(all_summaries)
    summary_path = args.output_dir / "nonstationary_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    aggregates = []
    for method, group in summary_df.groupby("method"):
        aggregates.append(
            {
                "method": method,
                "num_runs": int(len(group)),
                "epsilon": args.epsilon,
                "alpha": args.alpha,
                "source_a": args.name_a,
                "source_b": args.name_b,
                "a_size": args.a_size,
                "b_size": args.b_size,
                "mean_final_risk": float(group["final_risk"].mean()),
                "std_final_risk": float(group["final_risk"].std(ddof=0)),
                "mean_final_expert_call_ratio": float(group["final_expert_call_ratio"].mean()),
                "mean_final_token_ratio": float(group["final_token_ratio"].mean()),
            }
        )
    aggregate_path = args.output_dir / "nonstationary_aggregate.json"
    aggregate_path.write_text(json.dumps(aggregates, indent=2), encoding="utf-8")

    print(summary_df.groupby("method")[["final_risk", "final_expert_call_ratio", "final_token_ratio"]].agg(["mean", "std"]))
    print(f"Wrote {summary_path}")
    print(f"Wrote {aggregate_path}")


if __name__ == "__main__":
    main()
