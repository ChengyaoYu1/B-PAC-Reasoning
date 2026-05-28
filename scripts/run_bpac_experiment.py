#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import trange

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bpac import (
    BPAC,
    BPACConfig,
    IPSHoeffding,
    IPSHoeffdingConfig,
    OnlineNaive,
    OnlineNaiveConfig,
    run_simulation,
)


def read_table(path: Path) -> pd.DataFrame:
    if path.suffix == ".jsonl":
        return pd.read_json(path, lines=True)
    if path.suffix == ".json":
        return pd.read_json(path)
    if path.suffix == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"Unsupported input format: {path}")


def make_model(method: str, args: argparse.Namespace, seed: int):
    if method == "bpac":
        cfg = BPACConfig(
            alpha=args.alpha,
            epsilon=args.epsilon,
            rho_warm=args.rho_warm,
            rho_deploy=args.rho_deploy,
            warmup_steps=args.warmup_steps,
            beta=args.beta,
            c_clip=args.c_clip,
            num_thresholds=args.num_thresholds,
            seed=seed,
        )
        return cfg, BPAC(cfg)
    if method == "ips":
        cfg = IPSHoeffdingConfig(
            alpha=args.alpha,
            epsilon=args.epsilon,
            rho=args.rho_deploy,
            num_thresholds=args.num_thresholds,
            seed=seed,
        )
        return cfg, IPSHoeffding(cfg)
    if method == "naive":
        cfg = OnlineNaiveConfig(
            alpha=args.alpha,
            epsilon=args.epsilon,
            rho=args.rho_deploy,
            num_thresholds=args.num_thresholds,
            seed=seed,
        )
        return cfg, OnlineNaive(cfg)
    raise ValueError(method)


def summarize_run(logs: pd.DataFrame, run: int, method: str) -> dict:
    if logs.empty:
        return {"run": run, "method": method, "steps": 0}
    last = logs.iloc[-1]
    return {
        "run": run,
        "method": method,
        "steps": int(len(logs)),
        "final_risk": float(last["avg_risk"]),
        "final_expert_call_ratio": float(last["expert_call_ratio"]),
        "final_token_ratio": float(last["token_ratio"]),
        "final_threshold": float(last["threshold"]),
        "final_wealth": float(last["wealth"]),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run B-PAC simulations with paper-aligned defaults.")
    parser.add_argument("--input", type=Path, required=True, help="CSV/JSON/JSONL simulation table.")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--method", choices=["bpac", "ips", "naive"], default="bpac")
    parser.add_argument("--num-runs", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--shuffle", action=argparse.BooleanOptionalAction, default=True)

    parser.add_argument("--epsilon", type=float, default=0.08)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--rho-warm", type=float, default=0.7)
    parser.add_argument("--rho-deploy", type=float, default=0.05)
    parser.add_argument("--warmup-steps", type=int, default=200)
    parser.add_argument("--c-clip", type=float, default=0.9)
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--num-thresholds", type=int, default=1001)
    parser.add_argument("--save-logs", action=argparse.BooleanOptionalAction, default=False)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data = read_table(args.input)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    summaries = []
    for run in trange(args.num_runs, desc=f"{args.method} simulations"):
        run_seed = args.seed + run
        run_data = data.sample(frac=1.0, random_state=run_seed).reset_index(drop=True) if args.shuffle else data
        cfg, model = make_model(args.method, args, run_seed)
        logs, _ = run_simulation(run_data, cfg, model=model)
        summaries.append(summarize_run(logs, run, args.method))
        if args.save_logs:
            logs.to_csv(args.output_dir / f"{args.method}_run{run:03d}.csv", index=False)

    summary_df = pd.DataFrame(summaries)
    summary_path = args.output_dir / f"{args.method}_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    aggregate = {
        "method": args.method,
        "num_runs": args.num_runs,
        "epsilon": args.epsilon,
        "alpha": args.alpha,
        "rho_warm": args.rho_warm,
        "rho_deploy": args.rho_deploy,
        "warmup_steps": args.warmup_steps,
        "mean_final_risk": float(summary_df["final_risk"].mean()),
        "std_final_risk": float(summary_df["final_risk"].std(ddof=0)),
        "mean_final_expert_call_ratio": float(summary_df["final_expert_call_ratio"].mean()),
        "mean_final_token_ratio": float(summary_df["final_token_ratio"].mean()),
    }
    aggregate_path = args.output_dir / f"{args.method}_aggregate.json"
    aggregate_path.write_text(json.dumps(aggregate, indent=2), encoding="utf-8")

    print(summary_df.describe())
    print(f"Wrote {summary_path}")
    print(f"Wrote {aggregate_path}")


if __name__ == "__main__":
    main()
