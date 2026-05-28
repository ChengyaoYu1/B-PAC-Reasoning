#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import pandas as pd


def read_table(path: Path) -> pd.DataFrame:
    if path.suffix == ".jsonl":
        return pd.read_json(path, lines=True)
    if path.suffix == ".json":
        return pd.read_json(path)
    if path.suffix == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"Unsupported input format: {path}")


def write_table(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".jsonl":
        df.to_json(path, orient="records", lines=True, force_ascii=False)
    elif path.suffix == ".json":
        df.to_json(path, orient="records", indent=2, force_ascii=False)
    elif path.suffix == ".csv":
        df.to_csv(path, index=False)
    else:
        raise ValueError(f"Unsupported output format: {path}")


def read_many(paths: list[Path]) -> pd.DataFrame:
    frames = [read_table(path) for path in paths]
    return pd.concat(frames, ignore_index=True)


def as_bool_correct(value: Any) -> int:
    if isinstance(value, str) and value.lower() in {"no answer extracted", "none", "nan"}:
        return 0
    return int(bool(value))


def token_count(row: pd.Series, preferred: str | None) -> float:
    candidates = [preferred] if preferred else []
    candidates += ["gen_token_count", "token_count", "tokens"]
    for col in candidates:
        if col and col in row and pd.notna(row[col]):
            return float(row[col])
    raise KeyError("Cannot find token count column; pass --expert-token-column/--instant-token-column.")


def side_view(row: pd.Series, side: str) -> pd.Series:
    suffix = f"_{side}"
    values = {}
    for key, value in row.items():
        if key.endswith(suffix):
            values[key.removesuffix(suffix)] = value
        elif not key.endswith("_expert") and not key.endswith("_instant"):
            values[key] = value
    return pd.Series(values)


def uncertainty(row: pd.Series, uncertainty_column: str | None) -> float:
    if uncertainty_column:
        return float(row[uncertainty_column])
    if "token_probs" not in row or row["token_probs"] is None:
        raise KeyError("Cannot infer uncertainty; pass --uncertainty-column or provide token_probs.")
    probs = row["token_probs"]
    if isinstance(probs, str):
        probs = json.loads(probs)
    if isinstance(probs, (int, float)):
        avg_prob = float(probs)
    else:
        vals = [float(x) for x in probs if x is not None]
        avg_prob = sum(vals) / len(vals)
    return 1.0 - avg_prob


def choose_join_key(expert: pd.DataFrame, instant: pd.DataFrame, requested: str | None) -> str:
    if requested:
        return requested
    for key in ["session_id", "uuid", "question_id", "id"]:
        if key in expert.columns and key in instant.columns:
            return key
    raise ValueError("Cannot infer join key; pass --join-key or use --join-by-index.")


def build_verifiable(args: argparse.Namespace) -> pd.DataFrame:
    expert = read_many(args.expert_json)
    instant = read_table(args.instant_json)
    join_key = choose_join_key(expert, instant, args.join_key)
    expert = expert.drop_duplicates(subset=[join_key], keep="first")
    instant = instant.drop_duplicates(subset=[join_key], keep="first")

    if args.filter_expert_correct:
        expert = expert[expert[args.expert_correct_column].apply(as_bool_correct) == 1]

    merged = expert.merge(
        instant,
        on=join_key,
        suffixes=("_expert", "_instant"),
        how="inner",
    )
    records: list[dict[str, Any]] = []
    for _, row in merged.iterrows():
        expert_correct = as_bool_correct(row[f"{args.expert_correct_column}_expert"])
        instant_correct = as_bool_correct(row[f"{args.instant_correct_column}_instant"])
        records.append(
            {
                "id": row[join_key],
                "dataset": args.dataset,
                "uncertainty": uncertainty(side_view(row, "instant"), args.uncertainty_column),
                "instant_correct": instant_correct,
                "expert_correct": expert_correct,
                "loss": float(expert_correct * (1 - instant_correct)),
                "instant_token": token_count(
                    side_view(row, "instant"),
                    args.instant_token_column,
                ),
                "expert_token": token_count(
                    side_view(row, "expert"),
                    args.expert_token_column,
                ),
            }
        )
    return pd.DataFrame(records)


def build_magpie(args: argparse.Namespace) -> pd.DataFrame:
    expert = read_table(args.expert_json[0])
    instant = read_table(args.instant_json)
    join_key = choose_join_key(expert, instant, args.join_key)

    merged = expert.merge(
        instant,
        on=join_key,
        suffixes=("_expert", "_instant"),
        how="inner",
    )
    expert_score_col = f"{args.judge_score_column}_expert"
    instant_score_col = f"{args.judge_score_column}_instant"
    merged = merged[merged[expert_score_col].notna() & merged[instant_score_col].notna()]
    if args.filter_expert_preferred:
        merged = merged[merged[expert_score_col] >= merged[instant_score_col]]

    records: list[dict[str, Any]] = []
    for _, row in merged.iterrows():
        expert_score = float(row[expert_score_col])
        instant_score = float(row[instant_score_col])
        loss = math.sqrt(max(0.0, expert_score - instant_score) / args.score_range)
        records.append(
            {
                "id": row[join_key],
                "dataset": args.dataset,
                "uncertainty": uncertainty(side_view(row, "instant"), args.uncertainty_column),
                "loss": loss,
                "instant_score": instant_score,
                "expert_score": expert_score,
                "instant_token": token_count(
                    side_view(row, "instant"),
                    args.instant_token_column,
                ),
                "expert_token": token_count(
                    side_view(row, "expert"),
                    args.expert_token_column,
                ),
            }
        )
    return pd.DataFrame(records)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a B-PAC simulation table from evaluated outputs.")
    parser.add_argument("--task", choices=["verifiable", "magpie"], required=True)
    parser.add_argument("--dataset", required=True, help="Dataset name recorded in the output table.")
    parser.add_argument("--expert-json", nargs="+", type=Path, required=True)
    parser.add_argument("--instant-json", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--join-key", default=None)
    parser.add_argument("--uncertainty-column", default=None)
    parser.add_argument("--expert-token-column", default=None)
    parser.add_argument("--instant-token-column", default=None)
    parser.add_argument("--expert-correct-column", default="matched")
    parser.add_argument("--instant-correct-column", default="matched")
    parser.add_argument("--judge-score-column", default="gpt4_score")
    parser.add_argument("--score-range", type=float, default=9.0)
    parser.add_argument("--filter-expert-correct", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--filter-expert-preferred", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.task == "verifiable":
        output = build_verifiable(args)
    else:
        output = build_magpie(args)
    write_table(output, args.output)
    print(f"Wrote {len(output)} rows to {args.output}")


if __name__ == "__main__":
    main()
