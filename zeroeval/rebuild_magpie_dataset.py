import argparse
import json
from pathlib import Path

import pandas as pd


DROP_COLUMNS = {
    "chat_history",
    "model_input",
    "output",
    "generator",
    "token_probs",
    "topk_probs",
    "avg_logprob",
    "avg_topk_entropy",
    "configs",
    "gen_token_count",
    "latency_s",
    "gpt4_score",
    "gpt4_fail_reason",
    "dataset"
}

PREFERRED_COLUMNS = [
    "session_id",
    "question",
    "question_id",
    "options",
    "answer",
    "answer_index",
    "cot_content",
    "category",
    "src",
    "GPT_answer",
    "uuid",
    "verbalized_prob",
    "gpt4_score",
    "gpt4_fail_reason",
    "dataset",
]


def load_records(json_path: Path) -> list[dict]:
    with json_path.open("r") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"{json_path} does not contain a JSON list")
    return data


def is_missing_value(value) -> bool:
    if value is None:
        return True
    if isinstance(value, (str, list, dict)):
        return value in ("", [], {})
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def merge_rows(base: dict, new_row: dict) -> dict:
    merged = dict(base)
    for key, value in new_row.items():
        if key not in merged or is_missing_value(merged[key]):
            merged[key] = value
    return merged


def reconstruct_dataset(result_dir: Path) -> pd.DataFrame:
    merged_rows: dict[str, dict] = {}
    json_files = sorted(result_dir.glob("*.json"))
    if not json_files:
        raise FileNotFoundError(f"No JSON result files found in {result_dir}")

    for json_file in json_files:
        for row in load_records(json_file):
            session_id = row.get("session_id")
            if not session_id:
                continue
            filtered = {
                key: value
                for key, value in row.items()
                if key not in DROP_COLUMNS
            }
            if session_id in merged_rows:
                merged_rows[session_id] = merge_rows(merged_rows[session_id], filtered)
            else:
                merged_rows[session_id] = filtered

    df = pd.DataFrame(list(merged_rows.values()))
    ordered_cols = [col for col in PREFERRED_COLUMNS if col in df.columns]
    remaining_cols = sorted(col for col in df.columns if col not in ordered_cols)
    return df[ordered_cols + remaining_cols].sort_values("session_id").reset_index(drop=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--result-dir",
        default="result_dirs/magpie",
        help="Directory containing historical result JSON files.",
    )
    parser.add_argument(
        "--output-dir",
        default="data",
        help="Directory to save the reconstructed dataset.",
    )
    parser.add_argument(
        "--output-prefix",
        default=None,
        help="Output file prefix. Defaults to the name of result-dir.",
    )
    args = parser.parse_args()

    result_dir = Path(args.result_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_prefix = args.output_prefix or result_dir.name

    df = reconstruct_dataset(result_dir)
    parquet_path = output_dir / f"{output_prefix}_reconstructed.parquet"
    jsonl_path = output_dir / f"{output_prefix}_reconstructed.jsonl"

    df.to_parquet(parquet_path, index=False)
    df.to_json(jsonl_path, orient="records", lines=True, force_ascii=False)

    print(f"Saved {len(df)} rows to {parquet_path}")
    print(f"Saved {len(df)} rows to {jsonl_path}")
    print(f"Columns: {list(df.columns)}")


if __name__ == "__main__":
    main()
