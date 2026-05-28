#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
from openai import OpenAI
from tqdm import tqdm


EVAL_PROMPT = """You are a careful, unbiased and strict evaluator.

Given a user question and an assistant answer, evaluate the overall quality of the answer.
Consider correctness, helpfulness, relevance, clarity, completeness, and instruction following.

Provide a single numeric score from 1 to 10.
Do not provide any explanation. Output only the number.

User Question:
{question}

Assistant Answer:
{answer}
"""

THINK_OPEN_RE = re.compile(r"`?\s*<think>\s*`?", flags=re.IGNORECASE)
THINK_CLOSE_RE = re.compile(r"`?\s*</think>\s*`?", flags=re.IGNORECASE)


def make_client() -> OpenAI:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("Set OPENAI_API_KEY before running the judge script.")
    base_url = os.environ.get("OPENAI_BASE_URL")
    return OpenAI(api_key=api_key, base_url=base_url) if base_url else OpenAI(api_key=api_key)


def strip_think_content(answer: str) -> str:
    if not isinstance(answer, str) or not answer:
        return answer

    close_matches = list(THINK_CLOSE_RE.finditer(answer))
    if close_matches:
        tail = answer[close_matches[-1].end() :].strip()
        if tail:
            return tail
        return THINK_OPEN_RE.sub("", THINK_CLOSE_RE.sub("", answer)).strip()

    open_match = THINK_OPEN_RE.search(answer)
    if open_match:
        return answer[: open_match.start()].strip()

    return answer


def get_judge_score(client: OpenAI, prompt: str, *, model: str) -> int:
    completion = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You evaluate assistant answers."},
            {"role": "user", "content": prompt},
        ],
    )
    raw = completion.choices[0].message.content
    match = re.search(r"\d+(?:\.\d+)?", str(raw))
    if not match:
        raise ValueError(f"Judge returned a non-numeric score: {raw!r}")
    return int(float(match.group(0)))


def eval_one_row(row, retry: int, client: OpenAI, *, judge_model: str) -> float:
    question = row["chat_history"][0]
    answer = strip_think_content(row["output"][0])
    prompt = EVAL_PROMPT.format(question=question, answer=answer)
    scores = [get_judge_score(client, prompt, model=judge_model) for _ in range(retry)]
    return sum(scores) / len(scores)


def run_file(
    candidate_model_file_path: Path,
    *,
    max_workers: int,
    retry: int,
    checkpoint_every: int,
    judge_model: str,
) -> None:
    candidate_model_file_path = candidate_model_file_path.resolve()
    data = pd.read_json(candidate_model_file_path)

    if "gpt4_score" not in data.columns:
        data["gpt4_score"] = None
    if "gpt4_fail_reason" not in data.columns:
        data["gpt4_fail_reason"] = None

    pending_df = data[data["gpt4_score"].isna()]
    if len(pending_df) == 0:
        print(f"All samples already evaluated: {candidate_model_file_path}")
        return

    client = make_client()
    scores: dict = {}
    fail_logs: dict = {}
    completed = 0

    def flush_partial() -> None:
        if scores:
            data.loc[pd.Series(scores).index, "gpt4_score"] = pd.Series(scores, dtype=object)
        if fail_logs:
            data.loc[pd.Series(fail_logs).index, "gpt4_fail_reason"] = pd.Series(
                fail_logs, dtype=object
            )
        data.to_json(candidate_model_file_path, orient="records", indent=2, force_ascii=False)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(eval_one_row, row, retry, client, judge_model=judge_model): idx
            for idx, row in pending_df.iterrows()
        }
        with tqdm(total=len(futures), desc=f"Scoring {candidate_model_file_path.stem}") as pbar:
            for future in as_completed(futures):
                idx = futures[future]
                try:
                    scores[idx] = future.result()
                except Exception as exc:
                    scores[idx] = None
                    fail_logs[idx] = str(exc)

                completed += 1
                pbar.update(1)
                if completed % checkpoint_every == 0:
                    flush_partial()

    flush_partial()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LLM-as-judge scoring for Magpie JSON files.")
    parser.add_argument("--json", nargs="+", type=Path, required=True)
    parser.add_argument("--max-workers", type=int, default=8)
    parser.add_argument("--retry", type=int, default=3)
    parser.add_argument("--checkpoint-every", type=int, default=200)
    parser.add_argument("--judge-model", type=str, default="gpt-4o-mini")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for path in args.json:
        if not path.is_file():
            raise FileNotFoundError(path)
        run_file(
            path,
            max_workers=args.max_workers,
            retry=args.retry,
            checkpoint_every=args.checkpoint_every,
            judge_model=args.judge_model,
        )


if __name__ == "__main__":
    main()
