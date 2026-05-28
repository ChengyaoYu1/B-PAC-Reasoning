#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from pathlib import Path

import pandas as pd
from tqdm import tqdm


COD_SYSTEM_PROMPTS = {
    "math": "Think step by step, but only keep minimum draft for each thinking step, with 5 words at most.\nReturn the answer at the end of the response after a separator ####.",
    "mmlupro": "Think step by step, but only keep minimum draft for each thinking step, with 5 words at most.\nReturn the choice at the end of the response after a separator ####, e.g., #### A.",
    "bbh": "Think step by step, but only keep minimum draft for each thinking step, with 5 words at most.\nReturn the answer at the end of the response after a separator ####.",
}

NO_THINKING_TEMPLATE = """Think step by step, and answer the following question.
Return the answer at the end of the response after a separator ####.
Q: {question}
"""


def format_question(row: pd.Series, dataset: str) -> str:
    if dataset == "mmlupro":
        options = row.get("options", row.get("choices", []))
        option_text = "\n".join(f"{chr(65 + i)}. {opt}" for i, opt in enumerate(options))
        return f"{row['question']}\n{option_text}"
    if dataset == "math":
        return row.get("problem", row.get("question", ""))
    return row.get("question", row.get("problem", ""))


def build_prompt(row: pd.Series, tokenizer, dataset: str, mode: str) -> str:
    question = format_question(row, dataset)
    if mode == "cod":
        messages = [
            {"role": "system", "content": COD_SYSTEM_PROMPTS[dataset]},
            {"role": "user", "content": f"Q: {question}"},
        ]
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    messages = [{"role": "user", "content": NO_THINKING_TEMPLATE.format(question=question)}]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return prompt.rstrip() + "\nOkay, I think I have finished thinking.\n</think>\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate CoD or NoThinking baseline responses.")
    parser.add_argument("--mode", choices=["cod", "no-thinking"], required=True)
    parser.add_argument("--dataset", choices=["math", "mmlupro", "bbh"], required=True)
    parser.add_argument("--dataset-path", type=Path, required=True, help="Input parquet file.")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--output", type=Path, required=True, help="Output JSONL path.")
    parser.add_argument("--gpu-ids", default="0")
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--max-model-len", type=int, default=20000)
    parser.add_argument("--max-tokens", type=int, default=8192)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_ids

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    data = pd.read_parquet(args.dataset_path).copy()
    if "id" not in data.columns:
        data["id"] = [f"{args.dataset}#{i}" for i in range(len(data))]

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    llm = LLM(
        model=args.model_path,
        disable_log_stats=True,
        max_model_len=args.max_model_len,
        trust_remote_code=True,
    )
    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        max_tokens=args.max_tokens,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        args.output.unlink()

    prompts = [
        build_prompt(row, tokenizer, args.dataset, args.mode)
        for _, row in tqdm(data.iterrows(), total=len(data), desc="Formatting prompts")
    ]
    for start in tqdm(range(0, len(data), args.batch_size), desc="Generating"):
        end = min(start + args.batch_size, len(data))
        batch = data.iloc[start:end].copy()
        outputs = llm.generate(prompts[start:end], sampling_params, use_tqdm=True)
        batch["chat_history"] = prompts[start:end]
        batch["output"] = [out.outputs[0].text for out in outputs]
        batch["token_count"] = [len(out.outputs[0].token_ids) for out in outputs]
        batch.to_json(args.output, orient="records", lines=True, force_ascii=False, mode="a")


if __name__ == "__main__":
    main()
