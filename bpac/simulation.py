from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import numpy as np
import pandas as pd

from .algorithms import BPAC, BPACConfig, compute_step_loss


REQUIRED_COLUMNS = {
    "uncertainty",
    "instant_token",
    "expert_token",
}


def _normalize_records(data_sequence: Iterable[dict[str, Any]] | pd.DataFrame) -> list[dict[str, Any]]:
    if isinstance(data_sequence, pd.DataFrame):
        missing = REQUIRED_COLUMNS - set(data_sequence.columns)
        if missing:
            raise ValueError(f"Missing required columns: {sorted(missing)}")
        return data_sequence.to_dict("records")

    records = list(data_sequence)
    for i, record in enumerate(records):
        missing = REQUIRED_COLUMNS - set(record)
        if missing:
            raise ValueError(f"Record {i} is missing required keys: {sorted(missing)}")
    return records


def run_simulation(
    data_sequence: Iterable[dict[str, Any]] | pd.DataFrame,
    config: BPACConfig,
    model: Any | None = None,
) -> tuple[pd.DataFrame, Any]:
    """Run an online routing simulation.

    Input records must contain:
    uncertainty, instant_correct, expert_correct, instant_token, expert_token.
    """

    records = _normalize_records(data_sequence)
    model = model or BPAC(config)
    logs: list[dict[str, Any]] = []

    total_actual_tokens = 0.0
    total_baseline_tokens = 0.0
    cumulative_loss = 0.0
    expert_calls = 0
    metric_warmup_steps = int(getattr(config, "metric_warmup_steps", 0))

    for t, item in enumerate(records):
        u_t = float(item["uncertainty"])
        instant_token = float(item["instant_token"])
        expert_token = float(item["expert_token"])

        if isinstance(model, BPAC):
            model.current_rho = (
                model.cfg.rho_warm if t < model.cfg.warmup_steps else model.cfg.rho_deploy
            )

        action, propensity = model.get_action(u_t)
        if "loss" in item:
            true_loss = float(item["loss"])
        else:
            instant_correct = int(item["instant_correct"])
            expert_correct = int(item["expert_correct"])
            true_loss = compute_step_loss(expert_correct, instant_correct)
        observed_loss = true_loss if action == 1 else None
        model.update(u_t, action, observed_loss)

        if t < metric_warmup_steps:
            continue

        if action == 1:
            step_actual_tokens = instant_token + expert_token
            expert_calls += 1
        else:
            step_actual_tokens = instant_token
            cumulative_loss += true_loss

        total_actual_tokens += step_actual_tokens
        total_baseline_tokens += expert_token

        n_eval_steps = t - metric_warmup_steps + 1
        if hasattr(model, "wealth"):
            wealth = model.wealth[getattr(model, "current_u_idx", 0)]
        else:
            wealth = 0.0
        logs.append(
            {
                "step": t,
                "uncertainty": u_t,
                "threshold": model.current_u,
                "action": action,
                "propensity": propensity,
                "true_loss": true_loss,
                "observed_loss": observed_loss if observed_loss is not None else np.nan,
                "avg_risk": cumulative_loss / n_eval_steps,
                "token_ratio": total_actual_tokens / total_baseline_tokens
                if total_baseline_tokens > 0
                else 1.0,
                "expert_call_ratio": expert_calls / n_eval_steps,
                "wealth": float(wealth),
            }
        )

    return pd.DataFrame(logs), model
