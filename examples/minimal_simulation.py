from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bpac import BPACConfig, run_simulation


def make_toy_data(n: int = 800, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    uncertainty = rng.random(n)
    instant_correct = (rng.random(n) > uncertainty * 0.7).astype(int)
    expert_correct = (rng.random(n) > 0.08).astype(int)
    return pd.DataFrame(
        {
            "uncertainty": uncertainty,
            "instant_correct": instant_correct,
            "expert_correct": expert_correct,
            "instant_token": rng.integers(80, 180, size=n),
            "expert_token": rng.integers(900, 1800, size=n),
        }
    )


if __name__ == "__main__":
    data = make_toy_data()
    config = BPACConfig(alpha=0.1, epsilon=0.08, warmup_steps=200, seed=42)
    logs, model = run_simulation(data, config)

    print(logs.tail())
    print(f"final_threshold={model.current_u:.3f}")
    print(f"final_avg_risk={logs['avg_risk'].iloc[-1]:.3f}")
    print(f"final_expert_call_ratio={logs['expert_call_ratio'].iloc[-1]:.3f}")
