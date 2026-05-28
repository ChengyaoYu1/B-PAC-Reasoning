from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


def compute_step_loss(expert_correct: int, instant_correct: int) -> float:
    """Loss is one only when the expert is correct and the instant model is wrong."""

    return float(int(expert_correct) * (1 - int(instant_correct)))


@dataclass
class BPACConfig:
    alpha: float = 0.1
    epsilon: float = 0.08
    rho_deploy: float = 0.05
    rho_warm: float = 0.7
    warmup_steps: int = 200
    beta: float = 1.0
    c_clip: float = 0.9
    num_thresholds: int = 1001
    metric_warmup_steps: int = 0
    seed: Optional[int] = None


class BPAC:
    """Betting PAC threshold learner for online reasoning with partial feedback."""

    def __init__(self, config: BPACConfig):
        self.cfg = config
        self.rng = np.random.default_rng(config.seed)
        self.threshold_candidates = np.linspace(0, 1, self.cfg.num_thresholds)
        self.current_u_idx = 0
        self.current_u = float(self.threshold_candidates[self.current_u_idx])
        self.current_rho = self.cfg.rho_warm
        self.wealth = np.ones(self.cfg.num_thresholds)
        self.sum_D = np.zeros(self.cfg.num_thresholds)
        self.sum_D_sq = np.zeros(self.cfg.num_thresholds)

    def get_action(self, uncertainty_score: float) -> tuple[int, float]:
        """Return action and propensity. Action 1 calls the expert, 0 uses instant."""

        if uncertainty_score >= self.current_u:
            return 1, 1.0

        propensity = self.current_rho
        action = int(self.rng.random() < self.current_rho)
        return action, propensity

    def update(
        self,
        uncertainty_score: float,
        action: int,
        observed_loss: Optional[float],
    ) -> None:
        l_t = observed_loss if observed_loss is not None else 0.0
        xi_t = action

        indicator_less = (uncertainty_score < self.threshold_candidates).astype(float)
        pi_t = self.current_rho if uncertainty_score < self.current_u else 1.0

        rho_min = min(self.cfg.rho_deploy, self.cfg.rho_warm)
        weighted_loss = (1 - rho_min) * (l_t * xi_t * indicator_less) / pi_t
        D_t = self.cfg.epsilon - weighted_loss

        denom = self.sum_D_sq + self.cfg.beta
        denom[denom == 0] = 1e-9
        lambda_raw = self.sum_D / denom

        M_t = max(self.cfg.epsilon, ((1.0 - rho_min) / self.current_rho) - self.cfg.epsilon)
        upper_bound = self.cfg.c_clip / M_t
        lambda_t = np.clip(lambda_raw, 0, upper_bound)

        self.wealth *= 1.0 + lambda_t * D_t
        self.sum_D += D_t
        self.sum_D_sq += D_t**2

        is_safe_mask = self.wealth >= (1.0 / self.cfg.alpha)
        prefix_safe_mask = np.logical_and.accumulate(is_safe_mask)
        valid_indices = np.where(prefix_safe_mask)[0]

        if len(valid_indices) > 0:
            self.current_u_idx = int(valid_indices[-1])
            self.current_u = float(self.threshold_candidates[self.current_u_idx])
        else:
            self.current_u_idx = 0
            self.current_u = 0.0


@dataclass
class IPSHoeffdingConfig:
    alpha: float = 0.1
    epsilon: float = 0.08
    rho: float = 0.05
    num_thresholds: int = 1001
    metric_warmup_steps: int = 0
    seed: Optional[int] = None


class IPSHoeffding:
    """IPS plus Hoeffding baseline."""

    def __init__(self, config: IPSHoeffdingConfig):
        self.cfg = config
        self.rng = np.random.default_rng(config.seed)
        self.threshold_candidates = np.linspace(0, 1, self.cfg.num_thresholds)
        self.current_u_idx = 0
        self.current_u = float(self.threshold_candidates[self.current_u_idx])
        self.time_step = 0
        self.sum_Z = np.zeros(self.cfg.num_thresholds)
        self.M_tilde = (1.0 - self.cfg.rho) / self.cfg.rho

    def get_action(self, uncertainty_score: float) -> tuple[int, float]:
        if uncertainty_score >= self.current_u:
            return 1, 1.0
        return int(self.rng.random() < self.cfg.rho), self.cfg.rho

    def update(
        self,
        uncertainty_score: float,
        action: int,
        observed_loss: Optional[float],
    ) -> None:
        self.time_step += 1
        t = self.time_step

        l_t = observed_loss if observed_loss is not None else 0.0
        indicator_less = (uncertainty_score < self.threshold_candidates).astype(float)
        pi_t = self.cfg.rho if uncertainty_score < self.current_u else 1.0

        Z_t = (1.0 - self.cfg.rho) * (l_t * action * indicator_less) / pi_t
        self.sum_Z += Z_t

        mean_Z = self.sum_Z / t
        alpha_t = (6 * self.cfg.alpha) / (np.pi**2 * t**2)
        penalty = self.M_tilde * np.sqrt(np.log(1 / alpha_t + 1e-9) / (2 * t))
        ucb = mean_Z + penalty

        valid_indices = np.where(ucb <= self.cfg.epsilon)[0]
        if len(valid_indices) > 0:
            self.current_u_idx = int(valid_indices[-1])
            self.current_u = float(self.threshold_candidates[self.current_u_idx])
        else:
            self.current_u_idx = 0
            self.current_u = 0.0


@dataclass
class OnlineNaiveConfig:
    alpha: float = 0.1
    epsilon: float = 0.08
    rho: float = 0.05
    num_thresholds: int = 1001
    metric_warmup_steps: int = 0
    seed: Optional[int] = None


class OnlineNaive:
    """Naive online risk baseline without anytime-valid control."""

    def __init__(self, config: OnlineNaiveConfig):
        self.cfg = config
        self.rng = np.random.default_rng(config.seed)
        self.threshold_candidates = np.linspace(0, 1, self.cfg.num_thresholds)
        self.current_u_idx = 0
        self.current_u = float(self.threshold_candidates[self.current_u_idx])
        self.t = 1
        self.sum_risk_terms = np.zeros(self.cfg.num_thresholds)

    def get_action(self, uncertainty_score: float) -> tuple[int, float]:
        if uncertainty_score >= self.current_u:
            return 1, 1.0
        return int(self.rng.random() < self.cfg.rho), self.cfg.rho

    def update(
        self,
        uncertainty_score: float,
        action: int,
        observed_loss: Optional[float],
    ) -> None:
        self.t += 1
        l_t = observed_loss if observed_loss is not None else 0.0
        indicator_less = (uncertainty_score < self.threshold_candidates).astype(float)
        self.sum_risk_terms += action * l_t * indicator_less

        estimated_risk = self.sum_risk_terms / self.t
        valid_indices = np.where(estimated_risk <= self.cfg.epsilon)[0]
        if len(valid_indices) > 0:
            self.current_u_idx = int(valid_indices[-1])
            self.current_u = float(self.threshold_candidates[self.current_u_idx])
        else:
            self.current_u_idx = 0
            self.current_u = 0.0
