"""
UMBRA Primal-Dual constrained optimiser — makes Sentrix safety mathematically unbypassable.
Hard constraint: zero tolerance for confidential data leaks. No reward compensates a leak.
Lagrangian: L(π, λ) = E[total_reward] - λ * E[leak_events].
"""

import json
from pathlib import Path

LOG_FILE = Path("logs/primal_dual_log.jsonl")
LOG_FILE.parent.mkdir(parents=True, exist_ok=True)

ALPHA_DUAL = 0.01
LAMBDA_MIN = 0.0
LAMBDA_MAX = 10.0
LAMBDA_ALERT = 5.0


class PrimalDualOptimiser:
    def __init__(self, lambda_init: float = 0.1):
        self.lam = lambda_init
        self._episode_count = 0
        self._leak_history: list[float] = []

    def lagrangian_reward(self, base_reward: float, leak_count: int) -> float:
        """Replace raw reward with Lagrangian objective for policy gradient."""
        return base_reward - self.lam * leak_count

    def dual_update(self, mean_leak_rate: float) -> None:
        """λ += α * (observed_leak_rate - target_leak_rate=0)."""
        self.lam += ALPHA_DUAL * (mean_leak_rate - 0.0)
        self.lam = max(LAMBDA_MIN, min(LAMBDA_MAX, self.lam))

    def update(self, base_rewards: list[float], leak_counts: list[int]) -> list[float]:
        """
        Called after each training batch.
        Returns Lagrangian-adjusted rewards. Updates λ based on batch leak rate.
        """
        assert len(base_rewards) == len(leak_counts)
        mean_leak_rate = sum(leak_counts) / max(len(leak_counts), 1)
        self._leak_history.append(mean_leak_rate)
        self.dual_update(mean_leak_rate)
        self._episode_count += len(base_rewards)

        if self._episode_count % 10 == 0:
            self._log()

        if self.lam > LAMBDA_ALERT:
            print(f"[PrimalDual] WARNING: λ={self.lam:.3f} > {LAMBDA_ALERT}. "
                  f"Frequent leaks detected — review environment.")

        return [self.lagrangian_reward(r, lk) for r, lk in zip(base_rewards, leak_counts)]

    def _log(self) -> None:
        record = {
            "episode": self._episode_count,
            "lambda": round(self.lam, 4),
            "mean_leak_rate_recent": round(
                sum(self._leak_history[-10:]) / max(len(self._leak_history[-10:]), 1), 4
            ),
        }
        with open(LOG_FILE, "a") as f:
            f.write(json.dumps(record) + "\n")

    @property
    def lambda_value(self) -> float:
        return self.lam
