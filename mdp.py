"""
UMBRA MDP wrapper — formalises state/action/transition/reward for RL trainer.
Adds state_hash for belief deduplication. Independent NPC Q-tables maintained here.

Core RL contract (mirrors UmbraEnv)
────────────────────────────────────
reset()      → delegates to env.reset(), clears trajectory buffers
step(action) → delegates to env.step(), appends to obs/action history
observation  → property returning the latest observation in the trajectory
trajectory   → full obs/action/hash history for the current episode
compute_return() → discounted return G from a list of step rewards
"""

from typing import Any
import hashlib
import json


GAMMA = 0.95
NPC_ACTIONS = ["cooperate", "deceive", "escalate", "signal_ally", "stay_silent"]


def state_hash(obs: dict) -> str:
    """Deterministic hash over turn_count, uncertainty_bucket, belief_state keys+values."""
    belief = obs.get("belief_state", {})
    belief_tuple = tuple(sorted(
        (k, round(v.get("hidden_agenda_prob", 0), 2), round(v.get("trust_score", 0), 2))
        for k, v in belief.items()
    ))
    key = (
        obs.get("turn_count", 0),
        obs.get("uncertainty_bucket", "weak"),
        belief_tuple,
    )
    return hashlib.md5(json.dumps(key, default=str).encode()).hexdigest()[:12]


class MDPWrapper:
    """Wraps UmbraEnv to expose MDP-style interface for RL trainer."""

    def __init__(self, env, reward_model):
        self.env = env
        self.reward_model = reward_model
        self.gamma = GAMMA
        self._obs_history: list[dict] = []
        self._action_history: list[int] = []
        self._npc_qtables: dict[str, dict[str, dict[str, float]]] = {}

    # ── reset() ──────────────────────────────────────────────────────────────
    # Delegates to UmbraEnv.reset(). Clears trajectory history.
    # Returns the first observation of the new episode.
    def reset(self, config: dict = None) -> dict:
        obs, _ = self.env.reset(config=config)
        self._obs_history = [obs]
        self._action_history = []
        return obs

    # ── step(action) ──────────────────────────────────────────────────────────
    # Delegates to UmbraEnv.step(). Appends obs and action to trajectory.
    # Returns (observation, reward, done, truncated, info).
    # info['reward_breakdown'] = {step, episode, total} from RewardModel.
    def step(self, action: int) -> tuple[dict, float, bool, bool, dict]:
        obs, reward, done, truncated, info = self.env.step(action)
        self._obs_history.append(obs)
        self._action_history.append(action)
        return obs, reward, done, truncated, info

    def get_horizon(self) -> int:
        return self.env.config.get("max_turns", 5)

    def compute_return(self, rewards: list[float]) -> float:
        """Discounted return from a list of step rewards."""
        G = 0.0
        for r in reversed(rewards):
            G = r + self.gamma * G
        return G

    def update_npc_qtable(self, npc_id: str, s_hash: str, action: str,
                          reward: float, ns_hash: str) -> None:
        if npc_id not in self._npc_qtables:
            self._npc_qtables[npc_id] = {}
        for h in (s_hash, ns_hash):
            if h not in self._npc_qtables[npc_id]:
                self._npc_qtables[npc_id][h] = {a: 0.0 for a in NPC_ACTIONS}
        lr, gamma = 0.05, 0.9
        best_next = max(self._npc_qtables[npc_id][ns_hash].values())
        td = reward + gamma * best_next - self._npc_qtables[npc_id][s_hash][action]
        self._npc_qtables[npc_id][s_hash][action] += lr * td

    # ── observation ───────────────────────────────────────────────────────────
    # What the agent currently sees — the latest obs in the trajectory buffer.
    @property
    def observation(self) -> dict:
        """Latest observation from the current episode trajectory."""
        return self._obs_history[-1] if self._obs_history else {}

    # ── trajectory ────────────────────────────────────────────────────────────
    # Full record of the current episode: observations, actions, state hashes.
    @property
    def trajectory(self) -> dict:
        return {
            "obs":          self._obs_history,
            "actions":      self._action_history,
            "state_hashes": [state_hash(o) for o in self._obs_history],
        }
