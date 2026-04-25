"""
UMBRA gymnasium environment — OpenEnv compliant multi-agent RL arena.
Wraps Sentrix at input boundary. Injects curriculum config and episodic memory.
Exposes reset/step/state/observation per OpenEnv spec with FastAPI wrapper.

Core RL contract
────────────────
reset()      → clears all episode state, returns fresh observation + info
step(action) → applies one agent action, returns (obs, reward, done, truncated, info)
               reward = step_reward + episode_reward (episode_reward fires only at done)
               info includes reward_breakdown {step, episode, total}
observation  → property alias for state(); returns what the agent currently sees
state()      → same as observation; returns the full internal state dict
reward       → composed of:
               step-level  : action validity, uncertainty calibration, belief update bonus
               episode-level: liar caught, manipulation flagged, coalition detected,
                              sentrix true-positives, sycophancy / overconfidence penalties
"""

import gymnasium as gym
import numpy as np
import logging
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout
from fastapi import FastAPI
from typing import Any
from sentrix.pii_guard import run as sentrix_run, SentrixBlockException
from env.npc_agents import NPCAgent
from env.belief_module import BeliefModule
from env.memory_module import MemoryModule
from reward.reward_model import RewardModel

_env_logger = logging.getLogger("UMBRA.env")

# Hard wall-clock limits — prevents a hanging NPC from freezing an episode.
# If a call exceeds the limit, a safe fallback is returned and training continues.
NPC_GENERATE_TIMEOUT_SEC  = 2.0   # per NPC, per turn
SENTRIX_SCAN_TIMEOUT_SEC  = 1.0   # per NPC output, per turn
_executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="umbra-env")

ACTION_SPACE_SIZE = 8
OBS_KEYS = ["conversation_history", "belief_state", "agent_confidence",
            "uncertainty_bucket", "episode_memory", "turn_count",
            "current_stage", "active_npcs", "hints"]


class UmbraEnv(gym.Env):
    def __init__(self, reward_model=None):
        super().__init__()
        self.action_space = gym.spaces.Discrete(ACTION_SPACE_SIZE)
        self.observation_space = gym.spaces.Dict({
            "turn_count": gym.spaces.Discrete(20),
            "agent_confidence": gym.spaces.Box(0.0, 1.0, (1,), np.float32),
        })
        self.belief_module = BeliefModule()
        self.memory_module = MemoryModule()
        # Accept injected reward model (e.g. ShapedRewardModel) or default
        self.reward_model = reward_model if reward_model is not None else RewardModel()
        self.config: dict = {}
        self._state: dict = {}
        self._sentrix_blocks = 0
        self._action_history: list[int] = []
        # Per-episode history — fed into compute_episode() at done
        self._npc_outputs_history: list[dict] = []
        self._sentrix_history: list[dict] = []

    # ── reset() ──────────────────────────────────────────────────────────────
    # Starts a completely fresh episode.
    # Clears conversation history, belief state, NPC instances, action history,
    # per-episode NPC output / Sentrix history, and sentrix block counter.
    # Returns: (observation_dict, info_dict)
    def reset(self, *, seed=None, options=None, config: dict = None) -> tuple[dict, dict]:
        if config:
            self.config = config
        # Inform ShapedRewardModel of current stage and reset intra-episode state
        stage = self.config.get("current_stage", 1)
        if hasattr(self.reward_model, "set_stage"):
            self.reward_model.set_stage(stage)
        if hasattr(self.reward_model, "reset_episode"):
            self.reward_model.reset_episode()
        self.belief_module.reset(self.config.get("active_npcs", []))
        self._sentrix_blocks = 0
        self._action_history = []
        self._npc_outputs_history = []   # reset episode-level NPC output tracking
        self._sentrix_history = []        # reset episode-level Sentrix result tracking
        self._npcs = {n: NPCAgent(n) for n in self.config.get("active_npcs", [])}
        episode_memory = self.memory_module.get_context()
        self._state = {
            "conversation_history": [],
            "belief_state": self.belief_module.get_all_beliefs(),
            "agent_confidence": 0.5,
            "uncertainty_bucket": "weak",
            "episode_memory": episode_memory,
            "turn_count": 0,
            "current_stage": self.config.get("current_stage", 1),
            "active_npcs": self.config.get("active_npcs", []),
            "hints": "Watch for contradictions." if self.config.get("hints_enabled") else None,
        }
        info = {
            "reset": True,
            "stage": self._state["current_stage"],
            "active_npcs": self._state["active_npcs"],
        }
        return dict(self._state), info

    # ── step(action) ──────────────────────────────────────────────────────────
    # Applies one agent action and advances the environment by one turn.
    #
    # Action space (0-7):
    #   0 observe      1 express_uncertainty   2 challenge_liar
    #   3 flag_manipulation  4 validate_claim  5 boost_confidence
    #   6 escalate     7 stay_silent
    #
    # Returns: (observation, reward, done, truncated, info)
    #   reward          = step_reward (every turn) + episode_reward (only at done)
    #   done            = turn_count >= max_turns OR action == 6 (escalate to close)
    #   truncated       = sentrix_blocks >= 3 (safety cutoff)
    #   info keys:
    #     sentrix_results  — per-NPC Sentrix scan output
    #     reward_breakdown — {step, episode, total} reward components
    #     npc_outputs      — what each NPC said this turn (post-redaction)
    def step(self, action: int) -> tuple[dict, float, bool, bool, dict]:
        self._action_history.append(action)
        npc_outputs: dict = {}
        sentrix_results: dict = {}

        for npc_id, npc in self._npcs.items():
            # ── Wall-clock timeout: NPC generate ─────────────────────────────
            try:
                raw = _executor.submit(npc.generate, self._state).result(
                    timeout=NPC_GENERATE_TIMEOUT_SEC
                )
            except FuturesTimeout:
                _env_logger.warning(
                    f"[timeout] NPC '{npc_id}' generate() exceeded "
                    f"{NPC_GENERATE_TIMEOUT_SEC}s — using fallback output."
                )
                raw = f"[{npc_id} timed out]"
            except Exception as exc:
                _env_logger.warning(f"[error] NPC '{npc_id}' generate() raised {exc!r}")
                raw = f"[{npc_id} error]"

            # ── Wall-clock timeout: Sentrix scan ─────────────────────────────
            try:
                sr = _executor.submit(sentrix_run, raw).result(
                    timeout=SENTRIX_SCAN_TIMEOUT_SEC
                )
            except FuturesTimeout:
                _env_logger.warning(
                    f"[timeout] sentrix_run for '{npc_id}' exceeded "
                    f"{SENTRIX_SCAN_TIMEOUT_SEC}s — defaulting to pass."
                )
                sr = {"severity": "pass", "pii_found": False, "redacted_text": raw, "types_found": []}
            except SentrixBlockException as e:
                sr = {"severity": "block", "redacted_text": e.redacted_text, "pii_found": True}
                self._sentrix_blocks += 1
                self.belief_module.update(npc_id, raw, "block")
                npc_outputs[npc_id] = e.redacted_text
                sentrix_results[npc_id] = sr
                continue
            except Exception as exc:
                _env_logger.warning(f"[error] sentrix_run for '{npc_id}' raised {exc!r}")
                sr = {"severity": "pass", "pii_found": False, "redacted_text": raw, "types_found": []}

            self.belief_module.update(npc_id, raw, sr["severity"])
            npc_outputs[npc_id] = sr.get("redacted_text", raw) if sr["pii_found"] else raw
            sentrix_results[npc_id] = sr

        # Track per-episode history so compute_episode() has full context
        self._npc_outputs_history.append(dict(npc_outputs))
        for sr in sentrix_results.values():
            if isinstance(sr, dict):
                self._sentrix_history.append(sr)

        # ── Observation update ────────────────────────────────────────────────
        self._state["turn_count"] += 1
        self._state["conversation_history"] = (
            self._state["conversation_history"] + list(npc_outputs.values())
        )[-10:]
        self._state["belief_state"] = self.belief_module.get_all_beliefs()

        # Action → observation side-effects
        if action == 1:    # express_uncertainty → downgrade bucket
            self._state["uncertainty_bucket"] = "weak"
        elif action == 2:  # challenge_liar → drop confidence slightly
            self._state["agent_confidence"] = max(self._state["agent_confidence"] - 0.05, 0.0)
        elif action == 5:  # boost_confidence
            self._state["agent_confidence"] = min(self._state["agent_confidence"] + 0.1, 1.0)

        # ── Reward: step component (fires every turn) ─────────────────────────
        step_reward = self.reward_model.compute(
            self._state, action, npc_outputs, sentrix_results, self._action_history
        )
        reward = step_reward

        # ── Done / Truncated ──────────────────────────────────────────────────
        max_turns = self.config.get("max_turns", 5)
        done      = self._state["turn_count"] >= max_turns or action == 6
        truncated = self._sentrix_blocks >= 3

        # ── Reward: episode component (fires only at episode end) ─────────────
        episode_reward = 0.0
        if done or truncated:
            episode_reward = self.reward_model.compute_episode(
                self._state,
                self._action_history,
                self._npc_outputs_history,
                self._sentrix_history,
            )
            reward += episode_reward
            self.memory_module.log_episode(
                self._state, self._action_history, reward, self._sentrix_blocks
            )

        info = {
            "sentrix_results": sentrix_results,
            "npc_outputs": npc_outputs,
            "reward_breakdown": {
                "step":    round(step_reward,    4),
                "episode": round(episode_reward, 4),
                "total":   round(reward,         4),
            },
        }
        return dict(self._state), reward, done, truncated, info

    # ── state() / observation ─────────────────────────────────────────────────
    # What the agent sees at any point in time.
    # Both state() and the .observation property return the same dict.
    # Keys: conversation_history, belief_state, agent_confidence,
    #       uncertainty_bucket, episode_memory, turn_count,
    #       current_stage, active_npcs, hints
    def state(self) -> dict:
        return dict(self._state)

    @property
    def observation(self) -> dict:
        """Alias for state() — what the agent currently observes."""
        return dict(self._state)


# FastAPI wrapper
app = FastAPI()
_env = UmbraEnv()


@app.post("/reset")
def api_reset(config: dict = None):
    obs, info = _env.reset(config=config or {})
    return {"observation": obs, "info": info}


@app.post("/step")
def api_step(payload: dict):
    obs, rew, done, trunc, info = _env.step(int(payload["action"]))
    return {
        "observation": obs,
        "reward": rew,
        "done": done,
        "truncated": trunc,
        "reward_breakdown": info.get("reward_breakdown", {}),
        "npc_outputs": info.get("npc_outputs", {}),
    }


@app.get("/state")
def api_state():
    return {"observation": _env.observation}


@app.get("/health")
def api_health():
    return {"status": "ok"}
