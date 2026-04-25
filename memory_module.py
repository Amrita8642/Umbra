"""
UMBRA episodic memory — persists cross-episode learning context for the agent.
Last 5 episodes injected as memory_context string into each new episode's observation.
Local JSON storage only. No network. No external DB.
"""

import json
import os
from pathlib import Path

EPISODES_FILE = Path("logs/episodes.jsonl")


class MemoryModule:
    def __init__(self):
        EPISODES_FILE.parent.mkdir(parents=True, exist_ok=True)
        self._episode_id = self._count_episodes()

    def _count_episodes(self) -> int:
        if not EPISODES_FILE.exists():
            return 0
        with open(EPISODES_FILE) as f:
            return sum(1 for _ in f)

    def log_episode(self, state: dict, actions: list[int], total_reward: float, sentrix_blocks: int) -> None:
        record = {
            "episode_id": self._episode_id,
            "stage": state.get("current_stage", 1),
            "active_npcs": state.get("active_npcs", []),
            "actions_taken": actions,
            "signals_fired": [],
            "reward_breakdown": {},
            "verifier_flags": [],
            "total_reward": total_reward,
            "sentrix_blocks": sentrix_blocks,
            "sentrix_false_positives": 0,
        }
        with open(EPISODES_FILE, "a") as f:
            f.write(json.dumps(record) + "\n")
        self._episode_id += 1

    def get_context(self) -> str:
        if not EPISODES_FILE.exists():
            return "No prior episodes."
        episodes = []
        with open(EPISODES_FILE) as f:
            for line in f:
                try:
                    episodes.append(json.loads(line.strip()))
                except json.JSONDecodeError:
                    continue
        last5 = episodes[-5:]
        if not last5:
            return "No prior episodes."
        lines = []
        for ep in last5:
            npcs = ", ".join(ep.get("active_npcs", []))
            reward = ep.get("total_reward", 0.0)
            blocks = ep.get("sentrix_blocks", 0)
            lines.append(f"Episode {ep['episode_id']}: npcs=[{npcs}] reward={reward:.2f} sentrix_blocks={blocks}")
        return " | ".join(lines)

    def load_last_n(self, n: int = 5) -> list[dict]:
        if not EPISODES_FILE.exists():
            return []
        episodes = []
        with open(EPISODES_FILE) as f:
            for line in f:
                try:
                    episodes.append(json.loads(line.strip()))
                except json.JSONDecodeError:
                    continue
        return episodes[-n:]
