"""
UMBRA game theory layer — Nash equilibrium, coalition detection, meta-skepticism.
Models agent vs NPC as formal games. Detects coordinated deception patterns.
"""

import json
import math
from pathlib import Path
from typing import Any

AUDIT_FILE = Path("logs/game_theory_audit.jsonl")
AUDIT_FILE.parent.mkdir(parents=True, exist_ok=True)

AGENT_ACTIONS = list(range(8))
NPC_ACTIONS = ["cooperate", "deceive", "escalate", "signal_ally", "stay_silent"]


class GameTheoryLayer:
    def __init__(self):
        self._payoff_history: dict[str, list[float]] = {}
        self._npc_signal_history: dict[str, list[float]] = {}
        self._trust_history: dict[str, list[float]] = {}
        self._suboptimal_count = 0
        self._total_turns = 0

    # ── A) Nash Equilibrium ──────────────────────────────────────────────────
    def _estimate_payoffs(self, npc_id: str) -> list[list[float]]:
        history = self._payoff_history.get(npc_id, [0.0] * 10)[-10:]
        avg = sum(history) / max(len(history), 1)
        return [[avg + (i - j) * 0.1 for j in range(len(NPC_ACTIONS))]
                for i in range(len(AGENT_ACTIONS))]

    def nash_best_response(self, npc_id: str, npc_action_idx: int) -> int:
        matrix = self._estimate_payoffs(npc_id)
        return max(range(len(AGENT_ACTIONS)), key=lambda a: matrix[a][npc_action_idx])

    def check_suboptimal(self, npc_id: str, agent_action: int, npc_action_idx: int) -> bool:
        best = self.nash_best_response(npc_id, npc_action_idx)
        self._total_turns += 1
        if agent_action != best:
            self._suboptimal_count += 1
            return True
        return False

    # ── B) Coalition Detector ────────────────────────────────────────────────
    def record_npc_signal(self, npc_id: str, signal_value: float) -> None:
        if npc_id not in self._npc_signal_history:
            self._npc_signal_history[npc_id] = []
        self._npc_signal_history[npc_id].append(signal_value)

    def _pearson(self, xs: list[float], ys: list[float]) -> float:
        n = min(len(xs), len(ys), 5)
        if n < 2:
            return 0.0
        xs, ys = xs[-n:], ys[-n:]
        mx, my = sum(xs) / n, sum(ys) / n
        num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
        denom = math.sqrt(sum((x - mx) ** 2 for x in xs) * sum((y - my) ** 2 for y in ys))
        return num / denom if denom else 0.0

    def detect_coalition(self) -> tuple[bool, list[str]]:
        npc_ids = list(self._npc_signal_history.keys())
        alliances: list[str] = []
        for i in range(len(npc_ids)):
            for j in range(i + 1, len(npc_ids)):
                a, b = npc_ids[i], npc_ids[j]
                corr = self._pearson(self._npc_signal_history[a], self._npc_signal_history[b])
                if corr > 0.75:
                    alliances.append(f"{a}+{b}")
        if alliances:
            self._log({"event": "coalition_detected", "alliances": alliances})
        return bool(alliances), alliances

    # ── C) Meta-Skepticism ───────────────────────────────────────────────────
    def record_trust_delta(self, npc_id: str, trust_score: float) -> None:
        if npc_id not in self._trust_history:
            self._trust_history[npc_id] = []
        self._trust_history[npc_id].append(trust_score)

    def meta_skepticism_check(self) -> bool:
        if len(self._trust_history) < 2:
            return False
        deltas = []
        for hist in self._trust_history.values():
            if len(hist) >= 2:
                deltas.append(hist[-1] - hist[-2])
        if len(deltas) < 2:
            return False
        all_spiked = all(d > 0.3 for d in deltas)
        if all_spiked:
            self._log({"event": "meta_skepticism_trigger", "deltas": deltas})
        return all_spiked

    def suboptimal_rate(self) -> float:
        return self._suboptimal_count / max(self._total_turns, 1)

    def _log(self, record: dict) -> None:
        with open(AUDIT_FILE, "a") as f:
            f.write(json.dumps(record) + "\n")

    def reset(self) -> None:
        self._npc_signal_history.clear()
        self._trust_history.clear()
        self._suboptimal_count = 0
        self._total_turns = 0
