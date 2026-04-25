"""
UMBRA belief module — Theory-of-Mind tracker per NPC across turns.
Updates hidden agenda probability and trust score based on observed signals.
Rule-based only — no LLM-as-judge, prevents signal gaming.
"""

from typing import Any


class BeliefModule:
    def __init__(self):
        self._beliefs: dict[str, dict[str, Any]] = {}

    def reset(self, npc_ids: list[str]) -> None:
        self._beliefs = {
            npc_id: {
                "hidden_agenda_prob": 0.1,
                "trust_score": 0.5,
                "contradiction_count": 0,
                "hedged_response_count": 0,
                "signal_history": [],
                "consistent_streak": 0,
            }
            for npc_id in npc_ids
        }

    def _detect_contradiction(self, npc_id: str, text: str) -> bool:
        history = self._beliefs[npc_id]["signal_history"]
        text_words = set(text.lower().split())
        for past in history[-5:]:
            past_words = set(past.lower().split())
            negated = {"not", "never", "no", "false", "wrong", "incorrect"}
            if negated & text_words and (text_words - negated) & past_words:
                return True
            if negated & past_words and (past_words - negated) & text_words:
                return True
        return False

    def _is_hedged(self, text: str) -> bool:
        hedges = {"maybe", "perhaps", "possibly", "might", "could", "not sure", "unclear"}
        return any(h in text.lower() for h in hedges)

    def _is_instant_agreement(self, text: str) -> bool:
        agreements = {"absolutely", "completely agree", "you're right", "correct", "exactly"}
        return any(a in text.lower() for a in agreements)

    def update(self, npc_id: str, turn_output: str, sentrix_result: str) -> None:
        if npc_id not in self._beliefs:
            return
        b = self._beliefs[npc_id]

        if self._detect_contradiction(npc_id, turn_output):
            b["hidden_agenda_prob"] = min(b["hidden_agenda_prob"] + 0.2, 1.0)
            b["contradiction_count"] += 1
            b["consistent_streak"] = 0
        elif self._is_hedged(turn_output):
            b["hidden_agenda_prob"] = min(b["hidden_agenda_prob"] + 0.1, 1.0)
            b["hedged_response_count"] += 1
            b["consistent_streak"] = 0
        else:
            b["consistent_streak"] += 1

        if self._is_instant_agreement(turn_output):
            b["trust_score"] = max(b["trust_score"] - 0.15, 0.0)

        if b["consistent_streak"] >= 3:
            b["trust_score"] = min(b["trust_score"] + 0.1, 1.0)

        if sentrix_result == "pass":
            b["trust_score"] = min(b["trust_score"] + 0.05, 1.0)
        elif sentrix_result == "block":
            b["hidden_agenda_prob"] = min(b["hidden_agenda_prob"] + 0.35, 1.0)

        b["signal_history"].append(turn_output)
        if len(b["signal_history"]) > 10:
            b["signal_history"] = b["signal_history"][-10:]

    def get_belief(self, npc_id: str) -> dict:
        return dict(self._beliefs.get(npc_id, {}))

    def get_all_beliefs(self) -> dict:
        return {k: dict(v) for k, v in self._beliefs.items()}
