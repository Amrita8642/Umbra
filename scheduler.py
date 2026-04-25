"""
UMBRA Curriculum Scheduler — controls difficulty progression across training.
Unlocks NPCs and failure modes as agent crosses reward thresholds.
Prevents zero-reward stall that kills RL learning from the start.
"""

from collections import deque
import logging

logger = logging.getLogger(__name__)

STAGES = {
    1: {
        "active_npcs": ["Agreeable"],
        "active_failure_modes": ["sycophancy", "overconfidence", "uncertainty"],
        "hints_enabled": True,
        "max_turns": 5,
        "unlock_condition": 1.5,
    },
    2: {
        "active_npcs": ["Agreeable", "Liar", "Emotional"],
        "active_failure_modes": [
            "sycophancy", "overconfidence", "uncertainty",
            "deception", "EQ_failure", "calibration_error",
        ],
        "hints_enabled": False,
        "max_turns": 8,
        "unlock_condition": 3.5,
    },
    3: {
        "active_npcs": ["Agreeable", "Liar", "Manipulator", "Coalition_A", "Coalition_B", "Emotional"],
        "active_failure_modes": [
            "sycophancy", "overconfidence", "uncertainty", "deception",
            "EQ_failure", "calibration_error", "prompt_injection",
            "coalition_manipulation", "belief_manipulation",
            "reward_hacking", "meta_deception",
        ],
        "hints_enabled": False,
        "max_turns": 12,
        "unlock_condition": None,
    },
}


class CurriculumScheduler:
    def __init__(self):
        self.current_stage = 1
        self.steps_in_stage = 0
        self.rolling_reward_buffer: deque[float] = deque(maxlen=10)

    @property
    def unlock_progress(self) -> float:
        cond = STAGES[self.current_stage]["unlock_condition"]
        if cond is None or not self.rolling_reward_buffer:
            return 1.0
        avg = sum(self.rolling_reward_buffer) / len(self.rolling_reward_buffer)
        return min(avg / cond, 1.0)

    def update(self, episode_reward: float) -> None:
        self.rolling_reward_buffer.append(episode_reward)
        self.steps_in_stage += 1

        if self.current_stage >= 3:
            return

        cond = STAGES[self.current_stage]["unlock_condition"]
        if cond is None:
            return

        if len(self.rolling_reward_buffer) == 10:
            avg = sum(self.rolling_reward_buffer) / 10
            if avg >= cond:
                self.current_stage += 1
                self.steps_in_stage = 0
                self.rolling_reward_buffer.clear()
                logger.info(f"[Curriculum] Advanced to Stage {self.current_stage}")

    def get_config(self) -> dict:
        cfg = dict(STAGES[self.current_stage])
        cfg["current_stage"] = self.current_stage
        cfg["steps_in_stage"] = self.steps_in_stage
        cfg["unlock_progress"] = self.unlock_progress
        return cfg
