"""
UMBRA Curriculum Scheduler — controls difficulty progression across training.
Unlocks NPCs and failure modes as agent crosses reward thresholds.
Prevents zero-reward stall that kills RL learning from the start.

Stage promotion requires ALL of:
  1. Rolling average >= unlock_condition   (reward threshold)
  2. At least MIN_STAGE_EPISODES episodes in current stage  (enough data)
  3. No episode in last 5 had reward below REWARD_FLOOR     (no collapse)
"""

from collections import deque
import logging

logger = logging.getLogger(__name__)

# Minimum episodes before stage promotion is even considered.
# Stops the agent from graduating on a lucky streak before it truly learned.
MIN_STAGE_EPISODES = 20

# If any of the last 5 episodes scored below this, promotion is blocked —
# the agent is still failing on edge cases.
REWARD_FLOOR = 0.5

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
        # Tracks last 5 rewards to enforce the floor check
        self._recent5: deque[float] = deque(maxlen=5)

    @property
    def unlock_progress(self) -> float:
        cond = STAGES[self.current_stage]["unlock_condition"]
        if cond is None or not self.rolling_reward_buffer:
            return 1.0
        avg = sum(self.rolling_reward_buffer) / len(self.rolling_reward_buffer)
        return min(avg / cond, 1.0)

    def update(self, episode_reward: float) -> None:
        self.rolling_reward_buffer.append(episode_reward)
        self._recent5.append(episode_reward)
        self.steps_in_stage += 1

        if self.current_stage >= 3:
            return

        cond = STAGES[self.current_stage]["unlock_condition"]
        if cond is None:
            return

        # Gate 1: enough episodes in this stage
        if self.steps_in_stage < MIN_STAGE_EPISODES:
            return

        # Gate 2: rolling average meets the reward threshold
        if len(self.rolling_reward_buffer) < 10:
            return
        avg = sum(self.rolling_reward_buffer) / 10
        if avg < cond:
            return

        # Gate 3: no collapse in last 5 episodes (floor check)
        if any(r < REWARD_FLOOR for r in self._recent5):
            floor_violations = [r for r in self._recent5 if r < REWARD_FLOOR]
            logger.warning(
                f"[Curriculum] Stage {self.current_stage} avg={avg:.2f} meets threshold "
                f"but {len(floor_violations)} of last 5 episodes below floor "
                f"({REWARD_FLOOR}) — promotion blocked. Lowest={min(floor_violations):.3f}"
            )
            return

        # All gates passed — promote
        self.current_stage += 1
        self.steps_in_stage = 0
        self.rolling_reward_buffer.clear()
        self._recent5.clear()
        logger.info(
            f"[Curriculum] All gates passed (avg={avg:.2f} >= {cond}, "
            f"min_episodes={MIN_STAGE_EPISODES}, floor={REWARD_FLOOR}) "
            f"-- Advanced to Stage {self.current_stage}"
        )

    def get_config(self) -> dict:
        cfg = dict(STAGES[self.current_stage])
        cfg["current_stage"] = self.current_stage
        cfg["steps_in_stage"] = self.steps_in_stage
        cfg["unlock_progress"] = self.unlock_progress
        return cfg
