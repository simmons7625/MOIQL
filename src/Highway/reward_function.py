import numpy as np
from typing import Optional

# vec_reward = np.array(
#             [
#                 rewards["high_speed_reward"],
#                 rewards["right_lane_reward"], # will be 0
#                 -rewards["collision_reward"],
#             ],
#             dtype=np.float32,
#         )


class PreferenceFunction:
    def __init__(
        self,
        contenous_decay: float = 0.01,
        switch_decay: float = 0.5,
        init_speed_weight: float = 1.0,
        switch_time: Optional[int] = None,
    ):
        self.contenous_decay = contenous_decay
        self.switch_decay = switch_decay
        self.switch_time = switch_time
        self.init_speed_weight = init_speed_weight

    def reset(self):
        """Reset time step at episode start."""
        self.time_step = 0

    def drift(self):
        self.speed_weight = self.init_speed_weight * np.exp(
            -self.contenous_decay * self.time_step
        )

    def switch(self):
        self.speed_weight *= self.switch_decay

    def __call__(self) -> float:
        """Compute preference-based scalar reward."""
        if self.switch_time is not None and self.time_step >= self.switch_time:
            self.switch()
        else:
            self.drift()
        self.time_step += 1

        return np.array(
            [self.speed_weight, 0.0, 1.0 - self.speed_weight], dtype=np.float32
        )


class RewardFunction:
    """Combined reward function with preference and time-based shaping."""

    def __init__(
        self,
        preference_fn: Optional[PreferenceFunction] = None,
    ):
        """
        Initialize combined reward function.

        Args:
            preference_fn: Preference function for multi-objective rewards
            time_shaping: Time-based reward shaping
            preference_weight: Weight for preference component
            time_weight: Weight for time component
        """
        self.preference_fn = preference_fn

    def __call__(self, mo_reward: np.ndarray) -> float:
        """
        Compute final scalar reward.

        Args:
            mo_reward: Multi-objective reward vector

        Returns:
            Scalar reward combining preference and time components
        """
        weights = self.preference_fn()
        reward = np.dot(weights, mo_reward)
        return reward
