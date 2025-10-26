"""
Reward function for Deep Sea Treasure with time-varying preferences.
"""

import numpy as np


class DSTPreferenceFunction:
    def __init__(
        self,
        contenous_decay: float = 0.01,
        init_treasure_weight: float = 1.0,
    ):
        self.contenous_decay = contenous_decay
        self.init_treasure_weight = init_treasure_weight
        self.last_weights = None
        self.time_step = 0

    def reset(self):
        """Reset time step at episode start."""
        self.time_step = 0

    def __call__(self) -> np.ndarray:
        """Compute preference weights with continuous decay."""
        self.treasure_weight = max(
            0.0,
            self.init_treasure_weight * np.exp(-self.contenous_decay * self.time_step),
        )
        self.time_step += 1

        weights = np.array(
            [self.treasure_weight, 1.0 - self.treasure_weight], dtype=np.float32
        )
        self.last_weights = weights
        return weights


class RewardFunction:
    """Combined reward function with preference and time-based shaping."""

    def __init__(self, preference_fn: DSTPreferenceFunction):
        self.preference_fn = preference_fn

    def reset(self):
        """Reset at episode start."""
        self.preference_fn.reset()

    def __call__(self, mo_reward: np.ndarray, **kwargs) -> float:
        """
        Compute scalar reward from multi-objective reward.

        Args:
            mo_reward: Multi-objective reward [treasure_value, time_penalty]

        Returns:
            Scalar reward
        """
        # Get preference weights
        weights = self.preference_fn()

        # Compute weighted sum
        scalar_reward = float(np.dot(weights, mo_reward))

        return scalar_reward
