"""
Reward function for Deep Sea Treasure with time-varying preferences.
"""

import numpy as np


class DSTPreferenceFunction:
    def __init__(
        self,
        contenous_decay: float = 0.01,
        init_treasure_weight: float = 1.0,
        use_discontinuous_decay: bool = False,
        switching_factor: float = 0.5,
    ):
        self.contenous_decay = contenous_decay
        self.init_treasure_weight = init_treasure_weight
        self.use_discontinuous_decay = use_discontinuous_decay
        self.switching_factor = switching_factor
        self.last_weights = None
        self.time_step = 0
        self.treasure_weight = init_treasure_weight

    def reset(self):
        """Reset time step and treasure weight at episode start."""
        self.time_step = 0
        self.treasure_weight = self.init_treasure_weight

    def __call__(self, treasure_collected: bool = False) -> np.ndarray:
        """
        Compute preference weights.

        Args:
            treasure_collected: Whether a treasure was collected in this step (for discontinuous decay)

        Returns:
            Preference weights [treasure_weight, time_weight]
        """
        if self.use_discontinuous_decay:
            # Discontinuous decay: only update when treasure is collected
            if treasure_collected:
                self.treasure_weight *= self.switching_factor
                self.treasure_weight = max(0.0, self.treasure_weight)
        else:
            # Continuous decay: exponential decay over time
            self.treasure_weight = max(
                0.0,
                self.init_treasure_weight
                * np.exp(-self.contenous_decay * self.time_step),
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
            **kwargs: Additional arguments (e.g., treasure_collected for discontinuous decay)

        Returns:
            Scalar reward
        """
        # Check if treasure was collected (treasure_value > 0)
        treasure_collected = mo_reward[0] > 0

        # Get preference weights (pass treasure_collected for discontinuous decay)
        weights = self.preference_fn(treasure_collected=treasure_collected)

        # Compute weighted sum
        scalar_reward = float(np.dot(weights, mo_reward))

        return scalar_reward
