"""
Reward function for Highway environment with distance-based safety switching.
"""

import numpy as np


class HighwayPreferenceFunction:
    def __init__(
        self,
        init_speed_weight: float = 1.0,
        safety_distance_threshold: float = 10.0,
        safety_boost_factor: float = 1.5,
    ):
        self.init_speed_weight = init_speed_weight
        self.safety_distance_threshold = safety_distance_threshold
        self.safety_boost_factor = safety_boost_factor
        self.last_weights = None

    def reset(self):
        pass

    def __call__(self, nearest_distance: float = float("inf")) -> np.ndarray:
        """
        Compute preference weights with distance-based safety switching.

        Args:
            nearest_distance: Distance to nearest car. If closer than threshold, boost safety weight.

        Returns:
            Preference weights [speed_weight, safety_weight]
        """
        # Compute base speed weight
        base_speed_weight = self.init_speed_weight
        # Base safety weight (complement of speed weight)
        base_safety_weight = 1.0 - base_speed_weight

        # Check if we need to boost safety weight based on nearest car distance
        if nearest_distance < self.safety_distance_threshold:
            # Boost safety weight
            safety_weight = min(1.0, base_safety_weight * self.safety_boost_factor)
            speed_weight = 1.0 - safety_weight
        else:
            speed_weight = base_speed_weight
            safety_weight = base_safety_weight

        weights = np.array([speed_weight, safety_weight], dtype=np.float32)
        self.last_weights = weights
        return weights


class RewardFunction:
    """Combined reward function with preference and distance-based shaping."""

    def __init__(self, preference_fn: HighwayPreferenceFunction):
        self.preference_fn = preference_fn

    def reset(self):
        """Reset at episode start."""
        self.preference_fn.reset()

    def __call__(
        self, mo_reward: np.ndarray, nearest_distance: float = float("inf")
    ) -> float:
        """
        Compute scalar reward from multi-objective reward.

        Args:
            mo_reward: Multi-objective reward [speed_reward, safety_reward]
            nearest_distance: Distance to nearest car (for preference adjustment)

        Returns:
            Scalar reward
        """
        # Get preference weights based on nearest distance
        weights = self.preference_fn(nearest_distance=nearest_distance)

        # Compute weighted sum
        scalar_reward = float(np.dot(weights, mo_reward))

        return scalar_reward
