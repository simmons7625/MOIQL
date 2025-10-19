import numpy as np
from typing import Optional, Union


class DSTPreferenceFunction:
    def __init__(
        self,
        contenous_decay: float = 0.01,
        init_treasure_weight: float = 1.0,
    ):
        self.contenous_decay = contenous_decay
        self.init_treasure_weight = init_treasure_weight
        self.last_weights = None

    def reset(self):
        """Reset time step at episode start."""
        self.time_step = 0

    def __call__(self) -> np.ndarray:
        """Compute preference weights with continuous linear decay."""
        self.treasure_weight = max(
            0.0, self.init_treasure_weight - self.contenous_decay * self.time_step
        )
        self.time_step += 1

        weights = np.array(
            [self.treasure_weight, 1.0 - self.treasure_weight], dtype=np.float32
        )
        self.last_weights = weights
        return weights


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
        # Compute base speed weight with linear decay
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
    """Combined reward function with preference and time-based shaping."""

    def __init__(
        self,
        preference_fn: Optional[
            Union[HighwayPreferenceFunction, DSTPreferenceFunction]
        ] = None,
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

    def reset(self):
        """Reset preference function at episode start."""
        if self.preference_fn:
            self.preference_fn.reset()

    def __call__(self, mo_reward: np.ndarray, nearest_distance: float = None) -> float:
        """
        Compute final scalar reward.

        Args:
            mo_reward: Multi-objective reward vector
            nearest_distance: Optional distance to nearest car (for Highway preference function)

        Returns:
            Scalar reward combining preference and time components
        """
        # Call preference function with nearest_distance if it accepts it (Highway)
        # Otherwise call without it (DST)
        if nearest_distance is not None and isinstance(
            self.preference_fn, HighwayPreferenceFunction
        ):
            weights = self.preference_fn(nearest_distance=nearest_distance)
        else:
            weights = self.preference_fn()

        reward = np.dot(weights, mo_reward)
        return reward
