"""
Reward function for Overcooked with time-varying preferences.

Multi-objective rewards:
1. Task completion (dishes served)
2. Time efficiency (negative time penalty)
3. Coordination quality (smooth cooperation without collisions)
"""

import numpy as np


class OvercookedPreferenceFunction:
    def __init__(
        self,
        continuous_decay: float = 0.01,
        init_task_weight: float = 0.7,
        init_efficiency_weight: float = 0.2,
        init_coordination_weight: float = 0.1,
        use_discontinuous_decay: bool = False,
        switching_factor: float = 0.8,
    ):
        """
        Initialize preference function for Overcooked.

        Args:
            continuous_decay: Decay rate for continuous preference change
            init_task_weight: Initial weight for task completion objective
            init_efficiency_weight: Initial weight for time efficiency objective
            init_coordination_weight: Initial weight for coordination quality objective
            use_discontinuous_decay: Use discrete preference switches (e.g., after dish served)
            switching_factor: Multiply task weight by this when dish is served
        """
        self.continuous_decay = continuous_decay
        self.init_weights = np.array(
            [init_task_weight, init_efficiency_weight, init_coordination_weight],
            dtype=np.float32,
        )
        self.use_discontinuous_decay = use_discontinuous_decay
        self.switching_factor = switching_factor

        self.time_step = 0
        self.weights = self.init_weights.copy()
        self.last_weights = None

    def reset(self):
        """Reset preference function at episode start."""
        self.time_step = 0
        self.weights = self.init_weights.copy()
        self.last_weights = None

    def __call__(self, dish_served: bool = False) -> np.ndarray:
        """
        Compute preference weights.

        Args:
            dish_served: Whether a dish was served in this step (for discontinuous decay)

        Returns:
            Preference weights [task, efficiency, coordination]
        """
        if self.use_discontinuous_decay:
            # Discontinuous: update when dish is served
            if dish_served:
                # Reduce task weight, increase efficiency weight
                self.weights[0] *= self.switching_factor  # task
                self.weights[1] = 1.0 - self.weights[0] - self.weights[2]  # efficiency
                self.weights = np.clip(self.weights, 0.0, 1.0)
                # Renormalize
                self.weights = self.weights / np.sum(self.weights)
        else:
            # Continuous: exponential decay of task weight over time
            task_weight = self.init_weights[0] * np.exp(
                -self.continuous_decay * self.time_step
            )
            task_weight = max(0.1, task_weight)  # Don't go below 0.1

            # Redistribute remaining weight
            remaining = 1.0 - task_weight
            efficiency_ratio = self.init_weights[1] / (
                self.init_weights[1] + self.init_weights[2]
            )

            self.weights = np.array(
                [
                    task_weight,
                    remaining * efficiency_ratio,
                    remaining * (1 - efficiency_ratio),
                ],
                dtype=np.float32,
            )
            self.time_step += 1

        self.last_weights = self.weights.copy()
        return self.weights


class RewardFunction:
    """Reward function that computes scalar reward from multi-objective rewards."""

    def __init__(self, preference_fn: OvercookedPreferenceFunction):
        self.preference_fn = preference_fn

    def reset(self):
        """Reset at episode start."""
        self.preference_fn.reset()

    def __call__(self, mo_reward: np.ndarray, **kwargs) -> float:
        """
        Compute scalar reward from multi-objective reward.

        Args:
            mo_reward: Multi-objective reward [task_reward, efficiency_reward, coordination_reward]
            **kwargs: Additional arguments

        Returns:
            Scalar reward
        """
        # Check if dish was served (task_reward > 0)
        dish_served = mo_reward[0] > 0

        # Get preference weights
        weights = self.preference_fn(dish_served=dish_served)

        # Compute weighted sum
        scalar_reward = float(np.dot(weights, mo_reward))

        return scalar_reward
