import gymnasium as gym
import numpy as np
from typing import Tuple, Optional
from .reward_function import RewardFunction

TREASURE_VALUES = [
    [0.3, 0.5, 0.8],
    [3, 5, 8],
    [6, 10, 12],
    [13, 14, 16],
    [14, 15, 17],
    [16, 17, 18],
    [20, 25, 28],
    [24, 26, 29],
    [30, 32, 34],
    [35, 36, 38],
]


class DeepSeaTreasureWrapper(gym.Wrapper):
    """Wrapper for Deep Sea Treasure environment with multi-objective rewards."""

    def __init__(
        self,
        env: gym.Env,
        reward_fn: Optional[RewardFunction] = None,
        ignore_done: bool = False,
    ):
        super().__init__(env)
        self.n_objectives = 2
        self.reward_fn = reward_fn
        self.return_scalar_reward = reward_fn is not None
        self.ignore_done = ignore_done

        # Fix observation space dtype to float32 if needed
        if hasattr(self.observation_space, "dtype"):
            import gymnasium.spaces as spaces

            if isinstance(self.observation_space, spaces.Box):
                self.observation_space = spaces.Box(
                    low=self.observation_space.low.astype(np.float32),
                    high=self.observation_space.high.astype(np.float32),
                    shape=self.observation_space.shape,
                    dtype=np.float32,
                )

    def step(self, action: int) -> Tuple[np.ndarray, np.ndarray, bool, bool, dict]:
        obs, reward, terminated, truncated, info = self.env.step(action)

        if isinstance(reward, (list, np.ndarray)):
            mo_reward = np.array(reward, dtype=np.float32)
        else:
            mo_reward = np.array([reward, -1.0], dtype=np.float32)

        # Store original multi-objective reward in info
        info["mo_reward"] = mo_reward.copy()

        # Override done flags if ignore_done is enabled
        if self.ignore_done:
            terminated = False
            truncated = False

        # Apply reward function if provided
        if self.reward_fn:
            scalar_reward = self.reward_fn(mo_reward)
            info["preference_weights"] = self.reward_fn.preference_fn.last_weights
            return obs, scalar_reward, terminated, truncated, info

        return obs, mo_reward, terminated, truncated, info

    def reset(self, **kwargs) -> Tuple[np.ndarray, dict]:
        if self.reward_fn:
            self.reward_fn.reset()
        # Initialize environment map (randomly select dimension in each treasure)
        # Access the unwrapped environment to modify sea_map
        unwrapped_env = self.env.unwrapped
        treasure_values = np.random.randint(3, size=10)
        unwrapped_env.sea_map[1, 0] = TREASURE_VALUES[0][treasure_values[0]]
        unwrapped_env.sea_map[2, 1] = TREASURE_VALUES[1][treasure_values[1]]
        unwrapped_env.sea_map[3, 2] = TREASURE_VALUES[2][treasure_values[2]]
        unwrapped_env.sea_map[4, 3] = TREASURE_VALUES[3][treasure_values[3]]
        unwrapped_env.sea_map[4, 4] = TREASURE_VALUES[4][treasure_values[4]]
        unwrapped_env.sea_map[4, 5] = TREASURE_VALUES[5][treasure_values[5]]
        unwrapped_env.sea_map[7, 6] = TREASURE_VALUES[6][treasure_values[6]]
        unwrapped_env.sea_map[7, 7] = TREASURE_VALUES[7][treasure_values[7]]
        unwrapped_env.sea_map[9, 8] = TREASURE_VALUES[8][treasure_values[8]]
        unwrapped_env.sea_map[10, 9] = TREASURE_VALUES[9][treasure_values[9]]

        return self.env.reset(**kwargs)
