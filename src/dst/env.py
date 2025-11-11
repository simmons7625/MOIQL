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
        max_num_treasure: int = 1,
        max_timestep: Optional[int] = None,
        render_mode: Optional[str] = None,
        render_fps: int = 10,
        use_local_obs: bool = True,
        local_obs_size: int = 3,
    ):
        super().__init__(env)
        self.n_objectives = 2
        self.reward_fn = reward_fn
        self.return_scalar_reward = reward_fn is not None
        self.max_num_treasure = max_num_treasure
        self.max_timestep = max_timestep
        self._render_mode = render_mode
        self.render_fps = render_fps
        self.use_local_obs = use_local_obs
        self.local_obs_size = local_obs_size
        self.collected_treasures = 0
        self.timestep = 0

        # Initialize pygame clock if rendering is enabled
        self.clock = None
        if self._render_mode == "human":
            import pygame

            self.clock = pygame.time.Clock()

        # Update observation space to include local view
        import gymnasium.spaces as spaces

        if self.use_local_obs:
            # Observation includes:
            # - Position (row, col) - 2 values (normalized to [0, 1])
            # - Flattened local grid (e.g., 3x3 = 9 values or 5x5 = 25 values)
            # - Remaining treasure capacity (1 value, normalized) - knapsack constraint
            grid_size = local_obs_size * local_obs_size

            # Calculate max treasure value from TREASURE_VALUES
            max_treasure_value = float(np.max(TREASURE_VALUES))

            # Build observation space with proper bounds for each component
            # Position: [0, 1], Grid: [-10, max_treasure], Remaining capacity: [0, 1]
            low = np.concatenate(
                [
                    np.array([0.0, 0.0], dtype=np.float32),  # Position bounds
                    np.full(
                        grid_size, -10.0, dtype=np.float32
                    ),  # Grid bounds (obstacles = -10)
                    np.array([0.0], dtype=np.float32),  # Remaining capacity bound
                ]
            )
            high = np.concatenate(
                [
                    np.array([1.0, 1.0], dtype=np.float32),  # Position bounds
                    np.full(
                        grid_size, max_treasure_value, dtype=np.float32
                    ),  # Grid bounds (max treasure value)
                    np.array([1.0], dtype=np.float32),  # Remaining capacity bound
                ]
            )

            self.observation_space = spaces.Box(
                low=low,
                high=high,
                dtype=np.float32,
            )
        else:
            # Keep original observation space
            if hasattr(self.observation_space, "dtype"):
                if isinstance(self.observation_space, spaces.Box):
                    self.observation_space = spaces.Box(
                        low=self.observation_space.low.astype(np.float32),
                        high=self.observation_space.high.astype(np.float32),
                        shape=self.observation_space.shape,
                        dtype=np.float32,
                    )

    def _get_local_observation(self, position: np.ndarray) -> np.ndarray:
        """
        Extract a local observation centered on the agent's position.
        Returns position, local grid, and remaining capacity (knapsack constraint).

        Observation structure:
        - [0:1]: Position (row, col) - normalized to [0, 1]
        - [2:2+N]: Flattened local grid (N = local_obs_size^2)
        - [2+N]: Remaining treasure capacity (normalized by max_num_treasure)
        """
        unwrapped_env = self.env.unwrapped
        sea_map = unwrapped_env.sea_map
        map_height, map_width = sea_map.shape

        row, col = position[0], position[1]
        radius = self.local_obs_size // 2  # For 3x3, radius = 1

        # Normalize position to [0, 1]
        pos_normalized = np.array(
            [float(row) / max(1, map_height - 1), float(col) / max(1, map_width - 1)],
            dtype=np.float32,
        )

        # Initialize local observation with zeros (or a default value)
        local_obs = np.zeros(
            (self.local_obs_size, self.local_obs_size), dtype=np.float32
        )

        # Extract the local view with boundary handling
        for i in range(self.local_obs_size):
            for j in range(self.local_obs_size):
                # Calculate map coordinates
                map_row = row - radius + i
                map_col = col - radius + j

                # Check boundaries
                if 0 <= map_row < map_height and 0 <= map_col < map_width:
                    local_obs[i, j] = sea_map[map_row, map_col]
                else:
                    # Out of bounds = obstacle
                    local_obs[i, j] = -10.0

        # Flatten to 1D array
        local_grid = local_obs.flatten()

        # Add remaining capacity (normalized) - this is the knapsack constraint
        remaining_normalized = float(
            self.max_num_treasure - self.collected_treasures
        ) / max(1, self.max_num_treasure)

        # Concatenate: [position, local_grid, remaining_capacity]
        full_obs = np.concatenate(
            [
                pos_normalized,
                local_grid,
                np.array([remaining_normalized], dtype=np.float32),
            ]
        )

        return full_obs

    def step(self, action: int) -> Tuple[np.ndarray, np.ndarray, bool, bool, dict]:
        obs, reward, terminated, truncated, info = self.env.step(action)

        # Increment timestep
        self.timestep += 1

        if isinstance(reward, (list, np.ndarray)):
            mo_reward = np.array(reward, dtype=np.float32)
        else:
            mo_reward = np.array([reward, -1.0], dtype=np.float32)

        # Store original multi-objective reward in info
        info["mo_reward"] = mo_reward.copy()

        # Get current position
        unwrapped_env = self.env.unwrapped
        current_pos = unwrapped_env.current_state

        # Check if a treasure was collected (treasure_value > 0)
        treasure_value = mo_reward[0]
        if treasure_value > 0:
            self.collected_treasures += 1
            # Make the treasure disappear
            unwrapped_env.sea_map[current_pos[0], current_pos[1]] = 0.0

            # If max treasures collected, make ALL remaining treasures disappear
            if self.collected_treasures >= self.max_num_treasure:
                self._clear_all_treasures(unwrapped_env)

        # Convert observation to local view if enabled (AFTER treasure collection update)
        if self.use_local_obs:
            obs = self._get_local_observation(current_pos)

        # Termination logic:
        # Episode terminates when agent collects max_num_treasure treasures
        terminated = False
        if self.collected_treasures >= self.max_num_treasure:
            terminated = True

        # Check if max timestep is reached
        if self.max_timestep is not None and self.timestep >= self.max_timestep:
            truncated = True

        # Render if enabled
        if self._render_mode == "human":
            self.env.render()
            if self.clock is not None:
                self.clock.tick(self.render_fps)

        # Apply reward function if provided
        if self.reward_fn:
            scalar_reward = self.reward_fn(mo_reward)
            info["preference_weights"] = self.reward_fn.preference_fn.last_weights
            return obs, scalar_reward, terminated, truncated, info

        return obs, mo_reward, terminated, truncated, info

    def _clear_all_treasures(self, unwrapped_env):
        """Remove all treasures from the environment by setting them to 0."""
        # Iterate through the entire map and set all positive values (treasures) to 0
        for i in range(unwrapped_env.sea_map.shape[0]):
            for j in range(unwrapped_env.sea_map.shape[1]):
                if unwrapped_env.sea_map[i, j] > 0:
                    unwrapped_env.sea_map[i, j] = 0.0

    def reset(self, **kwargs) -> Tuple[np.ndarray, dict]:
        # Reset treasure counter and timestep
        self.collected_treasures = 0
        self.timestep = 0

        if self.reward_fn:
            self.reward_fn.reset()

        # Reset the environment first
        obs, info = self.env.reset(**kwargs)

        # Initialize environment map (randomly select dimension in each treasure)
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

        # Convert observation to local view if enabled
        if self.use_local_obs:
            current_pos = unwrapped_env.current_state
            obs = self._get_local_observation(current_pos)

        return obs, info
