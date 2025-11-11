"""
Overcooked multi-agent environment wrapper with multi-objective rewards.
"""

import numpy as np
from typing import Tuple, Optional, Dict, Any
from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld, OvercookedState
from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv
from overcooked_ai_py.mdp.actions import Action
from .reward_function import RewardFunction


class OvercookedWrapper:
    """
    Wrapper for Overcooked-AI environment with multi-objective rewards.

    Multi-objective rewards:
    1. Task completion: dishes served (sparse, +20 per dish)
    2. Time efficiency: negative time penalty (-0.1 per step)
    3. Coordination quality: collision avoidance, smooth handoffs (shaped)
    """

    def __init__(
        self,
        layout_name: str = "cramped_room",
        horizon: int = 400,
        reward_fn: Optional[RewardFunction] = None,
        agent_idx: int = 0,  # Which agent this wrapper controls (0 or 1)
    ):
        """
        Initialize Overcooked wrapper.

        Args:
            layout_name: Name of the Overcooked layout
            horizon: Maximum episode length
            reward_fn: Optional reward function for preference-based rewards
            agent_idx: Which agent this wrapper is for (0 or 1)
        """
        # Create MDP and environment
        self.mdp = OvercookedGridworld.from_layout_name(layout_name)
        self.env = OvercookedEnv.from_mdp(self.mdp, horizon=horizon)

        self.n_objectives = 3  # task, efficiency, coordination
        self.reward_fn = reward_fn
        self.agent_idx = agent_idx
        self.other_idx = 1 - agent_idx
        self.layout_name = layout_name
        self.horizon = horizon

        # Episode state
        self.timestep = 0
        self.total_dishes_served = 0
        self.prev_state = None

        # Observation and action space info
        self.n_actions = len(Action.ALL_ACTIONS)  # 6 actions

    def reset(self) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Reset environment.

        Returns:
            observation: Flattened state observation for the agent
            info: Additional information
        """
        self.env.reset()
        self.timestep = 0
        self.total_dishes_served = 0
        self.prev_state = self.env.state

        if self.reward_fn is not None:
            self.reward_fn.reset()

        obs = self._get_observation(self.env.state)
        info = {"state": self.env.state}

        return obs, info

    def step(
        self, joint_action: Tuple[int, int]
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Take a step in the environment.

        Args:
            joint_action: Tuple of actions for both agents (action_indices)

        Returns:
            observation: Next observation
            reward: Scalar reward (if reward_fn provided) or multi-objective
            terminated: Whether episode ended naturally
            truncated: Whether episode was truncated
            info: Additional information including mo_reward
        """
        # Convert action indices to Overcooked actions
        action_0 = Action.ALL_ACTIONS[joint_action[0]]
        action_1 = Action.ALL_ACTIONS[joint_action[1]]
        joint_action_tuple = (action_0, action_1)

        # Take step in environment
        next_state, sparse_reward, done, info = self.env.step(joint_action_tuple)

        self.timestep += 1

        # Compute multi-objective reward
        mo_reward = self._compute_mo_reward(
            self.prev_state, next_state, sparse_reward, info
        )

        # Compute scalar reward if reward function provided
        if self.reward_fn is not None:
            scalar_reward = self.reward_fn(mo_reward)
        else:
            scalar_reward = float(np.sum(mo_reward))

        # Get observation
        obs = self._get_observation(next_state)

        # Check termination
        terminated = done
        truncated = False
        if self.timestep >= self.horizon:
            truncated = True
            terminated = False

        # Update state
        self.prev_state = next_state

        # Prepare info
        info_out = {
            "state": next_state,
            "mo_reward": mo_reward,
            "sparse_reward": sparse_reward,
            "timestep": self.timestep,
            "dishes_served": self.total_dishes_served,
            "preference_weights": (
                self.reward_fn.preference_fn.last_weights
                if self.reward_fn is not None
                else None
            ),
        }

        return obs, scalar_reward, terminated, truncated, info_out

    def _compute_mo_reward(
        self,
        prev_state: OvercookedState,
        next_state: OvercookedState,
        sparse_reward: float,
        info: Dict,
    ) -> np.ndarray:
        """
        Compute multi-objective reward.

        Returns:
            mo_reward: [task_reward, efficiency_reward, coordination_reward]
        """
        # 1. Task completion: dishes served (sparse reward from environment)
        task_reward = float(sparse_reward)
        if task_reward > 0:
            self.total_dishes_served += 1

        # 2. Time efficiency: negative time penalty
        efficiency_reward = -0.1

        # 3. Coordination quality: detect collisions and cooperation
        coordination_reward = self._compute_coordination_reward(
            prev_state, next_state, info
        )

        mo_reward = np.array(
            [task_reward, efficiency_reward, coordination_reward], dtype=np.float32
        )

        return mo_reward

    def _compute_coordination_reward(
        self, prev_state: OvercookedState, next_state: OvercookedState, info: Dict
    ) -> float:
        """
        Compute coordination quality reward.

        Penalizes:
        - Collisions (agents trying to move to same location)
        - Inefficient movements

        Rewards:
        - Successful handoffs
        - Staying out of each other's way
        """
        coordination = 0.0

        # Check if agents collided (both at same position)
        if next_state is not None and hasattr(next_state, "players"):
            pos_0 = next_state.players[0].position
            pos_1 = next_state.players[1].position

            # Collision penalty
            if pos_0 == pos_1:
                coordination -= 1.0

            # Small reward for being apart (encourages spatial distribution)
            manhattan_dist = abs(pos_0[0] - pos_1[0]) + abs(pos_0[1] - pos_1[1])
            if manhattan_dist >= 2:
                coordination += 0.05

        return coordination

    def _get_observation(self, state: OvercookedState) -> np.ndarray:
        """
        Convert Overcooked state to observation vector.

        For now, we'll create a simple flattened representation:
        - Agent position (2 values)
        - Other agent position (2 values)
        - Agent holding item (one-hot, 4 values: nothing, onion, dish, soup)
        - Other agent holding item (one-hot, 4 values)
        - Pot states (simplified, e.g., 4 values per pot, assume 2 pots)
        - Order status (1 value: number of orders remaining)

        Total: 2 + 2 + 4 + 4 + 8 + 1 = 21 dimensions (can be expanded)
        """
        if state is None:
            # Return zero observation on reset
            return np.zeros(21, dtype=np.float32)

        obs_parts = []

        # Agent positions (normalized to [0, 1])
        my_pos = state.players[self.agent_idx].position
        other_pos = state.players[self.other_idx].position
        grid_shape = self.mdp.shape

        obs_parts.extend(
            [
                my_pos[0] / grid_shape[0],
                my_pos[1] / grid_shape[1],
                other_pos[0] / grid_shape[0],
                other_pos[1] / grid_shape[1],
            ]
        )

        # Agent held objects (one-hot encoding)
        def encode_held_object(player):
            # Encoding: [nothing, onion, dish, soup]
            held = [0.0, 0.0, 0.0, 0.0]
            if player.held_object is None:
                held[0] = 1.0
            elif player.held_object.name == "onion":
                held[1] = 1.0
            elif player.held_object.name == "dish":
                held[2] = 1.0
            elif player.held_object.name == "soup":
                held[3] = 1.0
            return held

        obs_parts.extend(encode_held_object(state.players[self.agent_idx]))
        obs_parts.extend(encode_held_object(state.players[self.other_idx]))

        # Pot states (simplified: cooking progress for up to 2 pots)
        # Each pot: [has_onions, num_onions/3, is_cooking, cook_time/20]
        pot_states = []
        if hasattr(state, "objects") and state.objects:
            pots = [
                obj
                for pos, obj in state.objects.items()
                if hasattr(obj, "name") and obj.name == "soup"
            ]
            for i in range(2):  # Assume max 2 pots
                if i < len(pots):
                    pot = pots[i]
                    has_onions = 1.0 if pot.ingredients else 0.0
                    num_onions = len(pot.ingredients) / 3.0 if pot.ingredients else 0.0
                    is_cooking = 1.0 if pot.is_cooking else 0.0
                    cook_time = pot.cook_time / 20.0 if pot.is_cooking else 0.0
                    pot_states.extend([has_onions, num_onions, is_cooking, cook_time])
                else:
                    pot_states.extend([0.0, 0.0, 0.0, 0.0])
        else:
            pot_states = [0.0] * 8

        obs_parts.extend(pot_states)

        # Order status (number of remaining orders, normalized)
        num_orders = 1.0  # Simplified: assume 1 order at a time
        obs_parts.append(num_orders)

        obs = np.array(obs_parts, dtype=np.float32)
        return obs

    @property
    def observation_space_dim(self) -> int:
        """Get observation space dimensionality."""
        return 21

    @property
    def action_space_dim(self) -> int:
        """Get action space dimensionality."""
        return self.n_actions
