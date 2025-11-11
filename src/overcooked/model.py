"""
Neural network models for Overcooked multi-agent RL.
"""

import torch.nn as nn
from typing import Optional


class ActorCritic(nn.Module):
    """Actor-Critic network for PPO in Overcooked."""

    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()

        # Shared feature extractor
        self.shared = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        # Actor head
        self.actor = nn.Linear(hidden_dim, action_dim)

        # Critic head
        self.critic = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        features = self.shared(x)
        return features

    def act(self, x):
        features = self.forward(x)
        logits = self.actor(features)
        value = self.critic(features)
        return logits, value

    def get_value(self, x):
        features = self.forward(x)
        return self.critic(features)


class RolloutBuffer:
    """Buffer for storing rollout data."""

    def __init__(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.dones = []
        self.mo_rewards = []  # Store multi-objective rewards

    def store(
        self,
        state,
        action,
        reward,
        value,
        log_prob,
        done,
        mo_reward: Optional[list] = None,
    ):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.values.append(value)
        self.log_probs.append(log_prob)
        self.dones.append(done)
        if mo_reward is not None:
            self.mo_rewards.append(mo_reward)

    def clear(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.dones = []
        self.mo_rewards = []

    def get(self):
        return (
            self.states,
            self.actions,
            self.rewards,
            self.values,
            self.log_probs,
            self.dones,
            self.mo_rewards,
        )
