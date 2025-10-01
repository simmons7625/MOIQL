import torch
import torch.nn as nn
import numpy as np
from typing import Optional


class MultiObjectiveQNetwork(nn.Module):
    """Q-Network for multi-objective reinforcement learning."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        n_objectives: int,
        hidden_dim: int = 256,
        n_hidden_layers: int = 2,
    ):
        super().__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.n_objectives = n_objectives

        layers = []
        input_dim = state_dim

        for _ in range(n_hidden_layers):
            layers.extend([nn.Linear(input_dim, hidden_dim), nn.ReLU()])
            input_dim = hidden_dim

        layers.append(nn.Linear(hidden_dim, action_dim * n_objectives))

        self.network = nn.Sequential(*layers)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through Q-network.

        Args:
            state: State tensor of shape (batch_size, state_dim)

        Returns:
            Q-values of shape (batch_size, action_dim, n_objectives)
        """
        q_values = self.network(state)
        return q_values.view(-1, self.action_dim, self.n_objectives)


class RewardNetwork(nn.Module):
    """Neural network to learn reward function for inverse Q-learning."""

    def __init__(
        self, state_dim: int, action_dim: int, n_objectives: int, hidden_dim: int = 128
    ):
        super().__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.n_objectives = n_objectives

        self.network = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_objectives),
        )

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """
        Predict reward for state-action pair.

        Args:
            state: State tensor of shape (batch_size, state_dim)
            action: Action tensor of shape (batch_size, action_dim)

        Returns:
            Reward predictions of shape (batch_size, n_objectives)
        """
        x = torch.cat([state, action], dim=-1)
        return self.network(x)


class InverseQLearningLoss(nn.Module):
    """Loss function for inverse Q-learning."""

    def __init__(self, margin: float = 1.0):
        super().__init__()
        self.margin = margin

    def forward(
        self,
        expert_q_values: torch.Tensor,
        expert_actions: torch.Tensor,
        other_q_values: torch.Tensor,
    ) -> torch.Tensor:
        """
        Calculate inverse Q-learning loss.

        Args:
            expert_q_values: Q-values for expert states (batch_size, action_dim, n_objectives)
            expert_actions: Expert actions (batch_size,)
            other_q_values: Q-values for all actions (batch_size, action_dim, n_objectives)

        Returns:
            Loss value
        """
        batch_size = expert_actions.size(0)

        expert_q = expert_q_values[torch.arange(batch_size), expert_actions]

        max_other_q, _ = torch.max(other_q_values.sum(dim=-1), dim=-1)
        expert_total_q = expert_q.sum(dim=-1)

        loss = torch.clamp(self.margin + max_other_q - expert_total_q, min=0.0)

        return loss.mean()


class MultiObjectivePolicy:
    """Policy for multi-objective Q-learning."""

    def __init__(
        self,
        q_network: MultiObjectiveQNetwork,
        epsilon: float = 0.1,
        weights: Optional[np.ndarray] = None,
    ):
        self.q_network = q_network
        self.epsilon = epsilon
        self.weights = (
            weights
            if weights is not None
            else np.ones(q_network.n_objectives) / q_network.n_objectives
        )

    def act(self, state: np.ndarray, deterministic: bool = False) -> int:
        """Select action using epsilon-greedy policy with weighted scalarization."""

        if not deterministic and np.random.rand() < self.epsilon:
            return np.random.randint(self.q_network.action_dim)

        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            q_values = self.q_network(state_tensor).squeeze(0).numpy()

            scalarized_q = q_values @ self.weights

            return int(np.argmax(scalarized_q))

    def set_weights(self, weights: np.ndarray):
        """Update preference weights."""
        self.weights = weights
