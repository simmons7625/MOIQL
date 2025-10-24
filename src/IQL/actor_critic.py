import torch.nn as nn


class ActorCritic(nn.Module):
    """Actor-Critic network for object dimensional Soft IQ-Learn."""

    def __init__(
        self, obs_dim: int, action_dim: int, hidden_dim: int = 256, n_objects: int = 2
    ):
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

        # Critic head - outputs Q-values for each (action, objective) pair
        self.critic = nn.Linear(hidden_dim, action_dim * n_objects)

        self.action_dim = action_dim
        self.n_objects = n_objects

    def forward(self, x):
        features = self.shared(x)
        return features

    def act(self, x):
        """
        Get policy logits and Q-values.

        Args:
            x: [batch, obs_dim]

        Returns:
            logits: [batch, action_dim] - policy logits
            q_values: [batch, action_dim, n_objectives] - Q-values for each action and objective
        """
        features = self.forward(x)
        logits = self.actor(features)  # [batch, action_dim]
        q_flat = self.critic(features)  # [batch, action_dim * n_objectives]

        # Reshape to [batch, action_dim, n_objectives]
        batch_size = x.shape[0]
        q_values = q_flat.view(batch_size, self.action_dim, self.n_objects)

        return logits, q_values
