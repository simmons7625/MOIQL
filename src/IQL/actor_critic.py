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

        # Critic head
        self.critic = nn.Linear(hidden_dim, n_objects)

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
