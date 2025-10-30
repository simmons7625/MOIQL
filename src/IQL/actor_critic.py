import torch
import torch.nn as nn


class ActorCritic(nn.Module):
    """Q-network for object dimensional Soft IQ-Learn (policy derived from Q-values via softmax)."""

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

        # Critic head - outputs Q-values for each (action, objective) pair
        self.critic = nn.Linear(hidden_dim, action_dim * n_objects)

        self.action_dim = action_dim
        self.n_objects = n_objects

        # Initialize parameters with Xavier initialization
        self.reset_params()

    def reset_params(self):
        """Initialize model parameters with Xavier/Kaiming initialization."""

        def init_weights(m):
            if isinstance(m, nn.Linear):
                # Xavier uniform initialization for linear layers
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        self.apply(init_weights)

    def forward(self, x):
        features = self.shared(x)
        return features

    def act(self, x, preference_weights=None):
        """
        Get Q-values and derive policy from them.

        Args:
            x: [batch, obs_dim]
            preference_weights: [batch, n_objectives] - optional preference weights for computing scalar Q

        Returns:
            logits: [batch, action_dim] - policy logits derived from Q-values
            q_values: [batch, action_dim, n_objectives] - Q-values for each action and objective
        """
        features = self.forward(x)
        q_flat = self.critic(features)  # [batch, action_dim * n_objectives]

        # Reshape to [batch, action_dim, n_objectives]
        batch_size = x.shape[0]
        q_values = q_flat.view(batch_size, self.action_dim, self.n_objects)

        # Derive policy logits from Q-values
        # If preference weights provided, use them to compute scalar Q
        if preference_weights is not None:
            # q_values: [batch, action_dim, n_objectives]
            # preference_weights: [batch, n_objectives]
            # Compute scalar Q for each action: sum over objectives weighted by preferences
            logits = torch.einsum(
                "bao,bo->ba", q_values, preference_weights
            )  # [batch, action_dim]
        else:
            # Use mean over objectives as default
            logits = q_values.mean(dim=-1)  # [batch, action_dim]

        return logits, q_values
