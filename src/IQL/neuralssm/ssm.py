import numpy as np
import torch
import torch.nn as nn


def compute_mismatch(vec1, vec2, eps: float = 1e-8):
    """
    Compute mismatch between two vectors (works with both numpy and torch).

    Args:
        vec1: First vector (preference weights)
        vec2: Second vector (Q-values)
        eps: Small constant to avoid division by zero

    Returns:
        Mismatch value(s)
    """
    # Check if inputs are torch tensors
    if isinstance(vec1, torch.Tensor):
        vec1_norm = torch.norm(vec1, dim=-1, keepdim=True) + eps
        vec1_normalized = vec1 / vec1_norm

        vec2_norm = torch.norm(vec2, dim=-1, keepdim=True) + eps
        vec2_normalized = vec2 / vec2_norm

        # Compute squared L2 distance
        mismatch = torch.sum((vec1_normalized - vec2_normalized) ** 2, dim=-1)
    else:
        # numpy version
        vec1_norm = np.linalg.norm(vec1, axis=-1, keepdims=True) + eps
        vec1_normalized = vec1 / vec1_norm

        vec2_norm = np.linalg.norm(vec2, axis=-1, keepdims=True) + eps
        vec2_normalized = vec2 / vec2_norm

        # Compute squared L2 distance
        mismatch = np.sum((vec1_normalized - vec2_normalized) ** 2, axis=-1)

    return mismatch


def compute_margin(
    preferences: torch.Tensor,
    q_values: torch.Tensor,
    expert_actions: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Compute margins for batch of timesteps.

    Args:
        preferences: [T, n_objectives]
        q_values: [T, action_dim, n_objectives]
        expert_actions: [T]
        eps: Small constant

    Returns:
        margins: [T]
    """
    action_dim = q_values.shape[1]

    # Normalize preferences: [T, n_objectives]
    pref_norm = torch.norm(preferences, dim=-1, keepdim=True) + eps
    pref_normalized = preferences / pref_norm

    # Normalize Q-values: [T, action_dim, n_objectives]
    q_norm = torch.norm(q_values, dim=-1, keepdim=True) + eps
    q_normalized = q_values / q_norm

    # Compute mismatches for all actions: [T, action_dim]
    # pref_normalized: [T, 1, n_objectives], q_normalized: [T, action_dim, n_objectives]
    pref_expanded = pref_normalized.unsqueeze(1)  # [T, 1, n_objectives]
    mismatches = torch.sum(
        (pref_expanded - q_normalized) ** 2, dim=-1
    )  # [T, action_dim]

    # Get expert mismatches: [T]
    expert_mismatches = mismatches.gather(1, expert_actions.unsqueeze(1)).squeeze(1)

    # Compute mean of other mismatches: [T]
    # Create mask for non-expert actions
    mask = torch.ones_like(mismatches, dtype=torch.bool)
    mask.scatter_(1, expert_actions.unsqueeze(1), False)

    # Sum over non-expert actions and divide by (action_dim - 1)
    other_mismatch_sum = (mismatches * mask).sum(dim=1)
    mean_other_mismatches = other_mismatch_sum / (action_dim - 1)

    # Compute margins
    margins = -(expert_mismatches - mean_other_mismatches)

    return margins


class MambaSSM(nn.Module):
    """
    Mamba State Space Model for preference weight prediction.

    Architecture:
    - Input: full trajectory sequences
    - Selective state-space mechanism with input-dependent parameters
    - Output: logits -> softmax -> preference weights (ensures simplex constraint)
    - Training: sequence-wise (no step-by-step update needed)

    Note: Does not inherit from StateSpaceModel as it uses sequence-wise processing
    rather than step-by-step predict/update interface.
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        n_objectives: int,
        hidden_dim: int = 128,
        learning_rate: float = 1e-3,
        device: str = "cuda",
    ):
        """
        Initialize Mamba SSM for sequence-wise processing.

        Args:
            obs_dim: Observation dimension
            action_dim: Action dimension (not used in input, but needed for interface)
            n_objectives: Number of objectives
            hidden_dim: Hidden dimension for Mamba
            learning_rate: Learning rate for optimizer
            device: Device to run on
        """
        super().__init__()

        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.n_objectives = n_objectives
        self.hidden_dim = hidden_dim
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

        # Input projection
        self.input_proj = nn.Linear(self.obs_dim, self.hidden_dim, bias=False)

        # Projections for input-dependent parameters (selective mechanism)
        self.B_proj = nn.Linear(self.obs_dim, self.hidden_dim, bias=False)
        self.C_proj = nn.Linear(self.obs_dim, self.hidden_dim, bias=False)
        self.delta_proj = nn.Linear(self.obs_dim, self.hidden_dim, bias=False)

        # SSM parameters
        self.A_log = nn.Parameter(torch.randn(self.hidden_dim))

        # Output projection
        self.out_proj = nn.Linear(self.hidden_dim, self.n_objectives)

        # Skip connection projection (if obs_dim != n_objectives)
        if self.obs_dim != self.n_objectives:
            self.skip_proj = nn.Linear(self.obs_dim, self.n_objectives, bias=False)
        else:
            self.skip_proj = None

        # Move to device
        self.to(self.device)

        # Initialize parameters with Xavier initialization
        self.reset_params()

        # Optimizer
        self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)

    def forward(self, x):
        """
        Forward pass through Mamba SSM for entire sequence.

        Args:
            x: Input sequence [T, obs_dim]

        Returns:
            output: Logits [T, n_objectives]
        """
        # Process entire sequence at once
        x_state = self.input_proj(x)  # [T, hidden_dim]
        B = self.B_proj(x)  # [T, hidden_dim]
        C = self.C_proj(x)  # [T, hidden_dim]
        delta = torch.sigmoid(self.delta_proj(x))  # [T, hidden_dim]
        A = -torch.exp(self.A_log.float())  # [hidden_dim]

        # Compute state evolution for entire sequence
        T = x.shape[0]
        h = torch.zeros(self.hidden_dim, device=x.device)  # Initial hidden state
        h_list = []

        for t in range(T):
            A_bar = torch.exp(delta[t] * A)  # [hidden_dim]
            h = A_bar * h + B[t] * x_state[t]  # [hidden_dim]
            h_list.append(h)

        h_seq = torch.stack(h_list)  # [T, hidden_dim]
        y = C * h_seq  # [T, hidden_dim]

        output = self.out_proj(y)  # [T, n_objectives]

        # Skip connection
        if self.skip_proj is not None:
            output = output + self.skip_proj(x)
        else:
            output = output + x

        return output

    def predict(self, observations: np.ndarray) -> np.ndarray:
        """
        Predict preferences for entire trajectory sequence.

        Args:
            observations: [T, obs_dim] - full trajectory observations

        Returns:
            preferences: [T, n_objectives] - predictions for all timesteps
        """
        obs_tensor = torch.FloatTensor(observations).to(self.device)  # [T, obs_dim]

        with torch.no_grad():
            logits = self.forward(obs_tensor)  # [T, n_objectives]
            preferences = torch.softmax(logits, dim=-1)  # [T, n_objectives]

        return preferences.cpu().numpy()

    def reset_params(self):
        """Initialize model parameters with Xavier/Kaiming initialization."""

        def init_weights(m):
            if isinstance(m, nn.Linear):
                # Xavier uniform initialization for linear layers
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        self.apply(init_weights)

        # Reinitialize A_log parameter
        nn.init.normal_(self.A_log, mean=0.0, std=1.0)

    def train(
        self,
        observations: np.ndarray,
        actions: np.ndarray,
        q_values_all: np.ndarray,
    ) -> torch.Tensor:
        """
        Train on full trajectory sequence.

        Args:
            observations: [T, obs_dim]
            actions: [T] - expert actions
            q_values_all: [T, action_dim, n_objectives]

        Returns:
            loss: Scalar loss for entire trajectory
        """
        obs_tensor = torch.FloatTensor(observations).to(self.device)  # [T, obs_dim]
        actions_tensor = torch.LongTensor(actions).to(self.device)  # [T]
        q_values_tensor = torch.FloatTensor(q_values_all).to(
            self.device
        )  # [T, action_dim, n_objectives]

        # Forward pass through entire sequence
        logits = self.forward(obs_tensor)  # [T, n_objectives]
        preferences = torch.softmax(logits, dim=-1)  # [T, n_objectives]

        # Compute margin-based loss
        margins = compute_margin(preferences, q_values_tensor, actions_tensor)  # [T]

        # Loss is negative margin (we want to maximize margin)
        loss = -margins.mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss
