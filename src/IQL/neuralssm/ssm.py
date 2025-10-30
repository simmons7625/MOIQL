"""
State Space Models for Preference Weight Prediction.

State space formulation:
    h_t = f(h_{t-1}, x_t) + noise_1,t    (state transition)
    y_t = g(h_t) + noise_2,t              (observation)

For preference weight estimation:
    x_t: (s_t, a_t) - state and action
    h_t: preference weights
    g: mismatch function

Mismatch function:
    mismatch = ||preference/|preference| - q_values_all/|q_values_all|||^2

Where:
    - preference: h_t (preference weights)
    - q_values_all: expert Q-values from demonstrations

Implements three approaches:
1. Particle Filter: Non-parametric sequential Monte Carlo
2. Extended Kalman Filter (EKF): Gaussian approximation with linearization
3. Neural SSM: GRU-based neural state space model
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Callable
from abc import ABC, abstractmethod


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


def compute_margin_vector(
    preference: np.ndarray,
    q_values: np.ndarray,
    eps: float = 1e-8,
) -> np.ndarray:
    """
    Compute margin for all actions.

    For each action a, compute: margin_a = -(mismatch(q_a, w) - mean_other_mismatch)
    Higher margin means the action is more aligned with the preference.

    Args:
        preference: Preference weights [n_objectives]
        q_values: Q-values for all actions [action_dim, n_objectives]
        eps: Small constant to avoid division by zero

    Returns:
        Margin vector [action_dim]
    """
    # Normalize preference
    pref_norm = np.linalg.norm(preference) + eps
    pref_normalized = preference / pref_norm

    # Compute mismatch for each action
    action_dim = q_values.shape[0]
    mismatches = np.zeros(action_dim)

    for a in range(action_dim):
        q_val = q_values[a]  # [n_objectives]
        q_norm = np.linalg.norm(q_val) + eps
        q_normalized = q_val / q_norm

        # Mismatch = ||w_norm - q_norm||^2
        mismatch = np.sum((pref_normalized - q_normalized) ** 2)
        mismatches[a] = mismatch

    # For each action, compute margin relative to others
    margins = np.zeros(action_dim)
    for a in range(action_dim):
        expert_mismatch = mismatches[a]
        other_mask = np.ones(action_dim, dtype=bool)
        other_mask[a] = False
        mean_other_mismatch = np.mean(mismatches[other_mask])
        margins[a] = -(expert_mismatch - mean_other_mismatch)

    return margins


def compute_margin_objective(
    preference: np.ndarray,
    q_values: np.ndarray,
    expert_action: int,
    eps: float = 1e-8,
) -> float:
    """
    Compute margin objective for a specific expert action.

    This is a convenience wrapper around compute_margin_vector that returns
    the margin for a single action.

    Args:
        preference: Preference weights [n_objectives]
        q_values: Q-values for all actions [action_dim, n_objectives]
        expert_action: Expert's chosen action index
        eps: Small constant to avoid division by zero

    Returns:
        Margin objective value for expert_action (higher is better)
    """
    margins = compute_margin_vector(preference, q_values, eps)
    return margins[expert_action]


def softmax_stable(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Numerically stable softmax."""
    xm = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(xm)
    return e / np.sum(e, axis=axis, keepdims=True)


def softmax_jacobian(s: np.ndarray) -> np.ndarray:
    """
    Compute Jacobian of softmax function.

    Args:
        s: Softmax output [n]

    Returns:
        Jacobian matrix [n, n] where J[i,j] = ds_i/dx_j
    """
    s = s.reshape(-1, 1)
    return np.diagflat(s.flatten()) - (s @ s.T)


def numeric_jacobian(f: Callable, x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """
    Compute Jacobian numerically using finite differences.

    Args:
        f: Function that takes x [n] and returns y [m]
        x: Input point [n]
        eps: Finite difference step size

    Returns:
        Jacobian matrix [m, n] where J[i,j] = dy_i/dx_j
    """
    y0 = f(x)
    m = y0.shape[0] if y0.ndim > 0 else 1
    n = x.shape[0]
    J = np.zeros((m, n))

    for j in range(n):
        x_plus = x.copy()
        x_plus[j] += eps
        y_plus = f(x_plus)
        J[:, j] = (y_plus - y0) / eps

    return J


class StateSpaceModel(ABC):
    """Abstract base class for state space models."""

    @abstractmethod
    def predict(self, observation: np.ndarray, hidden_state: np.ndarray) -> np.ndarray:
        """
        Predict preference weights given observation and action.

        Args:
            observation: Current observation
            hidden_state: Current hidden state

        Returns:
            Predicted preference weights
        """
        pass

    @abstractmethod
    def update(
        self,
        observation: np.ndarray,
        action: np.ndarray,
        q_values_all: np.ndarray,
    ):
        """
        Update the model to maximize margin: -(mismatch(q_expert, w) - mean(mismatch(q_other, w))).

        Args:
            observation: Current observation
            action: Action taken (discrete) - expert action
            next_observation: Next observation
            q_values_all: Q-values for all actions [action_dim, n_objectives]
        """
        pass

    @abstractmethod
    def reset(self):
        """Reset the model state."""
        pass


class Mamba(nn.Module):
    """Simplified Mamba block with selective state-space model."""

    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        # Input projection
        self.input_proj = nn.Linear(self.input_dim, self.hidden_dim, bias=False)

        # Projections for input-dependent parameters (selective mechanism)
        self.B_proj = nn.Linear(self.input_dim, self.hidden_dim, bias=False)
        self.C_proj = nn.Linear(self.input_dim, self.hidden_dim, bias=False)
        self.delta_proj = nn.Linear(self.input_dim, self.hidden_dim, bias=False)

        # SSM parameters
        self.A_log = nn.Parameter(torch.randn(self.hidden_dim))

        # Output projection
        self.out_proj = nn.Linear(self.hidden_dim, self.output_dim)

        # Skip connection projection (if input_dim != output_dim)
        if self.input_dim != self.output_dim:
            self.skip_proj = nn.Linear(self.input_dim, self.output_dim, bias=False)
        else:
            self.skip_proj = None

    def forward(self, x, h=None):
        batch = x.shape[0]
        if h is None:
            h = torch.zeros(batch, self.hidden_dim, device=x.device)

        x_state = self.input_proj(x)
        B = self.B_proj(x)
        C = self.C_proj(x)
        delta = torch.sigmoid(self.delta_proj(x))
        A = -torch.exp(self.A_log.float())

        A_bar = torch.exp(delta * A)
        h = A_bar * h + B * x_state
        y = C * h

        output = self.out_proj(y)

        # Skip connection
        if self.skip_proj is not None:
            output = output + self.skip_proj(x)
        else:
            output = output + x

        return output, h


class MambaSSM(StateSpaceModel):
    """
    Mamba State Space Model for preference weight prediction.

    Architecture:
    - Input: full trajectory sequences
    - Selective state-space mechanism with input-dependent parameters
    - Output: logits -> softmax -> preference weights (ensures simplex constraint)
    - Training: sequence-wise (no step-by-step update needed)
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
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.n_objectives = n_objectives
        self.hidden_dim = hidden_dim
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

        # Build model
        self.model = self._build_model().to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)

    def _build_model(self) -> nn.Module:
        """Build Mamba-based neural SSM."""
        return Mamba(
            input_dim=self.obs_dim,
            hidden_dim=self.hidden_dim,
            output_dim=self.n_objectives,
        )

    def _compute_margin_batch(
        self,
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

    def predict_sequence(self, observations: np.ndarray) -> np.ndarray:
        """
        Predict preferences for entire trajectory sequence.

        Args:
            observations: [T, obs_dim] - full trajectory observations

        Returns:
            preferences: [T, n_objectives] - predictions for all timesteps
        """
        obs_tensor = torch.FloatTensor(observations).to(self.device)  # [T, obs_dim]

        with torch.no_grad():
            T = obs_tensor.shape[0]
            h = None
            logits_list = []

            # Process sequence step by step to maintain hidden state
            for t in range(T):
                obs_t = obs_tensor[t : t + 1]  # [1, obs_dim]
                logits_t, h = self.model(obs_t, h)  # [1, n_objectives]
                logits_list.append(logits_t)

            logits = torch.cat(logits_list, dim=0)  # [T, n_objectives]
            preferences = torch.softmax(logits, dim=-1)  # [T, n_objectives]

        return preferences.cpu().numpy()

    def train_trajectory(
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

        T = obs_tensor.shape[0]
        h = None
        logits_list = []

        # Forward pass through sequence
        for t in range(T):
            obs_t = obs_tensor[t : t + 1]  # [1, obs_dim]
            logits_t, h = self.model(obs_t, h)  # [1, n_objectives]
            logits_list.append(logits_t)

        logits = torch.cat(logits_list, dim=0)  # [T, n_objectives]
        preferences = torch.softmax(logits, dim=-1)  # [T, n_objectives]

        # Compute margin-based loss
        margins = self._compute_margin_batch(
            preferences, q_values_tensor, actions_tensor
        )  # [T]

        # Loss is negative margin (we want to maximize margin)
        loss = -margins.mean()

        return loss

    def step(self, loss: torch.Tensor):
        """Perform backward pass and optimizer step."""
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
