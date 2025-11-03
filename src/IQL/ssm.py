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
from typing import Optional, Callable
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

    For each action a, compute: margin_a = max_other_mismatch - mismatch(q_a, w)
    Higher margin means the action is more aligned with the preference than the best alternative.

    This uses max instead of mean to enforce a stricter criterion: each action
    is compared against the BEST competing action, not the average.

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

    # For each action, compute margin relative to best alternative
    margins = np.zeros(action_dim)
    for a in range(action_dim):
        action_mismatch = mismatches[a]
        other_mask = np.ones(action_dim, dtype=bool)
        other_mask[a] = False
        max_other_mismatch = np.max(mismatches[other_mask])
        margins[a] = max_other_mismatch - action_mismatch

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


def compute_margin(
    preference_weights: torch.Tensor,
    q_values_all: torch.Tensor,
    actions: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Compute margin for expert actions (PyTorch version for gradient computation).

    Margin = max_other_mismatch - expert_mismatch
    Higher margin means expert action is more aligned with preference than the best alternative.

    This uses max instead of mean to enforce a stricter criterion: the expert action
    must be better than the BEST competing action, not just better than average.

    Args:
        preference_weights: Preference weights [batch, n_objectives]
        q_values_all: Q-values for all actions [batch, action_dim, n_objectives]
        actions: Expert actions [batch]
        eps: Small constant to avoid division by zero

    Returns:
        Margins [batch] (higher is better)
    """
    batch_size, action_dim, n_objectives = q_values_all.shape

    # Normalize preference weights
    pref_norm = torch.norm(preference_weights, dim=-1, keepdim=True) + eps
    pref_normalized = preference_weights / pref_norm  # [batch, n_objectives]

    # Compute mismatch for all actions
    # q_values_all: [batch, action_dim, n_objectives]
    q_norm = (
        torch.norm(q_values_all, dim=-1, keepdim=True) + eps
    )  # [batch, action_dim, 1]
    q_normalized = q_values_all / q_norm  # [batch, action_dim, n_objectives]

    # Expand preferences for broadcasting
    pref_expanded = pref_normalized.unsqueeze(1)  # [batch, 1, n_objectives]

    # Compute mismatch for all actions: ||pref_norm - q_norm||^2
    mismatches = torch.sum(
        (pref_expanded - q_normalized) ** 2, dim=-1
    )  # [batch, action_dim]

    # Get expert mismatch
    actions_idx = actions.long().unsqueeze(-1)  # [batch, 1]
    expert_mismatch = torch.gather(mismatches, 1, actions_idx).squeeze(-1)  # [batch]

    # Compute max of other mismatches (excluding expert action)
    # Create mask for other actions
    batch_indices = torch.arange(batch_size, device=q_values_all.device)
    mask = torch.ones_like(mismatches, dtype=torch.bool)  # [batch, action_dim]
    mask[batch_indices, actions.long()] = False

    # Get maximum mismatch among other actions
    # Set expert action mismatch to -inf so it's not selected
    masked_mismatches = mismatches.clone()
    masked_mismatches[~mask] = float("-inf")
    max_other_mismatch = torch.max(masked_mismatches, dim=1)[0]  # [batch]

    # Margin = max_other_mismatch - expert_mismatch
    # Positive margin means expert action has lower mismatch than the best alternative
    margins = max_other_mismatch - expert_mismatch  # [batch]

    return margins


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
    def predict(self) -> np.ndarray:
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


class ParticleFilter(StateSpaceModel):
    """
    Particle Filter for preference weight prediction.

    Uses Sequential Monte Carlo to maintain a distribution over preference weights.
    """

    def __init__(
        self,
        n_objectives: int,
        n_particles: int = 1000,
        process_noise: float = 0.01,
        observation_noise: float = 0.1,
        transition_fn: Optional[Callable] = None,
        seed: int = 42,
    ):
        """
        Initialize Particle Filter.

        Args:
            n_objectives: Number of objectives
            n_particles: Number of particles
            process_noise: Process noise std (for random walk dynamics)
            observation_noise: Observation noise std (for likelihood)
            transition_fn: Optional custom transition function for particles
            seed: Random seed
        """
        self.n_objectives = n_objectives
        self.n_particles = n_particles
        self.process_noise = process_noise
        self.observation_noise = observation_noise
        self.transition_fn = transition_fn
        self.rng = np.random.RandomState(seed)

        # Initialize particles and weights to uniform distribution
        self.reset()

    def _initialize_particles(self) -> np.ndarray:
        """
        Initialize particles uniformly on probability simplex.

        Uses Dirichlet(1,1,...,1) which is the uniform distribution over the simplex.
        """
        particles = self.rng.dirichlet(
            np.ones(self.n_objectives), size=self.n_particles
        )
        return particles

    def reset(self):
        """Reset particle filter."""
        self.particles = self._initialize_particles()
        self.weights = np.ones(self.n_particles) / self.n_particles
        self.current_estimate = np.mean(self.particles, axis=0)

    def _project_to_simplex(self, weights: np.ndarray) -> np.ndarray:
        """Project weights onto probability simplex (sum to 1, all positive)."""
        weights = np.maximum(weights, 0)
        weights_sum = np.sum(weights, axis=-1, keepdims=True)
        weights_sum = np.maximum(weights_sum, 1e-8)  # Avoid division by zero
        return weights / weights_sum

    def _transition(self, particles: np.ndarray) -> np.ndarray:
        """
        Transition particles forward in time.

        Uses random walk with projection onto simplex if no custom transition is provided.
        """
        if self.transition_fn is not None:
            return self.transition_fn(particles)

        # Random walk with Gaussian noise
        noise = self.rng.randn(self.n_particles, self.n_objectives) * self.process_noise
        new_particles = particles + noise

        # Project onto simplex
        new_particles = self._project_to_simplex(new_particles)

        return new_particles

    def _resample(self):
        """Resample particles based on weights (systematic resampling)."""
        cumsum = np.cumsum(self.weights)
        cumsum[-1] = 1.0  # Ensure last value is exactly 1

        # Systematic resampling
        positions = (self.rng.rand() + np.arange(self.n_particles)) / self.n_particles
        indices = np.searchsorted(cumsum, positions)

        self.particles = self.particles[indices]
        self.weights = np.ones(self.n_particles) / self.n_particles

    def predict(self) -> np.ndarray:
        """
        Predict preference weights (return current estimate).

        Args:
            observation: Current observation
            hidden_state: Current hidden state

        Returns:
            Predicted preference weights
        """
        return np.mean(self.particles, axis=0)

    def update(
        self,
        observation: np.ndarray,
        action: np.ndarray,
        q_values_all: np.ndarray,
    ):
        # Transition particles
        self.particles = self._transition(self.particles)

        # Compute margin objective for each particle
        expert_action = int(action)
        margins = np.zeros(self.n_particles)

        for i in range(self.n_particles):
            margins[i] = compute_margin_objective(
                self.particles[i], q_values_all, expert_action
            )
        # Convert margins to fitness (softmax-like)
        margins_shifted = margins - np.max(margins)
        fitness = np.exp(margins_shifted / self.observation_noise)

        # Update weights with fitness
        self.weights *= fitness
        self.weights /= np.sum(self.weights) + 1e-8  # Normalize weights

        # Resample if effective sample size is too low
        ess = 1.0 / np.sum(self.weights**2)
        if ess < self.n_particles / 2:
            self._resample()

        return None


class ExtendedKalmanFilter(StateSpaceModel):
    """
    Extended Kalman Filter for preference weight prediction using margin-based observation model.

    State representation: logits x in R^n, where w = softmax(x) are the preference weights.
    This ensures weights stay on the probability simplex automatically.

    Observation model: p(a|w) = softmax(beta * margin(w))
    where margin(w) is computed from Q-values and preferences.
    """

    def __init__(
        self,
        n_objectives: int,
        process_noise: float = 0.01,
        observation_noise: float = 0.1,
        initial_variance: float = 0.1,
        beta: float = 5.0,
        seed: int = 42,
    ):
        """
        Initialize Extended Kalman Filter.

        Args:
            n_objectives: Number of objectives
            process_noise: Process noise std on logits
            observation_noise: Observation noise std
            initial_variance: Initial covariance diagonal value for logits
            beta: Temperature parameter for margin->probability (higher = sharper)
            seed: Random seed
        """
        self.n_objectives = n_objectives
        self.process_noise = process_noise
        self.observation_noise = observation_noise
        self.initial_variance = initial_variance
        self.beta = beta
        self.rng = np.random.RandomState(seed)
        self.obs_noise_std = observation_noise
        self._eps = 1e-8

    def predict(self) -> np.ndarray:
        """
        Predict preference weights (return current estimate from logits).

        Args:
            observation: Current observation (not used in prediction)
            hidden_state: Current hidden state (not used in prediction)

        Returns:
            Predicted preference weights = softmax(x)
        """
        return softmax_stable(self.x)

    def _compute_observation_and_jacobian(
        self, x: np.ndarray, q_values_all: np.ndarray
    ) -> tuple:
        """
        Compute predicted observation h(x) and its Jacobian H.

        h(x) = softmax(beta * margin(softmax(x)))

        Args:
            x: Logits [n_objectives]
            q_values_all: Q-values [action_dim, n_objectives]

        Returns:
            h: Predicted action probabilities [action_dim]
            H: Jacobian dh/dx [action_dim, n_objectives]
        """
        # 1) w = softmax(x) and dw/dx
        w = softmax_stable(x)  # [n_objectives]
        S_w = softmax_jacobian(w)  # [n_objectives, n_objectives]

        # 2) margin vector m(w) and dm/dw
        def margin_fn(weights):
            return compute_margin_vector(weights, q_values_all, eps=self._eps)

        m = margin_fn(w)  # [action_dim]
        dm_dw = numeric_jacobian(margin_fn, w, eps=1e-6)  # [action_dim, n_objectives]

        # 3) p = softmax(beta * m) and dp/d(beta*m)
        u = self.beta * m  # [action_dim]
        p = softmax_stable(u)  # [action_dim]
        S_p = softmax_jacobian(p)  # [action_dim, action_dim]

        # Chain: dp/du * beta (since u = beta * m)
        dp_dm = S_p * self.beta  # [action_dim, action_dim]

        # 4) Full Jacobian: H = dp/dx = dp/dm @ dm/dw @ dw/dx
        H = dp_dm @ dm_dw @ S_w  # [action_dim, n_objectives]

        return p, H

    def update(
        self,
        observation: np.ndarray,
        action: np.ndarray,
        q_values_all: np.ndarray,
    ):
        expert_action = int(action)
        action_dim = q_values_all.shape[0]

        x_pred = self.x.copy()
        P_pred = self.P + self.Q

        try:
            h_pred, H = self._compute_observation_and_jacobian(x_pred, q_values_all)
        except Exception:
            self.x = x_pred
            self.P = P_pred
            return None

        z = np.zeros(action_dim)
        z[expert_action] = 1.0

        R = np.eye(action_dim) * (self.obs_noise_std**2 + self._eps)

        y = z - h_pred

        S = H @ P_pred @ H.T + R
        S += np.eye(action_dim) * self._eps

        try:
            S_inv = np.linalg.inv(S)
        except np.linalg.LinAlgError:
            S_inv = np.linalg.pinv(S)

        K = P_pred @ H.T @ S_inv

        self.x = x_pred + K @ y

        identity_matrix = np.eye(self.n_objectives)
        IKH = identity_matrix - K @ H
        self.P = IKH @ P_pred @ IKH.T + K @ R @ K.T

        self.P = (self.P + self.P.T) / 2
        self.P += np.eye(self.n_objectives) * self._eps

        return None

    def reset(self):
        """Reset EKF to initial state."""
        # Initialize to zero -> uniform weights after softmax
        self.x = np.zeros(self.n_objectives)

        # Covariance matrix [n_objectives, n_objectives]
        self.P = np.eye(self.n_objectives) * self.initial_variance

        # Process noise covariance (random walk on logits)
        self.Q = np.eye(self.n_objectives) * (self.process_noise**2)


class KalmanFilter(StateSpaceModel):
    """
    Standard Kalman Filter for preference weight prediction using linear observation model.

    Uses a simpler linear observation model compared to EKF:
    - State representation: preference weights directly (constrained to simplex via projection)
    - Process model: w_t = w_{t-1} + process_noise (random walk with projection)
    - Observation model: Linear relationship between preference weights and observed margin

    This is simpler and faster than EKF but assumes linearity in the observation.
    """

    def __init__(
        self,
        n_objectives: int,
        process_noise: float = 0.01,
        observation_noise: float = 0.1,
        initial_variance: float = 0.1,
        seed: int = 42,
    ):
        """
        Initialize Kalman Filter.

        Args:
            n_objectives: Number of objectives
            process_noise: Process noise std on preference weights
            observation_noise: Observation noise std
            initial_variance: Initial covariance diagonal value
            seed: Random seed
        """
        self.n_objectives = n_objectives
        self.process_noise = process_noise
        self.observation_noise = observation_noise
        self.initial_variance = initial_variance
        self.rng = np.random.RandomState(seed)
        self._eps = 1e-8

    def _project_to_simplex(self, w: np.ndarray) -> np.ndarray:
        """
        Project weights onto probability simplex using euclidean projection.

        Ensures: w >= 0 and sum(w) = 1

        Args:
            w: Weights to project [n_objectives]

        Returns:
            Projected weights [n_objectives]
        """
        # Sort weights in descending order
        w_sorted = np.sort(w)[::-1]
        cumsum_w = np.cumsum(w_sorted)

        # Find rho (largest j such that w_j + (1 - cumsum) / (j+1) > 0)
        rho = None
        for j in range(self.n_objectives):
            if w_sorted[j] + (1.0 - cumsum_w[j]) / (j + 1) > 0:
                rho = j

        if rho is None:
            # Fallback to uniform
            return np.ones(self.n_objectives) / self.n_objectives

        # Compute threshold
        theta = (1.0 - cumsum_w[rho]) / (rho + 1)

        # Project
        w_proj = np.maximum(w + theta, 0)

        # Normalize to ensure sum = 1 (numerical stability)
        w_sum = np.sum(w_proj)
        if w_sum > self._eps:
            w_proj = w_proj / w_sum
        else:
            w_proj = np.ones(self.n_objectives) / self.n_objectives

        return w_proj

    def predict(self) -> np.ndarray:
        """
        Predict preference weights (return current estimate).

        Args:
            observation: Current observation (not used in prediction)
            hidden_state: Current hidden state (not used in prediction)

        Returns:
            Predicted preference weights
        """
        return self.w.copy()

    def update(
        self,
        observation: np.ndarray,
        action: np.ndarray,
        q_values_all: np.ndarray,
    ):
        """
        Update preference weights using Kalman Filter with linear observation model.

        Observation: margin of expert action compared to others.

        Args:
            observation: Current observation (not used)
            action: Expert action (scalar)
            q_values_all: Q-values for all actions [action_dim, n_objectives]
        """
        expert_action = int(action)

        # ===== Prediction Step =====
        # State prediction: w_pred = w (no dynamics, random walk)
        w_pred = self.w.copy()

        # Covariance prediction: P_pred = P + Q
        P_pred = self.P + self.Q

        # ===== Update Step =====
        # Compute expected observation using linear model
        # H = q_expert - mean(q_other) [n_objectives]
        # This is the gradient of margin w.r.t. preferences
        q_expert = q_values_all[expert_action]  # [n_objectives]

        # Mean of other Q-values
        other_mask = np.ones(q_values_all.shape[0], dtype=bool)
        other_mask[expert_action] = False
        q_others = q_values_all[other_mask]  # [action_dim-1, n_objectives]
        q_mean_other = np.mean(q_others, axis=0)  # [n_objectives]

        # Observation matrix (linear relationship)
        H = (q_expert - q_mean_other).reshape(1, -1)  # [1, n_objectives]

        # Predicted observation: h_pred = H @ w_pred
        h_pred = H @ w_pred  # scalar

        # Actual observation: We want expert action to have positive margin
        # Set target observation = 1 (positive margin)
        z = 1.0

        # Innovation (measurement residual)
        y = z - h_pred  # scalar

        # Innovation covariance: S = H @ P_pred @ H^T + R
        S = (H @ P_pred @ H.T).item() + self.R  # scalar
        S = max(S, self._eps)  # Ensure positive

        # Kalman gain: K = P_pred @ H^T @ S^{-1}
        K = (P_pred @ H.T) / S  # [n_objectives, 1]
        K = K.flatten()  # [n_objectives]

        # State update: w = w_pred + K @ y
        w_updated = w_pred + K * y

        # Project onto simplex to maintain constraint
        self.w = self._project_to_simplex(w_updated)

        # Covariance update: P = (I - K @ H) @ P_pred
        identity = np.eye(self.n_objectives)
        self.P = (identity - np.outer(K, H.flatten())) @ P_pred

        # Ensure symmetry and positive definiteness
        self.P = (self.P + self.P.T) / 2
        self.P += np.eye(self.n_objectives) * self._eps

        return None

    def reset(self):
        """Reset KF to initial state."""
        # Initialize to uniform distribution
        self.w = np.ones(self.n_objectives) / self.n_objectives

        # Covariance matrix [n_objectives, n_objectives]
        self.P = np.eye(self.n_objectives) * self.initial_variance

        # Process noise covariance (random walk on weights)
        self.Q = np.eye(self.n_objectives) * (self.process_noise**2)

        # Observation noise (scalar for mismatch)
        self.R = self.observation_noise**2


class GaussianProcessSSM(StateSpaceModel):
    """
    Gaussian Process State Space Model for preference weight prediction.

    Uses Gaussian Process regression to model the temporal dynamics of preference weights.
    The GP captures non-linear temporal correlations and provides uncertainty estimates.

    Model:
        - State: preference weights w_t ~ simplex
        - Process: w_t ~ GP(m(t), k(t, t'))
        - Observation: action selection via softmax(beta * margin(w_t, Q))

    The GP kernel captures temporal smoothness and allows for adaptive noise estimation.
    """

    def __init__(
        self,
        n_objectives: int,
        length_scale: float = 1.0,
        signal_variance: float = 1.0,
        observation_noise: float = 0.1,
        beta: float = 5.0,
        max_history: int = 100,
        kernel_type: str = "rbf",
        seed: int = 42,
    ):
        """
        Initialize Gaussian Process SSM.

        Args:
            n_objectives: Number of objectives
            length_scale: GP kernel length scale (controls temporal smoothness)
            signal_variance: GP kernel signal variance (controls amplitude)
            observation_noise: Observation noise for likelihood
            beta: Temperature parameter for action probability
            max_history: Maximum number of observations to keep in history
            kernel_type: Type of kernel ("rbf", "matern32", "matern52")
            seed: Random seed
        """
        self.n_objectives = n_objectives
        self.length_scale = length_scale
        self.signal_variance = signal_variance
        self.observation_noise = observation_noise
        self.beta = beta
        self.max_history = max_history
        self.kernel_type = kernel_type
        self.rng = np.random.RandomState(seed)
        self._eps = 1e-8

        # History of observations
        self.timesteps = []  # List of timestep indices
        self.observations = []  # List of observations
        self.actions = []  # List of expert actions
        self.q_values_history = []  # List of Q-values

        # GP posterior
        self.mean_weights = None  # Posterior mean [n_objectives]
        self.cov_weights = None  # Posterior covariance [n_objectives, n_objectives]

        # Current timestep
        self.t = 0

        self.reset()

    def _kernel(self, t1: np.ndarray, t2: np.ndarray) -> np.ndarray:
        """
        Compute kernel matrix between timesteps.

        Args:
            t1: Timesteps [n1]
            t2: Timesteps [n2]

        Returns:
            Kernel matrix [n1, n2]
        """
        t1 = np.atleast_1d(t1).reshape(-1, 1)
        t2 = np.atleast_1d(t2).reshape(-1, 1)

        # Compute pairwise distances
        dists = np.abs(t1 - t2.T)

        if self.kernel_type == "rbf":
            # RBF (Gaussian) kernel
            K = self.signal_variance * np.exp(-0.5 * (dists / self.length_scale) ** 2)
        elif self.kernel_type == "matern32":
            # Matérn 3/2 kernel
            r = np.sqrt(3) * dists / self.length_scale
            K = self.signal_variance * (1 + r) * np.exp(-r)
        elif self.kernel_type == "matern52":
            # Matérn 5/2 kernel
            r = np.sqrt(5) * dists / self.length_scale
            K = self.signal_variance * (1 + r + r**2 / 3) * np.exp(-r)
        else:
            raise ValueError(f"Unknown kernel type: {self.kernel_type}")

        return K

    def _project_to_simplex(self, w: np.ndarray) -> np.ndarray:
        """
        Project weights onto probability simplex.

        Args:
            w: Weights to project [n_objectives]

        Returns:
            Projected weights [n_objectives]
        """
        # Sort weights in descending order
        w_sorted = np.sort(w)[::-1]
        cumsum_w = np.cumsum(w_sorted)

        # Find rho (largest j such that w_j + (1 - cumsum) / (j+1) > 0)
        rho = None
        for j in range(self.n_objectives):
            if w_sorted[j] + (1.0 - cumsum_w[j]) / (j + 1) > 0:
                rho = j

        if rho is None:
            # Fallback to uniform
            return np.ones(self.n_objectives) / self.n_objectives

        # Compute threshold
        theta = (1.0 - cumsum_w[rho]) / (rho + 1)

        # Project
        w_proj = np.maximum(w + theta, 0)

        # Normalize to ensure sum to 1
        w_proj = w_proj / (np.sum(w_proj) + self._eps)

        return w_proj

    def predict(self) -> np.ndarray:
        """
        Predict preference weights using GP posterior.

        Args:
            observation: Current observation (not used in GP prediction)
            hidden_state: Not used

        Returns:
            Predicted preference weights [n_objectives]
        """
        if self.mean_weights is None:
            # No observations yet, return uniform prior
            return np.ones(self.n_objectives) / self.n_objectives

        # Return current posterior mean projected to simplex
        return self._project_to_simplex(self.mean_weights)

    def update(
        self,
        observation: np.ndarray,
        action: np.ndarray,
        q_values_all: np.ndarray,
    ):
        """
        Update GP posterior with new observation.

        Uses the expert action as observation to update the GP posterior
        over preference weights.

        Args:
            observation: Current observation
            action: Expert action (discrete)
            q_values_all: Q-values for all actions [action_dim, n_objectives]
        """
        expert_action = int(action)

        # Add to history
        self.timesteps.append(self.t)
        self.observations.append(observation)
        self.actions.append(expert_action)
        self.q_values_history.append(q_values_all)

        # Trim history if needed
        if len(self.timesteps) > self.max_history:
            self.timesteps.pop(0)
            self.observations.pop(0)
            self.actions.pop(0)
            self.q_values_history.pop(0)

        # Update GP posterior
        n_history = len(self.timesteps)

        if n_history == 1:
            # First observation: initialize with prior
            self.mean_weights = np.ones(self.n_objectives) / self.n_objectives
            self.cov_weights = np.eye(self.n_objectives) * self.signal_variance
        else:
            # Perform GP regression for each objective dimension independently
            history_timesteps = np.array(self.timesteps[:-1])  # Previous timesteps
            current_timestep = np.array([self.t])  # Current timestep

            # Compute kernels
            K_hist_hist = self._kernel(history_timesteps, history_timesteps)
            K_hist_hist += np.eye(len(history_timesteps)) * (
                self.observation_noise**2 + self._eps
            )

            K_current_hist = self._kernel(current_timestep, history_timesteps)
            K_current_current = self._kernel(current_timestep, current_timestep)

            # Compute target weights from previous actions
            # Find weights that maximize margin for expert action using simple gradient ascent
            target_weights = []
            for i, (act, q_vals) in enumerate(
                zip(self.actions[:-1], self.q_values_history[:-1])
            ):
                # Start with normalized expert Q-values
                expert_q = q_vals[act]
                expert_q_norm = np.linalg.norm(expert_q) + self._eps
                w = expert_q / expert_q_norm

                # Simple gradient ascent to maximize margin
                # margin = max_other_mismatch - expert_mismatch
                # where mismatch = ||w_normalized - q_normalized||^2
                n_iters = 10
                lr = 0.1

                for _ in range(n_iters):
                    # Normalize current weights
                    w_norm = np.linalg.norm(w) + self._eps
                    w_normalized = w / w_norm

                    # Normalize expert Q-values
                    expert_q_normalized = expert_q / expert_q_norm

                    # Compute max mismatch for other actions
                    max_other_mismatch = 0
                    for a in range(q_vals.shape[0]):
                        if a != act:
                            q_a = q_vals[a]
                            q_a_norm = np.linalg.norm(q_a) + self._eps
                            q_a_normalized = q_a / q_a_norm
                            mismatch_a = np.sum((w_normalized - q_a_normalized) ** 2)
                            max_other_mismatch = max(max_other_mismatch, mismatch_a)

                    # Gradient of margin w.r.t w
                    # margin = max_other_mismatch - expert_mismatch
                    # ∂margin/∂w = -∂expert_mismatch/∂w (max_other is constant for this gradient step)
                    # ∂expert_mismatch/∂w = 2(w_normalized - expert_q_normalized) * ∂w_normalized/∂w
                    grad = -2 * (w_normalized - expert_q_normalized) / w_norm

                    # Update
                    w = w + lr * grad

                    # Project to simplex
                    w = self._project_to_simplex(w)

                target_weights.append(w)

            target_weights = np.array(target_weights)  # [n_history-1, n_objectives]

            # GP regression for each dimension
            new_mean = np.zeros(self.n_objectives)
            new_cov = np.zeros((self.n_objectives, self.n_objectives))

            try:
                K_hist_hist_inv = np.linalg.inv(K_hist_hist)
            except np.linalg.LinAlgError:
                K_hist_hist_inv = np.linalg.pinv(K_hist_hist)

            for d in range(self.n_objectives):
                # GP posterior for dimension d
                y = target_weights[:, d]  # [n_history-1]

                # Posterior mean
                new_mean[d] = K_current_hist @ K_hist_hist_inv @ y

                # Posterior variance (diagonal only for efficiency)
                new_cov[d, d] = (
                    K_current_current
                    - K_current_hist @ K_hist_hist_inv @ K_current_hist.T
                )

            self.mean_weights = new_mean
            self.cov_weights = new_cov + np.eye(self.n_objectives) * self._eps

        self.t += 1

        return None

    def reset(self):
        """Reset GP-SSM to initial state."""
        self.timesteps = []
        self.observations = []
        self.actions = []
        self.q_values_history = []
        self.mean_weights = None
        self.cov_weights = None
        self.t = 0
