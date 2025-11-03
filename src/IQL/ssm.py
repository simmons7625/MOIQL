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
from typing import Optional, Callable
from abc import ABC, abstractmethod


def project_to_simplex(w: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """
    Project weights onto probability simplex.

    Args:
        w: Weights to project [n_objectives]
        eps: Small constant to avoid division by zero

    Returns:
        Projected weights [n_objectives] summing to 1
    """
    n = len(w)

    # Sort weights in descending order
    w_sorted = np.sort(w)[::-1]
    cumsum_w = np.cumsum(w_sorted)

    # Find rho (largest j such that w_j + (1 - cumsum) / (j+1) > 0)
    rho = None
    for j in range(n):
        if w_sorted[j] + (1.0 - cumsum_w[j]) / (j + 1) > 0:
            rho = j

    if rho is None:
        # Fallback to uniform
        return np.ones(n) / n

    # Compute threshold
    theta = (1.0 - cumsum_w[rho]) / (rho + 1)

    # Project
    w_proj = np.maximum(w + theta, 0)

    # Normalize to ensure sum to 1
    w_proj = w_proj / (np.sum(w_proj) + eps)

    return w_proj


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

        Returns:
            Predicted preference weights projected to simplex
        """
        # Compute weighted mean of particles
        weighted_mean = np.average(self.particles, axis=0, weights=self.weights)
        # Project to simplex to ensure valid probability distribution
        return project_to_simplex(weighted_mean)

    def update(
        self,
        observation: np.ndarray,
        action: np.ndarray,
        q_values_all: np.ndarray,
    ):
        # Transition particles
        self.particles = self._transition(self.particles)

        # Compute mismatch for each particle (negative because lower is better)
        expert_action = int(action)
        expert_q = q_values_all[expert_action]  # [n_objectives]
        expert_q_norm = np.linalg.norm(expert_q) + 1e-8
        expert_q_normalized = expert_q / expert_q_norm

        mismatches = np.zeros(self.n_particles)
        for i in range(self.n_particles):
            # Normalize particle
            particle_norm = np.linalg.norm(self.particles[i]) + 1e-8
            particle_normalized = self.particles[i] / particle_norm
            # Compute mismatch (squared distance)
            mismatches[i] = np.sum((particle_normalized - expert_q_normalized) ** 2)

        # Convert mismatches to fitness (negative because we want low mismatch)
        negative_mismatches = -mismatches  # Higher is better
        shifted = negative_mismatches - np.max(negative_mismatches)
        fitness = np.exp(shifted / self.observation_noise)

        # Update weights with fitness
        self.weights *= fitness
        self.weights /= np.sum(self.weights) + 1e-8  # Normalize weights

        # Resample if effective sample size is too low
        ess = 1.0 / np.sum(self.weights**2)
        if ess < self.n_particles / 2:
            self._resample()


class ExtendedKalmanFilter(StateSpaceModel):
    """
    Extended Kalman Filter for preference weight prediction using mismatch-based observation model.

    State representation: logits x in R^n, where w = softmax(x) are the preference weights.
    This ensures weights stay on the probability simplex automatically.

    Observation model: Directly observes expert action's normalized Q-values as target preference.
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
        self.rng = np.random.RandomState(seed)
        self.obs_noise_std = observation_noise
        self._eps = 1e-8

    def predict(self) -> np.ndarray:
        """
        Predict preference weights (return current estimate from logits).

        Returns:
            Predicted preference weights projected to simplex
        """
        # Use softmax as intermediate step then project to ensure valid simplex
        w = softmax_stable(self.x)
        return project_to_simplex(w)

    def update(
        self,
        observation: np.ndarray,
        action: np.ndarray,
        q_values_all: np.ndarray,
    ):
        """
        Update EKF with observation: directly use normalized expert Q-values as target.

        Simpler observation model: z = normalize(Q[expert_action])
        Predicted observation: h(x) = softmax(x) = w
        """
        expert_action = int(action)

        # Process model: x_k = x_{k-1} + process_noise
        x_pred = self.x.copy()
        P_pred = self.P + self.Q

        # Observation: normalized expert Q-values
        expert_q = q_values_all[expert_action]  # [n_objectives]
        expert_q_norm = np.linalg.norm(expert_q) + self._eps
        z = expert_q / expert_q_norm  # Normalized observation [n_objectives]

        # Predicted observation: h(x) = softmax(x) = w
        w_pred = softmax_stable(x_pred)  # [n_objectives]

        # Jacobian: H = dw/dx = softmax_jacobian
        H = softmax_jacobian(w_pred)  # [n_objectives, n_objectives]

        # Observation noise
        R = np.eye(self.n_objectives) * (self.obs_noise_std**2 + self._eps)

        # Innovation
        y = z - w_pred  # [n_objectives]

        # Innovation covariance
        S = H @ P_pred @ H.T + R
        S += np.eye(self.n_objectives) * self._eps

        # Kalman gain
        try:
            S_inv = np.linalg.inv(S)
        except np.linalg.LinAlgError:
            S_inv = np.linalg.pinv(S)

        K = P_pred @ H.T @ S_inv

        # State update: w = w_pred + K @ y
        w_updated = w_pred + K @ y

        # Project onto simplex to maintain constraint
        self.w = w_updated / (np.sum(w_updated) + self._eps)

        identity_matrix = np.eye(self.n_objectives)
        IKH = identity_matrix - K @ H
        self.P = IKH @ P_pred @ IKH.T + K @ R @ K.T

        self.P = (self.P + self.P.T) / 2
        self.P += np.eye(self.n_objectives) * self._eps

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

    def predict(self) -> np.ndarray:
        """
        Predict preference weights (return current estimate).

        Returns:
            Predicted preference weights projected to simplex
        """
        return project_to_simplex(self.w).copy()

    def update(
        self,
        observation: np.ndarray,
        action: np.ndarray,
        q_values_all: np.ndarray,
    ):
        """
        Update preference weights using Kalman Filter.

        Observation: normalized expert Q-values as target preference.

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
        # Observation: normalized expert Q-values
        q_expert = q_values_all[expert_action]  # [n_objectives]
        q_expert_norm = np.linalg.norm(q_expert) + self._eps
        z = q_expert / q_expert_norm  # Normalized observation [n_objectives]

        # Observation model is identity: h(w) = w (directly observe preferences)
        # H = I, so H @ P @ H^T = P and H^T = I
        h_pred = w_pred

        # Innovation (measurement residual)
        y = z - h_pred  # [n_objectives]

        # Innovation covariance: S = H @ P_pred @ H^T + R = P_pred + R
        S = P_pred + np.eye(self.n_objectives) * self.R
        S += np.eye(self.n_objectives) * self._eps

        # Kalman gain: K = P_pred @ H^T @ S^{-1} = P_pred @ S^{-1}
        try:
            S_inv = np.linalg.inv(S)
        except np.linalg.LinAlgError:
            S_inv = np.linalg.pinv(S)

        K = P_pred @ S_inv  # [n_objectives, n_objectives]

        # State update: w = w_pred + K @ y
        w_updated = w_pred + K @ y

        # Project onto simplex to maintain constraint
        self.w = project_to_simplex(w_updated)

        # Covariance update: P = (I - K @ H) @ P_pred
        identity = np.eye(self.n_objectives)
        self.P = (identity - K) @ P_pred

        # Ensure symmetry and positive definiteness
        self.P = (self.P + self.P.T) / 2
        self.P += np.eye(self.n_objectives) * self._eps

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
        self.kernel_type = kernel_type
        self.rng = np.random.RandomState(seed)
        self._eps = 1e-8
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

    def predict(self) -> np.ndarray:
        """
        Predict preference weights using GP posterior.

        Returns:
            Predicted preference weights [n_objectives]
        """
        if self.mean_weights is None:
            # No observations yet, return uniform prior
            return np.ones(self.n_objectives) / self.n_objectives

        # Return current posterior mean projected to simplex
        return project_to_simplex(self.mean_weights)

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
        # Compute target weight for current observation
        expert_q = q_values_all[int(action)]  # [n_objectives]
        expert_q_norm = np.linalg.norm(expert_q) + self._eps
        w = expert_q / expert_q_norm
        target_weight = project_to_simplex(w)

        # Store in history
        self.timesteps.append(self.t)
        self.observations.append(observation)
        self.actions.append(int(action))
        self.targets_history.append(target_weight)
        self.t += 1

        # GP regression for each objective dimension independently
        n_history = len(self.timesteps)

        if n_history == 1:
            # First observation - just use the target weight
            self.mean_weights = target_weight.copy()
            self.cov_weights = np.eye(self.n_objectives) * self.observation_noise**2
            return

        # Get all timesteps and target weights
        all_timesteps = np.array(self.timesteps)  # [n_history]
        all_targets = np.array(self.targets_history)  # [n_history, n_objectives]

        # Compute kernel matrix for all observations
        K = self._kernel(all_timesteps, all_timesteps)  # [n_history, n_history]
        K += np.eye(n_history) * (self.observation_noise**2 + self._eps)

        # Invert kernel matrix
        try:
            K_inv = np.linalg.inv(K)
        except np.linalg.LinAlgError:
            K_inv = np.linalg.pinv(K)

        # For prediction at current time, we want the posterior mean
        # In GP regression: mean = K(t*, t) @ K(t, t)^-1 @ y
        # Since we just added current observation, we use it directly but smoothed

        # Compute weights for all history points to predict current
        current_time = np.array([self.t - 1])  # Current timestep (just added)
        k_star = self._kernel(current_time, all_timesteps)  # [1, n_history]

        # GP posterior mean for each dimension
        self.mean_weights = np.zeros(self.n_objectives)
        for d in range(self.n_objectives):
            # Observations for dimension d
            y = all_targets[:, d]  # [n_history]

            # Posterior mean: k* @ K^-1 @ y
            self.mean_weights[d] = (k_star @ K_inv @ y)[0]

        # Posterior covariance (simplified - just keep diagonal)
        k_star_star = self._kernel(current_time, current_time)[0, 0]
        posterior_var = k_star_star - k_star @ K_inv @ k_star.T
        self.cov_weights = np.eye(self.n_objectives) * max(
            posterior_var[0, 0], self._eps
        )

    def reset(self):
        """Reset GP-SSM to initial state."""
        # History of observations
        self.timesteps = []  # List of timestep indices
        self.observations = []  # List of observations
        self.actions = []  # List of expert actions
        self.targets_history = []  # List of Q-values

        # GP posterior
        self.mean_weights = None  # Posterior mean [n_objectives]
        self.cov_weights = None  # Posterior covariance [n_objectives, n_objectives]

        # Current timestep
        self.t = 0
