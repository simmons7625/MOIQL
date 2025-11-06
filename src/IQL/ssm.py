"""
State Space Models for Preference Weight Prediction.

State space formulation:
    h_t = f(h_{t-1}, x_t) + noise_1,t    (state transition)
    y_t = g(h_t) + noise_2,t              (observation)

For preference weight estimation:
    x_t: (s_t, a_t) - state and action
    h_t: preference weights
    g: mismatch margin function

Mismatch margin function:
    margin = mean(mismatch(preference, q_others)) - mismatch(preference, q_expert)
    where mismatch = ||preference/|preference| - q/|q|||^2

Where:
    - preference: h_t (preference weights)
    - q_expert: Q-values for expert action
    - q_others: Q-values for all other actions

The margin is positive when the expert action is better aligned with the preference.

Implements three approaches:
1. Particle Filter: Non-parametric sequential Monte Carlo
2. Extended Kalman Filter (EKF): Gaussian approximation with linearization
3. Gaussian Process SSM: GP-based temporal modeling
"""

import numpy as np
from typing import Optional, Callable
from abc import ABC, abstractmethod

EPS = 1e-8


def project_to_simplex(w: np.ndarray, eps: float = EPS) -> np.ndarray:
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


def compute_mismatch(
    preference: np.ndarray, q_values: np.ndarray, eps: float = EPS
) -> float:
    """
    Compute mismatch between preference and q_values.

    Args:
        preference: Preference weights [n_objectives]
        q_values: Q-values [n_objectives]
        eps: Small constant to avoid division by zero

    Returns:
        Mismatch (squared L2 distance between normalized vectors)
    """
    pref_norm = np.linalg.norm(preference) + eps
    pref_normalized = preference / pref_norm

    q_norm = np.linalg.norm(q_values) + eps
    q_normalized = q_values / q_norm

    return np.sum((pref_normalized - q_normalized) ** 2)


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
        weights_sum = np.maximum(weights_sum, EPS)  # Avoid division by zero
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

        # Compute mismatch margin for each particle
        expert_action = int(action)
        expert_q = q_values_all[expert_action]  # [n_objectives]

        # Normalize expert Q-values
        expert_q_norm = np.linalg.norm(expert_q) + EPS
        expert_q_normalized = expert_q / expert_q_norm

        # Normalize particles
        particle_norms = (
            np.linalg.norm(self.particles, axis=1, keepdims=True) + EPS
        )  # [n_particles, 1]
        particles_normalized = (
            self.particles / particle_norms
        )  # [n_particles, n_objectives]

        # Compute mismatch with expert action for each particle
        expert_mismatches = np.sum(
            (particles_normalized - expert_q_normalized) ** 2, axis=1
        )  # [n_particles]

        # Compute mean mismatch with other actions for each particle (vectorized)
        n_actions = q_values_all.shape[0]

        # Normalize all Q-values [n_actions, n_objectives]
        q_norms = (
            np.linalg.norm(q_values_all, axis=1, keepdims=True) + EPS
        )  # [n_actions, 1]
        q_normalized_all = q_values_all / q_norms  # [n_actions, n_objectives]

        # Compute mismatch for all actions and all particles
        # particles_normalized: [n_particles, n_objectives]
        # q_normalized_all: [n_actions, n_objectives]
        # Expand dimensions for broadcasting: [n_particles, 1, n_objectives] - [1, n_actions, n_objectives]
        diff = (
            particles_normalized[:, np.newaxis, :] - q_normalized_all[np.newaxis, :, :]
        )  # [n_particles, n_actions, n_objectives]
        all_mismatches = np.sum(diff**2, axis=2)  # [n_particles, n_actions]

        # Create mask for other actions (exclude expert)
        mask = np.ones(n_actions, dtype=bool)
        mask[expert_action] = False

        # Mean over other actions
        other_mismatches_mean = np.mean(
            all_mismatches[:, mask], axis=1
        )  # [n_particles]

        # Mismatch margin: positive margin means expert is better aligned
        # margin = mean(mismatch(others)) - mismatch(expert)
        # Higher margin is better (expert has lower mismatch than others)
        mismatch_margins = other_mismatches_mean - expert_mismatches  # [n_particles]

        # Convert margins to fitness (higher margin is better)
        # z-score normalization
        margins_mean = np.mean(mismatch_margins)
        margins_std = np.std(mismatch_margins) + EPS
        mismatch_margins = (mismatch_margins - margins_mean) / margins_std
        fitness = np.exp(mismatch_margins / self.observation_noise)

        # Update weights with fitness
        self.weights *= fitness
        self.weights /= np.sum(self.weights) + EPS  # Normalize weights

        # Resample if effective sample size is too low
        ess = 1.0 / np.sum(self.weights**2)
        if ess < 0.5 * self.n_particles:
            self._resample()


class ExtendedKalmanFilter(StateSpaceModel):
    """
    Extended Kalman Filter for preference weight prediction using nonlinear observation model.

    State representation: logits x in R^n
    Process model: f(x) = softmax(x) maps logits to simplex
    Observation model: h(x) = mismatch(softmax(x), q_expert_normalized)
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
            initial_variance: Initial covariance diagonal value
            seed: Random seed
        """
        self.n_objectives = n_objectives
        self.process_noise = process_noise
        self.observation_noise = observation_noise
        self.initial_variance = initial_variance
        self.rng = np.random.RandomState(seed)
        self._eps = EPS

    def f(self, x: np.ndarray) -> np.ndarray:
        """
        Process model: f(x) = softmax(x).

        Maps logits to probability simplex.

        Args:
            x: Logits [n_objectives]

        Returns:
            Weights on simplex [n_objectives]
        """
        # exp_x = np.exp(x)
        # return exp_x / (np.sum(exp_x) + self._eps)

        return project_to_simplex(x)

    def h(self, x: np.ndarray, q_expert: np.ndarray, q_values_all: np.ndarray) -> float:
        """
        Observation model: h(x) = mismatch margin (scalar).

        Computes mean(mismatch(preference, q_others)) - mismatch(preference, q_expert)

        Args:
            x: Logits [n_objectives]
            q_expert: Expert Q-values [n_objectives]
            q_values_all: All Q-values [n_actions, n_objectives]

        Returns:
            Mismatch margin (scalar) - positive when expert is better aligned
        """
        w = self.f(x)  # softmax to get preference weights
        w_norm = np.linalg.norm(w) + self._eps
        w_normalized = w / w_norm

        # Compute mismatch with expert action
        q_expert_norm = np.linalg.norm(q_expert) + self._eps
        q_expert_normalized = q_expert / q_expert_norm
        expert_mismatch = np.sum((w_normalized - q_expert_normalized) ** 2)

        # Compute mean mismatch with other actions (vectorized)
        n_actions = q_values_all.shape[0]

        # Normalize all Q-values [n_actions, n_objectives]
        q_norms = (
            np.linalg.norm(q_values_all, axis=1, keepdims=True) + self._eps
        )  # [n_actions, 1]
        q_normalized_all = q_values_all / q_norms  # [n_actions, n_objectives]

        # Compute mismatch for all actions
        # w_normalized: [n_objectives]
        # q_normalized_all: [n_actions, n_objectives]
        diff = (
            w_normalized[np.newaxis, :] - q_normalized_all
        )  # [n_actions, n_objectives]
        all_mismatches = np.sum(diff**2, axis=1)  # [n_actions]

        # Create mask for other actions (exclude expert)
        mask = np.ones(n_actions, dtype=bool)
        mask[
            np.argmin(np.linalg.norm(q_normalized_all - q_expert_normalized, axis=1))
        ] = False

        # Mean over other actions
        if np.sum(mask) > 0:
            other_mismatch_mean = np.mean(all_mismatches[mask])
        else:
            other_mismatch_mean = 0.0

        # Return margin (higher is better - expert has lower mismatch than others)
        return other_mismatch_mean - expert_mismatch

    def jacobian(self, func, x: np.ndarray, *args) -> np.ndarray:
        """
        Compute Jacobian of function numerically using finite differences.

        Args:
            func: Function to compute Jacobian for
            x: Input point [n]
            *args: Additional arguments to func
            eps: Finite difference step size

        Returns:
            Jacobian matrix [m, n] where J[i,j] = df_i/dx_j
            For scalar functions, returns [1, n]
        """
        f0 = func(x, *args)
        n = len(x)

        # Check if scalar or vector output
        if np.isscalar(f0):
            J = np.zeros((1, n))
            for j in range(n):
                x_plus = x.copy()
                x_plus[j] += self._eps
                f_plus = func(x_plus, *args)
                J[0, j] = (f_plus - f0) / self._eps
        else:
            m = len(f0)
            J = np.zeros((m, n))
            for j in range(n):
                x_plus = x.copy()
                x_plus[j] += self._eps
                f_plus = func(x_plus, *args)
                J[:, j] = (f_plus - f0) / self._eps

        return J

    def predict(self) -> np.ndarray:
        """
        Predict preference weights from current logits.

        Returns:
            Predicted preference weights on simplex
        """
        F = self.jacobian(self.f, self.x)
        self.x = self.f(self.x)
        self.P = F @ self.P @ F.T + self.Q

        return self.x

    def update(
        self,
        observation: np.ndarray,
        action: np.ndarray,
        q_values_all: np.ndarray,
    ):
        """
        Update EKF with observation using mismatch margin observation model.

        Args:
            observation: Current observation (not used)
            action: Expert action (scalar)
            q_values_all: Q-values for all actions [action_dim, n_objectives]
        """
        expert_action = int(action)

        # ===== Update Step =====
        # Observation: expert Q-values
        q_expert = q_values_all[expert_action]  # [n_objectives]

        # Expected observation should be positive (expert should have lower mismatch than others)
        # We use 0 as target but in practice positive margins are good
        z = np.array([0.0])  # Target margin baseline

        # Predicted observation: h(x_pred) returns scalar margin
        y_pred = np.array([self.h(self.x, q_expert, q_values_all)])

        # Jacobian of observation model with respect to x
        H = self.jacobian(self.h, self.x, q_expert, q_values_all)  # [1, n_objectives]

        # Kalman gain computation with scalar observation
        S = H @ self.P @ H.T + self.observation_noise**2  # [1, 1]
        K = self.P @ H.T / (S[0, 0] + self._eps)  # [n_objectives, 1]

        # Update state and covariance
        innovation = z - y_pred  # [1]
        self.x = self.x + (K * innovation[0]).flatten()
        self.P = (np.eye(self.n_objectives) - K @ H) @ self.P

    def reset(self):
        """Reset EKF to initial state."""
        # Initialize logits to zero -> uniform weights after softmax
        self.x = np.zeros(self.n_objectives)

        # Covariance matrix [n_objectives, n_objectives]
        self.P = np.eye(self.n_objectives) * self.initial_variance

        # Process noise covariance (random walk on logits)
        self.Q = np.eye(self.n_objectives) * (self.process_noise**2)

        # Note: observation_noise is used directly as scalar in update method


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
        self._eps = EPS
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
