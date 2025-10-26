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
from typing import Optional, Callable
from abc import ABC, abstractmethod


def compute_mismatch(preference, q_values_all, eps: float = 1e-8):
    """
    Compute mismatch function between normalized preference and expert Q-values.

    mismatch = ||preference/|preference| - q_values_all/|q_values_all|||^2

    Args:
        preference: Preference weights [n_objectives] (torch.Tensor or np.ndarray)
        q_values_all: Expert Q-values [n_objectives] (torch.Tensor or np.ndarray)
        eps: Small constant to avoid division by zero

    Returns:
        Mismatch value (scalar, same type as input)
    """
    # Handle both torch tensors and numpy arrays
    if isinstance(preference, torch.Tensor):
        # PyTorch implementation
        preference_norm = torch.norm(preference, dim=-1, keepdim=True) + eps
        preference_normalized = preference / preference_norm

        q_values_all_norm = torch.norm(q_values_all, dim=-1, keepdim=True) + eps
        q_values_all_normalized = q_values_all / q_values_all_norm

        # Compute squared L2 distance
        mismatch = torch.sum(
            (preference_normalized - q_values_all_normalized) ** 2, dim=-1
        )
    else:
        # NumPy implementation
        preference_norm = np.linalg.norm(preference, axis=-1, keepdims=True) + eps
        preference_normalized = preference / preference_norm

        q_values_all_norm = np.linalg.norm(q_values_all, axis=-1, keepdims=True) + eps
        q_values_all_normalized = q_values_all / q_values_all_norm

        # Compute squared L2 distance
        mismatch = np.sum(
            (preference_normalized - q_values_all_normalized) ** 2, axis=-1
        )

    return mismatch


class StateSpaceModel(ABC):
    """Abstract base class for state space models."""

    @abstractmethod
    def predict(self, observation: np.ndarray, action: np.ndarray) -> np.ndarray:
        """
        Predict preference weights given observation and action.

        Args:
            observation: Current observation
            action: Current action

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
        next_observation: np.ndarray,
    ):
        """
        Update the model given a transition and Q-values for all actions.

        Args:
            observation: Current observation
            action: Action taken (discrete)
            q_values_all: Q-values for all actions [action_dim, n_objectives]
            next_observation: Next observation
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

        # Initialize particles uniformly on simplex
        self.particles = self._initialize_particles()
        self.weights = np.ones(n_particles) / n_particles
        self.current_estimate = np.mean(self.particles, axis=0)

    def _initialize_particles(self) -> np.ndarray:
        """Initialize particles uniformly on probability simplex."""
        particles = self.rng.dirichlet(
            np.ones(self.n_objectives), size=self.n_particles
        )
        return particles

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

    def _compute_action_likelihood(
        self, particles: np.ndarray, q_values_all: np.ndarray, action: int
    ) -> np.ndarray:
        """
        Compute likelihood of observed action given particles and Q-values.

        Uses Boltzmann policy: P(a|s,w) = softmax(Q(s,:)^T w)

        Args:
            particles: Particle preference weights [n_particles, n_objectives]
            q_values_all: Q-values for all actions [action_dim, n_objectives]
            action: Observed action (discrete)

        Returns:
            Likelihoods [n_particles]
        """
        likelihoods = np.zeros(self.n_particles)
        for i in range(self.n_particles):
            # Compute scalar Q-values: Q^T w for all actions
            q_scalar = q_values_all @ particles[i]  # [action_dim]

            # Softmax to get action probabilities
            q_scalar = q_scalar - np.max(q_scalar)  # Numerical stability
            probs = np.exp(q_scalar / self.observation_noise) / np.sum(
                np.exp(q_scalar / self.observation_noise)
            )

            # Likelihood of observed action
            likelihoods[i] = probs[action] + 1e-10

        return likelihoods

    def _resample(self):
        """Resample particles based on weights (systematic resampling)."""
        cumsum = np.cumsum(self.weights)
        cumsum[-1] = 1.0  # Ensure last value is exactly 1

        # Systematic resampling
        positions = (self.rng.rand() + np.arange(self.n_particles)) / self.n_particles
        indices = np.searchsorted(cumsum, positions)

        self.particles = self.particles[indices]
        self.weights = np.ones(self.n_particles) / self.n_particles

    def predict(self, observation: np.ndarray, action: np.ndarray) -> np.ndarray:
        """
        Predict preference weights (return current estimate).

        Args:
            observation: Current observation
            action: Current action

        Returns:
            Predicted preference weights
        """
        return self.current_estimate

    def update(
        self,
        observation: np.ndarray,
        action: np.ndarray,
        q_values_all: np.ndarray,
        next_observation: np.ndarray,
    ):
        """
        Update particle filter using action likelihood.

        Args:
            observation: Current observation
            action: Action taken (discrete)
            q_values_all: Q-values for all actions [action_dim, n_objectives]
            next_observation: Next observation
        """
        # Transition particles
        self.particles = self._transition(self.particles)

        # Update weights based on action likelihood
        likelihoods = self._compute_action_likelihood(
            self.particles, q_values_all, int(action)
        )
        self.weights *= likelihoods
        self.weights += 1e-10  # Avoid zeros
        self.weights /= np.sum(self.weights)

        # Resample if effective sample size is too low
        ess = 1.0 / np.sum(self.weights**2)
        if ess < self.n_particles / 2:
            self._resample()

        # Update estimate
        self.current_estimate = np.mean(self.particles, axis=0)

    def reset(self):
        """Reset particle filter."""
        self.particles = self._initialize_particles()
        self.weights = np.ones(self.n_particles) / self.n_particles
        self.current_estimate = np.mean(self.particles, axis=0)


class ExtendedKalmanFilter(StateSpaceModel):
    """
    Extended Kalman Filter for preference weight prediction.

    Uses Gaussian approximation with linearization for non-linear dynamics.
    Maintains mean and covariance of preference weight distribution.
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
            process_noise: Process noise std
            observation_noise: Observation noise std
            initial_variance: Initial covariance diagonal value
            seed: Random seed
        """
        self.n_objectives = n_objectives
        self.process_noise = process_noise
        self.observation_noise = observation_noise
        self.rng = np.random.RandomState(seed)

        # State: preference weights [n_objectives]
        # Initialize with uniform weights
        self.mean = np.ones(n_objectives) / n_objectives

        # Covariance matrix [n_objectives, n_objectives]
        self.covariance = np.eye(n_objectives) * initial_variance

        # Process noise covariance
        self.Q = np.eye(n_objectives) * (process_noise**2)

        # Observation noise variance
        self.R = observation_noise**2

    def _project_to_simplex(self, weights: np.ndarray) -> np.ndarray:
        """Project weights onto probability simplex."""
        weights = np.maximum(weights, 0)
        weights_sum = np.sum(weights)
        if weights_sum < 1e-8:
            return np.ones(self.n_objectives) / self.n_objectives
        return weights / weights_sum

    def predict(self, observation: np.ndarray, action: np.ndarray) -> np.ndarray:
        """
        Predict preference weights (return current mean).

        Args:
            observation: Current observation
            action: Current action

        Returns:
            Predicted preference weights
        """
        return self.mean

    def update(
        self,
        observation: np.ndarray,
        action: np.ndarray,
        q_values_all: np.ndarray,
        next_observation: np.ndarray,
    ):
        """
        Update EKF using action likelihood.

        Args:
            observation: Current observation
            action: Action taken (discrete)
            q_values_all: Q-values for all actions [action_dim, n_objectives]
            next_observation: Next observation
        """
        # Prediction Step
        mean_pred = self.mean
        covariance_pred = self.covariance + self.Q

        # Update using action likelihood
        q_scalar = q_values_all @ mean_pred
        q_scalar = q_scalar - np.max(q_scalar)
        probs = np.exp(q_scalar / self.observation_noise) / np.sum(
            np.exp(q_scalar / self.observation_noise)
        )

        action_idx = int(action)
        q_action = q_values_all[action_idx]
        q_expected = q_values_all.T @ probs
        gradient = (q_action - q_expected) / self.observation_noise

        step_size = 0.1 * self.observation_noise
        self.mean = mean_pred + step_size * gradient
        self.covariance = 0.9 * covariance_pred + 0.1 * self.Q
        self.covariance = (self.covariance + self.covariance.T) / 2
        self.covariance += np.eye(self.n_objectives) * 1e-6
        self.mean = self._project_to_simplex(self.mean)

    def reset(self):
        """Reset EKF."""
        self.mean = np.ones(self.n_objectives) / self.n_objectives
        self.covariance = np.eye(self.n_objectives) * 0.1


class NeuralSSM(StateSpaceModel):
    """
    Neural State Space Model using GRU for preference weight prediction.

    Uses a GRU to learn temporal dynamics of preference weights from trajectories.
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        n_objectives: int,
        hidden_dim: int = 128,
        n_layers: int = 2,
        learning_rate: float = 1e-3,
        device: str = "cuda",
    ):
        """
        Initialize Neural SSM.

        Args:
            obs_dim: Observation dimension
            action_dim: Action dimension
            n_objectives: Number of objectives
            hidden_dim: GRU hidden dimension
            n_layers: Number of GRU layers
            learning_rate: Learning rate for optimizer
            device: Device to run on
        """
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.n_objectives = n_objectives
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

        # Build model
        self.model = self._build_model().to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)

        # Hidden state
        self.hidden = None

        # Loss function
        self.criterion = nn.MSELoss()

    def _build_model(self) -> nn.Module:
        """Build GRU-based neural SSM."""

        class GRUPreferenceModel(nn.Module):
            def __init__(self, obs_dim, action_dim, n_objectives, hidden_dim, n_layers):
                super().__init__()

                # Input: [obs, action, mo_reward]
                input_dim = obs_dim + action_dim + n_objectives

                # GRU for temporal dynamics
                self.gru = nn.GRU(
                    input_size=input_dim,
                    hidden_size=hidden_dim,
                    num_layers=n_layers,
                    batch_first=True,
                )

                # Output layer: predict preference weights
                self.fc = nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, n_objectives),
                    nn.Softmax(dim=-1),  # Ensure weights sum to 1
                )

            def forward(self, x, hidden=None):
                """
                Forward pass.

                Args:
                    x: Input [batch, seq_len, input_dim]
                    hidden: Hidden state [n_layers, batch, hidden_dim]

                Returns:
                    weights: Preference weights [batch, seq_len, n_objectives]
                    hidden: Updated hidden state
                """
                gru_out, hidden = self.gru(x, hidden)
                weights = self.fc(gru_out)
                return weights, hidden

        return GRUPreferenceModel(
            self.obs_dim,
            self.action_dim,
            self.n_objectives,
            self.hidden_dim,
            self.n_layers,
        )

    def _prepare_input(
        self, observation: np.ndarray, action: np.ndarray, reward: np.ndarray
    ) -> torch.Tensor:
        """
        Prepare input tensor from observation, action, and reward.

        Args:
            observation: Observation array
            action: Action (scalar or one-hot)
            reward: Multi-objective reward

        Returns:
            Input tensor [1, 1, input_dim]
        """
        # Convert action to array if scalar
        if np.isscalar(action):
            action_array = np.zeros(self.action_dim)
            action_array[action] = 1.0
        else:
            action_array = action

        # Concatenate [obs, action, mo_reward]
        input_array = np.concatenate([observation, action_array, reward])
        input_tensor = (
            torch.FloatTensor(input_array).unsqueeze(0).unsqueeze(0)
        )  # [1, 1, input_dim]

        return input_tensor.to(self.device)

    def predict(self, observation: np.ndarray, action: np.ndarray) -> np.ndarray:
        """
        Predict preference weights given observation and action.

        Note: Requires reward to be provided in update() for full context.
        This returns the last prediction.

        Args:
            observation: Current observation
            action: Current action

        Returns:
            Predicted preference weights
        """
        # For prediction without reward, use zeros as placeholder
        reward_placeholder = np.zeros(self.n_objectives)
        input_tensor = self._prepare_input(observation, action, reward_placeholder)

        with torch.no_grad():
            weights, self.hidden = self.model(input_tensor, self.hidden)
            weights_np = weights.squeeze().cpu().numpy()

        return weights_np

    def update(
        self,
        observation: np.ndarray,
        action: np.ndarray,
        q_values_all: np.ndarray,
        next_observation: np.ndarray,
    ):
        """
        Update neural SSM with new observation (online learning).

        Args:
            observation: Current observation
            action: Action taken (discrete)
            q_values_all: Q-values for all actions [action_dim, n_objectives]
            next_observation: Next observation
        """
        # Extract Q-value for taken action
        action_idx = int(action)
        q_action = q_values_all[action_idx]  # [n_objectives]

        # Prepare input
        input_tensor = self._prepare_input(observation, action, q_action)

        # Forward pass
        predicted_weights, self.hidden = self.model(input_tensor, self.hidden)

        # For online learning, we don't have ground truth weights
        # We could use self-supervised learning or inverse RL
        # For now, we just update the hidden state without backprop
        # True training should be done with train() method on batches

        # Detach hidden state for next iteration
        if self.hidden is not None:
            self.hidden = self.hidden.detach()

    def train_batch(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        target_weights: torch.Tensor,
    ) -> float:
        """
        Train on a batch of trajectories.

        Args:
            observations: [batch, seq_len, obs_dim]
            actions: [batch, seq_len, action_dim]
            rewards: [batch, seq_len, n_objectives]
            target_weights: [batch, seq_len, n_objectives] - ground truth weights

        Returns:
            Loss value
        """
        self.model.train()

        # Concatenate inputs
        inputs = torch.cat([observations, actions, rewards], dim=-1)

        # Forward pass
        predicted_weights, _ = self.model(inputs)

        # Compute loss
        loss = self.criterion(predicted_weights, target_weights)

        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def reset(self):
        """Reset hidden state."""
        self.hidden = None

    def save(self, path: str):
        """Save model checkpoint."""
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
            },
            path,
        )

    def load(self, path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
