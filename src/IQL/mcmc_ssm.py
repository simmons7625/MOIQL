"""
Batch Bayesian Auto-Tuning for State Space Models with MCMC.

Implements batch training for SSM preference estimators using multiple trajectories
with the same preference function. Based on:
"Batch Bayesian auto-tuning for nonlinear Kalman estimators" (Freitas Iglesias-Jr et al., 2025)

Key idea: Use MCMC to optimize SSM hyperparameters (process_noise, observation_noise, initial_variance)
at each timestep using all trajectories simultaneously. Since all trajectories share the same
preference dynamics, we can pool data across trajectories to get better hyperparameter estimates.
"""

import numpy as np
import torch
from typing import Dict, List

from .ssm import (
    ParticleFilter,
    KalmanFilter,
    ExtendedKalmanFilter,
    StateSpaceModel,
)
from .actor_critic import ActorCritic


class TrainableSSM(StateSpaceModel):
    """
    Wrapper for SSM classes that adds batch training with MCMC.

    Extends basic SSM classes with a train() method for Bayesian hyperparameter tuning.
    """

    def __init__(
        self, base_ssm: StateSpaceModel, ssm_type: str, n_objectives: int, **ssm_kwargs
    ):
        """
        Args:
            base_ssm: Base SSM instance (ParticleFilter, KalmanFilter, or ExtendedKalmanFilter)
            ssm_type: Type of SSM for reconstruction
            n_objectives: Number of objectives
            **ssm_kwargs: Additional SSM parameters (n_particles, beta, etc.)
        """
        self.base_ssm = base_ssm
        self.ssm_type = ssm_type
        self.n_objectives = n_objectives
        self.ssm_kwargs = ssm_kwargs

        # Current hyperparameters
        self.process_noise = base_ssm.process_noise
        self.observation_noise = base_ssm.observation_noise
        self.initial_variance = getattr(base_ssm, "initial_variance", 0.1)

    def predict(self, observation: np.ndarray, hidden_state: np.ndarray) -> np.ndarray:
        """Delegate to base SSM."""
        return self.base_ssm.predict(observation, hidden_state)

    def update(
        self, observation: np.ndarray, action: np.ndarray, q_values_all: np.ndarray
    ):
        """Delegate to base SSM."""
        return self.base_ssm.update(observation, action, q_values_all)

    def reset(self):
        """Delegate to base SSM."""
        return self.base_ssm.reset()

    def _create_ssm_with_params(
        self,
        process_noise: float,
        observation_noise: float,
        initial_variance: float,
        seed: int = 42,
    ) -> StateSpaceModel:
        """Create a new SSM instance with given parameters."""
        if self.ssm_type == "particle_filter":
            return ParticleFilter(
                n_objectives=self.n_objectives,
                n_particles=self.ssm_kwargs.get("n_particles", 1000),
                process_noise=process_noise,
                observation_noise=observation_noise,
                seed=seed,
            )
        elif self.ssm_type == "kalman_filter":
            return KalmanFilter(
                n_objectives=self.n_objectives,
                process_noise=process_noise,
                observation_noise=observation_noise,
                initial_variance=initial_variance,
                seed=seed,
            )
        elif self.ssm_type == "extended_kalman_filter":
            return ExtendedKalmanFilter(
                n_objectives=self.n_objectives,
                process_noise=process_noise,
                observation_noise=observation_noise,
                initial_variance=initial_variance,
                beta=self.ssm_kwargs.get("beta", 5.0),
                seed=seed,
            )
        else:
            raise ValueError(f"Unknown SSM type: {self.ssm_type}")

    def _evaluate_params(
        self,
        process_noise: float,
        observation_noise: float,
        initial_variance: float,
        trajectories: List[Dict],
        q_network: ActorCritic,
        device: str,
        seed: int = 42,
    ) -> float:
        """
        Evaluate SSM hyperparameters on trajectories.

        Returns negative MAE as log-likelihood proxy.
        """
        # Create temporary SSM with these parameters
        temp_ssm = self._create_ssm_with_params(
            process_noise, observation_noise, initial_variance, seed
        )

        total_mae = []

        for traj in trajectories:
            observations = traj["observations"]
            actions = traj["actions"]
            true_preferences = traj["preference_weights"]

            temp_ssm.reset()

            for t in range(len(observations) - 1):
                obs = observations[t]
                action = actions[t]
                true_pref = true_preferences[t]

                # Get Q-values
                obs_tensor = torch.FloatTensor(obs.flatten()).unsqueeze(0).to(device)
                with torch.no_grad():
                    _, q_values = q_network.act(obs_tensor)
                    q_values_all = q_values.squeeze(0).cpu().numpy()

                # Predict and compute MAE
                predicted_pref = temp_ssm.predict(obs, None)
                mae = np.mean(np.abs(predicted_pref - true_pref))
                total_mae.append(mae)

                # Update SSM
                temp_ssm.update(obs, action, q_values_all)

        mean_mae = np.mean(total_mae)
        return -mean_mae  # Negative MAE as log-likelihood

    def train(
        self,
        trajectories: List[Dict],
        q_network: ActorCritic,
        n_iterations: int = 1000,
        n_warmup: int = 200,
        proposal_scale: float = 0.1,
        device: str = "cuda",
        seed: int = 42,
    ) -> Dict[str, any]:
        """
        Train SSM hyperparameters using MCMC.

        Args:
            trajectories: List of trajectory dictionaries
            q_network: Trained Q-network for computing Q-values
            n_iterations: Number of MCMC iterations
            n_warmup: Number of warmup/burn-in iterations
            proposal_scale: Standard deviation of random walk proposals (in log space)
            device: Device for Q-network
            seed: Random seed

        Returns:
            Dictionary with best parameters, metrics, and MCMC samples
        """
        rng = np.random.RandomState(seed)

        # Define prior ranges (log-uniform)
        process_noise_range = (1e-4, 0.5)
        observation_noise_range = (1e-3, 1.0)
        initial_variance_range = (1e-3, 1.0)

        # Current parameters
        current_params = {
            "process_noise": self.process_noise,
            "observation_noise": self.observation_noise,
            "initial_variance": self.initial_variance,
        }

        # Evaluate current
        current_log_likelihood = self._evaluate_params(
            current_params["process_noise"],
            current_params["observation_noise"],
            current_params["initial_variance"],
            trajectories,
            q_network,
            device,
            seed,
        )

        # MCMC tracking
        mcmc_samples = []
        best_params = current_params.copy()
        best_log_likelihood = current_log_likelihood
        accepted = 0

        for iteration in range(n_iterations):
            # Propose in log space
            proposed_params_log = {
                "process_noise": np.log(current_params["process_noise"])
                + rng.normal(0, proposal_scale),
                "observation_noise": np.log(current_params["observation_noise"])
                + rng.normal(0, proposal_scale),
                "initial_variance": np.log(current_params["initial_variance"])
                + rng.normal(0, proposal_scale),
            }

            # Clip to prior ranges
            proposed_params_log["process_noise"] = np.clip(
                proposed_params_log["process_noise"],
                np.log(process_noise_range[0]),
                np.log(process_noise_range[1]),
            )
            proposed_params_log["observation_noise"] = np.clip(
                proposed_params_log["observation_noise"],
                np.log(observation_noise_range[0]),
                np.log(observation_noise_range[1]),
            )
            proposed_params_log["initial_variance"] = np.clip(
                proposed_params_log["initial_variance"],
                np.log(initial_variance_range[0]),
                np.log(initial_variance_range[1]),
            )

            # Convert to actual values
            proposed_params = {k: np.exp(v) for k, v in proposed_params_log.items()}

            # Evaluate
            proposed_log_likelihood = self._evaluate_params(
                proposed_params["process_noise"],
                proposed_params["observation_noise"],
                proposed_params["initial_variance"],
                trajectories,
                q_network,
                device,
                seed + iteration,
            )

            # Metropolis-Hastings acceptance
            log_acceptance_ratio = proposed_log_likelihood - current_log_likelihood

            if np.log(rng.rand()) < log_acceptance_ratio:
                current_params = proposed_params.copy()
                current_log_likelihood = proposed_log_likelihood
                accepted += 1

                if current_log_likelihood > best_log_likelihood:
                    best_log_likelihood = current_log_likelihood
                    best_params = current_params.copy()

            # Store sample after warmup
            if iteration >= n_warmup:
                mcmc_samples.append(
                    {
                        **current_params,
                        "log_likelihood": current_log_likelihood,
                        "mae": -current_log_likelihood,
                    }
                )

        # Update this SSM instance with best parameters
        self.process_noise = best_params["process_noise"]
        self.observation_noise = best_params["observation_noise"]
        self.initial_variance = best_params["initial_variance"]

        # Recreate base SSM with new parameters
        self.base_ssm = self._create_ssm_with_params(
            self.process_noise,
            self.observation_noise,
            self.initial_variance,
            seed,
        )

        return {
            "best_params": best_params,
            "best_log_likelihood": best_log_likelihood,
            "best_mae": -best_log_likelihood,
            "mcmc_samples": mcmc_samples,
            "acceptance_rate": accepted / n_iterations,
        }
