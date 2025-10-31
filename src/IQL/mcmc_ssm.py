"""
Batch Bayesian Auto-Tuning for State Space Models with MCMC.

Implements batch training for SSM preference estimators using multiple trajectories
with the same preference function. Based on:
"Batch Bayesian auto-tuning for nonlinear Kalman estimators" (Freitas Iglesias-Jr et al., 2025)

Key idea: Use MCMC to optimize Kalman Filter noise matrices (Q_t, P_t, R_t) at each timestep
using all trajectories simultaneously. Since all trajectories share the same preference dynamics,
we can pool data across trajectories to get better per-timestep noise parameter estimates.

The algorithm:
1. For each timestep t across all trajectories:
   - Use MCMC to find optimal Q_t (process noise), R_t (observation noise)
   - Update all trajectory SSMs with these optimized noise parameters
   - Process observations at timestep t
2. This allows the noise parameters to adapt to the specific dynamics at each timestep
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


class BatchSSM:
    """
    Batch SSM trainer that adjusts Q_t, R_t at each timestep using MCMC.

    This class manages multiple SSM instances (one per trajectory) and uses MCMC
    to find optimal noise parameters at each timestep by pooling data across all trajectories.
    """

    def __init__(
        self,
        ssm_type: str,
        n_objectives: int,
        n_trajectories: int,
        initial_process_noise: float = 0.01,
        initial_observation_noise: float = 0.1,
        initial_variance: float = 0.1,
        **ssm_kwargs,
    ):
        """
        Args:
            ssm_type: Type of SSM ('particle_filter', 'kalman_filter', or 'extended_kalman_filter')
            n_objectives: Number of objectives
            n_trajectories: Number of trajectories to manage
            initial_process_noise: Initial process noise value
            initial_observation_noise: Initial observation noise value
            initial_variance: Initial covariance diagonal value
            **ssm_kwargs: Additional SSM parameters (n_particles, beta, seed, etc.)
        """
        self.ssm_type = ssm_type
        self.n_objectives = n_objectives
        self.n_trajectories = n_trajectories
        self.ssm_kwargs = ssm_kwargs

        # Create one SSM per trajectory
        self.ssms = [
            self._create_ssm(
                initial_process_noise,
                initial_observation_noise,
                initial_variance,
                seed=ssm_kwargs.get("seed", 42) + i,
            )
            for i in range(n_trajectories)
        ]

        # Track current noise parameters
        self.current_process_noise = initial_process_noise
        self.current_observation_noise = initial_observation_noise

    def _create_ssm(
        self,
        process_noise: float,
        observation_noise: float,
        initial_variance: float,
        seed: int = 42,
    ) -> StateSpaceModel:
        """Create a single SSM instance."""
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

    def reset_all(self):
        """Reset all SSMs."""
        for ssm in self.ssms:
            ssm.reset()

    def _update_noise_parameters(self, process_noise: float, observation_noise: float):
        """
        Update Q and R matrices in all SSMs with new noise parameters.

        For Kalman Filter variants, this updates:
        - Q (process noise covariance): self.Q = np.eye(n) * (process_noise ** 2)
        - observation_noise attribute (used to compute R in update step)
        """
        for ssm in self.ssms:
            ssm.process_noise = process_noise
            ssm.observation_noise = observation_noise

            # Update Q matrix for KF and EKF
            if hasattr(ssm, "Q"):
                ssm.Q = np.eye(self.n_objectives) * (process_noise**2)

            # For EKF, also update obs_noise_std
            if hasattr(ssm, "obs_noise_std"):
                ssm.obs_noise_std = observation_noise

        self.current_process_noise = process_noise
        self.current_observation_noise = observation_noise

    def _evaluate_params_at_timestep(
        self,
        process_noise: float,
        observation_noise: float,
        timestep: int,
        trajectories: List[Dict],
        q_network: ActorCritic,
        device: str,
    ) -> float:
        """
        Evaluate noise parameters at a specific timestep across all trajectories.

        Creates temporary SSMs with the proposed parameters, runs them up to the timestep,
        then evaluates prediction MAE at that timestep.

        Args:
            process_noise: Process noise to evaluate
            observation_noise: Observation noise to evaluate
            timestep: Which timestep to evaluate
            trajectories: All trajectories
            q_network: Q-network for computing Q-values
            device: Device for computation

        Returns:
            Negative MAE (higher is better, for use as log-likelihood)
        """
        total_mae = []

        for traj_idx, traj in enumerate(trajectories):
            observations = traj["observations"]
            actions = traj["actions"]
            true_preferences = traj["preference_weights"]

            # Convert to numpy arrays if needed
            if isinstance(observations, list):
                observations = [np.array(obs) for obs in observations]
            if isinstance(actions, list):
                actions = np.array(actions)
            if isinstance(true_preferences, list):
                true_preferences = [np.array(pref) for pref in true_preferences]

            # Create temporary SSM
            temp_ssm = self._create_ssm(
                process_noise,
                observation_noise,
                getattr(self.ssms[0], "initial_variance", 0.1),
                seed=self.ssm_kwargs.get("seed", 42) + traj_idx * 10000,
            )

            # Run SSM up to timestep t
            for t in range(min(timestep + 1, len(observations) - 1)):
                obs = observations[t]
                action = actions[t]

                # Get Q-values
                obs_flat = (
                    obs.flatten()
                    if isinstance(obs, np.ndarray)
                    else np.array(obs).flatten()
                )
                obs_tensor = torch.FloatTensor(obs_flat).unsqueeze(0).to(device)
                with torch.no_grad():
                    _, q_values = q_network.act(obs_tensor)
                    q_values_all = q_values.squeeze(0).cpu().numpy()

                if t == timestep:
                    # Evaluate prediction at this timestep
                    true_pref = true_preferences[t]
                    predicted_pref = temp_ssm.predict(obs, None)
                    mae = np.mean(np.abs(predicted_pref - true_pref))
                    total_mae.append(mae)

                # Update SSM
                temp_ssm.update(obs, action, q_values_all)

        if len(total_mae) == 0:
            return -1.0  # Return poor score if no data

        mean_mae = np.mean(total_mae)
        return -mean_mae  # Negative MAE as log-likelihood

    def train(
        self,
        trajectories: List[Dict],
        q_network: ActorCritic,
        n_iterations_per_timestep: int = 50,
        proposal_scale: float = 0.1,
        device: str = "cuda",
        seed: int = 42,
    ) -> Dict[str, any]:
        """
        Train SSM noise parameters per-timestep using MCMC.

        At each timestep t:
        1. Use MCMC to find optimal process_noise and observation_noise
        2. Update all trajectory SSMs with these optimal parameters
        3. Process observations at timestep t

        Args:
            trajectories: List of trajectory dictionaries
            q_network: Trained Q-network for computing Q-values
            n_iterations_per_timestep: Number of MCMC iterations at each timestep
            proposal_scale: Standard deviation of random walk proposals (in log space)
            device: Device for Q-network
            seed: Random seed

        Returns:
            Dictionary with per-timestep parameters and metrics
        """
        rng = np.random.RandomState(seed)

        # Define prior ranges
        process_noise_range = (1e-4, 0.5)
        observation_noise_range = (1e-3, 1.0)

        # Find maximum trajectory length
        max_timesteps = max(len(traj["observations"]) - 1 for traj in trajectories)

        # Storage for per-timestep parameters
        timestep_params = []
        timestep_metrics = []

        # Reset all SSMs
        self.reset_all()

        # For each timestep
        for t in range(max_timesteps):
            # Initialize with current parameters
            current_params = {
                "process_noise": self.current_process_noise,
                "observation_noise": self.current_observation_noise,
            }

            # Evaluate current parameters at this timestep
            current_log_likelihood = self._evaluate_params_at_timestep(
                current_params["process_noise"],
                current_params["observation_noise"],
                t,
                trajectories,
                q_network,
                device,
            )

            best_params = current_params.copy()
            best_log_likelihood = current_log_likelihood
            accepted = 0

            # MCMC for this timestep
            for iteration in range(n_iterations_per_timestep):
                # Propose in log space
                proposed_log = {
                    "process_noise": np.log(current_params["process_noise"])
                    + rng.normal(0, proposal_scale),
                    "observation_noise": np.log(current_params["observation_noise"])
                    + rng.normal(0, proposal_scale),
                }

                # Clip to prior ranges
                proposed_log["process_noise"] = np.clip(
                    proposed_log["process_noise"],
                    np.log(process_noise_range[0]),
                    np.log(process_noise_range[1]),
                )
                proposed_log["observation_noise"] = np.clip(
                    proposed_log["observation_noise"],
                    np.log(observation_noise_range[0]),
                    np.log(observation_noise_range[1]),
                )

                # Convert to actual values
                proposed_params = {k: np.exp(v) for k, v in proposed_log.items()}

                # Evaluate proposed parameters
                proposed_log_likelihood = self._evaluate_params_at_timestep(
                    proposed_params["process_noise"],
                    proposed_params["observation_noise"],
                    t,
                    trajectories,
                    q_network,
                    device,
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

            # Update SSMs with best parameters for this timestep
            self._update_noise_parameters(
                best_params["process_noise"], best_params["observation_noise"]
            )

            # Store results
            timestep_params.append(
                {
                    "timestep": t,
                    "process_noise": best_params["process_noise"],
                    "observation_noise": best_params["observation_noise"],
                    "acceptance_rate": accepted / n_iterations_per_timestep,
                }
            )

            timestep_metrics.append(
                {
                    "timestep": t,
                    "best_mae": -best_log_likelihood,
                    "best_log_likelihood": best_log_likelihood,
                }
            )

            # Process this timestep on all trajectories with optimized parameters
            for traj_idx, traj in enumerate(trajectories):
                if t >= len(traj["observations"]) - 1:
                    continue

                obs = traj["observations"][t]
                action = traj["actions"][t]

                # Convert to numpy if needed
                obs = np.array(obs) if isinstance(obs, list) else obs

                # Get Q-values
                obs_flat = (
                    obs.flatten()
                    if isinstance(obs, np.ndarray)
                    else np.array(obs).flatten()
                )
                obs_tensor = torch.FloatTensor(obs_flat).unsqueeze(0).to(device)
                with torch.no_grad():
                    _, q_values = q_network.act(obs_tensor)
                    q_values_all = q_values.squeeze(0).cpu().numpy()

                # Update SSM
                self.ssms[traj_idx].update(obs, action, q_values_all)

        return {
            "timestep_params": timestep_params,
            "timestep_metrics": timestep_metrics,
            "max_timesteps": max_timesteps,
        }
