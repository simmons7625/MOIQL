"""
Batch Bayesian Auto-Tuning for State Space Models with MCMC.

Implements batch training for SSM preference estimators using multiple trajectories
with the same preference function. Based on:
"Batch Bayesian auto-tuning for nonlinear Kalman estimators" (Freitas Iglesias-Jr et al., 2025)

Key idea: Use MCMC to optimize SSM hyperparameters (process_noise, observation_noise, initial_variance)
at each timestep using all trajectories simultaneously. Since all trajectories share the same
preference dynamics, we can pool data across trajectories to get better hyperparameter estimates.
"""

import argparse
import json
import numpy as np
import torch
from pathlib import Path
from typing import Dict, List, Tuple
from tqdm import tqdm
import yaml

from .ssm import ParticleFilter, KalmanFilter, ExtendedKalmanFilter, compute_margin
from .actor_critic import ActorCritic


def load_trajectories(trajectories_path: str) -> List[Dict]:
    """Load trajectories from JSON file."""
    with open(trajectories_path, "r") as f:
        trajectories = json.load(f)

    # Convert lists back to numpy arrays
    for traj in trajectories:
        traj["observations"] = np.array(traj["observations"])
        traj["actions"] = np.array(traj["actions"])
        traj["mo_rewards"] = np.array(traj["mo_rewards"])
        traj["preference_weights"] = np.array(traj["preference_weights"])

    return trajectories


def evaluate_ssm_on_trajectories(
    ssm,
    trajectories: List[Dict],
    q_network: ActorCritic,
    device: str = "cuda",
) -> Dict[str, float]:
    """
    Evaluate SSM performance on multiple trajectories.

    Args:
        ssm: State space model (ParticleFilter, KalmanFilter, or ExtendedKalmanFilter)
        trajectories: List of trajectory dictionaries
        q_network: Trained Q-network for computing Q-values
        device: Device for Q-network

    Returns:
        Dictionary of evaluation metrics
    """
    total_preference_mae = []
    total_prediction_accuracy = []

    for traj in trajectories:
        observations = traj["observations"]
        actions = traj["actions"]
        true_preferences = traj["preference_weights"]

        # Reset SSM for this trajectory
        ssm.reset()

        traj_preference_mae = []
        traj_accuracy = []

        for t in range(len(observations) - 1):
            obs = observations[t]
            action = actions[t]
            true_pref = true_preferences[t]

            # Get Q-values from network
            obs_tensor = torch.FloatTensor(obs.flatten()).unsqueeze(0).to(device)
            with torch.no_grad():
                _, q_values = q_network.act(obs_tensor)
                q_values_all = (
                    q_values.squeeze(0).cpu().numpy()
                )  # [action_dim, n_objectives]

            # Predict preference with SSM
            predicted_pref = ssm.predict(obs, None)

            # Compute preference MAE
            pref_mae = np.mean(np.abs(predicted_pref - true_pref))
            traj_preference_mae.append(pref_mae)

            # Compute margin-based accuracy (does predicted preference prefer the expert action?)
            margins = compute_margin(predicted_pref, q_values_all)
            predicted_action = np.argmax(margins)
            accuracy = 1.0 if predicted_action == action else 0.0
            traj_accuracy.append(accuracy)

            # Update SSM with expert action
            ssm.update(obs, action, q_values_all)

        total_preference_mae.extend(traj_preference_mae)
        total_prediction_accuracy.extend(traj_accuracy)

    metrics = {
        "mean_preference_mae": float(np.mean(total_preference_mae)),
        "std_preference_mae": float(np.std(total_preference_mae)),
        "mean_accuracy": float(np.mean(total_prediction_accuracy)),
        "std_accuracy": float(np.std(total_prediction_accuracy)),
    }

    return metrics


def create_ssm(
    ssm_type: str,
    n_objectives: int,
    process_noise: float,
    observation_noise: float,
    initial_variance: float = 0.1,
    n_particles: int = 1000,
    beta: float = 5.0,
    seed: int = 42,
):
    """Create SSM model with given hyperparameters."""
    if ssm_type == "particle_filter":
        return ParticleFilter(
            n_objectives=n_objectives,
            n_particles=n_particles,
            process_noise=process_noise,
            observation_noise=observation_noise,
            seed=seed,
        )
    elif ssm_type == "kalman_filter":
        return KalmanFilter(
            n_objectives=n_objectives,
            process_noise=process_noise,
            observation_noise=observation_noise,
            initial_variance=initial_variance,
            seed=seed,
        )
    elif ssm_type == "extended_kalman_filter":
        return ExtendedKalmanFilter(
            n_objectives=n_objectives,
            process_noise=process_noise,
            observation_noise=observation_noise,
            initial_variance=initial_variance,
            beta=beta,
            seed=seed,
        )
    else:
        raise ValueError(f"Unknown SSM type: {ssm_type}")


def mcmc_tune_hyperparameters(
    ssm_type: str,
    trajectories: List[Dict],
    q_network: ActorCritic,
    n_objectives: int,
    n_iterations: int = 1000,
    n_warmup: int = 200,
    n_particles: int = 1000,
    beta: float = 5.0,
    device: str = "cuda",
    seed: int = 42,
) -> Tuple[Dict[str, float], Dict[str, float], List[Dict[str, float]]]:
    """
    MCMC-based hyperparameter tuning using Metropolis-Hastings algorithm.

    At each timestep, we pool data from all trajectories to estimate the best
    hyperparameters using MCMC sampling.

    Args:
        ssm_type: Type of SSM
        trajectories: List of trajectories
        q_network: Trained Q-network
        n_objectives: Number of objectives
        n_iterations: Number of MCMC iterations
        n_warmup: Number of warmup iterations (burn-in)
        n_particles: Number of particles (for PF only)
        beta: Temperature parameter (for EKF only)
        device: Device for Q-network
        seed: Random seed

    Returns:
        best_params: Best hyperparameters found
        best_metrics: Metrics with best hyperparameters
        mcmc_samples: List of accepted samples during MCMC
    """
    rng = np.random.RandomState(seed)

    # Define log-uniform prior ranges for hyperparameters
    process_noise_range = (1e-4, 0.5)
    observation_noise_range = (1e-3, 1.0)
    initial_variance_range = (1e-3, 1.0)

    # Initialize with random values from prior
    current_params = {
        "process_noise": rng.uniform(*np.log(process_noise_range)),
        "observation_noise": rng.uniform(*np.log(observation_noise_range)),
        "initial_variance": rng.uniform(*np.log(initial_variance_range)),
    }

    # Convert from log space to actual values
    current_params = {k: np.exp(v) for k, v in current_params.items()}

    # Evaluate current params
    ssm = create_ssm(
        ssm_type=ssm_type,
        n_objectives=n_objectives,
        process_noise=current_params["process_noise"],
        observation_noise=current_params["observation_noise"],
        initial_variance=current_params["initial_variance"],
        n_particles=n_particles,
        beta=beta,
        seed=seed,
    )

    current_metrics = evaluate_ssm_on_trajectories(ssm, trajectories, q_network, device)
    current_log_likelihood = -current_metrics[
        "mean_preference_mae"
    ]  # Negative MAE as log-likelihood proxy

    # MCMC sampling
    mcmc_samples = []
    best_params = current_params.copy()
    best_log_likelihood = current_log_likelihood
    best_metrics = current_metrics.copy()

    accepted = 0

    print(f"\nRunning MCMC with {n_iterations} iterations ({n_warmup} warmup)...")
    pbar = tqdm(total=n_iterations)

    for iteration in range(n_iterations):
        # Propose new parameters using random walk in log space
        proposal_scale = 0.1  # Standard deviation of proposal

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

        # Evaluate proposed parameters
        ssm_proposed = create_ssm(
            ssm_type=ssm_type,
            n_objectives=n_objectives,
            process_noise=proposed_params["process_noise"],
            observation_noise=proposed_params["observation_noise"],
            initial_variance=proposed_params["initial_variance"],
            n_particles=n_particles,
            beta=beta,
            seed=seed + iteration,
        )

        proposed_metrics = evaluate_ssm_on_trajectories(
            ssm_proposed, trajectories, q_network, device
        )
        proposed_log_likelihood = -proposed_metrics["mean_preference_mae"]

        # Metropolis-Hastings acceptance criterion
        log_acceptance_ratio = proposed_log_likelihood - current_log_likelihood

        if np.log(rng.rand()) < log_acceptance_ratio:
            # Accept proposal
            current_params = proposed_params.copy()
            current_log_likelihood = proposed_log_likelihood
            current_metrics = proposed_metrics.copy()
            accepted += 1

            # Update best if better than current best
            if current_log_likelihood > best_log_likelihood:
                best_log_likelihood = current_log_likelihood
                best_params = current_params.copy()
                best_metrics = current_metrics.copy()

        # Store sample after warmup
        if iteration >= n_warmup:
            mcmc_samples.append(
                {
                    **current_params,
                    "log_likelihood": current_log_likelihood,
                    "mae": current_metrics["mean_preference_mae"],
                    "accuracy": current_metrics["mean_accuracy"],
                }
            )

        pbar.update(1)
        pbar.set_postfix(
            {
                "accept_rate": f"{accepted/(iteration+1):.3f}",
                "best_mae": f"{best_metrics['mean_preference_mae']:.4f}",
                "current_mae": f"{current_metrics['mean_preference_mae']:.4f}",
            }
        )

    pbar.close()

    acceptance_rate = accepted / n_iterations
    print(f"\nMCMC acceptance rate: {acceptance_rate:.3f}")
    print(f"Collected {len(mcmc_samples)} posterior samples")

    return best_params, best_metrics, mcmc_samples


def grid_search_hyperparameters(
    ssm_type: str,
    trajectories: List[Dict],
    q_network: ActorCritic,
    n_objectives: int,
    process_noise_values: List[float],
    observation_noise_values: List[float],
    initial_variance_values: List[float] = None,
    n_particles: int = 1000,
    beta: float = 5.0,
    device: str = "cuda",
    seed: int = 42,
) -> Tuple[Dict[str, float], Dict[str, float]]:
    """
    Grid search over SSM hyperparameters using batch evaluation.

    Args:
        ssm_type: Type of SSM (particle_filter, kalman_filter, extended_kalman_filter)
        trajectories: List of trajectories for evaluation
        q_network: Trained Q-network
        n_objectives: Number of objectives
        process_noise_values: List of process noise values to try
        observation_noise_values: List of observation noise values to try
        initial_variance_values: List of initial variance values (for KF/EKF only)
        n_particles: Number of particles (for PF only)
        beta: Temperature parameter (for EKF only)
        device: Device for Q-network
        seed: Random seed

    Returns:
        best_params: Best hyperparameters found
        best_metrics: Metrics achieved with best hyperparameters
    """
    if initial_variance_values is None:
        initial_variance_values = [0.1]

    best_mae = float("inf")
    best_params = None
    best_metrics = None

    total_configs = (
        len(process_noise_values)
        * len(observation_noise_values)
        * len(initial_variance_values)
    )

    print(f"\nGrid search over {total_configs} configurations...")
    pbar = tqdm(total=total_configs)

    for process_noise in process_noise_values:
        for observation_noise in observation_noise_values:
            for initial_variance in initial_variance_values:
                # Create SSM with these hyperparameters
                ssm = create_ssm(
                    ssm_type=ssm_type,
                    n_objectives=n_objectives,
                    process_noise=process_noise,
                    observation_noise=observation_noise,
                    initial_variance=initial_variance,
                    n_particles=n_particles,
                    beta=beta,
                    seed=seed,
                )

                # Evaluate on all trajectories
                metrics = evaluate_ssm_on_trajectories(
                    ssm, trajectories, q_network, device
                )

                # Track best based on preference MAE
                if metrics["mean_preference_mae"] < best_mae:
                    best_mae = metrics["mean_preference_mae"]
                    best_params = {
                        "process_noise": process_noise,
                        "observation_noise": observation_noise,
                        "initial_variance": initial_variance,
                    }
                    best_metrics = metrics

                pbar.update(1)
                pbar.set_postfix(
                    {
                        "best_mae": f"{best_mae:.4f}",
                        "current_mae": f"{metrics['mean_preference_mae']:.4f}",
                    }
                )

    pbar.close()

    return best_params, best_metrics


def batch_train_ssm(config: dict):
    """
    Batch training for SSM using Bayesian auto-tuning.

    Args:
        config: Configuration dictionary
    """
    # Load configuration
    trajectories_path = config["trajectories_path"]
    model_path = config["model_path"]
    ssm_type = config["ssm_type"]
    output_dir = config["output_dir"]

    # Hyperparameter search space
    process_noise_values = config.get(
        "process_noise_values", [0.001, 0.01, 0.05, 0.1, 0.2]
    )
    observation_noise_values = config.get(
        "observation_noise_values", [0.01, 0.05, 0.1, 0.2, 0.5]
    )
    initial_variance_values = config.get(
        "initial_variance_values", [0.01, 0.05, 0.1, 0.2]
    )
    n_particles = config.get("n_particles", 1000)
    beta = config.get("beta", 5.0)
    seed = config.get("seed", 42)
    device = config.get("device", "cuda")

    # Load trajectories
    print(f"Loading trajectories from {trajectories_path}")
    trajectories = load_trajectories(trajectories_path)
    print(f"Loaded {len(trajectories)} trajectories")

    # Load Q-network
    print(f"Loading Q-network from {model_path}")
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)

    # Infer dimensions from checkpoint (handle both PPO and IQL formats)
    if "q_network" in checkpoint:
        # IQL format
        state_dict = checkpoint["q_network"]
    elif "ac_state_dict" in checkpoint:
        # PPO format
        state_dict = checkpoint["ac_state_dict"]
    else:
        raise ValueError(f"Unknown checkpoint format. Keys: {checkpoint.keys()}")

    first_layer_weight = state_dict["shared.0.weight"]
    obs_dim = first_layer_weight.shape[1]

    critic_weight = state_dict["critic.weight"]
    total_outputs = critic_weight.shape[0]

    n_objectives = 2  # Assuming 2 objectives (can be made configurable)
    action_dim = total_outputs // n_objectives

    print(
        f"Inferred dimensions: obs_dim={obs_dim}, action_dim={action_dim}, n_objectives={n_objectives}"
    )

    # Create Q-network
    q_network = ActorCritic(
        obs_dim=obs_dim,
        action_dim=action_dim,
        hidden_dim=256,  # Default hidden dim
        n_objects=n_objectives,
    ).to(device)

    q_network.load_state_dict(state_dict)
    q_network.eval()

    # MCMC settings
    use_mcmc = config.get("use_mcmc", True)
    n_iterations = config.get("n_iterations", 1000)
    n_warmup = config.get("n_warmup", 200)

    print(f"\nStarting batch Bayesian auto-tuning for {ssm_type}...")

    if use_mcmc:
        # MCMC-based hyperparameter tuning
        print(f"Using MCMC with {n_iterations} iterations ({n_warmup} warmup)")
        best_params, best_metrics, mcmc_samples = mcmc_tune_hyperparameters(
            ssm_type=ssm_type,
            trajectories=trajectories,
            q_network=q_network,
            n_objectives=n_objectives,
            n_iterations=n_iterations,
            n_warmup=n_warmup,
            n_particles=n_particles,
            beta=beta,
            device=device,
            seed=seed,
        )
    else:
        # Grid search for best hyperparameters
        print("Using grid search")
        best_params, best_metrics = grid_search_hyperparameters(
            ssm_type=ssm_type,
            trajectories=trajectories,
            q_network=q_network,
            n_objectives=n_objectives,
            process_noise_values=process_noise_values,
            observation_noise_values=observation_noise_values,
            initial_variance_values=initial_variance_values,
            n_particles=n_particles,
            beta=beta,
            device=device,
            seed=seed,
        )
        mcmc_samples = None

    # Print results
    print("\n" + "=" * 70)
    print("BATCH AUTO-TUNING RESULTS")
    print("=" * 70)
    print(f"SSM Type: {ssm_type}")
    print("\nBest Hyperparameters:")
    for key, value in best_params.items():
        print(f"  {key}: {value}")
    print("\nBest Performance:")
    for key, value in best_metrics.items():
        print(f"  {key}: {value:.4f}")
    print("=" * 70)

    # Save results
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    results = {
        "ssm_type": ssm_type,
        "method": "mcmc" if use_mcmc else "grid_search",
        "best_params": best_params,
        "best_metrics": best_metrics,
    }

    if use_mcmc:
        results["mcmc_info"] = {
            "n_iterations": n_iterations,
            "n_warmup": n_warmup,
            "n_samples": len(mcmc_samples) if mcmc_samples else 0,
        }
    else:
        results["search_space"] = {
            "process_noise_values": process_noise_values,
            "observation_noise_values": observation_noise_values,
            "initial_variance_values": initial_variance_values,
        }

    results_path = output_path / "batch_tuning_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved results to {results_path}")

    # Save MCMC samples if available
    if mcmc_samples:
        mcmc_samples_path = output_path / "mcmc_samples.json"
        with open(mcmc_samples_path, "w") as f:
            json.dump(mcmc_samples, f, indent=2)
        print(f"Saved MCMC samples to {mcmc_samples_path}")

    # Save config
    config_path = output_path / "batch_config.yaml"
    with open(config_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False)
    print(f"Saved config to {config_path}")


def main():
    """Main function for batch SSM training."""
    parser = argparse.ArgumentParser(
        description="Batch Bayesian auto-tuning for SSM models"
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to configuration file",
    )

    args = parser.parse_args()

    # Load configuration
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    # Run batch training
    batch_train_ssm(config)


if __name__ == "__main__":
    main()
