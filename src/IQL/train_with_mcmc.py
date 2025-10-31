"""
IQL Training with MCMC-based SSM Hyperparameter Tuning.

Integrates batch Bayesian auto-tuning into the IQL training loop:
1. Adjust SSM hyperparameters using MCMC on all trajectories
2. Train Q-network with current SSM
3. Repeat for num_updates iterations

This allows the SSM to adapt to the improving Q-network during training.
"""

import argparse
import json
import numpy as np
import torch
from pathlib import Path
from typing import Dict, List
from tqdm import tqdm
import yaml
from datetime import datetime

from .ssm import ParticleFilter, KalmanFilter, ExtendedKalmanFilter
from .trainer import SSMIQTrainer


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


def mcmc_update_ssm_params(
    current_params: Dict[str, float],
    trajectories: List[Dict],
    trainer: SSMIQTrainer,
    n_iterations: int = 50,
    proposal_scale: float = 0.1,
    device: str = "cuda",
    seed: int = 42,
) -> Dict[str, float]:
    """
    Single MCMC update step for SSM hyperparameters.

    Uses all trajectories to evaluate the current Q-network + SSM combination.

    Args:
        current_params: Current SSM hyperparameters
        trajectories: All training trajectories
        trainer: IQL trainer with current Q-network
        n_iterations: Number of MCMC iterations for this update
        proposal_scale: Scale of random walk proposals
        device: Device for computation
        seed: Random seed

    Returns:
        Updated SSM parameters
    """
    rng = np.random.RandomState(seed)

    # Define prior ranges
    process_noise_range = (1e-4, 0.5)
    observation_noise_range = (1e-3, 1.0)
    initial_variance_range = (1e-3, 1.0)

    # Evaluate current parameters
    current_log_likelihood = evaluate_params_on_trajectories(
        current_params, trajectories, trainer, device
    )

    best_params = current_params.copy()
    best_log_likelihood = current_log_likelihood
    accepted = 0

    for iteration in range(n_iterations):
        # Propose new parameters in log space
        proposed_params_log = {
            k: np.log(current_params[k]) + rng.normal(0, proposal_scale)
            for k in current_params.keys()
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
        proposed_log_likelihood = evaluate_params_on_trajectories(
            proposed_params, trajectories, trainer, device
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

    acceptance_rate = accepted / n_iterations

    return best_params, acceptance_rate


def evaluate_params_on_trajectories(
    params: Dict[str, float],
    trajectories: List[Dict],
    trainer: SSMIQTrainer,
    device: str,
) -> float:
    """
    Evaluate SSM parameters by running trajectories through current Q-network + SSM.

    Returns negative MAE as log-likelihood proxy.
    """
    # Create temporary SSM with these parameters
    temp_ssm = create_ssm(
        ssm_type=type(trainer.ssm).__name__.replace("Filter", "_filter").lower(),
        n_objectives=trainer.n_objectives,
        process_noise=params["process_noise"],
        observation_noise=params["observation_noise"],
        initial_variance=params["initial_variance"],
    )

    total_mae = []

    for traj in trajectories:
        observations = traj["observations"]
        actions = traj["actions"]
        true_preferences = traj["preference_weights"]

        # Reset SSM
        temp_ssm.reset()

        for t in range(len(observations) - 1):
            obs = observations[t]
            action = actions[t]
            true_pref = true_preferences[t]

            # Get Q-values from current Q-network
            obs_tensor = torch.FloatTensor(obs.flatten()).unsqueeze(0).to(device)
            with torch.no_grad():
                _, q_values = trainer.q_network.act(obs_tensor)
                q_values_all = q_values.squeeze(0).cpu().numpy()

            # Predict with SSM
            predicted_pref = temp_ssm.predict(obs, None)

            # Compute MAE
            mae = np.mean(np.abs(predicted_pref - true_pref))
            total_mae.append(mae)

            # Update SSM
            temp_ssm.update(obs, action, q_values_all)

    mean_mae = np.mean(total_mae)
    return -mean_mae  # Negative MAE as log-likelihood


def train_iql_with_mcmc(config: dict):
    """
    Train IQL with MCMC-based SSM hyperparameter tuning.

    Training loop:
    1. MCMC update: Adjust SSM hyperparameters using all trajectories
    2. Q-network update: Train Q-network on batch from trajectories
    3. Repeat for num_updates iterations
    """
    # Load configuration
    trajectories_path = config["trajectories_path"]
    ssm_type = config["ssm_type"]
    output_dir = config["output_dir"]

    # Training hyperparameters
    num_updates = config.get("num_updates", 10000)
    batch_size = config.get("batch_size", 256)
    mcmc_interval = config.get("mcmc_interval", 100)  # Run MCMC every N updates
    mcmc_iterations_per_update = config.get("mcmc_iterations_per_update", 20)

    # SSM hyperparameters (initial values)
    ssm_params = {
        "process_noise": config.get("initial_process_noise", 0.01),
        "observation_noise": config.get("initial_observation_noise", 0.1),
        "initial_variance": config.get("initial_initial_variance", 0.1),
    }

    # Model parameters
    n_objectives = config.get("n_objectives", 2)
    hidden_dim = config.get("hidden_dim", 256)
    lr = config.get("lr", 3e-4)
    gamma = config.get("gamma", 0.99)
    tau = config.get("tau", 0.005)
    mismatch_coef = config.get("mismatch_coef", 1.0)

    n_particles = config.get("n_particles", 1000)
    beta = config.get("beta", 5.0)
    device = config.get("device", "cuda")
    seed = config.get("seed", 42)

    # Load trajectories
    print(f"Loading trajectories from {trajectories_path}")
    trajectories = load_trajectories(trajectories_path)
    print(f"Loaded {len(trajectories)} trajectories")

    # Infer dimensions from first trajectory
    sample_obs = trajectories[0]["observations"][0]
    sample_actions = trajectories[0]["actions"]

    obs_dim = sample_obs.flatten().shape[0]
    action_dim = int(np.max(sample_actions) + 1)

    print(
        f"Inferred dimensions: obs_dim={obs_dim}, action_dim={action_dim}, n_objectives={n_objectives}"
    )

    # Create initial SSM
    ssm = create_ssm(
        ssm_type=ssm_type,
        n_objectives=n_objectives,
        process_noise=ssm_params["process_noise"],
        observation_noise=ssm_params["observation_noise"],
        initial_variance=ssm_params["initial_variance"],
        n_particles=n_particles,
        beta=beta,
        seed=seed,
    )

    # Create trainer
    trainer = SSMIQTrainer(
        obs_dim=obs_dim,
        action_dim=action_dim,
        n_objectives=n_objectives,
        ssm_model=ssm,
        hidden_dim=hidden_dim,
        lr=lr,
        gamma=gamma,
        tau=tau,
        mismatch_coef=mismatch_coef,
        device=device,
    )

    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = Path(output_dir) / timestamp
    save_dir.mkdir(parents=True, exist_ok=True)

    # Save initial config
    with open(save_dir / "config.yaml", "w") as f:
        yaml.dump(config, f, default_flow_style=False)

    # Training metrics
    train_metrics = []
    ssm_param_history = []

    print(f"\nStarting training for {num_updates} updates...")
    print(
        f"MCMC runs every {mcmc_interval} updates with {mcmc_iterations_per_update} iterations"
    )

    pbar = tqdm(total=num_updates)

    for update in range(num_updates):
        # MCMC update for SSM parameters (every mcmc_interval updates)
        if update % mcmc_interval == 0:
            pbar.set_description("MCMC SSM tuning")
            ssm_params, accept_rate = mcmc_update_ssm_params(
                current_params=ssm_params,
                trajectories=trajectories,
                trainer=trainer,
                n_iterations=mcmc_iterations_per_update,
                device=device,
                seed=seed + update,
            )

            # Update trainer's SSM with new parameters
            trainer.ssm = create_ssm(
                ssm_type=ssm_type,
                n_objectives=n_objectives,
                process_noise=ssm_params["process_noise"],
                observation_noise=ssm_params["observation_noise"],
                initial_variance=ssm_params["initial_variance"],
                n_particles=n_particles,
                beta=beta,
                seed=seed + update,
            )

            ssm_param_history.append(
                {
                    "update": update,
                    **ssm_params,
                    "acceptance_rate": accept_rate,
                }
            )

        # Sample batch from trajectories
        pbar.set_description("Training Q-network")
        batch = sample_batch_from_trajectories(trajectories, batch_size, seed + update)

        # Train Q-network on batch
        metrics = trainer.update(
            observations=batch["observations"],
            actions=batch["actions"],
            next_observations=batch["next_observations"],
            q_values_all_batch=batch["q_values_all"],
        )

        train_metrics.append(
            {
                "update": update,
                **metrics,
                **{f"ssm_{k}": v for k, v in ssm_params.items()},
            }
        )

        pbar.update(1)
        pbar.set_postfix(
            {
                "q_loss": f"{metrics.get('q_loss', 0):.4f}",
                "proc_noise": f"{ssm_params['process_noise']:.4f}",
            }
        )

        # Save checkpoint periodically
        if (update + 1) % 1000 == 0:
            torch.save(
                {
                    "q_network": trainer.q_network.state_dict(),
                    "q_target": trainer.q_target.state_dict(),
                    "q_optimizer": trainer.q_optimizer.state_dict(),
                    "ssm_params": ssm_params,
                    "update": update,
                },
                save_dir / f"checkpoint_{update+1}.pt",
            )

    pbar.close()

    # Save final model
    torch.save(
        {
            "q_network": trainer.q_network.state_dict(),
            "q_target": trainer.q_target.state_dict(),
            "ssm_params": ssm_params,
        },
        save_dir / "final_model.pt",
    )

    # Save metrics
    import pandas as pd

    pd.DataFrame(train_metrics).to_csv(save_dir / "train_metrics.csv", index=False)
    pd.DataFrame(ssm_param_history).to_csv(
        save_dir / "ssm_param_history.csv", index=False
    )

    print(f"\nTraining complete! Results saved to {save_dir}")
    print("Final SSM parameters:")
    for k, v in ssm_params.items():
        print(f"  {k}: {v:.6f}")


def sample_batch_from_trajectories(
    trajectories: List[Dict],
    batch_size: int,
    seed: int,
) -> Dict[str, np.ndarray]:
    """Sample a batch of transitions from trajectories."""
    rng = np.random.RandomState(seed)

    # Collect all transitions
    all_transitions = []
    for traj in trajectories:
        for t in range(len(traj["observations"]) - 1):
            all_transitions.append(
                {
                    "observation": traj["observations"][t],
                    "action": traj["actions"][t],
                    "next_observation": traj["observations"][t + 1],
                    "mo_reward": traj["mo_rewards"][t],
                    "preference_weights": traj["preference_weights"][t],
                }
            )

    # Sample batch
    indices = rng.choice(len(all_transitions), size=batch_size, replace=True)
    batch = [all_transitions[i] for i in indices]

    # Stack into arrays
    return {
        "observations": np.array([t["observation"] for t in batch]),
        "actions": np.array([t["action"] for t in batch]),
        "next_observations": np.array([t["next_observation"] for t in batch]),
        "mo_rewards": np.array([t["mo_reward"] for t in batch]),
        "preference_weights": np.array([t["preference_weights"] for t in batch]),
        "q_values_all": None,  # Will be computed by trainer
    }


def main():
    """Main function for IQL training with MCMC."""
    parser = argparse.ArgumentParser(
        description="Train IQL with MCMC-based SSM hyperparameter tuning"
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

    # Run training
    train_iql_with_mcmc(config)


if __name__ == "__main__":
    main()
