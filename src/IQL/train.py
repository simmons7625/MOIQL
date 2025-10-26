"""
Training script for Objective-Dimensional Soft Inverse Q-Learning (OD-SQIL).

This script:
1. Reads configuration from configs/iql.yaml
2. Loads expert trajectories from the reference directory
3. Trains the IQL agent with SSM preference prediction
4. Saves results and models
"""

import argparse
import json
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, Any

import numpy as np
import wandb
import yaml
from tqdm import tqdm

from src.IQL.trainer import ODSQILTrainer
from src.IQL.ssm import StateSpaceModel, ParticleFilter, ExtendedKalmanFilter, NeuralSSM

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning, module="pygame.pkgdata")
warnings.filterwarnings("ignore", category=UserWarning, module="gymnasium.spaces.box")
warnings.filterwarnings("ignore", category=RuntimeWarning, message="divide by zero")


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def infer_env_name(train_dir: str) -> str:
    """
    Infer environment name from train_dir path.

    Args:
        train_dir: Path like "dst_train/20251026_200603" or "highway_train/20251026_200603"

    Returns:
        Environment name: "deep_sea_treasure" or "mo-highway"
    """
    train_dir_lower = train_dir.lower()
    if (
        "dst" in train_dir_lower
        or "deep" in train_dir_lower
        or "treasure" in train_dir_lower
    ):
        return "deep_sea_treasure"
    elif "highway" in train_dir_lower:
        return "mo-highway"
    else:
        raise ValueError(
            f"Cannot infer environment from train_dir: {train_dir}. "
            "Expected 'dst_train' or 'highway_train' in path."
        )


def load_expert_config(expert_dir: str) -> Dict[str, Any]:
    """Load configuration from expert directory."""
    expert_path = Path(expert_dir)
    sim_config_path = expert_path / "config.yaml"

    if not sim_config_path.exists():
        raise FileNotFoundError(f"Expert config not found at {sim_config_path}")

    # Load simulation config to get train_dir
    with open(sim_config_path, "r") as f:
        sim_config = yaml.safe_load(f)

    # Load actual training config from train_dir
    train_dir = sim_config.get("train_dir")
    if not train_dir:
        raise ValueError(f"No train_dir found in {sim_config_path}")

    train_config_path = Path(train_dir) / "config.yaml"
    if not train_config_path.exists():
        raise FileNotFoundError(f"Training config not found at {train_config_path}")

    with open(train_config_path, "r") as f:
        training_config = yaml.safe_load(f)

    # Infer and add env_name to config
    env_name = infer_env_name(train_dir)
    training_config["env_name"] = env_name

    return training_config


def load_expert_trajectories(
    expert_dir: str, n_trajectories: int = None
) -> Dict[str, np.ndarray]:
    """
    Load expert trajectories from JSON file.

    Args:
        expert_dir: Directory containing trajectories.json
        n_trajectories: Number of trajectories to load (None = load all)

    Returns:
        Dictionary with states, actions, rewards, next_states, dones arrays
    """
    expert_path = Path(expert_dir)
    traj_path = expert_path / "trajectories.json"

    if not traj_path.exists():
        raise FileNotFoundError(f"Trajectories not found at {traj_path}")

    print(f"Loading expert trajectories from {traj_path}")
    with open(traj_path, "r") as f:
        trajectories = json.load(f)

    # Sample trajectories if requested
    if n_trajectories is not None and n_trajectories < len(trajectories):
        indices = np.random.choice(len(trajectories), n_trajectories, replace=False)
        trajectories = [trajectories[i] for i in indices]
        print(f"Sampled {n_trajectories} trajectories from {len(trajectories)} total")

    # Collect all transitions from selected trajectories
    states = []
    actions = []
    mo_rewards = []
    preference_weights = []
    next_states = []
    dones = []
    initial_states = []
    initial_actions = []
    initial_preferences = []

    skipped_count = 0
    for traj in trajectories:
        traj_states = np.array(traj["observations"])
        traj_actions = np.array(traj["actions"])
        traj_rewards = np.array(traj["mo_rewards"])
        traj_prefs = np.array(traj["preference_weights"])

        # Add all transitions from this trajectory
        T = len(traj_states) - 1  # number of transitions

        # Skip trajectories that are too short (need at least 1 transition)
        if T <= 0:
            skipped_count += 1
            continue

        states.append(traj_states[:-1])  # s_t
        actions.append(traj_actions)  # a_t
        mo_rewards.append(traj_rewards)  # r_t
        preference_weights.append(traj_prefs[:-1])  # pref at s_t
        next_states.append(traj_states[1:])  # s_{t+1}

        # Last transition is terminal
        done_flags = np.zeros(T, dtype=bool)
        done_flags[-1] = True
        dones.append(done_flags)

        # Store initial state separately for each trajectory
        initial_states.append(traj_states[0])  # First state
        initial_actions.append(traj_actions[0])  # First action
        initial_preferences.append(traj_prefs[0])  # First preference

    if skipped_count > 0:
        print(f"Skipped {skipped_count} trajectories with no transitions")

    # Convert to numpy arrays
    data = {
        "states": np.concatenate(states, axis=0),
        "actions": np.concatenate(actions, axis=0),
        "rewards": np.concatenate(mo_rewards, axis=0),
        "preference_weights": np.concatenate(preference_weights, axis=0),
        "next_states": np.concatenate(next_states, axis=0),
        "dones": np.concatenate(dones, axis=0),
        "initial_states": np.array(initial_states),
        "initial_actions": np.array(initial_actions),
        "initial_preferences": np.array(initial_preferences),
    }

    n_valid_trajectories = len(trajectories) - skipped_count
    print(
        f"Loaded {n_valid_trajectories} valid trajectories (skipped {skipped_count}) with {len(data['states'])} total transitions"
    )
    print(f"  States shape: {data['states'].shape}")
    print(f"  Actions shape: {data['actions'].shape}")
    print(f"  Rewards shape: {data['rewards'].shape}")

    return data


def create_ssm(
    config: Dict[str, Any], n_objectives: int, obs_dim: int, action_dim: int
) -> StateSpaceModel:
    """Create State Space Model for preference prediction."""
    ssm_type = config.get("ssm_type", "pf")

    # Map short names to full names for backward compatibility
    type_mapping = {
        "pf": "particle_filter",
        "particle_filter": "particle_filter",
        "ekf": "extended_kalman_filter",
        "extended_kalman_filter": "extended_kalman_filter",
        "neural": "neural_ssm",
        "neural_ssm": "neural_ssm",
    }

    if ssm_type not in type_mapping:
        raise ValueError(f"Unsupported SSM type: {ssm_type}. Options: pf, ekf, neural")

    full_type = type_mapping[ssm_type]

    if full_type == "particle_filter":
        pf_config = config.get("particle_filter", {})
        ssm = ParticleFilter(
            n_objectives=n_objectives,
            n_particles=pf_config.get("n_particles", 1000),
            process_noise=pf_config.get("process_noise", 0.01),
            observation_noise=pf_config.get("observation_noise", 0.1),
        )
    elif full_type == "extended_kalman_filter":
        ekf_config = config.get("ekf", {})
        ssm = ExtendedKalmanFilter(
            n_objectives=n_objectives,
            process_noise=ekf_config.get("process_noise", 0.01),
            observation_noise=ekf_config.get("observation_noise", 0.1),
            initial_variance=ekf_config.get("initial_variance", 0.1),
        )
    elif full_type == "neural_ssm":
        neural_config = config.get("neural_ssm", {})
        ssm = NeuralSSM(
            obs_dim=obs_dim,
            action_dim=action_dim,
            n_objectives=n_objectives,
            hidden_dim=neural_config.get("hidden_dim", 64),
            n_layers=neural_config.get("n_layers", 2),
            learning_rate=neural_config.get("learning_rate", 0.001),
            device=config.get("device", "cuda"),
        )
    else:
        raise ValueError(f"SSM type '{full_type}' not yet implemented")

    return ssm


def save_configs(
    save_dir: Path,
    config: Dict[str, Any],
    expert_config: Dict[str, Any],
):
    """Save training and expert configs before training starts."""
    save_dir.mkdir(parents=True, exist_ok=True)

    # Save training config
    config_path = save_dir / "config.yaml"
    with open(config_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False)
    print(f"Saved config to {config_path}")

    # Save expert config reference
    expert_config_path = save_dir / "expert_config.yaml"
    with open(expert_config_path, "w") as f:
        yaml.dump(expert_config, f, default_flow_style=False)
    print(f"Saved expert config to {expert_config_path}")


def save_model_and_info(
    save_dir: Path,
    trainer: ODSQILTrainer,
):
    """Save model and model info after training."""
    # Save model
    model_path = save_dir / "final_model.pt"
    trainer.save(str(model_path))
    print(f"Saved model to {model_path}")

    # Save model info
    model_info = {
        "total_parameters": sum(p.numel() for p in trainer.q_network.parameters()),
        "trainable_parameters": sum(
            p.numel() for p in trainer.q_network.parameters() if p.requires_grad
        ),
        "device": str(trainer.device),
    }

    model_info_path = save_dir / "model_info.json"
    with open(model_info_path, "w") as f:
        json.dump(model_info, f, indent=2)
    print(f"Saved model info to {model_info_path}")


def evaluate(
    trainer: ODSQILTrainer,
    expert_data: Dict[str, np.ndarray],
    n_samples: int = 100,
) -> Dict[str, Any]:
    """
    Evaluate preference prediction accuracy on expert data.

    Args:
        trainer: OD-SQIL trainer
        expert_data: Dictionary containing expert trajectories
        n_samples: Number of samples to evaluate

    Returns:
        Dictionary containing:
        - mean_preference_mae: Mean absolute error for preference prediction
        - std_preference_mae: Std of preference MAE
    """
    # Sample random transitions from expert data
    n_data = len(expert_data["states"])
    indices = np.random.choice(n_data, min(n_samples, n_data), replace=False)

    preference_errors = []

    for idx in indices:
        state = expert_data["states"][idx]
        action = expert_data["actions"][idx]
        true_pref = expert_data["preference_weights"][idx]

        # Get predicted preference from SSM
        pred_pref = trainer.ssm.predict(state, action)

        # Compute MAE for first objective (treasure weight)
        pref_error = np.abs(pred_pref[0] - true_pref[0])
        preference_errors.append(pref_error)

    results = {
        "mean_preference_mae": np.mean(preference_errors),
        "std_preference_mae": np.std(preference_errors),
    }

    return results


def train(config: Dict[str, Any]):
    """Main training function."""
    # Load expert configuration and trajectories
    expert_dir = Path(config["expert_dir"])
    if not expert_dir.exists():
        raise FileNotFoundError(f"Expert directory not found: {expert_dir}")

    print(f"Loading expert data from: {expert_dir}")
    expert_config = load_expert_config(str(expert_dir))
    expert_data = load_expert_trajectories(
        str(expert_dir), n_trajectories=config.get("n_trajectories")
    )

    # Get environment name from expert_config (auto-inferred from train_dir path)
    env_name = expert_config["env_name"]
    print(f"Environment: {env_name}")

    # Use expert config for environment settings to ensure exact match with expert data
    # Only override with IQL config for IQL-specific hyperparameters (lr, gamma, tau, etc.)
    # Expert config has priority for environment settings
    env_config = {**config, **expert_config}

    print("Environment settings loaded from expert config:")
    print(f"  use_local_obs: {env_config.get('use_local_obs')}")
    print(f"  local_obs_size: {env_config.get('local_obs_size')}")
    print(f"  max_num_treasure: {env_config.get('max_num_treasure')}")
    print(f"  max_timesteps: {env_config.get('max_timesteps')}")

    # Get dimensions from expert data
    obs_dim = expert_data["states"].shape[1]
    action_dim = int(expert_data["actions"].max() + 1)  # Discrete actions
    n_objectives = expert_data["preference_weights"].shape[1]

    print(f"Observation dim: {obs_dim}")
    print(f"Action dim: {action_dim}")
    print(f"Number of objectives: {n_objectives}")

    # Create SSM
    ssm = create_ssm(config, n_objectives, obs_dim, action_dim)

    # Create trainer
    trainer = ODSQILTrainer(
        obs_dim=obs_dim,
        action_dim=action_dim,
        n_objectives=n_objectives,
        ssm_model=ssm,
        hidden_dim=config["hidden_dim"],
        lr=config["lr"],
        gamma=config["gamma"],
        tau=config["tau"],
        mismatch_coef=config["mismatch_coef"],
        max_timesteps=config.get("max_timesteps"),
        device=config["device"],
    )

    print(f"Device: {trainer.device}")
    print(f"Model parameters: {sum(p.numel() for p in trainer.q_network.parameters())}")

    # Initialize wandb
    if config.get("use_wandb", False):
        wandb.init(
            project=config.get("wandb_project", "MOIQL"),
            name=f"iql_{env_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            config={**config, **{"expert_config": expert_config}},
        )

    # Create results directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = Path(config["save_dir"]) / timestamp
    results_dir.mkdir(parents=True, exist_ok=True)

    # Save configs before training
    save_configs(results_dir, config, expert_config)

    # Training loop
    print("\n" + "=" * 70)
    print("STARTING TRAINING")
    print("=" * 70)

    batch_size = config["batch_size"]
    n_updates = config["n_updates"]
    n_data = len(expert_data["states"])

    # Evaluation settings
    eval_interval = config.get("eval_interval", 100)
    eval_episodes = config.get("eval_episodes", 10)

    # Metrics tracking
    import csv

    metrics_file = results_dir / "metrics.csv"
    csv_writer = None
    csv_file = None

    # Get number of initial states available
    n_initial_states = len(expert_data["initial_states"])

    for update in tqdm(range(n_updates), desc="Training Updates"):
        # Sample batch from expert trajectories
        indices = np.random.choice(n_data, batch_size, replace=True)

        batch_states = expert_data["states"][indices]
        batch_actions = expert_data["actions"][indices]
        batch_rewards = expert_data["rewards"][indices]
        batch_next_states = expert_data["next_states"][indices]
        batch_dones = expert_data["dones"][indices]

        # Sample initial states (sample randomly from available initial states)
        init_indices = np.random.choice(n_initial_states, size=batch_size, replace=True)
        batch_initial_states = expert_data["initial_states"][init_indices]
        batch_initial_actions = expert_data["initial_actions"][init_indices]
        batch_initial_preferences = expert_data["initial_preferences"][init_indices]

        # Update Q-network with expert data
        losses = trainer.update(
            states=batch_states,
            actions=batch_actions,
            rewards=batch_rewards,
            next_states=batch_next_states,
            dones=batch_dones,
            is_expert=np.ones(batch_size, dtype=np.float32),  # All expert data
            initial_states=batch_initial_states,
            initial_actions=batch_initial_actions,
            initial_preferences=batch_initial_preferences,
        )

        # Log training losses
        if (update + 1) % 10 == 0:
            log_dict = {
                "update": update + 1,
                "total_loss": losses["total_loss"],
                "soft_iq_loss": losses["soft_iq_loss"],
                "mismatch_loss": losses["mismatch_loss"],
            }

            # Log preference weights
            for i, w in enumerate(losses["mean_preference"]):
                log_dict[f"preference_obj{i}"] = w

            if config.get("use_wandb", False):
                wandb.log(log_dict, step=update + 1)

        # Evaluate periodically
        if (update + 1) % eval_interval == 0 or update == 0:
            print(f"\n{'='*70}")
            print(f"Evaluation at update {update + 1}/{n_updates}")
            print(f"{'='*70}")

            # Evaluate preference prediction accuracy on expert data
            eval_metrics = evaluate(trainer, expert_data, n_samples=eval_episodes * 10)

            # Combine metrics
            all_metrics = {
                "update": update + 1,
                **losses,
                **eval_metrics,
            }

            # Print key metrics
            print("Preference Prediction Performance:")
            print(
                f"  Preference MAE: {eval_metrics['mean_preference_mae']:.4f} ± {eval_metrics['std_preference_mae']:.4f}"
            )

            # Log to wandb
            if config.get("use_wandb", False):
                wandb.log(
                    {
                        "eval/preference_mae_mean": eval_metrics["mean_preference_mae"],
                        "eval/preference_mae_std": eval_metrics["std_preference_mae"],
                    },
                    step=update + 1,
                )

            # Save to CSV
            if csv_writer is None:
                csv_file = open(metrics_file, "w", newline="")
                fieldnames = list(all_metrics.keys())
                csv_writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
                csv_writer.writeheader()

            csv_writer.writerow(all_metrics)
            csv_file.flush()

        # Progress update
        if (update + 1) % eval_interval == 0:
            pref_str = ", ".join([f"{w:.3f}" for w in losses["mean_preference"]])
            print(
                f"Update {update + 1}/{n_updates} | "
                f"Loss: {losses['total_loss']:.4f} | "
                f"Pref: [{pref_str}]"
            )

    # Close CSV file
    if csv_file is not None:
        csv_file.close()

    print("\n" + "=" * 70)
    print("TRAINING COMPLETED")
    print("=" * 70)

    # Save final model and info
    save_model_and_info(results_dir, trainer)

    if config.get("use_wandb", False):
        wandb.finish()

    print(f"\nAll results saved to: {results_dir}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Train Objective-Dimensional Soft IQL")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/iql.yaml",
        help="Path to configuration file",
    )

    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)
    print(f"Loaded config from {args.config}")

    # Train
    train(config)


if __name__ == "__main__":
    main()
