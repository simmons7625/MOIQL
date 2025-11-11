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

import wandb
import yaml
from tqdm import tqdm

from src.IQL.trainer import SSMIQTrainer
from src.IQL.ssm import (
    ParticleFilter,
    ExtendedKalmanFilter,
    GaussianProcessSSM,
    StateSpaceModel,
)

# Common evaluation function

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


def load_expert_trajectories(expert_dir: str, n_trajectories: int = None) -> list:
    """
    Load expert trajectories in raw format (list of trajectories).

    Args:
        expert_dir: Directory containing trajectories.json
        n_trajectories: Number of trajectories to load (None = load all)

    Returns:
        List of trajectory dictionaries
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
        trajectories = [trajectories[i] for i in range(n_trajectories)]
        print(f"Sampled {n_trajectories} trajectories from total")

    print(f"Loaded {len(trajectories)} trajectories")
    return trajectories


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
        "gp": "gaussian_process",
        "gaussian_process": "gaussian_process",
        "gpssm": "gaussian_process",
    }

    if ssm_type not in type_mapping:
        raise ValueError(f"Unsupported SSM type: {ssm_type}. Options: pf, ekf, gp")

    full_type = type_mapping[ssm_type]

    if full_type == "particle_filter":
        pf_config = config.get("pf", {})
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
    elif full_type == "gaussian_process":
        gp_config = config.get("gp", {})
        ssm = GaussianProcessSSM(
            n_objectives=n_objectives,
            length_scale=gp_config.get("length_scale", 1.0),
            signal_variance=gp_config.get("signal_variance", 1.0),
            observation_noise=gp_config.get("observation_noise", 0.1),
            kernel_type=gp_config.get("kernel_type", "rbf"),
        )
    else:
        raise ValueError(f"SSM type '{full_type}' not yet implemented")

    return ssm


def save_configs(
    save_dir: Path,
    config: Dict[str, Any],
    expert_config: Dict[str, Any],
    obs_dim: int = None,
    action_dim: int = None,
    n_objectives: int = None,
):
    """Save training and expert configs before training starts."""
    save_dir.mkdir(parents=True, exist_ok=True)

    # Add dimensions to config if provided
    if obs_dim is not None:
        config["obs_dim"] = obs_dim
    if action_dim is not None:
        config["action_dim"] = action_dim
    if n_objectives is not None:
        config["n_objectives"] = n_objectives

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
    trainer,
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


def train(config: Dict[str, Any]):
    """Main training function."""
    # Load expert configuration and trajectories
    expert_dir = Path(config["expert_dir"])
    if not expert_dir.exists():
        raise FileNotFoundError(f"Expert directory not found: {expert_dir}")

    print(f"Loading expert data from: {expert_dir}")
    expert_config = load_expert_config(str(expert_dir))
    expert_trajectories = load_expert_trajectories(
        str(expert_dir),
        n_trajectories=config.get("n_trajectories"),
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

    # Get dimensions from expert trajectories
    sample_traj = expert_trajectories[0]
    obs_dim = len(sample_traj["observations"][0])
    action_dim = max(max(traj["actions"]) for traj in expert_trajectories) + 1
    n_objectives = len(sample_traj["preference_weights"][0])

    print(f"Observation dim: {obs_dim}")
    print(f"Action dim: {action_dim}")
    print(f"Number of objectives: {n_objectives}")

    # Create SSM
    ssm = create_ssm(config, n_objectives, obs_dim, action_dim)

    # Extract IQL config (support both new nested format and old flat format for backward compatibility)
    iql_config = config.get("iql", config)

    # Create trainer
    trainer = SSMIQTrainer(
        obs_dim=obs_dim,
        action_dim=action_dim,
        n_objectives=n_objectives,
        ssm_model=ssm,
        hidden_dim=iql_config.get("hidden_dim", 256),
        lr=iql_config.get("lr", 1e-4),
        gamma=iql_config.get("gamma", 0.99),
        tau=iql_config.get("tau", 0.005),
        mismatch_coef=iql_config.get("mismatch_coef", 0.0),
        weight_decay=iql_config.get("weight_decay", 0.0),
        max_timesteps=config.get("max_timesteps"),
        device=config.get("device", "cuda"),
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
    save_configs(results_dir, config, expert_config, obs_dim, action_dim, n_objectives)

    # Training loop
    print("\n" + "=" * 70)
    print("STARTING TRAINING")
    print("=" * 70)

    n_updates = iql_config.get("n_updates", 100)
    save_step = config.get("save_step", 1000)

    # Metrics tracking
    import csv

    train_metrics_file = results_dir / "train_metrics.csv"
    train_csv_writer = None
    train_csv_file = None

    # Create checkpoints directory
    checkpoints_dir = results_dir / "checkpoints"
    checkpoints_dir.mkdir(exist_ok=True)

    # Trajectory-wise training
    n_trajectories = len(expert_trajectories)
    for update in tqdm(range(n_updates), desc="Training Updates"):
        # Select trajectory (cycle through trajectories)
        traj_idx = update % n_trajectories
        traj = expert_trajectories[traj_idx]

        # Train on single trajectory
        train_metrics = trainer.train(traj)

        # Log training metrics
        train_log = {
            "update": update + 1,
            "train_loss": train_metrics["loss"],
            "train_preference_mae": train_metrics["preference_mae"],
            "train_cross_entropy": train_metrics["cross_entropy"],
        }

        # Save to train CSV
        if train_csv_writer is None:
            train_csv_file = open(train_metrics_file, "w", newline="")
            train_fieldnames = list(train_log.keys())
            train_csv_writer = csv.DictWriter(
                train_csv_file, fieldnames=train_fieldnames
            )
            train_csv_writer.writeheader()

        train_csv_writer.writerow(train_log)
        train_csv_file.flush()

        # Save checkpoint at regular intervals
        if (update + 1) % save_step == 0:
            checkpoint_path = checkpoints_dir / f"model_update_{update + 1}.pt"
            trainer.save(str(checkpoint_path))
            print(f"\nSaved checkpoint at update {update + 1} to {checkpoint_path}")

        # Log to wandb
        if config.get("use_wandb", False):
            wandb_log = {
                "train/loss": train_metrics["loss"],
                "train/preference_mae": train_metrics["preference_mae"],
                "train/cross_entropy": train_metrics["cross_entropy"],
            }
            wandb.log(wandb_log, step=update + 1)

    # Close CSV files
    if train_csv_file is not None:
        train_csv_file.close()

    print(f"\nTraining completed. Checkpoints saved to: {checkpoints_dir}")

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
