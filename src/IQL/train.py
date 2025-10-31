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

from src.IQL.trainer import SSMIQTrainer
from src.IQL.ssm import (
    ParticleFilter,
    ExtendedKalmanFilter,
    KalmanFilter,
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
        indices = np.random.choice(len(trajectories), n_trajectories, replace=False)
        trajectories = [trajectories[i] for i in indices]
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
        "kf": "kalman_filter",
        "kalman_filter": "kalman_filter",
        "ekf": "extended_kalman_filter",
        "extended_kalman_filter": "extended_kalman_filter",
    }

    if ssm_type not in type_mapping:
        raise ValueError(f"Unsupported SSM type: {ssm_type}. Options: pf, kf, ekf")

    full_type = type_mapping[ssm_type]

    if full_type == "particle_filter":
        pf_config = config.get("particle_filter", {})
        ssm = ParticleFilter(
            n_objectives=n_objectives,
            n_particles=pf_config.get("n_particles", 1000),
            process_noise=pf_config.get("process_noise", 0.01),
            observation_noise=pf_config.get("observation_noise", 0.1),
        )
    elif full_type == "kalman_filter":
        kf_config = config.get("kf", {})
        ssm = KalmanFilter(
            n_objectives=n_objectives,
            process_noise=kf_config.get("process_noise", 0.01),
            observation_noise=kf_config.get("observation_noise", 0.1),
            initial_variance=kf_config.get("initial_variance", 0.1),
        )
    elif full_type == "extended_kalman_filter":
        ekf_config = config.get("ekf", {})
        ssm = ExtendedKalmanFilter(
            n_objectives=n_objectives,
            process_noise=ekf_config.get("process_noise", 0.01),
            observation_noise=ekf_config.get("observation_noise", 0.1),
            initial_variance=ekf_config.get("initial_variance", 0.1),
            beta=ekf_config.get("beta", 5.0),
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
        mismatch_coef=iql_config.get("mismatch_coef", 1.0),
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

    n_updates = iql_config.get("n_updates", 10000)
    n_trajectories = len(expert_trajectories)

    # Evaluation settings
    eval_interval = config.get("eval_interval", 100)

    # Metrics tracking
    import csv

    eval_metrics_file = results_dir / "eval_metrics.csv"
    train_metrics_file = results_dir / "train_metrics.csv"
    eval_csv_writer = None
    eval_csv_file = None
    train_csv_writer = None
    train_csv_file = None

    # Best model tracking and early stopping
    best_eval_score = float("inf")
    best_cross_entropy = None
    best_preference_mae = None
    best_update = 0
    patience_counter = 0
    early_stopping_patience = config.get("early_stopping_patience", 0)

    for update in tqdm(range(n_updates), desc="Training Updates"):
        # Sample one trajectory randomly
        traj_idx = np.random.randint(0, n_trajectories)
        trajectory = expert_trajectories[traj_idx]

        # Update Q-network with this trajectory
        train_metrics = trainer.update(trajectory)

        # Log training metrics
        train_log = {
            "update": update + 1,
            "train_loss": train_metrics["loss"],
            "train_preference_mae": train_metrics["preference_mae"],
            "train_cross_entropy": train_metrics["cross_entropy"],
        }

        # Add SSM loss if available
        if "ssm_loss" in train_metrics:
            train_log["train_ssm_loss"] = train_metrics["ssm_loss"]

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

        # Log to wandb
        if config.get("use_wandb", False):
            wandb_log = {
                "train/loss": train_metrics["loss"],
                "train/preference_mae": train_metrics["preference_mae"],
                "train/cross_entropy": train_metrics["cross_entropy"],
            }

            # Add SSM loss if available
            if "ssm_loss" in train_metrics:
                wandb_log["train/ssm_loss"] = train_metrics["ssm_loss"]

            wandb.log(wandb_log, step=update + 1)

        # Evaluate periodically
        if (update + 1) % eval_interval == 0 or update == 0:
            print(f"\n{'='*70}")
            print(f"Evaluation at update {update + 1}/{n_updates}")
            print(f"{'='*70}")

            # Evaluate preference prediction accuracy on expert trajectories
            eval_weights_config = config.get("eval_weights")
            eval_weights_np = (
                np.array(eval_weights_config)
                if eval_weights_config is not None
                else None
            )
            eval_metrics = trainer.evaluate(
                expert_dir=str(expert_dir),
                n_trajectories=config.get("n_trajectories"),
                save_dir=str(results_dir / "eval_predictions"),
                update_step=update + 1,
                eval_weights=eval_weights_np,
            )

            # Combine metrics
            all_metrics = {
                "update": update + 1,
                **eval_metrics,
            }

            # Print key metrics
            print("Evaluation Performance:")
            print(
                f"  Preference MAE: {eval_metrics['mean_preference_mae']:.4f} ± {eval_metrics['std_preference_mae']:.4f}"
            )
            print(
                f"  Cross-Entropy (Imitation): {eval_metrics['mean_cross_entropy']:.4f} ± {eval_metrics['std_cross_entropy']:.4f}"
            )
            if "mean_eval_score" in eval_metrics:
                print(
                    f"  Eval Score (w1*CE + w2*MAE): {eval_metrics['mean_eval_score']:.4f} ± {eval_metrics['std_eval_score']:.4f}"
                )

            # Log to wandb
            if config.get("use_wandb", False):
                wandb_log_dict = {
                    "eval/preference_mae_mean": eval_metrics["mean_preference_mae"],
                    "eval/preference_mae_std": eval_metrics["std_preference_mae"],
                    "eval/cross_entropy_mean": eval_metrics["mean_cross_entropy"],
                    "eval/cross_entropy_std": eval_metrics["std_cross_entropy"],
                }
                if "mean_eval_score" in eval_metrics:
                    wandb_log_dict["eval/eval_score_mean"] = eval_metrics[
                        "mean_eval_score"
                    ]
                    wandb_log_dict["eval/eval_score_std"] = eval_metrics[
                        "std_eval_score"
                    ]

                wandb.log(wandb_log_dict, step=update + 1)

            # Save to eval CSV
            if eval_csv_writer is None:
                eval_csv_file = open(eval_metrics_file, "w", newline="")
                fieldnames = list(all_metrics.keys())
                eval_csv_writer = csv.DictWriter(eval_csv_file, fieldnames=fieldnames)
                eval_csv_writer.writeheader()

            eval_csv_writer.writerow(all_metrics)
            eval_csv_file.flush()

            # Check if this is the best model so far based on eval_score
            current_cross_entropy = eval_metrics["mean_cross_entropy"]
            current_preference_mae = eval_metrics["mean_preference_mae"]

            # Use eval_score if available, otherwise fall back to combined criteria
            if "mean_eval_score" in eval_metrics:
                current_eval_score = eval_metrics["mean_eval_score"]
                is_improvement = current_eval_score < best_eval_score

                if is_improvement:
                    best_eval_score = current_eval_score
                    best_cross_entropy = current_cross_entropy
                    best_preference_mae = current_preference_mae
                    best_update = update + 1
                    patience_counter = 0  # Reset patience counter

                    # Save best model
                    best_model_path = results_dir / "best_model.pt"
                    trainer.save(str(best_model_path))
                    print(f"\n{'='*70}")
                    print("NEW BEST MODEL!")
                    print(f"  Eval Score: {best_eval_score:.4f}")
                    print(f"  Cross-Entropy: {best_cross_entropy:.4f}")
                    print(f"  Preference MAE: {best_preference_mae:.4f}")
                    print(f"Saved to {best_model_path}")
                    print(f"{'='*70}\n")

                    # Save best model info
                    best_info = {
                        "best_eval_score": float(best_eval_score),
                        "best_cross_entropy": float(best_cross_entropy),
                        "best_preference_mae": float(best_preference_mae),
                        "best_update": int(best_update),
                        "total_updates": n_updates,
                    }
                    best_info_path = results_dir / "best_model_info.json"
                    with open(best_info_path, "w") as f:
                        json.dump(best_info, f, indent=2)
                else:
                    # No improvement
                    patience_counter += 1
                    if early_stopping_patience > 0:
                        print(
                            f"No improvement for {patience_counter}/{early_stopping_patience} evaluations"
                        )
                        print(
                            f"  Current Eval Score: {current_eval_score:.4f} (CE={current_cross_entropy:.4f}, MAE={current_preference_mae:.4f})"
                        )
                        print(
                            f"  Best Eval Score:    {best_eval_score:.4f} (CE={best_cross_entropy:.4f}, MAE={best_preference_mae:.4f})"
                        )
            else:
                # Fallback: use original logic if eval_score not available
                cross_entropy_improved = current_cross_entropy < (
                    best_cross_entropy
                    if best_cross_entropy is not None
                    else float("inf")
                )
                preference_mae_improved = current_preference_mae < (
                    best_preference_mae
                    if best_preference_mae is not None
                    else float("inf")
                )

                is_improvement = (
                    cross_entropy_improved
                    and (
                        best_preference_mae is None
                        or current_preference_mae <= best_preference_mae
                    )
                ) or (
                    preference_mae_improved
                    and (
                        best_cross_entropy is None
                        or current_cross_entropy <= best_cross_entropy
                    )
                )

                if is_improvement:
                    best_cross_entropy = current_cross_entropy
                    best_preference_mae = current_preference_mae
                    best_update = update + 1
                    patience_counter = 0

                    best_model_path = results_dir / "best_model.pt"
                    trainer.save(str(best_model_path))
                    print(f"\n{'='*70}")
                    print("NEW BEST MODEL!")
                    print(f"  Cross-Entropy: {best_cross_entropy:.4f}")
                    print(f"  Preference MAE: {best_preference_mae:.4f}")
                    print(f"Saved to {best_model_path}")
                    print(f"{'='*70}\n")

                    best_info = {
                        "best_cross_entropy": float(best_cross_entropy),
                        "best_preference_mae": float(best_preference_mae),
                        "best_update": int(best_update),
                        "total_updates": n_updates,
                    }
                    best_info_path = results_dir / "best_model_info.json"
                    with open(best_info_path, "w") as f:
                        json.dump(best_info, f, indent=2)
                else:
                    patience_counter += 1
                    if early_stopping_patience > 0:
                        print(
                            f"No improvement for {patience_counter}/{early_stopping_patience} evaluations"
                        )
                        print(
                            f"  Current: CE={current_cross_entropy:.4f}, MAE={current_preference_mae:.4f}"
                        )
                        print(
                            f"  Best:    CE={best_cross_entropy:.4f}, MAE={best_preference_mae:.4f}"
                        )

            # Check early stopping
            if (
                early_stopping_patience > 0
                and patience_counter >= early_stopping_patience
            ):
                print(f"\n{'='*70}")
                print("EARLY STOPPING TRIGGERED")
                print(f"No improvement for {patience_counter} consecutive evaluations")
                print(f"Best model was at update {best_update}:")
                if best_eval_score != float("inf"):
                    print(f"  Eval Score: {best_eval_score:.4f}")
                if best_cross_entropy is not None:
                    print(f"  Cross-Entropy: {best_cross_entropy:.4f}")
                if best_preference_mae is not None:
                    print(f"  Preference MAE: {best_preference_mae:.4f}")
                print(f"{'='*70}\n")
                break

    # Close CSV files
    if train_csv_file is not None:
        train_csv_file.close()
    if eval_csv_file is not None:
        eval_csv_file.close()

    print("\n" + "=" * 70)
    print("TRAINING COMPLETED")
    if early_stopping_patience > 0 and patience_counter >= early_stopping_patience:
        print(f"Stopped early after {update + 1} updates (patience exhausted)")
    else:
        print(f"Completed all {n_updates} updates")
    print("=" * 70)
    print(f"\nBest model at update {best_update}:")
    if best_eval_score != float("inf"):
        print(f"  Eval Score: {best_eval_score:.4f}")
    if best_cross_entropy is not None:
        print(f"  Cross-Entropy: {best_cross_entropy:.4f}")
    if best_preference_mae is not None:
        print(f"  Preference MAE: {best_preference_mae:.4f}")

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
