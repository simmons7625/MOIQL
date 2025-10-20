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
import torch
import wandb
import yaml
from tqdm import tqdm

import mo_gymnasium as gym
from src.Env.deepseatreasure import DeepSeaTreasureWrapper
from src.Env.highway import HighwayWrapper
from src.Env.reward_function import (
    RewardFunction,
    DSTPreferenceFunction,
    HighwayPreferenceFunction,
)
from src.IQL.trainer import ODSQILTrainer
from src.IQL.ssm import StateSpaceModel, ParticleFilter

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning, module="pygame.pkgdata")
warnings.filterwarnings("ignore", category=UserWarning, module="gymnasium.spaces.box")
warnings.filterwarnings("ignore", category=RuntimeWarning, message="divide by zero")


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


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
        return yaml.safe_load(f)


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

    for traj in trajectories:
        traj_states = np.array(traj["observations"])
        traj_actions = np.array(traj["actions"])
        traj_rewards = np.array(traj["mo_rewards"])
        traj_prefs = np.array(traj["preference_weights"])

        # Add all transitions from this trajectory
        T = len(traj_states) - 1  # number of transitions

        states.append(traj_states[:-1])  # s_t
        actions.append(traj_actions)  # a_t
        mo_rewards.append(traj_rewards)  # r_t
        preference_weights.append(traj_prefs[:-1])  # pref at s_t
        next_states.append(traj_states[1:])  # s_{t+1}

        # Last transition is terminal
        done_flags = np.zeros(T, dtype=bool)
        done_flags[-1] = True
        dones.append(done_flags)

    # Convert to numpy arrays
    data = {
        "states": np.concatenate(states, axis=0),
        "actions": np.concatenate(actions, axis=0),
        "rewards": np.concatenate(mo_rewards, axis=0),
        "preference_weights": np.concatenate(preference_weights, axis=0),
        "next_states": np.concatenate(next_states, axis=0),
        "dones": np.concatenate(dones, axis=0),
    }

    print(
        f"Loaded {len(trajectories)} trajectories with {len(data['states'])} total transitions"
    )
    print(f"  States shape: {data['states'].shape}")
    print(f"  Actions shape: {data['actions'].shape}")
    print(f"  Rewards shape: {data['rewards'].shape}")

    return data


def create_environment(env_name: str, expert_config: Dict[str, Any]):
    """Create environment with same configuration as expert."""
    if env_name == "deep_sea_treasure":
        env = gym.make("deep-sea-treasure-v0")

        # Create preference function
        preference_fn = DSTPreferenceFunction(
            contenous_decay=expert_config.get("contenous_decay", 0.01),
            init_treasure_weight=expert_config.get("init_weight", [0.8, 0.2])[0],
        )
        reward_fn = RewardFunction(preference_fn=preference_fn)

        env = DeepSeaTreasureWrapper(env=env, reward_fn=reward_fn)

    elif env_name == "mo-highway":
        env = gym.make("mo-highway-v0")

        # Create preference function
        preference_fn = HighwayPreferenceFunction(
            init_speed_weight=expert_config.get("init_weight", [0.8, 0.2])[0],
            safety_distance_threshold=expert_config.get(
                "safety_distance_threshold", 10.0
            ),
            safety_boost_factor=expert_config.get("safety_boost_factor", 1.5),
        )
        reward_fn = RewardFunction(preference_fn=preference_fn)

        env = HighwayWrapper(env=env, reward_fn=reward_fn)
    else:
        raise ValueError(f"Unsupported environment: {env_name}")

    return env


def create_ssm(config: Dict[str, Any], n_objectives: int) -> StateSpaceModel:
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


def evaluate_policy(
    trainer: ODSQILTrainer,
    env,
    n_episodes: int = 10,
    expert_config: Dict[str, Any] = None,
) -> Dict[str, Any]:
    """
    Evaluate learned policy.

    Returns:
        Dictionary containing:
        - mean_episode_reward: Average episode reward
        - std_episode_reward: Std of episode reward
        - mean_episode_length: Average episode length
        - preference_accuracy: MSE between predicted and true preferences
    """
    episode_rewards = []
    episode_lengths = []
    preference_errors = []

    for ep in range(n_episodes):
        obs, _ = env.reset()
        episode_reward = 0
        episode_length = 0
        done = False

        # Flatten obs if needed
        if len(obs.shape) > 1:
            obs = obs.flatten()

        while not done and episode_length < 1000:
            # Get action from policy
            obs_t = torch.FloatTensor(obs).unsqueeze(0).to(trainer.device)

            with torch.no_grad():
                logits, _ = trainer.q_network.act(obs_t)
                action_probs = torch.softmax(logits, dim=-1)
                action = torch.argmax(action_probs, dim=-1).item()

            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            # Track reward
            if isinstance(reward, np.ndarray):
                episode_reward += reward.sum()
            else:
                episode_reward += reward

            # Check preference accuracy if available
            if "preference_weights" in info and "mo_reward" in info:
                true_pref = np.array(info["preference_weights"])
                # mo_reward = np.array(info["mo_reward"])  # Not currently used

                # Get predicted preference from SSM
                pred_pref = trainer.ssm.predict(obs, action)

                # Compute MAE
                pref_error = np.mean(np.abs(pred_pref - true_pref))
                preference_errors.append(pref_error)

            obs = next_obs if len(next_obs.shape) == 1 else next_obs.flatten()
            episode_length += 1

        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)

    results = {
        "mean_episode_reward": np.mean(episode_rewards),
        "std_episode_reward": np.std(episode_rewards),
        "mean_episode_length": np.mean(episode_lengths),
        "std_episode_length": np.std(episode_lengths),
    }

    if len(preference_errors) > 0:
        results["mean_preference_mae"] = np.mean(preference_errors)
        results["std_preference_mae"] = np.std(preference_errors)

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

    # Get environment info from expert config
    env_name = expert_config["env_name"]
    print(f"Environment: {env_name}")

    # Create environment
    env = create_environment(env_name, expert_config)

    # Get dimensions
    if len(env.observation_space.shape) == 1:
        obs_dim = env.observation_space.shape[0]
    else:
        obs_dim = int(np.prod(env.observation_space.shape))

    action_dim = env.action_space.n
    n_objectives = env.n_objectives

    print(f"Observation dim: {obs_dim}")
    print(f"Action dim: {action_dim}")
    print(f"Number of objectives: {n_objectives}")

    # Create SSM
    ssm = create_ssm(config, n_objectives)

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

    for update in tqdm(range(n_updates), desc="Training Updates"):
        # Sample batch from expert trajectories
        indices = np.random.choice(n_data, batch_size, replace=True)

        batch_states = expert_data["states"][indices]
        batch_actions = expert_data["actions"][indices]
        batch_rewards = expert_data["rewards"][indices]
        batch_next_states = expert_data["next_states"][indices]
        batch_dones = expert_data["dones"][indices]

        # Update Q-network with expert data
        losses = trainer.update(
            states=batch_states,
            actions=batch_actions,
            rewards=batch_rewards,
            next_states=batch_next_states,
            dones=batch_dones,
            is_expert=np.ones(batch_size, dtype=np.float32),  # All expert data
        )

        # Update actor with self-generated trajectories
        n_rollouts = config.get("n_rollouts_per_update", 100)
        rollout_data = {"states": [], "preference_weights": []}
        for _ in range(n_rollouts):
            data = trainer.collect_rollout(env)
        rollout_data["states"].extend(data["states"])
        rollout_data["preference_weights"].extend(data["preference_weights"])
        actor_loss = trainer.update_actor(
            states=rollout_data["states"],
            preference_weights=rollout_data["preference_weights"],
        )

        # Combine losses
        losses["actor_loss"] = actor_loss

        # Log training losses
        if (update + 1) % 10 == 0:
            log_dict = {
                "update": update + 1,
                "total_loss": losses["total_loss"],
                "soft_iq_loss": losses["soft_iq_loss"],
                "mismatch_loss": losses["mismatch_loss"],
                "actor_loss": losses["actor_loss"],
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

            # Episode reward achievement
            eval_metrics = evaluate_policy(
                trainer, env, n_episodes=eval_episodes, expert_config=expert_config
            )

            # Combine metrics
            all_metrics = {
                "update": update + 1,
                **losses,
                **eval_metrics,
            }

            # Print key metrics
            print("Episode Performance:")
            print(
                f"  Reward: {eval_metrics['mean_episode_reward']:.2f} ± {eval_metrics['std_episode_reward']:.2f}"
            )
            print(
                f"  Length: {eval_metrics['mean_episode_length']:.1f} ± {eval_metrics['std_episode_length']:.1f}"
            )

            if "mean_preference_mae" in eval_metrics:
                print(f"  Online Pref MAE: {eval_metrics['mean_preference_mae']:.6f}")

            # Log to wandb
            if config.get("use_wandb", False):
                wandb.log(
                    {
                        "eval/episode_reward_mean": eval_metrics["mean_episode_reward"],
                        "eval/episode_reward_std": eval_metrics["std_episode_reward"],
                        "eval/episode_length_mean": eval_metrics["mean_episode_length"],
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
        if (update + 1) % 100 == 0:
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
