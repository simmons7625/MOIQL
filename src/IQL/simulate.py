"""
Simulation script to collect trajectory data using trained IQL model.

This script loads a trained IQL model (Q-network + SSM) and simulates episodes,
recording:
- observations
- actions
- mo_rewards (multi-objective rewards from environment)
- preference_weights_truth (ground truth from environment)
- preference_weights_predicted (predicted by SSM)
"""

import argparse
import json
from pathlib import Path
from typing import Dict

import numpy as np
import torch
import yaml
from tqdm import tqdm

import mo_gymnasium as gym


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def infer_dimensions_from_checkpoint(checkpoint_path: str) -> dict:
    """Infer obs_dim, action_dim, n_objectives from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    # Get dimensions from q_network state dict
    # The first layer should be: input_dim -> hidden_dim
    first_layer_weight = checkpoint["q_network"]["shared.0.weight"]
    obs_dim = first_layer_weight.shape[1]

    # The critic layer should be: hidden_dim -> action_dim * n_objectives
    critic_weight = checkpoint["q_network"]["critic.weight"]
    total_outputs = critic_weight.shape[0]

    # We need to infer action_dim and n_objectives
    # For deep sea treasure: action_dim=4, n_objectives=2
    # We'll assume n_objectives=2 for now (common case)
    n_objectives = 2
    action_dim = total_outputs // n_objectives

    return {
        "obs_dim": obs_dim,
        "action_dim": action_dim,
        "n_objectives": n_objectives,
    }


def load_iql_model(
    model_path: str,
    training_config: dict,
    device: str = "cuda",
):
    """Load trained IQL model (Q-network + SSM) from checkpoint."""
    # Get dimensions - either from config or infer from checkpoint
    if (
        "obs_dim" not in training_config
        or "action_dim" not in training_config
        or "n_objectives" not in training_config
    ):
        print("Inferring dimensions from checkpoint...")
        dims = infer_dimensions_from_checkpoint(model_path)
        obs_dim = dims["obs_dim"]
        action_dim = dims["action_dim"]
        n_objectives = dims["n_objectives"]
        print(
            f"Inferred: obs_dim={obs_dim}, action_dim={action_dim}, n_objectives={n_objectives}"
        )
    else:
        obs_dim = training_config["obs_dim"]
        action_dim = training_config["action_dim"]
        n_objectives = training_config["n_objectives"]

    # Import based on SSM type
    ssm_type = training_config.get("ssm_type", "pf")

    from src.IQL.trainer import SSMIQTrainer
    from src.IQL.ssm import ParticleFilter, ExtendedKalmanFilter, KalmanFilter

    # Create SSM based on type
    if ssm_type == "pf":
        pf_config = training_config.get("particle_filter", {})
        ssm = ParticleFilter(
            n_objectives=n_objectives,
            n_particles=pf_config.get("n_particles", 1000),
            process_noise=pf_config.get("process_noise", 0.01),
            observation_noise=pf_config.get("observation_noise", 0.1),
        )
    elif ssm_type == "kf":
        kf_config = training_config.get("kf", {})
        ssm = KalmanFilter(
            n_objectives=n_objectives,
            process_noise=kf_config.get("process_noise", 0.01),
            observation_noise=kf_config.get("observation_noise", 0.1),
            initial_variance=kf_config.get("initial_variance", 0.1),
        )
    elif ssm_type == "ekf":
        ekf_config = training_config.get("ekf", {})
        ssm = ExtendedKalmanFilter(
            n_objectives=n_objectives,
            process_noise=ekf_config.get("process_noise", 0.01),
            observation_noise=ekf_config.get("observation_noise", 0.1),
            initial_variance=ekf_config.get("initial_variance", 0.1),
            beta=ekf_config.get("beta", 1.0),
        )
    else:
        raise ValueError(f"Unknown SSM type: {ssm_type}")

    # Create trainer
    trainer = SSMIQTrainer(
        obs_dim=obs_dim,
        action_dim=action_dim,
        n_objectives=n_objectives,
        ssm_model=ssm,
        hidden_dim=training_config.get("hidden_dim", 256),
        lr=training_config.get("lr", 3e-4),
        gamma=training_config.get("gamma", 0.99),
        tau=training_config.get("tau", 0.005),
        mismatch_coef=training_config.get("mismatch_coef", 1.0),
        max_timesteps=training_config.get("max_timesteps"),
        device=device,
    )

    # Load checkpoint
    trainer.load(str(model_path))
    trainer.q_network.eval()

    return trainer


def create_environment(
    env_name: str, training_config: dict, render: bool = False, render_fps: int = 10
):
    """Create environment based on training config."""
    if "dst" in env_name.lower() or "deep" in env_name.lower():
        from src.dst.env import DeepSeaTreasureWrapper
        from src.dst.reward_function import RewardFunction, DSTPreferenceFunction

        # Handle init_weight - can be a list or scalar
        init_weight = training_config.get("init_weight", 0.8)
        if isinstance(init_weight, list):
            init_weight = init_weight[0]

        # Create reward function
        preference_fn = DSTPreferenceFunction(
            contenous_decay=training_config["contenous_decay"],
            init_treasure_weight=init_weight,
        )
        reward_fn = RewardFunction(preference_fn=preference_fn)

        # Create environment
        env = DeepSeaTreasureWrapper(
            env=gym.make(
                "deep-sea-treasure-v0", render_mode="human" if render else None
            ),
            reward_fn=reward_fn,
            max_num_treasure=training_config.get("max_num_treasure", 1),
            max_timestep=training_config.get("max_timesteps"),
            use_local_obs=training_config.get("use_local_obs", True),
            local_obs_size=training_config.get("local_obs_size", 3),
            render_mode="human" if render else None,
            render_fps=render_fps,
        )

        return env, reward_fn

    elif "highway" in env_name.lower():
        from src.highway.env import HighwayWrapper
        from src.highway.reward_function import (
            RewardFunction,
            HighwayPreferenceFunction,
        )

        # Create reward function
        preference_fn = HighwayPreferenceFunction(
            change_interval=training_config.get("change_interval", 10),
        )
        reward_fn = RewardFunction(preference_fn=preference_fn)

        # Create environment
        env = HighwayWrapper(
            reward_fn=reward_fn,
            max_timestep=training_config.get("max_timesteps"),
            normalize_obs=training_config.get("normalize_obs", True),
            render_mode="human" if render else None,
        )

        return env, reward_fn
    else:
        raise ValueError(f"Unknown environment: {env_name}")


def collect_trajectory(
    env,
    trainer,
    reward_fn,
    max_episode_steps: int = None,
    deterministic: bool = False,
) -> Dict:
    """
    Collect a single trajectory using IQL model.

    Records both ground truth preferences from environment and predicted preferences from SSM.

    Args:
        env: Environment
        trainer: IQL trainer (contains Q-network and SSM)
        reward_fn: Reward function
        max_episode_steps: Maximum timesteps per episode (None = use done signal only)
        deterministic: If True, take argmax action; otherwise sample from softmax policy

    Returns:
        Dictionary containing trajectory data with both truth and predicted preferences
    """
    trajectory = {
        "observations": [],
        "actions": [],
        "mo_rewards": [],
        "preference_weights_truth": [],  # Ground truth from environment
        "preference_weights_predicted": [],  # Predicted by SSM
    }

    obs, info = env.reset()
    reward_fn.reset()

    # Reset SSM (for step-wise SSMs like PF/EKF)
    if hasattr(trainer.ssm, "reset"):
        trainer.ssm.reset()

    episode_step = 0
    observations_list = []  # Store for sequence-wise prediction

    while True:
        # Store observation
        trajectory["observations"].append(obs.copy())
        observations_list.append(obs.copy())

        # Get predicted preference from SSM
        if hasattr(trainer.ssm, "predict_sequence"):
            # Sequence-wise SSM - predict on full sequence so far
            obs_array = np.array(observations_list)
            predicted_prefs = trainer.ssm.predict_sequence(obs_array)
            predicted_pref = predicted_prefs[-1]  # Get last prediction
        else:
            # Step-wise SSM (PF/EKF)
            if hasattr(trainer.ssm, "hidden"):
                result = trainer.ssm.predict(obs, trainer.ssm.hidden)
                if isinstance(result, tuple):
                    predicted_pref, _ = result
                else:
                    predicted_pref = result
            else:
                predicted_pref = trainer.ssm.predict()

        # Get action from Q-network using predicted preference
        with torch.no_grad():
            obs_flat = obs.flatten() if len(obs.shape) > 1 else obs
            obs_tensor = torch.FloatTensor(obs_flat).unsqueeze(0).to(trainer.device)
            pred_pref_tensor = (
                torch.FloatTensor(predicted_pref).unsqueeze(0).to(trainer.device)
            )

            logits, q_values_all = trainer.q_network.act(obs_tensor, pred_pref_tensor)

            if deterministic:
                action = torch.argmax(logits, dim=1).item()
            else:
                # Sample from softmax policy
                probs = torch.softmax(logits, dim=-1)
                dist = torch.distributions.Categorical(probs=probs)
                action = dist.sample().item()

        # Take action in environment
        next_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        # Get multi-objective reward and ground truth preference
        mo_reward = info.get("mo_reward", None)
        preference_truth = info.get("preference_weights", None)

        # Store transition
        trajectory["actions"].append(action)
        trajectory["mo_rewards"].append(mo_reward)
        trajectory["preference_weights_truth"].append(preference_truth)
        trajectory["preference_weights_predicted"].append(predicted_pref.copy())

        # Update SSM for step-wise models
        if hasattr(trainer.ssm, "update"):
            q_values_np = q_values_all[0].cpu().numpy()
            trainer.ssm.update(observation=obs, action=action, q_values_all=q_values_np)

        obs = next_obs
        episode_step += 1

        # Check termination
        if max_episode_steps is not None:
            if episode_step >= max_episode_steps:
                break
        else:
            if done:
                break

    # Convert lists to numpy arrays
    trajectory["observations"] = np.array(trajectory["observations"])
    trajectory["actions"] = np.array(trajectory["actions"])

    if trajectory["mo_rewards"][0] is not None:
        trajectory["mo_rewards"] = np.array(trajectory["mo_rewards"])

    if trajectory["preference_weights_truth"][0] is not None:
        trajectory["preference_weights_truth"] = np.array(
            trajectory["preference_weights_truth"]
        )

    trajectory["preference_weights_predicted"] = np.array(
        trajectory["preference_weights_predicted"]
    )

    return trajectory


def simulate(config: dict):
    """
    Run IQL simulation to collect trajectory dataset with SSM predictions.

    Args:
        config: Simulation configuration dictionary
    """
    # Extract simulation parameters
    train_dir = Path(config["train_dir"])
    model_filename = config["model_path"]
    training_config_filename = config["training_config"]

    # Construct full paths
    model_path = train_dir / model_filename
    training_config_path = train_dir / training_config_filename

    # Load training configuration
    training_config = load_config(str(training_config_path))
    print(f"Loaded training config from {training_config_path}")

    # Load expert configuration (environment parameters)
    expert_config_path = train_dir / "expert_config.yaml"
    expert_config = load_config(str(expert_config_path))
    print(f"Loaded expert config from {expert_config_path}")

    # Extract simulation parameters
    n_episodes = config["n_episodes"]
    max_episode_steps = config.get("max_episode_steps")
    output_dir = config["output_dir"]
    deterministic = config.get("deterministic", False)
    device = config.get("device", "cuda")
    render = config.get("render", False)
    render_fps = config.get("render_fps", 10)

    # Create environment
    env_name = expert_config.get("env_name", "dst")
    env, reward_fn = create_environment(
        env_name=env_name,
        training_config=expert_config,  # Use expert_config for environment parameters
        render=render,
        render_fps=render_fps,
    )
    print(f"Created environment: {env_name}")

    # Load IQL model
    trainer = load_iql_model(
        model_path=model_path,
        training_config=training_config,
        device=device,
    )
    print(f"Loaded IQL model from {model_path}")
    print(f"SSM type: {training_config.get('ssm_type', 'pf')}")
    print(f"Using device: {device}")
    print(f"Deterministic: {deterministic}")

    # Create output directory structure matching train_dir
    # Extract the last directory name from train_dir (e.g., "ekf_coef=0" from "moiql_results/ekf_coef=0/")
    train_dir_name = train_dir.name if train_dir.name else train_dir.parts[-1]

    output_path = Path(output_dir) / train_dir_name
    output_path.mkdir(parents=True, exist_ok=True)

    # Collect trajectories
    trajectories = []
    episode_lengths = []
    episode_returns = []
    preference_maes = []

    print(f"\nCollecting {n_episodes} episodes...")
    for episode in tqdm(range(n_episodes)):
        trajectory = collect_trajectory(
            env=env,
            trainer=trainer,
            reward_fn=reward_fn,
            max_episode_steps=max_episode_steps,
            deterministic=deterministic,
        )
        trajectories.append(trajectory)

        # Track statistics
        episode_length = len(trajectory["actions"])
        episode_lengths.append(episode_length)

        # Compute true return: sum of (preference_weights_truth * mo_rewards)
        if isinstance(trajectory["mo_rewards"], np.ndarray) and isinstance(
            trajectory["preference_weights_truth"], np.ndarray
        ):
            # Compute scalar rewards from preference weights and multi-objective rewards
            scalar_rewards = np.sum(
                trajectory["preference_weights_truth"] * trajectory["mo_rewards"],
                axis=1,
            )
            episode_return = np.sum(scalar_rewards)
            episode_returns.append(episode_return)

        # Compute preference MAE
        if isinstance(
            trajectory["preference_weights_truth"], np.ndarray
        ) and isinstance(trajectory["preference_weights_predicted"], np.ndarray):
            mae = np.mean(
                np.abs(
                    trajectory["preference_weights_truth"][:, 0]
                    - trajectory["preference_weights_predicted"][:, 0]
                )
            )
            preference_maes.append(mae)

    # Compute statistics
    stats = {
        "n_episodes": n_episodes,
        "mean_length": float(np.mean(episode_lengths)),
        "std_length": float(np.std(episode_lengths)),
        "min_length": int(np.min(episode_lengths)),
        "max_length": int(np.max(episode_lengths)),
        "total_timesteps": int(np.sum(episode_lengths)),
    }

    if episode_returns:
        stats.update(
            {
                "mean_return": float(np.mean(episode_returns)),
                "std_return": float(np.std(episode_returns)),
                "min_return": float(np.min(episode_returns)),
                "max_return": float(np.max(episode_returns)),
            }
        )

    if preference_maes:
        stats.update(
            {
                "mean_preference_mae": float(np.mean(preference_maes)),
                "std_preference_mae": float(np.std(preference_maes)),
                "min_preference_mae": float(np.min(preference_maes)),
                "max_preference_mae": float(np.max(preference_maes)),
            }
        )

    # Save trajectories as JSON
    trajectories_json = []
    for traj in trajectories:
        traj_dict = {
            "observations": traj["observations"].tolist(),
            "actions": traj["actions"].tolist(),
            "mo_rewards": traj["mo_rewards"].tolist()
            if isinstance(traj["mo_rewards"], np.ndarray)
            else traj["mo_rewards"],
            "preference_weights_truth": traj["preference_weights_truth"].tolist()
            if isinstance(traj["preference_weights_truth"], np.ndarray)
            else traj["preference_weights_truth"],
            "preference_weights_predicted": traj[
                "preference_weights_predicted"
            ].tolist()
            if isinstance(traj["preference_weights_predicted"], np.ndarray)
            else traj["preference_weights_predicted"],
        }
        trajectories_json.append(traj_dict)

    trajectories_json_path = output_path / "trajectories.json"
    with open(trajectories_json_path, "w") as f:
        json.dump(trajectories_json, f, indent=2)
    print(f"\nSaved {n_episodes} trajectories to {trajectories_json_path}")

    # Save statistics
    stats_path = output_path / "statistics.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"Saved statistics to {stats_path}")

    # Save configuration
    config_save_path = output_path / "config.yaml"
    with open(config_save_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False)
    print(f"Saved config to {config_save_path}")

    # Print summary
    print("\n" + "=" * 70)
    print("IQL SIMULATION SUMMARY")
    print("=" * 70)
    print(f"Episodes collected: {stats['n_episodes']}")
    print(f"Total timesteps: {stats['total_timesteps']}")
    print(f"Mean length: {stats['mean_length']:.1f} +/- {stats['std_length']:.1f}")
    print(f"Length range: [{stats['min_length']}, {stats['max_length']}]")

    if episode_returns:
        print("\nTrue Return Statistics:")
        print(f"Mean return: {stats['mean_return']:.4f} +/- {stats['std_return']:.4f}")
        print(f"Return range: [{stats['min_return']:.4f}, {stats['max_return']:.4f}]")

    if preference_maes:
        print("\nPreference Prediction Performance:")
        print(
            f"Mean MAE: {stats['mean_preference_mae']:.4f} +/- {stats['std_preference_mae']:.4f}"
        )
        print(
            f"MAE range: [{stats['min_preference_mae']:.4f}, {stats['max_preference_mae']:.4f}]"
        )

    print("=" * 70)

    return trajectories, stats


def main():
    """Main function for IQL simulation."""
    parser = argparse.ArgumentParser(
        description="Simulate trajectories using trained IQL model with SSM predictions"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/simulate-iql.yaml",
        help="Path to simulation configuration file",
    )

    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)
    print(f"Loaded simulation config from {args.config}")

    # Run simulation
    simulate(config=config)


if __name__ == "__main__":
    main()
