"""
Simulation script to collect trajectory data using trained PPO model for Deep Sea Treasure.
"""

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from tqdm import tqdm
from PIL import Image

import mo_gymnasium as gym
from src.dst.env import DeepSeaTreasureWrapper
from src.dst.reward_function import RewardFunction, DSTPreferenceFunction
from src.dst.model import ActorCritic


def save_gif(frames: List[np.ndarray], path: Path, fps: int = 10):
    """Save a list of frames as a GIF.

    Args:
        frames: List of numpy arrays representing frames (H, W, C)
        path: Path to save the GIF
        fps: Frames per second
    """
    # Convert numpy arrays to PIL Images
    pil_frames = [Image.fromarray(frame) for frame in frames]

    # Save as GIF
    duration = int(1000 / fps)  # Duration in milliseconds
    pil_frames[0].save(
        path,
        save_all=True,
        append_images=pil_frames[1:],
        duration=duration,
        loop=0,
    )


def create_preference_weights_gif(
    preference_weights: np.ndarray,
    path: Path,
    fps: int = 5,
    figsize: tuple = (6.65, 6.4),
):
    """Create animated GIF showing preference weights evolution over time.

    Args:
        preference_weights: Array of shape (n_timesteps, n_objectives) with preference weights
        path: Path to save the GIF
        fps: Frames per second
        figsize: Figure size in inches (width, height)
    """
    n_timesteps = len(preference_weights)
    treasure_weights = preference_weights[:, 0]
    time_weights = preference_weights[:, 1]

    frames = []

    for t in range(n_timesteps):
        fig, ax = plt.subplots(figsize=figsize, dpi=80)

        # Plot the full trajectory as lines
        timesteps = np.arange(n_timesteps)
        ax.plot(
            timesteps,
            treasure_weights,
            "b-",
            linewidth=2,
            label="Treasure Weight",
            alpha=0.7,
        )
        ax.plot(
            timesteps, time_weights, "g-", linewidth=2, label="Time Weight", alpha=0.7
        )

        # Plot the current position as a red point
        ax.plot(t, treasure_weights[t], "ro", markersize=12)
        ax.plot(t, time_weights[t], "ro", markersize=12)

        # Formatting
        ax.set_xlabel("Timestep", fontsize=12)
        ax.set_ylabel("Preference Weight", fontsize=12)
        ax.set_title(
            f"Preference Weights Evolution (t={t}/{n_timesteps-1})", fontsize=14
        )
        ax.set_xlim(-0.5, n_timesteps - 0.5)
        ax.set_ylim(-0.05, 1.05)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best", fontsize=10)

        # Convert plot to image
        fig.canvas.draw()
        image = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
        image = image.reshape(fig.canvas.get_width_height()[::-1] + (4,))
        # Convert RGBA to RGB
        image_rgb = image[:, :, :3]
        frames.append(Image.fromarray(image_rgb))

        plt.close(fig)

    # Save as GIF
    duration = int(1000 / fps)  # Duration in milliseconds
    frames[0].save(
        path,
        save_all=True,
        append_images=frames[1:],
        duration=duration,
        loop=0,
    )


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def load_model(
    model_path: str,
    obs_dim: int,
    action_dim: int,
    hidden_dim: int,
    device: str = "cuda",
) -> ActorCritic:
    """Load trained PPO model from checkpoint."""
    model = ActorCritic(obs_dim, action_dim, hidden_dim).to(device)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint["ac_state_dict"])
    model.eval()
    return model


def create_environment(
    contenous_decay: float,
    init_weight: float,
    use_discontinuous_decay: bool = False,
    switching_factor: float = 0.5,
    max_timesteps: int = None,
    max_num_treasure: int = 1,
    use_local_obs: bool = True,
    local_obs_size: int = 3,
    render_mode: str = None,
    render_fps: int = 10,
):
    """Create Deep Sea Treasure environment with reward function."""
    # Create reward function with time-varying preference
    preference_fn = DSTPreferenceFunction(
        contenous_decay=contenous_decay,
        init_treasure_weight=init_weight,
        use_discontinuous_decay=use_discontinuous_decay,
        switching_factor=switching_factor,
    )
    reward_fn = RewardFunction(preference_fn=preference_fn)

    # Create environment
    env = DeepSeaTreasureWrapper(
        env=gym.make("deep-sea-treasure-v0", render_mode=render_mode),
        reward_fn=reward_fn,
        max_num_treasure=max_num_treasure,
        max_timestep=max_timesteps,
        use_local_obs=use_local_obs,
        local_obs_size=local_obs_size,
        render_mode=render_mode,
        render_fps=render_fps,
    )

    return env, reward_fn


def collect_trajectory(
    env,
    model: ActorCritic,
    reward_fn: RewardFunction,
    device: str,
    max_episode_steps: int = None,
    deterministic: bool = False,
    save_render: bool = False,
) -> Dict:
    """
    Collect a single trajectory.

    Args:
        env: Environment
        model: Trained PPO policy model
        reward_fn: Reward function for computing scalar rewards
        device: Device to run model on
        max_episode_steps: Maximum timesteps per episode (None = use done signal only)
        deterministic: If True, take argmax action; otherwise sample
        save_render: If True, capture frames for rendering

    Returns:
        Dictionary containing trajectory data
    """
    trajectory = {
        "observations": [],
        "actions": [],
        "rewards": [],
        "mo_rewards": [],
        "preference_weights": [],
        "dones": [],
        "infos": [],
    }

    # For rendering
    frames = [] if save_render else None

    obs, info = env.reset()
    reward_fn.reset()
    episode_step = 0

    # Capture initial frame if rendering
    if save_render:
        frame = env.render()
        if frame is not None:
            frames.append(frame)

    while True:
        # Store observation
        trajectory["observations"].append(obs.copy())

        # Get action from model
        with torch.no_grad():
            # Flatten observation if multi-dimensional
            obs_flat = obs.flatten() if len(obs.shape) > 1 else obs
            obs_tensor = torch.FloatTensor(obs_flat).unsqueeze(0).to(device)
            logits, _ = model.act(obs_tensor)

            if deterministic:
                action = torch.argmax(logits, dim=1).item()
            else:
                dist = torch.distributions.Categorical(logits=logits)
                action = dist.sample().item()

        # Take action in environment
        next_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        # Store transition
        mo_reward = info.get("mo_reward", None)
        reward = reward_fn(mo_reward)  # Recompute reward from multi-objective reward
        trajectory["actions"].append(action)
        trajectory["rewards"].append(reward)
        trajectory["mo_rewards"].append(mo_reward)
        trajectory["preference_weights"].append(info.get("preference_weights", None))
        trajectory["dones"].append(done)
        trajectory["infos"].append(info)

        obs = next_obs
        episode_step += 1

        # Capture frame after action if rendering
        if save_render:
            frame = env.render()
            if frame is not None:
                frames.append(frame)

        # Check termination: use max_episode_steps if set, otherwise use done signal
        if max_episode_steps is not None:
            if episode_step >= max_episode_steps:
                break
        else:
            if done:
                break

    # Convert lists to numpy arrays where appropriate
    trajectory["observations"] = np.array(trajectory["observations"])
    trajectory["actions"] = np.array(trajectory["actions"])
    trajectory["rewards"] = np.array(trajectory["rewards"])
    if trajectory["mo_rewards"][0] is not None:
        trajectory["mo_rewards"] = np.array(trajectory["mo_rewards"])
    if trajectory["preference_weights"][0] is not None:
        trajectory["preference_weights"] = np.array(trajectory["preference_weights"])
    trajectory["dones"] = np.array(trajectory["dones"])

    # Add frames to trajectory if rendering
    if save_render:
        trajectory["frames"] = frames

    return trajectory


def simulate(config: dict):
    """
    Run simulation to collect trajectory dataset.

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

    # Load training configuration for environment setup
    training_config = load_config(str(training_config_path))
    print(f"Loaded training config from {training_config_path}")

    # Extract other simulation parameters
    n_episodes = config["n_episodes"]
    max_episode_steps = config.get("max_episode_steps")  # None means use done signal
    output_dir = config["output_dir"]
    deterministic = config.get("deterministic", False)
    device = config.get("device", "cuda")
    render = config.get("render", False)
    save_render = config.get("save_render", False)
    render_fps = config.get("render_fps", 10)

    # Handle init_weight - can be a list or scalar
    init_weight = training_config.get("init_weight", 0.8)
    if isinstance(init_weight, list):
        init_weight = init_weight[0]  # Use first element for initial weight

    # Create environment
    # Use rgb_array mode if saving render, otherwise use human mode for rendering
    render_mode = None
    if save_render:
        render_mode = "rgb_array"
    elif render:
        render_mode = "human"

    env, reward_fn = create_environment(
        contenous_decay=training_config["contenous_decay"],
        init_weight=init_weight,
        use_discontinuous_decay=training_config.get("use_discontinuous_decay", False),
        switching_factor=training_config.get("switching_factor", 0.5),
        max_timesteps=training_config.get("max_timesteps"),
        max_num_treasure=training_config.get("max_num_treasure", 1),
        use_local_obs=training_config.get("use_local_obs", True),
        local_obs_size=training_config.get("local_obs_size", 3),
        render_mode=render_mode,
        render_fps=render_fps,
    )
    print("Created environment: Deep Sea Treasure")

    # Get environment dimensions
    if len(env.observation_space.shape) == 1:
        obs_dim = env.observation_space.shape[0]
    else:
        obs_dim = int(np.prod(env.observation_space.shape))

    action_dim = env.action_space.n

    # Load model
    model = load_model(
        model_path=str(model_path),
        obs_dim=obs_dim,
        action_dim=action_dim,
        hidden_dim=training_config["hidden_dim"],
        device=device,
    )
    print(f"Loaded model from {model_path}")
    print(f"Using device: {device}")
    print(f"Deterministic: {deterministic}")

    # Create output directory with timestamp to avoid overwriting
    save_path = Path(output_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = save_path / f"{timestamp}"
    output_path.mkdir(parents=True, exist_ok=True)

    # Collect trajectories
    trajectories = []
    episode_returns = []
    episode_lengths = []

    print(f"\nCollecting {n_episodes} episodes...")
    for episode in tqdm(range(n_episodes)):
        # Only save render for the first episode
        should_save_render = save_render and episode == 0

        trajectory = collect_trajectory(
            env=env,
            model=model,
            reward_fn=reward_fn,
            device=device,
            max_episode_steps=max_episode_steps,
            deterministic=deterministic,
            save_render=should_save_render,
        )
        trajectories.append(trajectory)

        # Track statistics
        episode_return = np.sum(trajectory["rewards"])
        episode_length = len(trajectory["rewards"])
        episode_returns.append(episode_return)
        episode_lengths.append(episode_length)

        # Save GIFs for first episode if requested
        if should_save_render and "frames" in trajectory:
            # Save episode GIF
            gif_path = output_path / "episode.gif"
            save_gif(trajectory["frames"], gif_path, fps=render_fps)
            print(f"Saved episode rendering to {gif_path}")

            # Save preference weights GIF
            if isinstance(trajectory["preference_weights"], np.ndarray):
                pref_gif_path = output_path / "preference_weights.gif"
                create_preference_weights_gif(
                    trajectory["preference_weights"], pref_gif_path, fps=render_fps
                )
                print(f"Saved preference weights visualization to {pref_gif_path}")

    # Compute statistics
    stats = {
        "n_episodes": n_episodes,
        "mean_return": float(np.mean(episode_returns)),
        "std_return": float(np.std(episode_returns)),
        "min_return": float(np.min(episode_returns)),
        "max_return": float(np.max(episode_returns)),
        "mean_length": float(np.mean(episode_lengths)),
        "std_length": float(np.std(episode_lengths)),
        "min_length": int(np.min(episode_lengths)),
        "max_length": int(np.max(episode_lengths)),
        "total_timesteps": int(np.sum(episode_lengths)),
    }

    # Save trajectories as JSON (state, action, mo_reward, preference_weights)
    trajectories_json = []
    for traj in trajectories:
        traj_dict = {
            "observations": traj["observations"].tolist(),
            "actions": traj["actions"].tolist(),
            "mo_rewards": traj["mo_rewards"].tolist()
            if isinstance(traj["mo_rewards"], np.ndarray)
            else traj["mo_rewards"],
            "preference_weights": traj["preference_weights"].tolist()
            if isinstance(traj["preference_weights"], np.ndarray)
            else traj["preference_weights"],
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
    print("SIMULATION SUMMARY")
    print("=" * 70)
    print(f"Episodes collected: {stats['n_episodes']}")
    print(f"Total timesteps: {stats['total_timesteps']}")
    print(f"Mean return: {stats['mean_return']:.2f} +/- {stats['std_return']:.2f}")
    print(f"Return range: [{stats['min_return']:.2f}, {stats['max_return']:.2f}]")
    print(f"Mean length: {stats['mean_length']:.1f} +/- {stats['std_length']:.1f}")
    print(f"Length range: [{stats['min_length']}, {stats['max_length']}]")
    print("=" * 70)

    return trajectories, stats


def main():
    """Main function for simulation."""
    parser = argparse.ArgumentParser(
        description="Simulate trajectories using trained SAC model for Deep Sea Treasure"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/simulate.yaml",
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
