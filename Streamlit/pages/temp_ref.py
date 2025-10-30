"""
Streamlit Dashboard for IQL Training Results Visualization.

Features:
- Select experiment from results directory
- Plot training metrics (losses, preference accuracy, episode rewards)
- Play policy rollout videos via pygame
"""

import streamlit as st
import pandas as pd
import numpy as np
import yaml
import torch
from pathlib import Path
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pygame
import mo_gymnasium as gym
from typing import Dict, Any

# Add parent directory to path for imports
import sys

sys.path.append(str(Path(__file__).parent.parent.parent))

from src.Env.deepseatreasure import DeepSeaTreasureWrapper
from src.Env.highway import HighwayWrapper
from src.Env.reward_function import (
    RewardFunction,
    DSTPreferenceFunction,
    HighwayPreferenceFunction,
)
from IQL.simplessm.trainer import ODSQILTrainer
from IQL.simplessm.ssm import StateSpaceModel


st.set_page_config(page_title="IQL Dashboard", layout="wide")


def load_experiment_dirs(base_dir: str = "moiql_results") -> list:
    """Load all experiment directories from results folder."""
    base_path = Path(base_dir)
    if not base_path.exists():
        return []

    # Get all subdirectories with timestamps
    exp_dirs = [d for d in base_path.iterdir() if d.is_dir()]
    exp_dirs.sort(reverse=True)  # Most recent first
    return exp_dirs


def load_config(exp_dir: Path) -> Dict[str, Any]:
    """Load experiment configuration."""
    config_path = exp_dir / "config.yaml"
    if not config_path.exists():
        return {}

    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def load_expert_config(exp_dir: Path) -> Dict[str, Any]:
    """Load expert configuration."""
    expert_config_path = exp_dir / "expert_config.yaml"
    if not expert_config_path.exists():
        return {}

    with open(expert_config_path, "r") as f:
        return yaml.safe_load(f)


def load_metrics(exp_dir: Path) -> pd.DataFrame:
    """Load metrics CSV."""
    metrics_path = exp_dir / "metrics.csv"
    if not metrics_path.exists():
        return pd.DataFrame()

    return pd.read_csv(metrics_path)


def plot_training_losses(df: pd.DataFrame):
    """Plot training losses over time."""
    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=("Total Loss", "Soft IQ Loss", "Actor Loss", "Mismatch Loss"),
    )

    # Total Loss
    fig.add_trace(
        go.Scatter(x=df["update"], y=df["total_loss"], name="Total Loss", mode="lines"),
        row=1,
        col=1,
    )

    # Soft IQ Loss
    fig.add_trace(
        go.Scatter(
            x=df["update"], y=df["soft_iq_loss"], name="Soft IQ Loss", mode="lines"
        ),
        row=1,
        col=2,
    )

    # Actor Loss
    fig.add_trace(
        go.Scatter(x=df["update"], y=df["actor_loss"], name="Actor Loss", mode="lines"),
        row=2,
        col=1,
    )

    # Mismatch Loss
    fig.add_trace(
        go.Scatter(
            x=df["update"], y=df["mismatch_loss"], name="Mismatch Loss", mode="lines"
        ),
        row=2,
        col=2,
    )

    fig.update_xaxes(title_text="Update", row=2, col=1)
    fig.update_xaxes(title_text="Update", row=2, col=2)
    fig.update_yaxes(title_text="Loss", row=1, col=1)
    fig.update_yaxes(title_text="Loss", row=1, col=2)
    fig.update_yaxes(title_text="Loss", row=2, col=1)
    fig.update_yaxes(title_text="Loss", row=2, col=2)

    fig.update_layout(height=600, showlegend=False)

    st.plotly_chart(fig, use_container_width=True)


def plot_preference_accuracy(df: pd.DataFrame):
    """Plot preference prediction accuracy metrics."""
    fig = make_subplots(
        rows=1, cols=2, subplot_titles=("Preference MSE", "Preference MAE")
    )

    # MSE with std
    fig.add_trace(
        go.Scatter(
            x=df["update"],
            y=df["preference_mse"],
            name="MSE",
            mode="lines",
            line=dict(color="blue"),
        ),
        row=1,
        col=1,
    )

    if "preference_mse_std" in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df["update"].tolist() + df["update"].tolist()[::-1],
                y=(df["preference_mse"] + df["preference_mse_std"]).tolist()
                + (df["preference_mse"] - df["preference_mse_std"]).tolist()[::-1],
                fill="toself",
                fillcolor="rgba(0,100,200,0.2)",
                line=dict(color="rgba(255,255,255,0)"),
                showlegend=False,
                name="MSE ± std",
            ),
            row=1,
            col=1,
        )

    # MAE with std
    fig.add_trace(
        go.Scatter(
            x=df["update"],
            y=df["preference_mae"],
            name="MAE",
            mode="lines",
            line=dict(color="green"),
        ),
        row=1,
        col=2,
    )

    if "preference_mae_std" in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df["update"].tolist() + df["update"].tolist()[::-1],
                y=(df["preference_mae"] + df["preference_mae_std"]).tolist()
                + (df["preference_mae"] - df["preference_mae_std"]).tolist()[::-1],
                fill="toself",
                fillcolor="rgba(0,200,100,0.2)",
                line=dict(color="rgba(255,255,255,0)"),
                showlegend=False,
                name="MAE ± std",
            ),
            row=1,
            col=2,
        )

    fig.update_xaxes(title_text="Update")
    fig.update_yaxes(title_text="MSE", row=1, col=1)
    fig.update_yaxes(title_text="MAE", row=1, col=2)

    fig.update_layout(height=400, showlegend=True)

    st.plotly_chart(fig, use_container_width=True)


def plot_episode_rewards(df: pd.DataFrame):
    """Plot episode reward metrics."""
    fig = go.Figure()

    # Mean reward
    fig.add_trace(
        go.Scatter(
            x=df["update"],
            y=df["mean_episode_reward"],
            name="Mean Reward",
            mode="lines+markers",
            line=dict(color="purple"),
        )
    )

    # Std band
    if "std_episode_reward" in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df["update"].tolist() + df["update"].tolist()[::-1],
                y=(df["mean_episode_reward"] + df["std_episode_reward"]).tolist()
                + (df["mean_episode_reward"] - df["std_episode_reward"]).tolist()[::-1],
                fill="toself",
                fillcolor="rgba(200,0,200,0.2)",
                line=dict(color="rgba(255,255,255,0)"),
                showlegend=False,
                name="Reward ± std",
            )
        )

    fig.update_layout(
        title="Episode Reward Over Training",
        xaxis_title="Update",
        yaxis_title="Episode Reward",
        height=400,
    )

    st.plotly_chart(fig, use_container_width=True)


def create_environment(env_name: str, expert_config: Dict[str, Any]):
    """Create environment for visualization."""
    if env_name == "deep_sea_treasure":
        env = gym.make("deep-sea-treasure-v0", render_mode="rgb_array")

        preference_fn = DSTPreferenceFunction(
            contenous_decay=expert_config.get("contenous_decay", 0.01),
            init_treasure_weight=expert_config.get("init_weight", [0.8, 0.2])[0],
        )
        reward_fn = RewardFunction(preference_fn=preference_fn)
        env = DeepSeaTreasureWrapper(env=env, reward_fn=reward_fn)

    elif env_name == "mo-highway":
        env = gym.make("mo-highway-v0", render_mode="rgb_array")

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


def load_model(exp_dir: Path, config: Dict[str, Any], expert_config: Dict[str, Any]):
    """Load trained model."""
    # Get environment dimensions
    env_name = expert_config["env_name"]
    env = create_environment(env_name, expert_config)

    if len(env.observation_space.shape) == 1:
        obs_dim = env.observation_space.shape[0]
    else:
        obs_dim = int(np.prod(env.observation_space.shape))

    action_dim = env.action_space.n
    n_objectives = env.n_objectives

    # Create SSM
    ssm = StateSpaceModel(
        n_objectives=n_objectives,
        n_particles=config.get("particle_filter", {}).get("n_particles", 1000),
        process_noise=config.get("particle_filter", {}).get("process_noise", 0.01),
        observation_noise=config.get("particle_filter", {}).get(
            "observation_noise", 0.1
        ),
        initial_noise=config.get("particle_filter", {}).get("initial_noise", 0.5),
    )

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
        device="cpu",  # Use CPU for visualization
    )

    # Load model weights
    model_path = exp_dir / "final_model.pt"
    if model_path.exists():
        trainer.load(str(model_path))

    return trainer, env


def play_episode_pygame(trainer, env, max_steps: int = 1000):
    """Play an episode and render with pygame."""
    pygame.init()

    # Get initial frame to determine screen size
    obs, _ = env.reset()
    frame = env.render()

    if frame is None:
        st.error("Environment does not support rendering!")
        return

    # Create pygame window
    screen_height, screen_width = frame.shape[:2]
    screen = pygame.display.set_mode((screen_width, screen_height))
    pygame.display.set_caption("IQL Policy Rollout")
    clock = pygame.time.Clock()

    # Episode info
    episode_reward = 0
    episode_length = 0
    done = False

    # Info display
    font = pygame.font.Font(None, 24)

    # Flatten obs if needed
    if len(obs.shape) > 1:
        obs = obs.flatten()

    running = True
    while running and not done and episode_length < max_steps:
        # Handle pygame events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # Get action from policy
        obs_t = torch.FloatTensor(obs).unsqueeze(0).to(trainer.device)
        with torch.no_grad():
            logits, _ = trainer.q_network.act(obs_t)
            action_probs = torch.softmax(logits, dim=-1)
            action = torch.argmax(action_probs, dim=-1).item()

        # Step environment
        next_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        # Track reward
        if isinstance(reward, np.ndarray):
            episode_reward += reward.sum()
        else:
            episode_reward += reward

        # Render
        frame = env.render()

        # Convert frame to pygame surface
        frame_surface = pygame.surfarray.make_surface(frame.swapaxes(0, 1))
        screen.blit(frame_surface, (0, 0))

        # Display info
        info_text = [
            f"Step: {episode_length}",
            f"Reward: {episode_reward:.2f}",
            f"Action: {action}",
        ]

        if "mo_reward" in info:
            mo_reward = info["mo_reward"]
            info_text.append(f"MO Reward: [{mo_reward[0]:.2f}, {mo_reward[1]:.2f}]")

        if "preference_weights" in info:
            pref = info["preference_weights"]
            info_text.append(f"Preference: [{pref[0]:.2f}, {pref[1]:.2f}]")

        # Render text
        y_offset = 10
        for text in info_text:
            text_surface = font.render(text, True, (255, 255, 255))
            text_rect = text_surface.get_rect()
            text_rect.topleft = (10, y_offset)

            # Draw background for text
            bg_rect = text_rect.inflate(10, 5)
            pygame.draw.rect(screen, (0, 0, 0), bg_rect)
            screen.blit(text_surface, text_rect)
            y_offset += 25

        pygame.display.flip()
        clock.tick(30)  # 30 FPS

        obs = next_obs if len(next_obs.shape) == 1 else next_obs.flatten()
        episode_length += 1

    pygame.quit()

    return {
        "episode_reward": episode_reward,
        "episode_length": episode_length,
    }


def main():
    """Main Streamlit app."""
    st.title("🤖 IQL Training Results Dashboard")

    # Sidebar for experiment selection
    st.sidebar.header("Experiment Selection")

    base_dir = st.sidebar.text_input("Results Directory", value="moiql_results")
    exp_dirs = load_experiment_dirs(base_dir)

    if not exp_dirs:
        st.warning(f"No experiments found in {base_dir}")
        return

    # Select experiment
    exp_names = [d.name for d in exp_dirs]
    selected_exp = st.sidebar.selectbox("Select Experiment", exp_names)

    if selected_exp:
        exp_dir = Path(base_dir) / selected_exp

        # Load data
        config = load_config(exp_dir)
        expert_config = load_expert_config(exp_dir)
        metrics_df = load_metrics(exp_dir)

        # Display experiment info
        st.header(f"📊 Experiment: {selected_exp}")

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Environment", expert_config.get("env_name", "Unknown"))
        with col2:
            st.metric("Total Updates", config.get("n_updates", "N/A"))
        with col3:
            if not metrics_df.empty:
                final_reward = metrics_df.iloc[-1]["mean_episode_reward"]
                st.metric("Final Episode Reward", f"{final_reward:.2f}")

        # Tabs for different views
        tab1, tab2, tab3, tab4 = st.tabs(
            [
                "📈 Training Metrics",
                "🎯 Preference Accuracy",
                "🎮 Policy Visualization",
                "⚙️ Configuration",
            ]
        )

        with tab1:
            st.subheader("Training Losses")
            if not metrics_df.empty:
                plot_training_losses(metrics_df)

                st.subheader("Episode Rewards")
                plot_episode_rewards(metrics_df)
            else:
                st.info("No metrics data available")

        with tab2:
            st.subheader("Preference Prediction Accuracy")
            if not metrics_df.empty:
                plot_preference_accuracy(metrics_df)

                # Summary statistics
                st.subheader("Summary Statistics")
                col1, col2 = st.columns(2)
                with col1:
                    st.metric(
                        "Final Preference MSE",
                        f"{metrics_df.iloc[-1]['preference_mse']:.6f}",
                    )
                with col2:
                    st.metric(
                        "Final Preference MAE",
                        f"{metrics_df.iloc[-1]['preference_mae']:.6f}",
                    )
            else:
                st.info("No metrics data available")

        with tab3:
            st.subheader("Policy Rollout Visualization")

            if st.button("▶️ Play Episode with Pygame"):
                with st.spinner("Loading model and environment..."):
                    try:
                        trainer, env = load_model(exp_dir, config, expert_config)
                        st.info(
                            "Playing episode in pygame window... (Close window to stop)"
                        )

                        result = play_episode_pygame(trainer, env)

                        st.success("Episode completed!")
                        st.write(f"**Total Reward:** {result['episode_reward']:.2f}")
                        st.write(f"**Episode Length:** {result['episode_length']}")

                    except Exception as e:
                        st.error(f"Error playing episode: {e}")
                        import traceback

                        st.code(traceback.format_exc())

        with tab4:
            st.subheader("Training Configuration")
            st.json(config)

            st.subheader("Expert Configuration")
            st.json(expert_config)


if __name__ == "__main__":
    main()
