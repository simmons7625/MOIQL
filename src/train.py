import argparse
import yaml
import torch
import torch.optim as optim
import numpy as np
import wandb
from pathlib import Path
from typing import Dict, Any

from environment import make_env
from model import (
    MultiObjectiveQNetwork,
    InverseQLearningLoss,
    MultiObjectivePolicy,
)
from metrics import evaluate_policy


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def collect_expert_demonstrations(env, n_episodes: int):
    """Collect expert demonstrations (placeholder for actual expert policy)."""
    demonstrations = []

    for _ in range(n_episodes):
        obs, _ = env.reset()
        episode_data = []
        terminated = False
        truncated = False

        while not (terminated or truncated):
            action = env.action_space.sample()
            next_obs, reward, terminated, truncated, info = env.step(action)

            episode_data.append(
                {
                    "state": obs.copy(),
                    "action": action,
                    "reward": reward.copy()
                    if isinstance(reward, np.ndarray)
                    else np.array([reward]),
                    "next_state": next_obs.copy(),
                    "terminated": terminated,
                }
            )

            obs = next_obs

        demonstrations.append(episode_data)

    return demonstrations


def train_inverse_q_learning(config: Dict[str, Any]):
    """Main training loop for inverse Q-learning."""

    wandb.init(
        project=config.get("wandb_project", "moiql"),
        name=config.get("wandb_run_name"),
        config=config,
        mode=config.get("wandb_mode", "online"),
    )

    device = torch.device(
        config.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    )

    env = make_env(config["env_name"], **config.get("env_kwargs", {}))

    state_dim = np.prod(env.observation_space.shape)
    action_dim = env.action_space.n
    n_objectives = env.n_objectives

    q_network = MultiObjectiveQNetwork(
        state_dim=state_dim,
        action_dim=action_dim,
        n_objectives=n_objectives,
        hidden_dim=config.get("hidden_dim", 256),
        n_hidden_layers=config.get("n_hidden_layers", 2),
    ).to(device)

    q_optimizer = optim.Adam(q_network.parameters(), lr=config.get("q_lr", 1e-3))

    criterion = InverseQLearningLoss(margin=config.get("margin", 1.0))

    print("Collecting expert demonstrations...")
    demonstrations = collect_expert_demonstrations(
        env, config.get("n_expert_episodes", 100)
    )

    expert_total_rewards = []
    for demo in demonstrations:
        total_reward = sum(t["reward"] for t in demo)
        expert_total_rewards.append(total_reward)
    expert_mean_reward = np.mean(expert_total_rewards, axis=0)

    print("Starting training...")
    for epoch in range(config.get("n_epochs", 1000)):
        epoch_q_loss = 0.0

        for demo in demonstrations:
            for transition in demo:
                state = (
                    torch.FloatTensor(transition["state"])
                    .flatten()
                    .unsqueeze(0)
                    .to(device)
                )
                action = torch.LongTensor([transition["action"]]).to(device)

                q_values = q_network(state)

                q_loss = criterion(q_values, action, q_values)

                q_optimizer.zero_grad()
                q_loss.backward()
                q_optimizer.step()

                epoch_q_loss += q_loss.item()

        avg_loss = epoch_q_loss / len(demonstrations)

        policy = MultiObjectivePolicy(q_network, epsilon=0.0)
        mean_rewards, episode_rewards = evaluate_policy(
            policy, env, n_episodes=config.get("eval_episodes", 10)
        )

        regret = expert_mean_reward - mean_rewards

        wandb.log(
            {
                "epoch": epoch + 1,
                "q_loss": avg_loss,
                "reward_obj0": mean_rewards[0] if len(mean_rewards) > 0 else 0,
                "reward_obj1": mean_rewards[1] if len(mean_rewards) > 1 else 0,
                "regret_obj0": regret[0] if len(regret) > 0 else 0,
                "regret_obj1": regret[1] if len(regret) > 1 else 0,
                "total_reward": np.sum(mean_rewards),
                "total_regret": np.sum(regret),
            }
        )

        if (epoch + 1) % config.get("log_interval", 100) == 0:
            print(f"Epoch {epoch + 1}/{config.get('n_epochs', 1000)}")
            print(f"  Q-Loss: {avg_loss:.4f}")
            print(f"  Mean Rewards: {mean_rewards}")
            print(f"  Regret: {regret}")

    save_dir = Path(config.get("save_dir", "checkpoints"))
    save_dir.mkdir(parents=True, exist_ok=True)

    torch.save(q_network.state_dict(), save_dir / "q_network.pt")

    print(f"Models saved to {save_dir}")

    wandb.finish()

    env.close()


def main():
    parser = argparse.ArgumentParser(
        description="Train Multi-Objective Inverse Q-Learning"
    )
    parser.add_argument(
        "--config", type=str, required=True, help="Path to config YAML file"
    )

    args = parser.parse_args()

    config = load_config(args.config)

    train_inverse_q_learning(config)


if __name__ == "__main__":
    main()
