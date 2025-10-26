import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from typing import Dict
from pathlib import Path
import wandb
from tqdm import tqdm
import csv

import mo_gymnasium as gym
from src.highway.env import HighwayWrapper
from src.highway.reward_function import RewardFunction, HighwayPreferenceFunction
from src.highway.model import ActorCritic, RolloutBuffer


class PPOTrainer:
    """PPO trainer for policy learning."""

    def __init__(
        self,
        contenous_decay: float = 0.01,
        init_treasure_weight: float = 1.0,
        safety_distance_threshold: float = 10.0,
        safety_boost_factor: float = 1.5,
        hidden_dim: int = 256,
        lr: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_epsilon: float = 0.2,
        vf_coef: float = 0.5,
        ent_coef: float = 0.01,
        max_grad_norm: float = 0.5,
        batch_size: int = 64,
        n_epochs: int = 10,
        n_rollouts_per_update: int = 10,
        max_timesteps: int = None,
        max_num_treasure: int = 1,
        use_local_obs: bool = True,
        local_obs_size: int = 3,
        device: str = "cuda",
        use_wandb: bool = True,
    ):
        """
        Initialize PPO trainer for Highway environment.

        Args:
            contenous_decay: Continuous linear decay rate for preference weight
            init_treasure_weight: Initial weight for treasure/speed objective
            safety_distance_threshold: Distance threshold for safety switching (Highway only)
            safety_boost_factor: Factor to boost safety weight when close to cars (Highway only)
            hidden_dim: Hidden dimension for networks
            lr: Learning rate
            gamma: Discount factor
            gae_lambda: GAE lambda parameter
            clip_epsilon: PPO clip epsilon
            vf_coef: Value function coefficient
            ent_coef: Entropy coefficient
            max_grad_norm: Max gradient norm for clipping
            batch_size: Batch size for updates
            n_epochs: Number of epochs per update
            n_rollouts_per_update: Number of rollouts to collect per update
            max_timesteps: Max timesteps per episode (None = use done signal only)
            device: Device to use
            use_wandb: Whether to use wandb logging
        """
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.max_timesteps = max_timesteps
        print(f"Using device: {self.device}")

        # Create reward function with time-varying preference
        preference_fn = HighwayPreferenceFunction(
            init_speed_weight=init_treasure_weight,
            safety_distance_threshold=safety_distance_threshold,
            safety_boost_factor=safety_boost_factor,
        )
        reward_fn = RewardFunction(preference_fn=preference_fn)

        # Create Highway environment
        self.env = HighwayWrapper(
            env=gym.make("mo-highway-v0"),
            reward_fn=reward_fn,
        )
        # Handle observation space - flatten if multi-dimensional
        if len(self.env.observation_space.shape) == 1:
            obs_dim = self.env.observation_space.shape[0]
        else:
            # Flatten multi-dimensional observations (e.g., highway's (5,5) -> 25)
            obs_dim = int(np.prod(self.env.observation_space.shape))

        action_dim = self.env.action_space.n

        # Create actor-critic network
        self.ac = ActorCritic(obs_dim, action_dim, hidden_dim).to(self.device)
        self.optimizer = optim.Adam(self.ac.parameters(), lr=lr)

        # PPO hyperparameters
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.vf_coef = vf_coef
        self.ent_coef = ent_coef
        self.max_grad_norm = max_grad_norm
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.n_rollouts_per_update = n_rollouts_per_update

        # Rollout buffer
        self.buffer = RolloutBuffer()

        # Logging
        self.use_wandb = use_wandb

        # Track preference weights over time
        self.preference_weights_history = []
        self.timesteps_history = []

        # Store reward function for tracking
        self.reward_fn = reward_fn

    def collect_rollout(self) -> Dict:
        """Collect rollout data - collects multiple episodes and returns mean/std statistics."""
        self.buffer.clear()

        # Track statistics across all episodes
        all_episode_rewards = []
        all_episode_mo_rewards = []
        all_episode_lengths = []

        for i in range(self.n_rollouts_per_update):
            episode_reward = 0
            episode_mo_reward = np.array([0.0, 0.0], dtype=np.float32)

            state, _ = self.env.reset()
            self.reward_fn.reset()
            episode_step = 0

            while True:
                # Flatten state if multi-dimensional
                state_flat = state.flatten() if len(state.shape) > 1 else state
                state_t = torch.FloatTensor(state_flat).unsqueeze(0).to(self.device)

                with torch.no_grad():
                    logits, value = self.ac.act(state_t)
                    dist = Categorical(logits=logits)
                    action = dist.sample()
                    log_prob = dist.log_prob(action)

                next_state, reward, terminated, truncated, info = self.env.step(
                    action.item()
                )
                done = terminated or truncated

                # Store transition
                mo_reward = info.get("mo_reward", None)
                reward = self.reward_fn(mo_reward)
                # Store flattened state
                state_to_store = state.flatten() if len(state.shape) > 1 else state
                self.buffer.store(
                    state_to_store,
                    action.item(),
                    reward,
                    value.item(),
                    log_prob.item(),
                    done,
                    mo_reward,
                )

                episode_reward += reward
                if mo_reward is not None:
                    episode_mo_reward = episode_mo_reward + mo_reward

                state = next_state
                episode_step += 1

                # Check termination: use max_timesteps if set, otherwise use done signal
                if self.max_timesteps is not None:
                    if episode_step >= self.max_timesteps:
                        break
                else:
                    if done:
                        break

            # Store episode statistics
            all_episode_rewards.append(episode_reward)
            all_episode_mo_rewards.append(episode_mo_reward)
            all_episode_lengths.append(episode_step)

        # Compute mean and std statistics
        mean_reward = np.mean(all_episode_rewards)
        std_reward = np.std(all_episode_rewards)
        mean_length = np.mean(all_episode_lengths)
        std_length = np.std(all_episode_lengths)

        # Compute mean and std for multi-objective rewards
        mean_mo_rewards = np.mean(all_episode_mo_rewards, axis=0)
        std_mo_rewards = np.std(all_episode_mo_rewards, axis=0)

        # Return episode statistics with mean and std
        return {
            "episode_reward": mean_reward,
            "episode_reward_std": std_reward,
            "episode_mo_rewards": mean_mo_rewards,
            "episode_mo_rewards_std": std_mo_rewards,
            "episode_length": mean_length,
            "episode_length_std": std_length,
        }

    def compute_gae(self, rewards, values, dones, next_value):
        """Compute Generalized Advantage Estimation."""
        advantages = []
        gae = 0

        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_val = next_value
            else:
                next_val = values[t + 1]

            delta = rewards[t] + self.gamma * next_val * (1 - dones[t]) - values[t]
            gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * gae
            advantages.insert(0, gae)

        return advantages

    def update(self):
        """Update policy using PPO."""
        states, actions, rewards, values, old_log_probs, dones, _ = self.buffer.get()

        # Compute next value for GAE
        with torch.no_grad():
            next_value = self.ac.get_value(
                torch.FloatTensor(states[-1]).unsqueeze(0).to(self.device)
            ).item()

        # Compute advantages
        advantages = self.compute_gae(rewards, values, dones, next_value)
        returns = [adv + val for adv, val in zip(advantages, values)]

        # Convert to tensors
        states_t = torch.FloatTensor(np.array(states)).to(self.device)
        actions_t = torch.LongTensor(actions).to(self.device)
        old_log_probs_t = torch.FloatTensor(old_log_probs).to(self.device)
        returns_t = torch.FloatTensor(returns).to(self.device)
        advantages_t = torch.FloatTensor(advantages).to(self.device)

        # Normalize advantages (only if we have enough samples)
        if len(advantages) > 1:
            adv_mean = advantages_t.mean()
            adv_std = advantages_t.std()
            if adv_std > 1e-8:
                advantages_t = (advantages_t - adv_mean) / (adv_std + 1e-8)

        # PPO update
        update_info = {"policy_loss": [], "value_loss": [], "entropy": []}

        for _ in range(self.n_epochs):
            # Random mini-batch sampling
            indices = np.random.permutation(len(states))
            # Use full batch if episode is shorter than batch_size
            batch_size = min(self.batch_size, len(states))

            for start in range(0, len(states), batch_size):
                end = min(start + batch_size, len(states))
                batch_indices = indices[start:end]

                batch_states = states_t[batch_indices]
                batch_actions = actions_t[batch_indices]
                batch_old_log_probs = old_log_probs_t[batch_indices]
                batch_returns = returns_t[batch_indices]
                batch_advantages = advantages_t[batch_indices]

                # Forward pass
                logits, values = self.ac.act(batch_states)
                dist = Categorical(logits=logits)
                log_probs = dist.log_prob(batch_actions)
                entropy = dist.entropy().mean()

                # Policy loss
                ratio = torch.exp(log_probs - batch_old_log_probs)
                surr1 = ratio * batch_advantages
                surr2 = (
                    torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon)
                    * batch_advantages
                )
                policy_loss = -torch.min(surr1, surr2).mean()

                # Value loss
                value_loss = nn.MSELoss()(values.squeeze(), batch_returns)

                # Total loss
                loss = policy_loss + self.vf_coef * value_loss - self.ent_coef * entropy

                # Update
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.ac.parameters(), self.max_grad_norm)
                self.optimizer.step()

                update_info["policy_loss"].append(policy_loss.item())
                update_info["value_loss"].append(value_loss.item())
                update_info["entropy"].append(entropy.item())

        return {
            "policy_loss": np.mean(update_info["policy_loss"]),
            "value_loss": np.mean(update_info["value_loss"]),
            "entropy": np.mean(update_info["entropy"]),
        }

    def train(
        self,
        n_updates: int = 1000,
        save_dir: str = None,
    ) -> Dict:
        """Train the agent."""
        global_step = 0

        # Track metrics for final report and CSV export
        all_episode_rewards = []
        all_episode_rewards_std = []
        all_episode_lengths = []
        all_episode_lengths_std = []
        all_mo_rewards_obj1 = []
        all_mo_rewards_obj1_std = []
        all_mo_rewards_obj2 = []
        all_mo_rewards_obj2_std = []
        all_policy_losses = []
        all_value_losses = []
        all_entropy = []
        all_updates = []
        all_global_steps = []

        # Create progress bar
        pbar = tqdm(range(n_updates), desc="Training")

        for update in pbar:
            # Collect rollout
            rollout_info = self.collect_rollout()

            # Update policy
            update_info = self.update()

            global_step += int(rollout_info["episode_length"])

            # Track metrics
            all_updates.append(update)
            all_global_steps.append(global_step)
            all_policy_losses.append(update_info["policy_loss"])
            all_value_losses.append(update_info["value_loss"])
            all_entropy.append(update_info["entropy"])
            all_episode_rewards.append(rollout_info["episode_reward"])
            all_episode_rewards_std.append(rollout_info["episode_reward_std"])
            all_episode_lengths.append(rollout_info["episode_length"])
            all_episode_lengths_std.append(rollout_info["episode_length_std"])

            # Track multi-objective rewards
            if rollout_info["episode_mo_rewards"] is not None:
                all_mo_rewards_obj1.append(rollout_info["episode_mo_rewards"][0])
                all_mo_rewards_obj1_std.append(
                    rollout_info["episode_mo_rewards_std"][0]
                )
                all_mo_rewards_obj2.append(rollout_info["episode_mo_rewards"][1])
                all_mo_rewards_obj2_std.append(
                    rollout_info["episode_mo_rewards_std"][1]
                )

            self.timesteps_history.append(global_step)

            # Logging
            log_dict = {
                "train/update": update,
                "train/policy_loss": update_info["policy_loss"],
                "train/value_loss": update_info["value_loss"],
                "train/entropy": update_info["entropy"],
                "train/episode_reward": rollout_info["episode_reward"],
                "train/episode_reward_std": rollout_info["episode_reward_std"],
                "train/episode_length": rollout_info["episode_length"],
                "train/episode_length_std": rollout_info["episode_length_std"],
            }

            # Log multi-objective rewards if available
            if rollout_info["episode_mo_rewards"] is not None:
                mo_rewards = rollout_info["episode_mo_rewards"]
                mo_rewards_std = rollout_info["episode_mo_rewards_std"]
                log_dict["train/mo_reward_treasure"] = mo_rewards[0]
                log_dict["train/mo_reward_treasure_std"] = mo_rewards_std[0]
                log_dict["train/mo_reward_time"] = mo_rewards[1]
                log_dict["train/mo_reward_time_std"] = mo_rewards_std[1]

            if self.use_wandb:
                wandb.log(log_dict, step=global_step)

        # Close progress bar
        pbar.close()

        # Save metrics to CSV if save_dir is provided
        if save_dir is not None:
            save_path = Path(save_dir)
            save_path.mkdir(parents=True, exist_ok=True)
            csv_path = save_path / "metrics.csv"

            with open(csv_path, "w", newline="") as f:
                writer = csv.writer(f)

                # Write header
                header = [
                    "update",
                    "global_step",
                    "policy_loss",
                    "value_loss",
                    "entropy",
                    "episode_reward",
                    "episode_reward_std",
                    "episode_length",
                    "episode_length_std",
                ]

                # Add multi-objective columns if available
                if len(all_mo_rewards_obj1) > 0:
                    header.extend(
                        [
                            "mo_reward_obj1",
                            "mo_reward_obj1_std",
                            "mo_reward_obj2",
                            "mo_reward_obj2_std",
                        ]
                    )

                writer.writerow(header)

                # Write data rows
                for i in range(len(all_updates)):
                    row = [
                        all_updates[i],
                        all_global_steps[i],
                        all_policy_losses[i],
                        all_value_losses[i],
                        all_entropy[i],
                        all_episode_rewards[i],
                        all_episode_rewards_std[i],
                        all_episode_lengths[i],
                        all_episode_lengths_std[i],
                    ]

                    # Add multi-objective data if available
                    if len(all_mo_rewards_obj1) > 0:
                        row.extend(
                            [
                                all_mo_rewards_obj1[i],
                                all_mo_rewards_obj1_std[i],
                                all_mo_rewards_obj2[i],
                                all_mo_rewards_obj2_std[i],
                            ]
                        )

                    writer.writerow(row)

            print(f"Metrics saved to {csv_path}")

        # Return final metrics
        final_metrics = {
            "policy_loss": all_policy_losses,
            "value_loss": all_value_losses,
            "episode_reward": all_episode_rewards,
        }

        return final_metrics

    def save(self, filename: str):
        """Save model checkpoint."""
        save_path = Path(filename)

        # Ensure parent directory exists
        save_path.parent.mkdir(parents=True, exist_ok=True)

        torch.save(
            {"ac_state_dict": self.ac.state_dict()},
            save_path,
        )

    def load(self, filename: str):
        """Load model checkpoint."""
        checkpoint = torch.load(filename, map_location=self.device)
        self.ac.load_state_dict(checkpoint["ac_state_dict"])
