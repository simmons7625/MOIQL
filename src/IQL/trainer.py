"""
Objective-Dimensional Soft Inverse Q-Learning (OD-SQIL) Trainer.

Implements soft IQL with:
- Objective-dimensional Q-values output
- Mismatch function as regularization in loss
- Preference weight prediction using SSM
"""

import numpy as np
import torch
from pathlib import Path
from typing import Dict

from .actor_critic import ActorCritic
from .ssm import StateSpaceModel, compute_mismatch


class ODSQILTrainer:
    """
    Objective-Dimensional Soft Inverse Q-Learning Trainer.

    Combines soft IQL with preference weight prediction for multi-objective RL.
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        n_objectives: int,
        ssm_model: StateSpaceModel,
        hidden_dim: int = 256,
        lr: float = 3e-4,
        gamma: float = 0.99,
        tau: float = 0.005,
        alpha: float = 0.2,
        mismatch_coef: float = 1.0,
        device: str = "cuda",
    ):
        """
        Initialize OD-SQIL trainer.

        Args:
            obs_dim: Observation dimension
            action_dim: Action dimension
            n_objectives: Number of objectives
            ssm_model: State space model for preference weight prediction
            hidden_dim: Hidden dimension for networks
            lr: Learning rate
            gamma: Discount factor
            tau: Soft update coefficient
            alpha: Temperature parameter for soft Q-learning
            mismatch_coef: Coefficient for mismatch regularization
            device: Device to run on
        """
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.n_objectives = n_objectives
        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha
        self.mismatch_coef = mismatch_coef
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

        # State space model for preference prediction
        self.ssm = ssm_model

        # Q-network (outputs objective-dimensional Q-values)
        self.q_network = ActorCritic(
            obs_dim=obs_dim,
            action_dim=action_dim,
            hidden_dim=hidden_dim,
            n_objects=n_objectives,
        ).to(self.device)

        # Target Q-network
        self.q_target = ActorCritic(
            obs_dim=obs_dim,
            action_dim=action_dim,
            hidden_dim=hidden_dim,
            n_objects=n_objectives,
        ).to(self.device)
        self.q_target.load_state_dict(self.q_network.state_dict())

        # Separate optimizers for Q-network and actor
        self.q_optimizer = torch.optim.Adam(self.q_network.critic.parameters(), lr=lr)
        self.actor_optimizer = torch.optim.Adam(
            list(self.q_network.shared.parameters())
            + list(self.q_network.actor.parameters()),
            lr=lr,
        )

        # Tracking
        self.current_preference = np.ones(n_objectives) / n_objectives

    def _soft_update(self):
        """Soft update of target network."""
        for param, target_param in zip(
            self.q_network.parameters(), self.q_target.parameters()
        ):
            target_param.data.copy_(
                self.tau * param.data + (1 - self.tau) * target_param.data
            )

    def compute_actor_loss(
        self,
        states: torch.Tensor,
        preference_weights: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute SAC actor loss: E[-Q(s,a) + log π(a|s)]

        Args:
            states: [batch, obs_dim]
            preference_weights: [batch, n_objectives]

        Returns:
            Actor loss tensor
        """
        # Get policy logits
        logits, _ = self.q_network.act(states)  # logits: [batch, action_dim]

        # Get action probabilities
        action_probs = torch.softmax(logits, dim=-1)  # [batch, action_dim]
        log_probs = torch.log_softmax(logits, dim=-1)  # [batch, action_dim]

        # Compute Q-values for all actions
        q_all_actions = []
        for _ in range(self.action_dim):
            _, q_a = self.q_network.act(states)
            q_all_actions.append(q_a)
        q_all_actions = torch.stack(
            q_all_actions, dim=1
        )  # [batch, action_dim, n_objectives]

        # Convert to scalar Q using preference weights
        q_scalar = torch.einsum(
            "bao,bo->ba", q_all_actions, preference_weights
        )  # [batch, action_dim]

        # SAC policy loss: E_a~π[-Q(s,a) + log π(a|s)]
        # = sum_a π(a|s) * [-Q(s,a) + log π(a|s)]
        actor_loss = (action_probs * (-q_scalar + log_probs)).sum(dim=-1).mean()

        return actor_loss

    def compute_soft_iq_loss(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        next_states: torch.Tensor,
        preference_weights: torch.Tensor,
    ) -> Dict[str, float]:
        """
        Compute Soft IQ loss with mismatch regularization.

        Loss = Soft IQ loss + mismatch_coef * mismatch(preference, Q_expert)

        Args:
            states: [batch, obs_dim]
            actions: [batch] (discrete actions)
            next_states: [batch, obs_dim]
            preference_weights: [batch, n_objectives]

        Returns:
            Dictionary of loss values
        """
        # ===== Compute Q-values (objective-dimensional) =====
        # Q(s, a) for all actions: [batch, action_dim, n_objectives]
        q_all_actions = []
        for _ in range(self.action_dim):
            # Create dummy action inputs (not used in current architecture)
            state_action = states  # Simplified: assuming Q depends only on state
            _, q_values = self.q_network.act(state_action)  # [batch, n_objectives]
            q_all_actions.append(q_values)
        q_all_actions = torch.stack(
            q_all_actions, dim=1
        )  # [batch, action_dim, n_objectives]

        # Get Q(s, a) for taken actions
        actions_idx = actions.long().unsqueeze(-1).unsqueeze(-1)  # [batch, 1, 1]
        actions_idx = actions_idx.expand(
            -1, -1, self.n_objectives
        )  # [batch, 1, n_objectives]
        q_current = q_all_actions.gather(1, actions_idx).squeeze(
            1
        )  # [batch, n_objectives]

        # ===== Compute v_init = logsumexp(q(s_0, a)) over all actions =====
        # Use the first state in the batch as s_0 (initial state)
        # Q(s_0, a) for all actions from q_all_actions[0]: [action_dim, n_objectives]
        q_init_all = q_all_actions[0]  # [action_dim, n_objectives]

        # Convert to scalar Q using preference weights for initial state
        q_init_scalar = torch.einsum(
            "ao,o->a", q_init_all, preference_weights[0]
        )  # [action_dim]

        # Compute v_init = alpha * logsumexp(Q(s_0, a) / alpha)
        v_init = self.alpha * torch.logsumexp(
            q_init_scalar / self.alpha, dim=0
        )  # scalar

        # ===== Compute target Q-values =====
        with torch.no_grad():
            # Next state Q-values: [batch, action_dim, n_objectives]
            q_next_all = []
            for _ in range(self.action_dim):
                _, q_next = self.q_target.act(next_states)
                q_next_all.append(q_next)
            q_next_all = torch.stack(q_next_all, dim=1)

            # Soft max over actions: V(s') = alpha * log sum_a exp(Q(s',a) / alpha)
            # For objective-dimensional Q, we compute scalar Q using preference weights
            # q_scalar = preference^T * q_objectives
            q_next_scalar = torch.einsum(
                "bao,bo->ba", q_next_all, preference_weights
            )  # [batch, action_dim]

            # Soft value: V(s') = alpha * logsumexp(Q(s',a) / alpha)
            v_next = self.alpha * torch.logsumexp(
                q_next_scalar / self.alpha, dim=1
            )  # [batch]

        # ===== Compute current Q scalar =====
        q_expert = torch.einsum("bo,bo->b", q_current, preference_weights)

        # Soft IQ loss: -(Q(s,a) - gamma * V(s')) + (1 - gamma) * V(s_0)
        soft_iq_loss = (
            -(q_expert - self.gamma * v_next) + (1 - self.gamma) * v_init
        ).mean()

        # ===== Mismatch Regularization =====
        # For expert data, add mismatch between predicted preference and expert Q direction
        mismatch_loss = compute_mismatch(
            preference=preference_weights, q_expert=q_current
        ).mean()

        # ===== Total Loss =====
        total_loss = soft_iq_loss + self.mismatch_coef * mismatch_loss

        return {
            "total_loss": total_loss,
            "soft_iq_loss": soft_iq_loss,
            "mismatch_loss": mismatch_loss,
        }

    def update(
        self,
        states: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        next_states: np.ndarray,
        dones: np.ndarray,
        is_expert: np.ndarray,
    ) -> Dict[str, float]:
        """
        Update Q-network with a batch of data.

        Args:
            states: [batch, obs_dim]
            actions: [batch]
            rewards: [batch, n_objectives]
            next_states: [batch, obs_dim]
            dones: [batch]
            is_expert: [batch] - 1 for expert, 0 for agent

        Returns:
            Dictionary of loss values
        """
        # Convert to tensors
        states_t = torch.FloatTensor(states).to(self.device)
        actions_t = torch.FloatTensor(actions).to(self.device)
        _rewards_t = torch.FloatTensor(rewards).to(self.device)  # For future use
        next_states_t = torch.FloatTensor(next_states).to(self.device)
        _dones_t = torch.FloatTensor(dones).to(self.device)  # For future use
        _is_expert_t = torch.FloatTensor(is_expert).to(self.device)  # For future use

        # Update SSM to get current preference estimate for each transition
        batch_size = states.shape[0]
        preference_weights = np.zeros((batch_size, self.n_objectives))

        for i in range(batch_size):
            # Get Q-values to use as "expert Q" for SSM update
            with torch.no_grad():
                _, q_values = self.q_network.act(states_t[i : i + 1])
                q_expert = q_values.squeeze().cpu().numpy()

            # Update SSM
            self.ssm.update(
                observation=states[i],
                action=actions[i],
                q_expert=q_expert,
                next_observation=next_states[i],
            )

            # Get current preference estimate
            preference_weights[i] = self.ssm.predict(states[i], actions[i])

        self.current_preference = np.mean(preference_weights, axis=0)
        preference_weights_t = torch.FloatTensor(preference_weights).to(self.device)

        # ===== Update Q-network =====
        q_loss_dict = self.compute_soft_iq_loss(
            states_t,
            actions_t,
            next_states_t,
            preference_weights_t,
        )

        # Backward pass for Q
        self.q_optimizer.zero_grad()
        q_total_loss = q_loss_dict["total_loss"]
        q_total_loss.backward()
        self.q_optimizer.step()

        # ===== Update Actor =====
        actor_loss = self.compute_actor_loss(states_t, preference_weights_t)

        # Backward pass for actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Soft update target network
        self._soft_update()

        # Convert losses to scalars for logging
        losses = {
            "total_loss": q_loss_dict["total_loss"].item(),
            "soft_iq_loss": q_loss_dict["soft_iq_loss"].item(),
            "mismatch_loss": q_loss_dict["mismatch_loss"].item(),
            "actor_loss": actor_loss.item(),
            "mean_preference": self.current_preference.tolist(),
        }

        return losses

    def save(self, path: str):
        """Save model checkpoint."""
        save_path = Path(path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        torch.save(
            {
                "q_network": self.q_network.state_dict(),
                "q_target": self.q_target.state_dict(),
                "q_optimizer": self.q_optimizer.state_dict(),
                "actor_optimizer": self.actor_optimizer.state_dict(),
                "current_preference": self.current_preference,
            },
            save_path,
        )

    def load(self, path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.q_network.load_state_dict(checkpoint["q_network"])
        self.q_target.load_state_dict(checkpoint["q_target"])
        self.q_optimizer.load_state_dict(checkpoint["q_optimizer"])
        self.actor_optimizer.load_state_dict(checkpoint["actor_optimizer"])
        self.current_preference = checkpoint["current_preference"]
