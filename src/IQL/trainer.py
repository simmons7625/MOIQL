"""
Objective-Dimensional Soft Inverse Q-Learning (OD-SQIL) Trainer.

Implements soft IQL with:
- Objective-dimensional Q-values output
- Mismatch-based regularization to align expert action Q-values with preferences
- Preference weight prediction using SSM
"""

import numpy as np
import torch
from pathlib import Path
from typing import Dict

from .actor_critic import ActorCritic
from .ssm import StateSpaceModel


class SSMIQTrainer:
    """
    Simple SSM-based IQL Trainer for non-neural SSMs (ParticleFilter, EKF).

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
        mismatch_coef: float = 0.0,
        weight_decay: float = 0.0,
        max_timesteps: int = None,
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
            mismatch_coef: Coefficient for mismatch update ratio term (0.0 = disable)
            weight_decay: Weight decay (L2 penalty) for AdamW optimizer
            max_timesteps: Max timesteps per episode (None = use done signal only)
            device: Device to run on
        """
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.n_objectives = n_objectives
        self.gamma = gamma
        self.tau = tau
        self.mismatch_coef = mismatch_coef
        self.max_timesteps = max_timesteps
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

        # Optimizer for Q-network (no separate actor optimizer)
        self.q_optimizer = torch.optim.AdamW(
            self.q_network.parameters(), lr=lr, weight_decay=weight_decay
        )

    def save(self, path: str):
        """Save model checkpoint."""
        save_path = Path(path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        checkpoint = {
            "q_network": self.q_network.state_dict(),
            "q_target": self.q_target.state_dict(),
            "q_optimizer": self.q_optimizer.state_dict(),
        }

        # Save Q-network
        torch.save(checkpoint, save_path)

    def load(self, path: str):
        """Load model checkpoint."""
        # Load Q-network (weights_only=False for numpy arrays)
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.q_network.load_state_dict(checkpoint["q_network"])
        self.q_target.load_state_dict(checkpoint["q_target"])
        self.q_optimizer.load_state_dict(checkpoint["q_optimizer"])

    def _soft_update(self):
        """Soft update of target network."""
        for param, target_param in zip(
            self.q_network.parameters(), self.q_target.parameters()
        ):
            target_param.data.copy_(
                self.tau * param.data + (1 - self.tau) * target_param.data
            )

    def _compute_soft_iq_loss(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        current_preferences: torch.Tensor,
        next_states: torch.Tensor,
        next_preferences: torch.Tensor,
        initial_states: torch.Tensor,
        initial_preferences: torch.Tensor,
    ) -> Dict[str, float]:
        """
        Compute Soft IQ loss with margin regularization.

        Loss = Soft IQ loss - mismatch_coef * margin
        where margin = max_other_mismatch - expert_mismatch

        This encourages the Q-network to maximize the margin between expert
        action and the BEST competing action based on preference alignment.
        Using max (instead of mean) enforces a stricter criterion.

        Args:
            states: [batch, obs_dim]
            actions: [batch] (discrete actions)
            current_preferences: [batch, n_objectives]
            next_states: [batch, obs_dim]
            next_preferences: [batch, n_objectives]
            initial_states: [batch, obs_dim] - initial states from episodes
            initial_preferences: [batch, n_objectives] - preferences at initial states

        Returns:
            Dictionary of loss values
        """
        # ===== Compute Q-values (objective-dimensional) =====
        # Q(s, a) for all actions: [batch, action_dim, n_objectives]
        _, q_all_actions = self.q_network.act(
            states
        )  # [batch, action_dim, n_objectives]

        # Get Q(s, a) for taken actions
        actions_idx = actions.long().unsqueeze(-1).unsqueeze(-1)  # [batch, 1, 1]
        actions_idx = actions_idx.expand(
            -1, -1, self.n_objectives
        )  # [batch, 1, n_objectives]
        q_current = q_all_actions.gather(1, actions_idx).squeeze(
            1
        )  # [batch, n_objectives]

        with torch.no_grad():
            # ===== Compute v_init = E_{a~π}[Q(s_0, a)] =====
            # Compute Q-values for initial states: [batch, action_dim, n_objectives]
            _, q_init_all_actions = self.q_target.act(
                initial_states
            )  # [batch, action_dim, n_objectives]

            # Convert to scalar Q using preference weights for initial states
            # q_init_all_actions: [batch, action_dim, n_objectives]
            # initial_preferences: [batch, n_objectives]
            q_init_scalar = torch.einsum(
                "bao,bo->ba", q_init_all_actions, initial_preferences
            )  # [batch, action_dim]
            v_init = torch.logsumexp(q_init_scalar, dim=1).mean()  # Scalar

            # ===== Compute target V(s') =====
            # Next state Q-values: [batch, action_dim, n_objectives]
            _, q_next_all = self.q_target.act(
                next_states
            )  # [batch, action_dim, n_objectives]

            # Expectation over policy: V(s') = E_{a~π}[Q(s', a)]
            # For objective-dimensional Q, we compute scalar Q using preference weights
            # q_scalar = preference^T * q_objectivesmoiql_results/20251105_155332
            q_next_scalar = torch.einsum(
                "bao,bo->ba", q_next_all, next_preferences
            )  # [batch, action_dim]
            # Value: V(s') = log Σ_a Q(s', a)
            v_next = torch.logsumexp(q_next_scalar, dim=1).mean()  # [batch]

        # ===== Compute current Q scalar =====
        q_expert = torch.einsum("bo,bo->b", q_current, current_preferences)  # [batch]

        # Soft IQ loss: -(Q(s,a) - gamma * V(s')) + (1 - gamma) * V(s_0)
        # where V(s) = E_{a~π(·|s)}[Q(s,a)] = Σ_a π(a|s) * Q(s,a)
        soft_iq_loss = -(q_expert - self.gamma * v_next) + (1 - self.gamma) * v_init

        # Compute mismatch regularization term
        # Normalize both to unit vectors for comparison
        q_current_norm = q_current / (torch.norm(q_current, dim=1, keepdim=True) + 1e-8)
        current_preferences_norm = current_preferences / (
            torch.norm(current_preferences, dim=1, keepdim=True) + 1e-8
        )
        # Compute mismatch (MSE between normalized vectors)
        mismatch_loss = torch.sum(
            (q_current_norm - current_preferences_norm) ** 2, dim=1
        )  # [batch]

        # Total loss: soft IQ loss + coefficient * mismatch improvement ratio
        loss = soft_iq_loss + self.mismatch_coef * mismatch_loss

        return loss.mean()

    def _ssm_rollout(
        self,
        trajectory: dict,
    ) -> list:
        """
        Rollout SSM on a trajectory to predict preference weights.

        Args:
            trajectory: Trajectory dictionary with 'observations' and 'actions'

        Returns:
            List of predicted preference weights for each timestep
        """
        self.ssm.reset()
        traj_states = np.array(trajectory["observations"])
        traj_actions = np.array(trajectory["actions"])
        traj_true_prefs = np.array(trajectory["preference_weights"])
        predicted_prefs = []
        preference_errors = []
        cross_entropies = []

        for t in range(len(traj_states)):
            state = traj_states[t]
            action = traj_actions[t]
            true_pref = traj_true_prefs[t]

            # Predict preference (before update)
            pred_pref = self.ssm.predict()
            predicted_prefs.append(pred_pref)

            # Compute metrics
            pref_error = np.mean(np.abs(pred_pref - true_pref))
            preference_errors.append(pref_error)

            # Compute cross-entropy
            with torch.no_grad():
                self.q_target.eval()
                state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                _, q_values = self.q_target.act(state_t)
                q_values_all = q_values.squeeze(0).cpu().numpy()
                pred_pref_t = torch.FloatTensor(pred_pref).unsqueeze(0).to(self.device)
                action_t = torch.LongTensor([action]).to(self.device)
                logits, _ = self.q_target.act(state_t, pred_pref_t)
                ce_loss = torch.nn.functional.cross_entropy(
                    logits, action_t, reduction="none"
                )
                cross_entropies.append(ce_loss.item())
                self.q_target.train()

            # Update SSM
            self.ssm.update(observation=state, action=action, q_values_all=q_values_all)
        return predicted_prefs, preference_errors, cross_entropies

    def train(self, trajectory: dict) -> Dict[str, float]:
        """
        Train on a single trajectory.

        Process:
        1. SSM rollout to predict preferences for each step
        2. Update Q-network on each transition
        3. Update SSM with expert actions and Q-values
        4. Soft update target network after trajectory

        Args:
            trajectory: Single trajectory dictionary with observations, actions, rewards

        Returns:
            Dictionary containing metrics (loss, preference_mae, cross_entropy)
        """
        # Step 1: SSM rollout to predict preferences
        predicted_preferences, preference_errors, cross_entropies = self._ssm_rollout(
            trajectory
        )

        # Step 2: Update Q-network on trajectory transitions
        observations = trajectory["observations"]
        actions = trajectory["actions"]
        n_steps = len(actions)

        trajectory_loss = 0.0
        for t in range(n_steps - 1):
            state = torch.tensor(
                observations[t], dtype=torch.float32, device=self.device
            ).unsqueeze(0)
            action = torch.tensor([actions[t]], dtype=torch.long, device=self.device)
            next_state = torch.tensor(
                observations[t + 1], dtype=torch.float32, device=self.device
            ).unsqueeze(0)
            current_pref = torch.tensor(
                predicted_preferences[t], dtype=torch.float32, device=self.device
            ).unsqueeze(0)
            next_pref = torch.tensor(
                predicted_preferences[t + 1], dtype=torch.float32, device=self.device
            ).unsqueeze(0)
            initial_state = torch.tensor(
                observations[0], dtype=torch.float32, device=self.device
            ).unsqueeze(0)
            initial_pref = torch.tensor(
                predicted_preferences[0], dtype=torch.float32, device=self.device
            ).unsqueeze(0)

            # Compute loss
            loss = self._compute_soft_iq_loss(
                states=state,
                actions=action,
                current_preferences=current_pref,
                next_states=next_state,
                next_preferences=next_pref,
                initial_states=initial_state,
                initial_preferences=initial_pref,
            )

            trajectory_loss += loss.item()

            # Backward and optimize
            self.q_optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=0.5)
            self.q_optimizer.step()

        # Step 3: Soft update target network after trajectory
        self._soft_update()

        # Compute metrics
        mean_loss = trajectory_loss / n_steps if n_steps > 0 else 0.0
        metrics = {
            "loss": mean_loss,
            "preference_mae": np.mean(preference_errors),
            "cross_entropy": np.mean(cross_entropies),
        }

        return metrics
