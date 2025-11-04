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
from .dataset import create_iql_dataloader


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
            # ===== Compute Q(s, a) from target network =====
            _, q_all_actions_target = self.q_target.act(
                states
            )  # [batch, action_dim, n_objectives]

            # Get Q(s, a) for taken actions from target network
            actions_idx_target = (
                actions.long().unsqueeze(-1).unsqueeze(-1)
            )  # [batch, 1, 1]
            actions_idx_target = actions_idx_target.expand(
                -1, -1, self.n_objectives
            )  # [batch, 1, n_objectives]
            q_current_old = q_all_actions_target.gather(1, actions_idx_target).squeeze(
                1
            )  # [batch, n_objectives]

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
            # q_scalar = preference^T * q_objectives
            q_next_scalar = torch.einsum(
                "bao,bo->ba", q_next_all, next_preferences
            )  # [batch, action_dim]
            # Value: V(s') = log Σ_a Q(s', a)
            v_next = torch.logsumexp(q_next_scalar, dim=1).mean()  # [batch]

        # ===== Compute current Q scalar =====
        q_expert = torch.einsum("bo,bo->b", q_current, current_preferences)  # [batch]

        # Soft IQ loss: -(Q(s,a) - gamma * V(s')) + (1 - gamma) * V(s_0)
        # where V(s) = E_{a~π(·|s)}[Q(s,a)] = Σ_a π(a|s) * Q(s,a)
        soft_iq_loss = (
            -(q_expert - self.gamma * v_next) + (1 - self.gamma) * v_init
        ).mean()

        # Compute mismatch regularization term
        # Normalize both to unit vectors for comparison
        q_current_norm = q_current / (torch.norm(q_current, dim=1, keepdim=True) + 1e-8)
        current_preferences_norm = current_preferences / (
            torch.norm(current_preferences, dim=1, keepdim=True) + 1e-8
        )
        # Compute mismatch (MSE between normalized vectors)
        mismatch = torch.sum(
            (q_current_norm - current_preferences_norm) ** 2, dim=1
        )  # [batch]

        # Compute mismatch_old using target network
        q_current_old_norm = q_current_old / (
            torch.norm(q_current_old, dim=1, keepdim=True) + 1e-8
        )
        mismatch_old = torch.sum(
            (q_current_old_norm - current_preferences_norm) ** 2, dim=1
        )  # [batch]

        # Compute update ratio: (improvement) / (current mismatch)
        mismatch_update_ratio = (mismatch_old - mismatch) / (mismatch + 1e-8)

        # Total loss: soft IQ loss + coefficient * mismatch improvement ratio
        loss = soft_iq_loss + self.mismatch_coef * mismatch_update_ratio.mean()

        return loss

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
                state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                _, q_values = self.q_network.act(state_t)
                q_values_all = q_values.squeeze(0).cpu().numpy()
                pred_pref_t = torch.FloatTensor(pred_pref).unsqueeze(0).to(self.device)
                action_t = torch.LongTensor([action]).to(self.device)
                logits, _ = self.q_network.act(state_t, pred_pref_t)
                ce_loss = torch.nn.functional.cross_entropy(
                    logits, action_t, reduction="none"
                )
                cross_entropies.append(ce_loss.item())

            # Update SSM
            self.ssm.update(observation=state, action=action, q_values_all=q_values_all)
        return predicted_prefs, preference_errors, cross_entropies

    def train(
        self,
        trajectories: list,
        predicted_preferences: list = None,
        batch_size: int = 256,
    ) -> tuple[Dict[str, float], list]:
        """
        Train for one epoch (batch-based training).

        Process:
        1. Use provided predictions or predict preferences for all trajectories (first epoch)
        2. Create IQL dataloader from trajectories with predicted preferences
        3. Update Q-network on batches
        4. Update SSM with expert actions and Q-values, collecting new predictions

        Args:
            trajectories: List of trajectory dictionaries
            batch_size: Batch size for training
            predicted_preferences: Optional list of predicted preferences from previous epoch

        Returns:
            Tuple of (metrics dict, new predicted preferences for next epoch)
        """
        # Step 1: Predict preferences if not provided
        if predicted_preferences is None:
            predicted_preferences = []
            for traj in trajectories:
                pref, _, _ = self._ssm_rollout(traj)
                predicted_preferences.append(pref)

        # Step 2: Create dataloader with predicted preferences
        train_loader = create_iql_dataloader(
            trajectories=trajectories,
            predicted_preferences=predicted_preferences,
            batch_size=batch_size,
            shuffle=True,
        )

        # Step 3: Update Q-network on batches
        epoch_loss = 0.0
        n_batches = 0

        for batch in train_loader:
            states = batch["states"].to(self.device)
            actions = batch["actions"].to(self.device)
            next_states = batch["next_states"].to(self.device)
            current_preferences = batch["current_preferences"].to(self.device)
            next_preferences = batch["next_preferences"].to(self.device)
            initial_states = batch["initial_states"].to(self.device)
            initial_preferences = batch["initial_preferences"].to(self.device)

            # Compute loss
            loss = self._compute_soft_iq_loss(
                states=states,
                actions=actions,
                current_preferences=current_preferences,
                next_states=next_states,
                next_preferences=next_preferences,
                initial_states=initial_states,
                initial_preferences=initial_preferences,
            )

            epoch_loss += loss.item()
            n_batches += 1

            # Backward and optimize
            self.q_optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=1.0)
            self.q_optimizer.step()

            # Soft update target network
            self._soft_update()

        # Compute mean loss for epoch
        mean_epoch_loss = epoch_loss / n_batches if n_batches > 0 else 0.0

        # Step 4: Update SSM and collect new predictions
        predicted_preferences = []
        preference_errors = []
        cross_entropies = []

        for traj in trajectories:
            pref, error, ce = self._ssm_rollout(traj)
            predicted_preferences.append(pref)
            preference_errors.extend(error)
            cross_entropies.extend(ce)

        metrics = {
            "loss": mean_epoch_loss,
            "preference_mae": np.mean(preference_errors),
            "cross_entropy": np.mean(cross_entropies),
        }

        return metrics, predicted_preferences

    def run(
        self,
        trajectories: list,
        n_epochs: int,
        batch_size: int = 256,
        eval_fn=None,
        eval_interval: int = 10,
    ) -> list:
        """
        Run training for multiple epochs.

        Args:
            trajectories: List of trajectory dictionaries
            n_epochs: Number of epochs to train
            batch_size: Batch size for training
            eval_fn: Optional evaluation function to call periodically
            eval_interval: Evaluate every N epochs

        Returns:
            List of dictionaries containing metrics for each epoch
        """
        history = []
        predicted_prefs = None  # Will be initialized in first epoch

        for epoch in range(n_epochs):
            # Train for one epoch, passing predictions from previous epoch
            metrics, predicted_prefs = self.train(
                trajectories,
                batch_size=batch_size,
                predicted_preferences=predicted_prefs,
            )
            metrics["epoch"] = epoch + 1

            history.append(metrics)

            # Print progress
            if (epoch + 1) % 10 == 0:
                print(
                    f"Epoch {epoch + 1}/{n_epochs} - "
                    f"Loss: {metrics['loss']:.4f}, "
                    f"Pref MAE: {metrics['preference_mae']:.4f}, "
                    f"CE: {metrics['cross_entropy']:.4f}"
                )

            # Evaluate periodically
            if eval_fn is not None and (epoch + 1) % eval_interval == 0:
                eval_metrics = eval_fn()
                print(f"Evaluation at epoch {epoch + 1}:")
                for key, value in eval_metrics.items():
                    print(f"  {key}: {value:.4f}")

        return history
