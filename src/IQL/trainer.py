"""
Objective-Dimensional Soft Inverse Q-Learning (OD-SQIL) Trainer.

Implements soft IQL with:
- Objective-dimensional Q-values output
- Margin-based regularization to maximize alignment of expert actions with preferences
- Preference weight prediction using SSM
"""

import numpy as np
import torch
from pathlib import Path
from typing import Dict

from .actor_critic import ActorCritic
from .ssm import StateSpaceModel, compute_margin


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
        mismatch_coef: float = 1.0,
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
            mismatch_coef: Coefficient for mismatch regularization
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
        self.q_optimizer = torch.optim.Adam(self.q_network.parameters(), lr=lr)

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

    def compute_soft_iq_loss(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        next_states: torch.Tensor,
        preference_weights: torch.Tensor,
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
            next_states: [batch, obs_dim]
            preference_weights: [batch, n_objectives]
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

        # ===== Compute v_init = E_{a~π}[Q(s_0, a)] =====
        # Compute Q-values for initial states: [batch, action_dim, n_objectives]
        _, q_init_all_actions = self.q_network.act(
            initial_states
        )  # [batch, action_dim, n_objectives]

        # Convert to scalar Q using preference weights for initial states
        # q_init_all_actions: [batch, action_dim, n_objectives]
        # initial_preferences: [batch, n_objectives]
        q_init_scalar = torch.einsum(
            "bao,bo->ba", q_init_all_actions, initial_preferences
        )  # [batch, action_dim]

        # Get policy probabilities for initial states
        with torch.no_grad():
            logits_init, _ = self.q_network.act(initial_states)
            pi_init = torch.softmax(logits_init, dim=-1)  # [batch, action_dim]

        # Compute v_init = average over initial states: Σ_a π(a|s_0) * Q(s_0, a)
        v_init_per_state = torch.sum(pi_init * q_init_scalar, dim=1)  # [batch]
        v_init = v_init_per_state.mean()  # scalar

        # ===== Compute target Q-values =====
        with torch.no_grad():
            # Next state Q-values: [batch, action_dim, n_objectives]
            _, q_next_all = self.q_target.act(
                next_states
            )  # [batch, action_dim, n_objectives]

            # Expectation over policy: V(s') = E_{a~π}[Q(s', a)]
            # For objective-dimensional Q, we compute scalar Q using preference weights
            # q_scalar = preference^T * q_objectives
            q_next_scalar = torch.einsum(
                "bao,bo->ba", q_next_all, preference_weights
            )  # [batch, action_dim]

            # Get policy probabilities for next states
            logits_next, _ = self.q_target.act(next_states)
            pi_next = torch.softmax(logits_next, dim=-1)  # [batch, action_dim]

            # Value: V(s') = Σ_a π(a|s') * Q(s', a)
            v_next = torch.sum(pi_next * q_next_scalar, dim=1)  # [batch]

        # ===== Compute current Q scalar =====
        q_expert = torch.einsum("bo,bo->b", preference_weights, q_current)  # [batch]

        # Soft IQ loss: -(Q(s,a) - gamma * V(s')) + (1 - gamma) * V(s_0)
        # where V(s) = E_{a~π(·|s)}[Q(s,a)] = Σ_a π(a|s) * Q(s,a)
        soft_iq_loss = (
            -(q_expert - self.gamma * v_next) + (1 - self.gamma) * v_init
        ).mean()

        # ===== Margin Regularization =====
        # Maximize margin: max_other_mismatch - expert_mismatch
        # Higher margin means expert action is better aligned than the BEST alternative
        # We want to MAXIMIZE margin, so loss = -margin
        margins = compute_margin(preference_weights, q_all_actions, actions)  # [batch]
        margin_loss = -margins.mean()  # Negative because we want to maximize margin

        # ===== Total Loss =====
        total_loss = soft_iq_loss + self.mismatch_coef * margin_loss

        return total_loss

    def update(self, trajectory: Dict) -> Dict[str, float]:
        """
        Update Q-network with a single trajectory.

        Processes trajectory sequentially to respect SSM's temporal structure.

        Args:
            trajectory: Dictionary containing:
                - observations: [T, obs_dim]
                - actions: [T]
                - mo_rewards: [T, n_objectives]
                - preference_weights: [T, n_objectives]

        Returns:
            Dictionary of metrics including:
                - loss: Training loss
                - preference_mae: Mean absolute error for preference prediction
                - cross_entropy: Cross-entropy loss (imitation metric)
        """
        traj_states = np.array(trajectory["observations"])
        traj_actions = np.array(trajectory["actions"])
        traj_prefs = np.array(trajectory["preference_weights"])
        T = len(traj_states)

        # Reset SSM for this trajectory
        self.ssm.reset()

        # Initial state info for v_init computation
        initial_state = traj_states[0]
        initial_preference = self.ssm.predict(initial_state, None)

        losses = []
        ssm_losses = []
        preference_errors = []
        cross_entropies = []

        for t in range(T - 1):
            state = traj_states[t]
            action = traj_actions[t]
            next_state = traj_states[t + 1]
            true_preference = traj_prefs[t]

            # Get Q-values for SSM update
            with torch.no_grad():
                states = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                _, q_values_all_batch = self.q_network.act(states)
                q_values_all = q_values_all_batch[0].cpu().numpy()

            # Update SSM with Q-values (returns loss without backward)
            ssm_loss = self.ssm.update(
                observation=state, action=action, q_values_all=q_values_all
            )
            if ssm_loss is not None:
                ssm_losses.append(ssm_loss)

            # Predict preference
            if hasattr(self.ssm, "hidden"):
                predicted_preference, _ = self.ssm.predict(state, self.ssm.hidden)
            else:
                predicted_preference = self.ssm.predict(state, None)

            # Compute preference MAE
            pref_error = np.abs(predicted_preference[0] - true_preference[0])
            preference_errors.append(pref_error)

            # Compute cross-entropy for imitation learning metric
            with torch.no_grad():
                state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                pred_pref_t = (
                    torch.FloatTensor(predicted_preference).unsqueeze(0).to(self.device)
                )
                action_t = torch.LongTensor([action]).to(self.device)

                logits, _ = self.q_network.act(state_t, pred_pref_t)

                ce_loss = torch.nn.functional.cross_entropy(
                    logits, action_t, reduction="none"
                )
                cross_entropies.append(ce_loss.item())

            # Compute training loss
            loss = self.compute_soft_iq_loss(
                states=torch.FloatTensor(state).unsqueeze(0).to(self.device),
                actions=torch.LongTensor([action]).to(self.device),
                next_states=torch.FloatTensor(next_state).unsqueeze(0).to(self.device),
                preference_weights=torch.FloatTensor(predicted_preference)
                .unsqueeze(0)
                .to(self.device),
                initial_states=torch.FloatTensor(initial_state)
                .unsqueeze(0)
                .to(self.device),
                initial_preferences=torch.FloatTensor(initial_preference)
                .unsqueeze(0)
                .to(self.device),
            )

            losses.append(loss)

        # Average metrics across trajectory
        mean_loss = torch.stack(losses).mean()

        # Update Q-network
        self.q_optimizer.zero_grad()
        mean_loss.backward()
        self.q_optimizer.step()
        self._soft_update()

        metrics = {
            "loss": mean_loss.item(),
            "preference_mae": np.mean(preference_errors),
            "cross_entropy": np.mean(cross_entropies),
        }

        return metrics

    def save(self, path: str):
        """Save model checkpoint."""
        save_path = Path(path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        checkpoint = {
            "q_network": self.q_network.state_dict(),
            "q_target": self.q_target.state_dict(),
            "q_optimizer": self.q_optimizer.state_dict(),
            "current_preference": self.current_preference,
        }

        # Save Q-network
        torch.save(checkpoint, save_path)

        # Non-neural SSMs (PF, EKF) don't need separate saving

    def load(self, path: str):
        """Load model checkpoint."""
        # Load Q-network (weights_only=False for numpy arrays)
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.q_network.load_state_dict(checkpoint["q_network"])
        self.q_target.load_state_dict(checkpoint["q_target"])
        self.q_optimizer.load_state_dict(checkpoint["q_optimizer"])
        self.current_preference = checkpoint["current_preference"]

        # Non-neural SSMs (PF, EKF) don't need separate loading

    def evaluate(
        self,
        expert_dir: str,
        n_trajectories: int = 10,
        save_dir: str = None,
        update_step: int = None,
        eval_weights: np.ndarray = None,
    ):
        """
        Evaluate preference prediction accuracy on expert trajectories.

        Uses step-wise prediction for simple SSMs (ParticleFilter, EKF).

        Args:
            expert_dir: Directory containing trajectories.json
            n_trajectories: Number of trajectories to evaluate
            save_dir: Directory to save detailed predictions (optional)
            update_step: Current update step for logging (optional)
            eval_weights: Weights [w1, w2] for eval_score = w1*cross_entropy + w2*preference_mae

        Returns:
            Dictionary containing evaluation metrics
        """
        import json

        # Set Q-network to eval mode
        self.q_network.eval()

        # Load trajectories
        expert_path = Path(expert_dir)
        traj_path = expert_path / "trajectories.json"

        with open(traj_path, "r") as f:
            all_trajectories = json.load(f)

        # Sample trajectories
        n_available = len(all_trajectories)
        n_eval = min(n_trajectories, n_available)
        indices = np.random.choice(n_available, n_eval, replace=False)
        eval_trajectories = [all_trajectories[i] for i in indices]

        preference_errors = []
        cross_entropy_losses = []
        detailed_predictions = []
        eval_scores = []

        # Process each trajectory
        for traj_idx, traj in enumerate(eval_trajectories):
            traj_states = np.array(traj["observations"])
            traj_actions = np.array(traj["actions"])
            traj_prefs = np.array(traj["preference_weights"])

            # Reset SSM for step-wise processing
            self.ssm.reset()

            traj_predictions = []
            traj_cross_entropies = []

            for t in range(len(traj_actions)):
                state = traj_states[t]
                action = traj_actions[t]
                true_pref = traj_prefs[t]

                # Get predicted preference (before update)
                if hasattr(self.ssm, "hidden"):
                    result = self.ssm.predict(state, self.ssm.hidden)
                    if isinstance(result, tuple):
                        pred_pref, _ = result
                    else:
                        pred_pref = result
                else:
                    pred_pref = self.ssm.predict(state, None)

                with torch.no_grad():
                    state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                    pred_pref_t = (
                        torch.FloatTensor(pred_pref).unsqueeze(0).to(self.device)
                    )
                    action_t = torch.LongTensor([action]).to(self.device)

                    logits, q_values_all_batch = self.q_network.act(
                        state_t, pred_pref_t
                    )
                    q_values_all = q_values_all_batch[0].cpu().numpy()

                    policy_probs = torch.softmax(logits[0], dim=0)

                    ce_loss = torch.nn.functional.cross_entropy(
                        logits, action_t, reduction="none"
                    )
                    traj_cross_entropies.append(ce_loss.item())

                # Update SSM with Q-values
                self.ssm.update(
                    observation=state, action=action, q_values_all=q_values_all
                )

                pref_error = np.abs(pred_pref[0] - true_pref[0])
                preference_errors.append(pref_error)

                traj_predictions.append(
                    {
                        "timestep": int(t),
                        "state": state.tolist(),
                        "action": int(action),
                        "ground_truth_preference": true_pref.tolist(),
                        "predicted_preference": pred_pref.tolist(),
                        "policy_probs": policy_probs.cpu().numpy().tolist(),
                        "mae_preference": float(pref_error),
                        "cross_entropy": float(traj_cross_entropies[-1]),
                    }
                )

            mean_traj_ce = np.mean(traj_cross_entropies)
            cross_entropy_losses.append(mean_traj_ce)

            if eval_weights is not None:
                traj_pref_errors = preference_errors[-(len(traj_actions)) :]
                traj_pref_mae = np.mean(traj_pref_errors)
                scores = np.array([mean_traj_ce, traj_pref_mae])
                traj_eval_score = np.dot(eval_weights, scores)
                eval_scores.append(traj_eval_score)

            traj_pred_dict = {
                "trajectory_idx": int(traj_idx),
                "trajectory_length": len(traj_actions),
                "mean_cross_entropy": float(mean_traj_ce),
                "predictions": traj_predictions,
            }
            if eval_weights is not None:
                traj_pred_dict["eval_score"] = float(traj_eval_score)

            detailed_predictions.append(traj_pred_dict)

        results = {
            "mean_preference_mae": np.mean(preference_errors),
            "std_preference_mae": np.std(preference_errors),
            "mean_cross_entropy": np.mean(cross_entropy_losses),
            "std_cross_entropy": np.std(cross_entropy_losses),
        }

        if eval_weights is not None:
            results["mean_eval_score"] = np.mean(eval_scores)
            results["std_eval_score"] = np.std(eval_scores)

        # Save detailed predictions
        if save_dir is not None:
            save_path = Path(save_dir)
            save_path.mkdir(parents=True, exist_ok=True)

            if update_step is not None:
                json_filename = f"step_{update_step}.json"
            else:
                json_filename = "eval_predictions.json"

            json_path = save_path / json_filename

            eval_data = {
                "mean_preference_mae": results["mean_preference_mae"],
                "std_preference_mae": results["std_preference_mae"],
                "mean_cross_entropy": results["mean_cross_entropy"],
                "std_cross_entropy": results["std_cross_entropy"],
                "trajectories": detailed_predictions,
            }

            if "mean_eval_score" in results:
                eval_data["mean_eval_score"] = results["mean_eval_score"]
                eval_data["std_eval_score"] = results["std_eval_score"]

            with open(json_path, "w") as f:
                json.dump(eval_data, f, indent=2)

            print(f"Saved detailed predictions to {json_path}")

        # Set Q-network back to train mode
        self.q_network.train()

        return results
