"""
Common evaluation functions for IQL trainers.
"""

import json
import numpy as np
import torch
from pathlib import Path
from typing import Dict, Any


def evaluate(
    trainer,
    expert_dir: str,
    n_trajectories: int = 10,
    save_dir: str = None,
    update_step: int = None,
    eval_weights: np.ndarray = None,
) -> Dict[str, Any]:
    """
    Evaluate preference prediction accuracy on expert trajectories.

    Processes complete trajectories sequentially to respect SSM's hidden state.

    Args:
        trainer: IQL trainer (SSMIQTrainer or NeuralSSMIQTrainer)
        expert_dir: Directory containing trajectories.json
        n_trajectories: Number of trajectories to evaluate
        save_dir: Directory to save detailed predictions (optional)
        update_step: Current update step for logging (optional)
        eval_weights: Weights [w1, w2] for computing eval_score = w1*cross_entropy + w2*preference_mae (optional)

    Returns:
        Dictionary containing:
        - mean_preference_mae: Mean absolute error for preference prediction
        - std_preference_mae: Std of preference MAE
        - mean_cross_entropy: Mean cross-entropy loss
        - std_cross_entropy: Std of cross-entropy loss
        - mean_eval_score: Mean weighted sum (w1*CE + w2*MAE) if eval_weights provided
        - std_eval_score: Std of weighted sum (w1*CE + w2*MAE) if eval_weights provided
    """
    # Load raw trajectories
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

    # Check if using MambaSSM (sequence-wise) or simple SSM (step-wise)
    from src.IQL.neuralssm.ssm import MambaSSM

    is_mamba = isinstance(trainer.ssm, MambaSSM)

    # Process each trajectory sequentially
    for traj_idx, traj in enumerate(eval_trajectories):
        traj_states = np.array(traj["observations"])
        traj_actions = np.array(traj["actions"])
        traj_prefs = np.array(traj["preference_weights"])

        traj_predictions = []
        traj_cross_entropies = []

        if is_mamba:
            # Sequence-wise prediction for MambaSSM
            predicted_prefs = trainer.ssm.predict_sequence(
                traj_states
            )  # [T, n_objectives]

            for t in range(len(traj_actions)):
                state = traj_states[t]
                action = traj_actions[t]
                true_pref = traj_prefs[t]
                pred_pref = predicted_prefs[t]

                # Get Q-values for policy evaluation
                with torch.no_grad():
                    state_t = torch.FloatTensor(state).unsqueeze(0).to(trainer.device)
                    pred_pref_t = (
                        torch.FloatTensor(pred_pref).unsqueeze(0).to(trainer.device)
                    )
                    action_t = torch.LongTensor([action]).to(trainer.device)

                    # Get Q-values and compute policy logits using predicted preference
                    logits, q_values_all_batch = trainer.q_network.act(
                        state_t, pred_pref_t
                    )

                    # Compute softmax policy
                    policy_probs = torch.softmax(logits[0], dim=0)

                    # Compute cross-entropy loss
                    ce_loss = torch.nn.functional.cross_entropy(
                        logits, action_t, reduction="none"
                    )
                    traj_cross_entropies.append(ce_loss.item())

                # Compute MAE for first objective
                pref_error = np.abs(pred_pref[0] - true_pref[0])
                preference_errors.append(pref_error)

                # Store detailed prediction
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
        else:
            # Step-wise prediction for simple SSMs (ParticleFilter, EKF)
            trainer.ssm.reset()

            for t in range(len(traj_actions)):
                state = traj_states[t]
                action = traj_actions[t]
                true_pref = traj_prefs[t]

                # Get predicted preference from SSM (before update)
                if hasattr(trainer.ssm, "hidden"):
                    result = trainer.ssm.predict(state, trainer.ssm.hidden)
                    if isinstance(result, tuple):
                        pred_pref, _ = result
                    else:
                        pred_pref = result
                else:
                    pred_pref = trainer.ssm.predict(state, None)

                # Get Q-values for policy evaluation
                with torch.no_grad():
                    state_t = torch.FloatTensor(state).unsqueeze(0).to(trainer.device)
                    pred_pref_t = (
                        torch.FloatTensor(pred_pref).unsqueeze(0).to(trainer.device)
                    )
                    action_t = torch.LongTensor([action]).to(trainer.device)

                    # Get Q-values and compute policy logits using predicted preference
                    logits, q_values_all_batch = trainer.q_network.act(
                        state_t, pred_pref_t
                    )
                    q_values_all = q_values_all_batch[0].cpu().numpy()

                    # Compute softmax policy
                    policy_probs = torch.softmax(logits[0], dim=0)

                    # Compute cross-entropy loss
                    ce_loss = torch.nn.functional.cross_entropy(
                        logits, action_t, reduction="none"
                    )
                    traj_cross_entropies.append(ce_loss.item())

                # Update SSM with Q-values
                trainer.ssm.update(
                    observation=state, action=action, q_values_all=q_values_all
                )

                # Compute MAE for first objective
                pref_error = np.abs(pred_pref[0] - true_pref[0])
                preference_errors.append(pref_error)

                # Store detailed prediction
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

        # Compute mean cross-entropy for this trajectory
        mean_traj_ce = np.mean(traj_cross_entropies)
        cross_entropy_losses.append(mean_traj_ce)

        # Compute eval_score if eval_weights provided
        if eval_weights is not None:
            traj_pref_errors = preference_errors[-(len(traj_actions)) :]
            traj_pref_mae = np.mean(traj_pref_errors)

            scores = np.array([mean_traj_ce, traj_pref_mae])
            traj_eval_score = np.dot(eval_weights, scores)
            eval_scores.append(traj_eval_score)

        # Store trajectory-level predictions
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

    # Add eval_score metrics if eval_weights were provided
    if eval_weights is not None:
        results["mean_eval_score"] = np.mean(eval_scores)
        results["std_eval_score"] = np.std(eval_scores)

    # Save detailed predictions to JSON file if save_dir is provided
    if save_dir is not None:
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)

        # Create filename with update step if provided
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

        # Add eval_score if available
        if "mean_eval_score" in results:
            eval_data["mean_eval_score"] = results["mean_eval_score"]
            eval_data["std_eval_score"] = results["std_eval_score"]

        with open(json_path, "w") as f:
            json.dump(eval_data, f, indent=2)

        print(f"Saved detailed predictions to {json_path}")

    return results
