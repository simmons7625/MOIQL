import torch
import numpy as np


class IQLDataset(torch.utils.data.Dataset):
    """Dataset for IQL training with predicted preferences."""

    def __init__(self, trajectories, predicted_preferences):
        """
        Args:
            trajectories: List of trajectory dicts with keys:
                - observations: [T, obs_dim]
                - actions: [T]
                - preference_weights: [T, n_objectives] (ground truth)
            predicted_preferences: List of arrays [T, n_objectives] - SSM predictions
        """
        self.transitions = []

        # Extract all transitions from trajectories
        for traj_idx, traj in enumerate(trajectories):
            states = np.array(traj["observations"])
            actions = np.array(traj["actions"])
            pred_prefs = predicted_preferences[traj_idx]

            # Get initial state and preference for this trajectory
            initial_state = states[0]
            initial_pref = pred_prefs[0]

            # Create transitions (s, a, s', w) for timesteps t=0..T-2
            for t in range(len(states) - 1):
                transition = {
                    "state": states[t],
                    "action": actions[t],
                    "next_state": states[t + 1],
                    "preference": pred_prefs[t],
                    "initial_state": initial_state,
                    "initial_preference": initial_pref,
                }
                self.transitions.append(transition)

    def __len__(self):
        return len(self.transitions)

    def __getitem__(self, idx):
        return self.transitions[idx]


def collate_fn(batch):
    """Collate batch of transitions into tensors."""
    return {
        "states": torch.FloatTensor(np.array([t["state"] for t in batch])),
        "actions": torch.LongTensor(np.array([t["action"] for t in batch])),
        "next_states": torch.FloatTensor(np.array([t["next_state"] for t in batch])),
        "preference_weights": torch.FloatTensor(
            np.array([t["preference"] for t in batch])
        ),
        "initial_states": torch.FloatTensor(
            np.array([t["initial_state"] for t in batch])
        ),
        "initial_preferences": torch.FloatTensor(
            np.array([t["initial_preference"] for t in batch])
        ),
    }


def create_iql_dataloader(
    trajectories, predicted_preferences, batch_size, shuffle=True
):
    """Create dataloader for IQL training.

    Args:
        trajectories: List of trajectory dictionaries
        predicted_preferences: List of predicted preference arrays [T, n_objectives]
        batch_size: Batch size
        shuffle: Whether to shuffle data

    Returns:
        DataLoader that yields batches with keys:
            - states: [batch, obs_dim]
            - actions: [batch]
            - next_states: [batch, obs_dim]
            - preference_weights: [batch, n_objectives]
            - initial_states: [batch, obs_dim]
            - initial_preferences: [batch, n_objectives]
    """
    dataset = IQLDataset(trajectories, predicted_preferences)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn,
    )
    return dataloader
