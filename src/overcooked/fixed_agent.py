"""
Fixed-preference agent for Overcooked (Agent 1 - human-like agent).

This agent has a fixed multi-objective preference and acts according to it.
"""

import numpy as np
import torch
from typing import Optional
from .model import ActorCritic


class FixedPreferenceAgent:
    """
    Agent with fixed preference weights for multi-objective rewards.

    This agent represents a "human" player with consistent preferences.
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        preference_weights: np.ndarray,
        model_path: Optional[str] = None,
        hidden_dim: int = 256,
        device: str = "cuda",
    ):
        """
        Initialize fixed-preference agent.

        Args:
            obs_dim: Observation space dimension
            action_dim: Action space dimension
            preference_weights: Fixed preference weights [task, efficiency, coordination]
            model_path: Path to pre-trained model (if None, random policy)
            hidden_dim: Hidden layer dimension
            device: Device to run on
        """
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.preference_weights = np.array(preference_weights, dtype=np.float32)
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.hidden_dim = hidden_dim

        # Create actor-critic network
        self.model = ActorCritic(obs_dim, action_dim, hidden_dim).to(self.device)

        # Load pre-trained model if provided
        if model_path is not None:
            self.load(model_path)

        self.model.eval()  # Set to evaluation mode

    def get_action(
        self, obs: np.ndarray, deterministic: bool = False
    ) -> tuple[int, float]:
        """
        Get action from observation.

        Args:
            obs: Observation array
            deterministic: If True, take argmax action; otherwise sample

        Returns:
            action: Action index
            log_prob: Log probability of action
        """
        with torch.no_grad():
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
            logits, value = self.model.act(obs_tensor)

            if deterministic:
                action = torch.argmax(logits, dim=1).item()
                log_prob = 0.0
            else:
                dist = torch.distributions.Categorical(logits=logits)
                action = dist.sample().item()
                log_prob = dist.log_prob(torch.tensor([action]).to(self.device)).item()

        return action, log_prob

    def get_preference(self) -> np.ndarray:
        """Get agent's preference weights."""
        return self.preference_weights.copy()

    def load(self, model_path: str):
        """Load model from checkpoint."""
        checkpoint = torch.load(model_path, map_location=self.device)
        if "ac_state_dict" in checkpoint:
            self.model.load_state_dict(checkpoint["ac_state_dict"])
        else:
            self.model.load_state_dict(checkpoint)
        print(f"Loaded fixed-preference agent model from {model_path}")

    def save(self, model_path: str):
        """Save model to checkpoint."""
        torch.save(
            {
                "ac_state_dict": self.model.state_dict(),
                "preference_weights": self.preference_weights,
                "obs_dim": self.obs_dim,
                "action_dim": self.action_dim,
                "hidden_dim": self.hidden_dim,
            },
            model_path,
        )
        print(f"Saved fixed-preference agent model to {model_path}")


class RandomAgent:
    """Random agent for testing purposes."""

    def __init__(
        self, action_dim: int, preference_weights: Optional[np.ndarray] = None
    ):
        """
        Initialize random agent.

        Args:
            action_dim: Number of actions
            preference_weights: Optional preference weights (for compatibility)
        """
        self.action_dim = action_dim
        self.preference_weights = (
            preference_weights
            if preference_weights is not None
            else np.array([1 / 3, 1 / 3, 1 / 3], dtype=np.float32)
        )

    def get_action(
        self, obs: np.ndarray, deterministic: bool = False
    ) -> tuple[int, float]:
        """
        Get random action.

        Args:
            obs: Observation (ignored)
            deterministic: If True, always return action 0

        Returns:
            action: Random action index
            log_prob: Log probability (uniform)
        """
        if deterministic:
            action = 0
        else:
            action = np.random.randint(0, self.action_dim)

        log_prob = -np.log(self.action_dim)  # Uniform distribution
        return action, log_prob

    def get_preference(self) -> np.ndarray:
        """Get agent's preference weights."""
        return self.preference_weights.copy()


def create_agent(
    agent_type: str,
    obs_dim: int,
    action_dim: int,
    preference_weights: np.ndarray,
    model_path: Optional[str] = None,
    hidden_dim: int = 256,
    device: str = "cuda",
):
    """
    Factory function to create agents.

    Args:
        agent_type: Type of agent ("fixed", "random")
        obs_dim: Observation dimension
        action_dim: Action dimension
        preference_weights: Preference weights
        model_path: Path to model checkpoint
        hidden_dim: Hidden dimension
        device: Device to run on

    Returns:
        Agent instance
    """
    if agent_type == "fixed":
        return FixedPreferenceAgent(
            obs_dim, action_dim, preference_weights, model_path, hidden_dim, device
        )
    elif agent_type == "random":
        return RandomAgent(action_dim, preference_weights)
    else:
        raise ValueError(f"Unknown agent type: {agent_type}")
