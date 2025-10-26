"""
Highway environment module with PPO trainer.
"""

from src.highway.env import HighwayWrapper
from src.highway.reward_function import HighwayPreferenceFunction, RewardFunction
from src.highway.model import ActorCritic
from src.highway.trainer import PPOTrainer

__all__ = [
    "HighwayWrapper",
    "HighwayPreferenceFunction",
    "RewardFunction",
    "ActorCritic",
    "PPOTrainer",
]
