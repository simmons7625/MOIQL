import numpy as np
from typing import List, Tuple


def hypervolume(pareto_front: np.ndarray, reference_point: np.ndarray) -> float:
    """
    Calculate hypervolume indicator for multi-objective optimization.

    Args:
        pareto_front: Array of shape (n_points, n_objectives) containing Pareto front points
        reference_point: Reference point for hypervolume calculation

    Returns:
        Hypervolume value
    """
    if len(pareto_front) == 0:
        return 0.0

    if pareto_front.shape[1] == 2:
        return _hypervolume_2d(pareto_front, reference_point)
    else:
        return _hypervolume_nd(pareto_front, reference_point)


def _hypervolume_2d(pareto_front: np.ndarray, reference_point: np.ndarray) -> float:
    """Efficient 2D hypervolume calculation."""
    sorted_front = pareto_front[np.argsort(pareto_front[:, 0])]

    hv = 0.0
    for i, point in enumerate(sorted_front):
        if i == 0:
            width = point[0] - reference_point[0]
        else:
            width = point[0] - sorted_front[i - 1][0]
        height = point[1] - reference_point[1]
        hv += width * height

    return hv


def _hypervolume_nd(pareto_front: np.ndarray, reference_point: np.ndarray) -> float:
    """N-dimensional hypervolume using WFG algorithm (simplified)."""

    dominated_volume = 0.0
    for point in pareto_front:
        box_volume = np.prod(point - reference_point)
        dominated_volume = max(dominated_volume, box_volume)

    return dominated_volume


def sparsity(pareto_front: np.ndarray) -> float:
    """
    Calculate sparsity metric for Pareto front distribution.

    Args:
        pareto_front: Array of shape (n_points, n_objectives)

    Returns:
        Sparsity value (lower is better)
    """
    if len(pareto_front) < 2:
        return 0.0

    distances = []
    for i in range(len(pareto_front)):
        min_dist = float("inf")
        for j in range(len(pareto_front)):
            if i != j:
                dist = np.linalg.norm(pareto_front[i] - pareto_front[j])
                min_dist = min(min_dist, dist)
        distances.append(min_dist)

    return np.std(distances)


def inverted_generational_distance(
    pareto_front: np.ndarray, reference_front: np.ndarray
) -> float:
    """
    Calculate Inverted Generational Distance (IGD) between obtained and reference Pareto fronts.

    Args:
        pareto_front: Obtained Pareto front
        reference_front: True/reference Pareto front

    Returns:
        IGD value (lower is better)
    """
    if len(pareto_front) == 0:
        return float("inf")

    distances = []
    for ref_point in reference_front:
        min_dist = min(
            np.linalg.norm(ref_point - pf_point) for pf_point in pareto_front
        )
        distances.append(min_dist)

    return np.mean(distances)


def evaluate_policy(
    policy, env, n_episodes: int = 10, render: bool = False
) -> Tuple[np.ndarray, List[np.ndarray]]:
    """
    Evaluate a policy on an environment.

    Args:
        policy: Policy to evaluate
        env: Environment
        n_episodes: Number of evaluation episodes
        render: Whether to render the environment

    Returns:
        Mean rewards and list of episode reward vectors
    """
    episode_rewards = []

    for _ in range(n_episodes):
        obs, _ = env.reset()
        episode_reward = np.zeros(
            env.n_objectives if hasattr(env, "n_objectives") else 1
        )
        terminated = False
        truncated = False

        while not (terminated or truncated):
            action = policy.act(obs)
            obs, reward, terminated, truncated, _ = env.step(action)
            episode_reward += reward

            if render:
                env.render()

        episode_rewards.append(episode_reward)

    return np.mean(episode_rewards, axis=0), episode_rewards
