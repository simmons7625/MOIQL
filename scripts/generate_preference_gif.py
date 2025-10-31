"""
Generate animated GIF showing preference weights evolution over time.
"""

import argparse
import json

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def create_preference_gif(
    trajectories_path: str,
    output_path: str,
    fps: int = 5,
    trajectory_idx: int = 0,
    figsize: tuple = (6.65, 6.4),
):
    """
    Create animated GIF showing preference weights over time.

    Args:
        trajectories_path: Path to trajectories.json file
        output_path: Path to save the output GIF
        fps: Frames per second
        trajectory_idx: Which trajectory to visualize (default: first one)
        figsize: Figure size in inches (width, height) for matching resolution
    """
    # Load trajectories
    with open(trajectories_path, "r") as f:
        trajectories = json.load(f)

    if trajectory_idx >= len(trajectories):
        raise ValueError(
            f"Trajectory index {trajectory_idx} out of range (max: {len(trajectories) - 1})"
        )

    trajectory = trajectories[trajectory_idx]
    preference_weights = np.array(trajectory["preference_weights"])

    # Extract weights for each objective
    n_timesteps = len(preference_weights)
    treasure_weights = preference_weights[:, 0]
    time_weights = preference_weights[:, 1]

    # Create frames
    frames = []

    for t in range(n_timesteps):
        fig, ax = plt.subplots(figsize=figsize, dpi=80)

        # Plot the full trajectory as lines
        timesteps = np.arange(n_timesteps)
        ax.plot(
            timesteps,
            treasure_weights,
            "b-",
            linewidth=2,
            label="Treasure Weight",
            alpha=0.7,
        )
        ax.plot(
            timesteps, time_weights, "g-", linewidth=2, label="Time Weight", alpha=0.7
        )

        # Plot the current position as a red point
        ax.plot(t, treasure_weights[t], "ro", markersize=12, label="Current (Treasure)")
        ax.plot(t, time_weights[t], "ro", markersize=12, label="Current (Time)")

        # Formatting
        ax.set_xlabel("Timestep", fontsize=12)
        ax.set_ylabel("Preference Weight", fontsize=12)
        ax.set_title(
            f"Preference Weights Evolution (t={t}/{n_timesteps-1})", fontsize=14
        )
        ax.set_xlim(-0.5, n_timesteps - 0.5)
        ax.set_ylim(-0.05, 1.05)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best", fontsize=10)

        # Convert plot to image
        fig.canvas.draw()
        image = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
        image = image.reshape(fig.canvas.get_width_height()[::-1] + (4,))
        # Convert RGBA to RGB
        image_rgb = image[:, :, :3]
        frames.append(Image.fromarray(image_rgb))

        plt.close(fig)

    # Save as GIF
    duration = int(1000 / fps)  # Duration in milliseconds
    frames[0].save(
        output_path,
        save_all=True,
        append_images=frames[1:],
        duration=duration,
        loop=0,
    )

    print(f"Saved preference weights GIF to {output_path}")
    print(f"  Timesteps: {n_timesteps}")
    print(f"  Frames: {len(frames)}")
    print(f"  FPS: {fps}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate preference weights visualization GIF"
    )
    parser.add_argument(
        "--trajectories",
        type=str,
        default="Streamlit/dataset/dst/trajectories.json",
        help="Path to trajectories.json file",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="Streamlit/dataset/dst/preference_weights.gif",
        help="Output GIF path",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=5,
        help="Frames per second",
    )
    parser.add_argument(
        "--trajectory-idx",
        type=int,
        default=0,
        help="Which trajectory to visualize",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=532,
        help="Output width in pixels",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=512,
        help="Output height in pixels",
    )

    args = parser.parse_args()

    # Calculate figsize from pixel dimensions (assuming 80 DPI)
    dpi = 80
    figsize = (args.width / dpi, args.height / dpi)

    create_preference_gif(
        trajectories_path=args.trajectories,
        output_path=args.output,
        fps=args.fps,
        trajectory_idx=args.trajectory_idx,
        figsize=figsize,
    )


if __name__ == "__main__":
    main()
