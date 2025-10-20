import argparse
import json
import warnings
from datetime import datetime
from pathlib import Path

import wandb
import yaml

from src.RL.trainer import PPOTrainer

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning, module="pygame.pkgdata")
warnings.filterwarnings("ignore", category=UserWarning, module="gymnasium.spaces.box")
warnings.filterwarnings("ignore", category=RuntimeWarning, message="divide by zero")
warnings.filterwarnings(
    "ignore", category=UserWarning, message="std.*degrees of freedom"
)
warnings.filterwarnings("ignore", category=UserWarning, message="Using a target size")


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def save_results(results_dir: str, config: dict, trainer: PPOTrainer):
    """
    Save training results including config and final model.
    Note: Metrics CSV is already saved during training.

    Args:
        results_dir: Directory to save results (already created with timestamp)
        config: Training configuration
        trainer: Trained PPOTrainer instance
    """
    results_dir = Path(results_dir)

    # Save configuration
    config_path = results_dir / "config.yaml"
    with open(config_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False)
    print(f"Saved config to {config_path}")

    # Save final model
    model_path = results_dir / "final_model.pt"
    trainer.save(str(model_path))
    print(f"Saved model to {model_path}")

    # Save model architecture info
    model_info = {
        "total_parameters": sum(p.numel() for p in trainer.ac.parameters()),
        "trainable_parameters": sum(
            p.numel() for p in trainer.ac.parameters() if p.requires_grad
        ),
        "device": str(trainer.device),
    }

    model_info_path = results_dir / "model_info.json"
    with open(model_info_path, "w") as f:
        json.dump(model_info, f, indent=2)
    print(f"Saved model info to {model_info_path}")

    # plot preference weights
    trainer.plot_preference_weights(save_path=results_dir / "preference_weights.png")
    print(f"Saved preference weights plot to {results_dir / 'preference_weights.png'}")

    return results_dir


def main():
    """Main training function for PPO."""
    parser = argparse.ArgumentParser(
        description="Train PPO agent on multi-objective environment"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/rl.yaml",
        help="Path to configuration file",
    )

    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)
    print(f"Loaded config from {args.config}")

    # Initialize wandb if enabled
    if config.get("use_wandb", False):
        wandb.init(
            project=config.get("wandb_project", "moiql-ppo"),
            name=config.get("wandb_run_name", None),
            config=config,
        )

    # Create trainer
    print("\nInitializing PPO trainer...")

    # Handle init_weight - can be a list or scalar
    init_weight = config["init_weight"]
    if isinstance(init_weight, list):
        init_weight = init_weight[0]  # Use first element for initial weight

    trainer = PPOTrainer(
        env_name=config["env_name"],
        contenous_decay=config["contenous_decay"],
        init_treasure_weight=init_weight,
        safety_distance_threshold=config.get("safety_distance_threshold", 10.0),
        safety_boost_factor=config.get("safety_boost_factor", 1.5),
        hidden_dim=config["hidden_dim"],
        lr=config["lr"],
        gamma=config["gamma"],
        gae_lambda=config["gae_lambda"],
        clip_epsilon=config["clip_epsilon"],
        vf_coef=config["vf_coef"],
        ent_coef=config["ent_coef"],
        max_grad_norm=config["max_grad_norm"],
        batch_size=config["batch_size"],
        n_epochs=config["n_epochs"],
        device=config["device"],
        use_wandb=config.get("use_wandb", False),
    )

    print(f"Device: {trainer.device}")
    print(f"Model parameters: {sum(p.numel() for p in trainer.ac.parameters())}")

    # Train
    print("\n" + "=" * 70)
    print("STARTING TRAINING")
    print("=" * 70)

    # Create results directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = Path(config["save_dir"]) / timestamp
    results_dir.mkdir(parents=True, exist_ok=True)

    trainer.train(
        n_updates=config["n_updates"],
        save_dir=str(results_dir),
    )

    print("\n" + "=" * 70)
    print("TRAINING COMPLETED")
    print("=" * 70)

    # Save results
    print("\nSaving results...")
    save_results(
        results_dir=str(results_dir),
        config=config,
        trainer=trainer,
    )

    # Log results directory to wandb
    if config.get("use_wandb", False):
        wandb.log({"results_dir": str(results_dir)})
        wandb.finish()
        print("WandB logging finished")

    print("\n" + "=" * 70)
    print(f"All results saved to: {results_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()
