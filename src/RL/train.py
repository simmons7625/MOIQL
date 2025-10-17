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


def save_results(save_dir: str, config: dict, final_metrics: dict, trainer: PPOTrainer):
    """
    Save training results including config, metrics, and final model.

    Args:
        save_dir: Directory to save results
        config: Training configuration
        final_metrics: Final training metrics
        trainer: Trained PPOTrainer instance
    """
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    # Create results subdirectory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = save_path / f"results_{timestamp}"
    results_dir.mkdir(parents=True, exist_ok=True)

    # Save configuration
    config_path = results_dir / "config.yaml"
    with open(config_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False)
    print(f"Saved config to {config_path}")

    # Save final metrics
    metrics_path = results_dir / "final_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(final_metrics, f, indent=2)
    print(f"Saved metrics to {metrics_path}")

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

    # Create summary file
    summary_path = results_dir / "summary.txt"
    with open(summary_path, "w") as f:
        f.write("=" * 70 + "\n")
        f.write("PPO TRAINING SUMMARY\n")
        f.write("=" * 70 + "\n\n")

        f.write(f"Timestamp: {timestamp}\n")
        f.write(f"Environment: {config['env_name']}\n")
        f.write(f"Continuous decay: {config['contenous_decay']}\n")
        f.write(f"Switch decay: {config['switch_decay']}\n")
        f.write(f"Init treasure weight: {config['init_treasure_weight']}\n")
        f.write(f"Switch time: {config.get('switch_time', 'None')}\n\n")

        f.write("Final Metrics:\n")
        for key, value in final_metrics.items():
            f.write(f"  {key}: {value}\n")

        f.write("\n" + "=" * 70 + "\n")

    print(f"Saved summary to {summary_path}")
    print(f"\nAll results saved to: {results_dir}")

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
    trainer = PPOTrainer(
        env_name=config["env_name"],
        contenous_decay=config["contenous_decay"],
        switch_decay=config["switch_decay"],
        init_treasure_weight=config["init_treasure_weight"],
        switch_time=config.get("switch_time", None),
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

    final_metrics = trainer.train(
        n_updates=config["n_updates"],
    )

    print("\n" + "=" * 70)
    print("TRAINING COMPLETED")
    print("=" * 70)

    # Save results
    print("\nSaving results...")
    results_dir = save_results(
        save_dir=config["save_dir"],
        config=config,
        final_metrics=final_metrics,
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
