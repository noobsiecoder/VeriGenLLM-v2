"""
Code to log info

This module provides a centralized logging system for the VeriGen LLM project.
It creates both file and console logs with consistent formatting across all
components of the evaluation framework.

Author: Abhishek Sriram <noobsiecoder@gmail.com>
Date:   Aug 14th, 2025
Place:  Boston, MA

Note: LLM was used to generate comments
"""

import json
import os
import logging
import wandb
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional


class Logger:
    """
    Logger class

    A custom logger implementation that provides:
    - Automatic log file creation with daily rotation
    - Consistent formatting across the project
    - Both file and console output
    - Prevention of duplicate handlers

    This ensures all components (LLM clients, evaluation scripts, etc.)
    have consistent logging behavior.
    """

    def __init__(self, name: str, level=logging.INFO):
        """
        Initialize a logger instance with file and console handlers

        Parameters:
        -----------
        name : str
            Logger name, typically the module or component name
            (e.g., 'claude_client', 'pass-k_metrics', 'local_llm')
            This name appears in log messages and determines the log filename
        level : int, default=logging.INFO
            Minimum logging level to capture
            - DEBUG: Detailed information for diagnosing problems
            - INFO: General informational messages
            - WARNING: Warning messages
            - ERROR: Error messages
            - CRITICAL: Critical problems

        Notes:
        ------
        Log files are created in the 'logs' directory at the project root
        with the format: {name}_{YYYY-MM-DD}.log
        """
        # Create or get a logger with the specified name
        # Using getLogger ensures we reuse existing loggers with the same name
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)

        # Check if handlers already exist to prevent duplicates
        # This is important when the same logger is initialized multiple times
        if not self.logger.handlers:
            # Define log directory path relative to this file
            # __file__ is the current file (logger.py)
            # parents[2] goes up two levels to reach project root
            # Assumes structure: project_root/
            parent_dir = Path(__file__).resolve().parents[1]
            log_dir = parent_dir / "logs"

            # Create logs directory if it doesn't exist
            # parents=True creates intermediate directories if needed
            # exist_ok=True prevents error if directory already exists
            log_dir.mkdir(parents=True, exist_ok=True)

            # Define log file name with timestamp for daily rotation
            # This creates separate log files for each day, making it easier to:
            # - Track issues by date
            # - Manage log file sizes
            # - Archive old logs
            log_file = log_dir / f"{name}_{datetime.now().strftime('%Y-%m-%d')}.log"

            # File handler - writes log messages to file
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(level)

            # Console handler - displays log messages in terminal
            # Useful for real-time monitoring during development/debugging
            console_handler = logging.StreamHandler()
            console_handler.setLevel(level)

            # Define log message format
            # Format: 2024-01-20 10:30:45,123 - module_name - INFO - Log message here
            # Components:
            # - asctime: Timestamp with milliseconds
            # - name: Logger name (module/component)
            # - levelname: Log level (INFO, ERROR, etc.)
            # - message: The actual log message
            formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )

            # Apply formatter to both handlers
            file_handler.setFormatter(formatter)
            console_handler.setFormatter(formatter)

            # Add handlers to logger
            # Both handlers will process each log message
            self.logger.addHandler(file_handler)
            self.logger.addHandler(console_handler)

    def get_logger(self):
        """
        Return the configured logger instance

        Returns:
        --------
        logging.Logger
            The configured logger ready for use

        Example:
        --------
        >>> log = Logger("my_module").get_logger()
        >>> log.info("Starting process...")
        >>> log.error("An error occurred: %s", error_msg)
        """
        return self.logger


class WeightsAndBiases:
    """
    Logger for PPO training with W&B and local file logging support
    """

    def __init__(
        self,
        project_name: str,
        model_name: str,
        config: Dict,
        run_name: Optional[str] = None,
        use_wandb: bool = True,
        log_dir: str = "./logs",
    ):
        """
        Initialize logger

        Args:
            run_name: Name for this run (auto-generated if None)
            config: Configuration dictionary to log
            use_wandb: Whether to use W&B (set False for local logging only)
            log_dir: Directory for local logs
        """
        self.use_wandb = use_wandb
        self.log = Logger("wandb").get_logger()
        self.step = 0
        self.project_name: str = project_name
        self.config = config

        # Create run name if not provided
        if run_name is None:
            run_name = f"{model_name}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        self.run_name = run_name

        # Initialize W&B if requested
        if self.use_wandb:
            try:
                self.run = wandb.init(
                    project=self.project_name,
                    name=run_name,
                    config=self.config or {},
                    reinit=True,
                )
            except Exception as err:
                self.log.error(f"Failed to initialize W&B: {err}")
                self.log.info("Falling back to local logging only...")
                self.use_wandb = False

        # Store config locally
        if self.config:
            with open(os.path.join(log_dir, f"{run_name}_config.json"), "w") as f:
                json.dump(self.config, f, indent=2)

    def log_batch(
        self,
        losses: Dict[str, float],
        batch_idx: int,
        epoch: int,
        additional_metrics: Optional[Dict] = None,
    ):
        """Log metrics for a single batch"""
        metrics = {
            "train/policy_loss": losses.get("policy_loss", 0),
            "train/value_loss": losses.get("value_loss", 0),
            "train/total_loss": losses.get("total_loss", 0),
            "train/mean_reward": losses.get("mean_reward", 0),
            "train/approx_kl": losses.get("approx_kl", 0),
            "train/clip_fraction": losses.get("clip_fraction", 0),
            "train/advantage_mean": losses.get("advantage_mean", 0),
            "train/advantage_std": losses.get("advantage_std", 0),
            "train/epoch": epoch,
            "train/batch": batch_idx,
            "train/step": self.step,
        }

        # Add any additional metrics
        if additional_metrics:
            metrics.update(additional_metrics)

        # Log to W&B
        if self.use_wandb:
            wandb.log(metrics, step=self.step)

        # Print key metrics
        self.log.info(
            f"Step {self.step} | Epoch {epoch} | Batch {batch_idx} | "
            f"Policy Loss: {losses.get('policy_loss', 0):.4f} | "
            f"Mean Reward: {losses.get('mean_reward', 0):.4f} | "
            f"KL: {losses.get('approx_kl', 0):.4f}"
        )

        self.step += 1

    def log_epoch(self, epoch_metrics: Dict[str, float], epoch: int):
        """Log epoch-level summaries"""
        metrics = {f"epoch/{k}": v for k, v in epoch_metrics.items()}
        metrics["epoch/number"] = epoch

        if self.use_wandb:
            wandb.log(metrics, step=self.step)

        self.log.info(f"METRIC: {metrics}")

        self.log.info(f"\nEpoch {epoch} Summary:")
        for k, v in epoch_metrics.items():
            self.log.info(f"  {k}: {v:.4f}")
        self.log.info("-" * 50)

    def log_examples(
        self,
        prompts: List[str],
        generations: List[str],
        rewards: List[float],
        epoch: int,
        num_examples: int = 3,
    ):
        """Log example generations"""
        examples = list(
            zip(
                prompts[:num_examples],
                generations[:num_examples],
                rewards[:num_examples],
            )
        )

        if self.use_wandb:
            table = wandb.Table(
                columns=["prompt", "generation", "reward"], data=examples
            )
            wandb.log({"examples": table, "examples/epoch": epoch}, step=self.step)

        # Log examples locally
        examples_data = {
            "step": self.step,
            "epoch": epoch,
            "examples": [
                {"prompt": p, "generation": g, "reward": float(r)}
                for p, g, r in examples
            ],
        }
        self._log_local(examples_data)

        # Print one example
        if examples:
            self.log.info(f"\nExample generation:")
            self.log.info(f"Prompt: {examples[0][0][:50]}...")
            self.log.info(f"Generation: {examples[0][1][:100]}...")
            self.log.info(f"Reward: {examples[0][2]:.3f}\n")

    def log_model_checkpoint(self, epoch: int, metrics: Dict[str, float]):
        """Log when model checkpoint is saved"""
        checkpoint_info = {
            "checkpoint/epoch": epoch,
            "checkpoint/step": self.step,
            **{f"checkpoint/{k}": v for k, v in metrics.items()},
        }

        if self.use_wandb:
            wandb.log(checkpoint_info, step=self.step)

        self.log.info(f"CHECKPOINT INFO: {checkpoint_info}")
        self.log.info(f"Checkpoint saved at epoch {epoch}, step {self.step}")
