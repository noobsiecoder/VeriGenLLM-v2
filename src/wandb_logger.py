"""
Weights & Biases Client Python API

Author: Abhishek Sriram <noobsiecoder@gmail.com>
Date:   Aug 26th, 2025
Place:  Boston, MA
"""

import wandb
from typing import Dict, Optional, List
import os


class WandbLogger:
    """Handles Weights & Biases logging for RLFT"""
    
    def __init__(
        self,
        project_name: str = "verilog-rlft",
        run_name: Optional[str] = None,
        config: Optional[Dict] = None,
        tags: Optional[List[str]] = None,
    ):
        self.project_name = project_name
        self.run_name = run_name
        self.config = config or {}
        self.tags = tags or []
        self.run = None
        
    def init(self):
        """Initialize W&B run"""
        self.run = wandb.init(
            project=self.project_name,
            name=self.run_name,
            config=self.config,
            tags=self.tags,
            reinit=True,
        )
        return self.run
    
    def log(self, metrics: Dict, step: Optional[int] = None):
        """Log metrics to W&B"""
        if self.run is None:
            raise RuntimeError("WandbLogger not initialized. Call init() first.")
        wandb.log(metrics, step=step)
    
    def log_batch_metrics(
        self,
        epoch: int,
        batch_idx: int,
        avg_reward: float,
        policy_loss: float,
        value_loss: float,
        rewards_dist: List[float],
    ):
        """Log batch-level training metrics"""
        metrics = {
            "train/avg_reward": avg_reward,
            "train/policy_loss": policy_loss,
            "train/value_loss": value_loss,
            "train/min_reward": min(rewards_dist),
            "train/max_reward": max(rewards_dist),
            "train/epoch": epoch,
            "train/batch": batch_idx,
        }
        
        # Create histogram of rewards
        wandb.log({
            **metrics,
            "train/reward_distribution": wandb.Histogram(rewards_dist)
        })
    
    def log_epoch_metrics(
        self,
        epoch: int,
        avg_epoch_reward: float,
        compilation_rate: float,
        functional_rate: float,
    ):
        """Log epoch-level metrics"""
        metrics = {
            "epoch/avg_reward": avg_epoch_reward,
            "epoch/compilation_rate": compilation_rate,
            "epoch/functional_rate": functional_rate,
            "epoch/number": epoch,
        }
        wandb.log(metrics)
    
    def save_model(self, model_path: str, aliases: Optional[List[str]] = None):
        """Save model artifact to W&B"""
        artifact = wandb.Artifact(
            name=f"model-{wandb.run.id}",
            type="model",
            description="RLFT fine-tuned Verilog generation model",
        )
        artifact.add_dir(model_path)
        wandb.log_artifact(artifact, aliases=aliases or ["latest"])
    
    def finish(self):
        """Finish W&B run"""
        if self.run:
            self.run.finish()
