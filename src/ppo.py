"""
This file contains the code to perform PPO based RLFT
It is independent of reward function calculation and policy's response generation
Only responsible in updated policy

Author: Abhishek Sriram <noobsiecoder@gmail.com>
Date:   Aug 28th, 2025
Place:  Boston, MA
"""

from typing import List
import torch
from torch import nn
from constants import BaseRL, PPO_CONFIG, RLFT_TRAIN_CONFIG
from src.models import Policy
from src.logger import Logger


class ValueHead(nn.Module):
    """
    This Value-head acts as a critic to calculate the advantage function in PPO
    """

    def __init__(self, hidden_size: int = 4):
        """
        Initialize constants
        """
        super().__init__()
        self.value_head = nn.Linear(hidden_size, 1)

    def forward(self, hidden_states):
        """
        Move forward in MLP
        """
        last_hidden = hidden_states[-1].mean(dim=1)
        return self.value_head(last_hidden)


class PPO(BaseRL):
    """
    Base class/abstract class for all RL algorithms
    """

    def __init__(
        self, policy: Policy, ref_policy: Policy, optimizer, hidden_dim: int = 1024
    ):
        """
        Store constants to run PPO
        """
        self.ref_policy = ref_policy  # frozen policy for reference
        self.policy = policy  # policy for update
        self.device = self.policy.device
        self.tokenizer = policy.tokenizer
        self.optimizer = optimizer  # Optimizer for the RL
        self.clip_epsilon = PPO_CONFIG.get("clip_epsilon", 0.2)
        self.value_coefficient = PPO_CONFIG.get("value_coefficient", 0.5)
        self.entropy_coefficient = PPO_CONFIG.get("entropy_coefficient", 0.01)
        self.max_grad_norm = PPO_CONFIG.get("max_grad_norm", 0.5)
        self.precision = RLFT_TRAIN_CONFIG.get("precision", torch.float16)
        self.log = Logger("PPO").get_logger()
        self.total_loss = None
        self.value_head = None
        self._value_head_init(hidden_dim)

    def compute_loss(self, batch: List, rewards: List):
        """
        PPO Loss function:
            1. Compute Advantage function
            2. Log probabilities/tokens
            3. Apply clipping function
        """
        # Get hidden states per batch
        hidden_states = torch.stack(batch["hidden_states"]).to(self.device)
        # # Get rewards per batch
        rewards = torch.tensor(rewards, dtype=self.precision, device=self.device)
        # # Calculate Advantage function
        advantage = self._advantage_function(rewards, hidden_states)
        # Normalize advantages
        norm_advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)
        # Log prob of the all tokens
        ref_log_probs = torch.nn.functional.log_softmax(
            batch["old_prob_seq_logits"], dim=-1
        )
        new_log_probs = torch.nn.functional.log_softmax(
            batch["new_prob_seq_logits"], dim=-1
        )
        # Log prob of the actual token at each position
        ref_selected_log_probs = torch.gather(
            ref_log_probs, dim=2, index=batch["sequences"].unsqueeze(-1)
        ).squeeze(-1)  # Result: [4, 512]
        new_selected_log_probs = torch.gather(
            new_log_probs, dim=2, index=batch["sequences"].unsqueeze(-1)
        ).squeeze(-1)  # Result: [4, 512]
        # Create action mask
        # In compute_loss, before the action_mask loop:
        self.log.info(f"batch['sequences'] shape: {batch['sequences'].shape}")
        self.log.info(f"batch['prompts_token_length'] shape: {batch['prompts_token_length'].shape}")
        self.log.info(f"batch['prompts_token_length']: {batch['prompts_token_length']}")
        action_mask = torch.zeros_like(batch["sequences"], dtype=self.precision)
        for i, prompt_len in enumerate(batch["prompts_token_length"]):
            self.log.info(f"i: {i}, prompt_len: {prompt_len}")
            action_mask[i, prompt_len:] = 1.0
        # Get attention mask (you calculated earlier)
        attention_mask = (
            batch["sequences"] != self.policy.tokenizer.pad_token_id
        ).float()
        # Final mask: only generated tokens that are not padding
        final_mask = action_mask * attention_mask
        # Sum only for generated tokens
        ref_log_probs_sum = (ref_selected_log_probs * final_mask).sum(dim=1)
        new_log_probs_sum = (new_selected_log_probs * final_mask).sum(dim=1)
        # Log probability ratio
        log_ratio = new_log_probs_sum - ref_log_probs_sum
        ratio = torch.exp(torch.clamp(log_ratio, -10, 10))
        # KL divergence approximation
        # approx_kl = (log_ratio).mean()
        # Clip fraction (how often clipping is active)
        # clip_fraction = ((ratio - 1).abs() > self.clip_epsilon).float().mean()
        surr1 = ratio * norm_advantage
        surr2 = (
            torch.clamp(ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon)
            * norm_advantage
        )
        policy_loss = -torch.min(surr1, surr2).mean()
        # Get value predictions (you already have the value head)
        values = self.value_head(hidden_states).squeeze(-1)
        value_targets = rewards.detach()
        # MSE loss for value function
        value_loss = torch.nn.functional.mse_loss(values, value_targets)
        # Combine policy and value losses
        total_loss = policy_loss + self.value_coefficient * value_loss
        # Accounting entropy loss
        entropy = -(new_log_probs * torch.exp(new_log_probs)).sum(dim=-1).mean()
        self.total_loss = total_loss + self.entropy_coefficient * entropy

        # Return metrics for logging
        return {
            "policy_loss": policy_loss.item(),
            "value_loss": value_loss.item(),
            "total_loss": total_loss.item(),
            # "approx_kl": approx_kl,
            # "clip_fraction": clip_fraction,
            "mean_reward": rewards.mean().item(),
        }

    def update(self):
        """
        Backpropagate in a model, re-compute weights and update the model
        """
        # Clear gradients from previous step
        self.optimizer.zero_grad()
        # Backpropagate
        self.total_loss.backward()
        # Optional but recommended: Gradient clipping
        torch.nn.utils.clip_grad_norm_(
            self.policy.model.parameters(), self.max_grad_norm
        )
        # Update model parameters
        self.optimizer.step()

    def _value_head_init(self, hidden_dim):
        """
        Initialize ValueHead to estimate advantage
        """
        self.value_head = ValueHead(hidden_size=hidden_dim).to(
            self.device, self.precision
        )

    def _advantage_function(self, rewards, hidden_states):
        """
        Calulcate Advantage Function using Actor-Critic Method (Value Head)
        """
        values = self.value_head(hidden_states).squeeze(-1)
        return rewards - values
