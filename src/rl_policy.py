"""
All the RL policies resides here (minus the reward function as it is equivalent b/w all)

Author: Abhishek Sriram <noobsiecoder@gmail.com>
Date:   Aug 26th, 2025
Place:  Boston, MA
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModel


class ValueHead(torch.nn.Module):
    """Value head for critic network in PPO"""

    def __init__(self, hidden_size: int, dropout: float = 0.1):
        super().__init__()
        self.dropout = torch.nn.Dropout(dropout)
        self.summary = torch.nn.Linear(hidden_size, hidden_size)
        self.activation = torch.nn.Tanh()
        self.value = torch.nn.Linear(hidden_size, 1)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Parameters:
        -----------
        hidden_states: torch.Tensor
            Shape: [batch_size, seq_len, hidden_size]

        Returns:
        --------
        values: torch.Tensor
            Shape: [batch_size]
        """
        # Use the last token's hidden state
        last_hidden = hidden_states[:, -1, :]

        # Two-layer MLP with dropout
        dropped = self.dropout(last_hidden)
        summary = self.activation(self.summary(dropped))
        values = self.value(summary).squeeze(-1)

        return values


class ActorCriticModel(torch.nn.Module):
    """Wrapper model that adds value head to language model"""

    def __init__(self, base_model, tokenizer):
        super().__init__()
        self.base_model = base_model
        self.tokenizer = tokenizer

        # Get hidden size from config
        if hasattr(base_model.config, "hidden_size"):
            hidden_size = base_model.config.hidden_size
        elif hasattr(base_model.config, "n_embd"):
            hidden_size = base_model.config.n_embd
        else:
            raise ValueError("Cannot determine hidden size from model config")

        self.value_head = ValueHead(hidden_size)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor = None,
        return_dict: bool = True,
        output_hidden_states: bool = True,
    ):
        """Forward pass through actor and critic"""
        # Get base model outputs
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=return_dict,
            output_hidden_states=True,  # Always need hidden states for value head
        )

        # Extract hidden states from last layer
        hidden_states = outputs.hidden_states[-1]

        # Get value estimates
        values = self.value_head(hidden_states)

        # Add values to outputs
        if return_dict:
            outputs.values = values
            return outputs
        else:
            return outputs.logits, values

    def generate(self, *args, **kwargs):
        """Pass through to base model's generate method"""
        return self.base_model.generate(*args, **kwargs)


class BaseRLPolicy(ABC):
    """Base class for RL policies"""

    @abstractmethod
    def update(self, states, actions, rewards):
        """Update policy parameters"""
        pass

    @abstractmethod
    def compute_loss(self, *args, **kwargs):
        """Compute policy loss"""
        pass


class PPO(BaseRLPolicy):
    """
    This class implements Proximal Policy Optimisation
    Refer paper for more info: https://arxiv.org/pdf/1707.06347
    For web version: https://openai.com/index/openai-baselines-ppo/
    """

    def __init__(
        self,
        clip_epsilon: float = 0.2,
        value_coef: float = 0.5,
        entropy_coef: float = 0.01,
        max_grad_norm: float = 0.5,
        ppo_epochs: int = 4,
        mini_batch_size: int = 4,
        gamma: float = 0.99,
        lam: float = 0.95,
    ):
        self.clip_epsilon = clip_epsilon
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.ppo_epochs = ppo_epochs
        self.mini_batch_size = mini_batch_size
        self.gamma = gamma
        self.lam = lam

    def compute_advantages(
        self,
        rewards: torch.Tensor,
        values: torch.Tensor,
        dones: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute Generalized Advantage Estimation (GAE) and returns
        """
        advantages = torch.zeros_like(rewards)
        returns = torch.zeros_like(rewards)
        
        # Last advantage and return
        lastgaelam = 0
        last_value = values[-1]
        
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = 0  # No next value at the end
                next_not_done = 0
            else:
                next_value = values[t + 1]
                next_not_done = 1 - dones[t + 1]
            
            # TD error
            delta = rewards[t] + self.gamma * next_value * next_not_done - values[t]
            
            # GAE
            advantages[t] = lastgaelam = delta + self.gamma * self.lam * next_not_done * lastgaelam
            
            # Returns for value loss
            returns[t] = advantages[t] + values[t]

        return advantages, returns

    def compute_loss(
        self,
        logprobs: torch.Tensor,
        old_logprobs: torch.Tensor,
        advantages: torch.Tensor,
        values: torch.Tensor,
        old_values: torch.Tensor,
        returns: torch.Tensor,
        entropy: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute PPO loss components
        """
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Policy loss with clipping
        ratio = torch.exp(logprobs - old_logprobs)
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()

        # Value loss with clipping
        values_clipped = old_values + torch.clamp(
            values - old_values, -self.clip_epsilon, self.clip_epsilon
        )
        value_loss_unclipped = F.mse_loss(values, returns)
        value_loss_clipped = F.mse_loss(values_clipped, returns)
        value_loss = torch.max(value_loss_unclipped, value_loss_clipped)

        # Entropy bonus
        entropy_loss = -entropy.mean()

        # Total loss
        loss = (
            policy_loss 
            + self.value_coef * value_loss 
            + self.entropy_coef * entropy_loss
        )

        return {
            "loss": loss,
            "policy_loss": policy_loss,
            "value_loss": value_loss,
            "entropy_loss": entropy_loss,
            "ratio": ratio.mean(),
            "advantages": advantages.mean(),
        }

    def update(
        self,
        model_engine,
        optimizer,
        prompts: List[str],
        responses: List[Dict],
        rewards: List[Dict],
        accumulation_steps: int = 4,    
    ) -> Dict[str, float]:
        """
        Update policy using PPO algorithm

        Parameters:
        -----------
        model_engine: DeepSpeed model engine
        optimizer: DeepSpeed optimizer
        prompts: List of input prompts
        responses: List of generated responses
        rewards: List of reward dictionaries
        """
        # Check if model has value head
        model = model_engine.module
        has_value_head = hasattr(model, 'value_head')
        if not has_value_head:
            raise ValueError(
                "Model must have value head for PPO. Use ActorCriticModel wrapper."
            )
        
        tokenizer = model_engine.tokenizer

        # Prepare batch data
        batch_data = self._prepare_batch_data(tokenizer, prompts, responses, rewards)
        
        # Move to device
        device = model_engine.device
        input_ids = batch_data['input_ids'].to(device)
        attention_masks = batch_data['attention_masks'].to(device)
        action_masks = batch_data['action_masks'].to(device)
        rewards_tensor = batch_data['rewards'].to(device)
        
        # Get old values and log probs
        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_masks,
                return_dict=True,
            )
            old_logits = outputs.logits
            old_values = outputs.values
            old_logprobs, old_entropy = self._get_logprobs_and_entropy(
                old_logits, input_ids, action_masks
            )
        
        # Compute advantages and returns
        dones = torch.zeros_like(rewards_tensor)  # Episodes don't end mid-generation
        advantages, returns = self.compute_advantages(rewards_tensor, old_values, dones)
        
        # Store metrics
        total_metrics = {
            "loss": 0.0,
            "policy_loss": 0.0,
            "value_loss": 0.0,
            "entropy_loss": 0.0,
        }
        
        # PPO epochs
        for epoch in range(self.ppo_epochs):
            # Mini-batch training
            indices = torch.randperm(len(input_ids))
            batch_count = 0  # Add counter for gradient accumulation
            
            for start in range(0, len(indices), self.mini_batch_size):
                end = start + self.mini_batch_size
                mb_indices = indices[start:end]
                
                # Mini-batch data
                mb_input_ids = input_ids[mb_indices]
                mb_attention_mask = attention_masks[mb_indices]
                mb_action_masks = action_masks[mb_indices]
                mb_advantages = advantages[mb_indices]
                mb_returns = returns[mb_indices]
                mb_old_logprobs = old_logprobs[mb_indices]
                mb_old_values = old_values[mb_indices]
                
                # Forward pass
                outputs = model(
                    input_ids=mb_input_ids,
                    attention_mask=mb_attention_mask,
                    return_dict=True,
                )
                logits = outputs.logits
                values = outputs.values
                
                # Get log probs and entropy
                logprobs, entropy = self._get_logprobs_and_entropy(
                    logits, mb_input_ids, mb_action_masks
                )
                
                # Compute loss
                loss_dict = self.compute_loss(
                    logprobs=logprobs,
                    old_logprobs=mb_old_logprobs,
                    advantages=mb_advantages,
                    values=values,
                    old_values=mb_old_values,
                    returns=mb_returns,
                    entropy=entropy,
                )

                # Scale loss by accumulation steps
                loss = loss_dict["loss"] / accumulation_steps
                
                # Backward pass
                model_engine.backward(loss)
                batch_count += 1
                
                # Only update weights every accumulation_steps batches
                if batch_count % accumulation_steps == 0 or (start + self.mini_batch_size) >= len(indices):
                    # Gradient clipping
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), self.max_grad_norm
                    )
                    
                    # Optimizer step
                    model_engine.step()
                    
                    # Clear gradients for next accumulation
                    optimizer.zero_grad()
                
                # Accumulate metrics (scale back up)
                for key, value in loss_dict.items():
                    if key in total_metrics:
                        total_metrics[key] += value.item()
            
            # Clear cache after each epoch to save memory
            torch.cuda.empty_cache()
        
        # Average metrics
        num_updates = self.ppo_epochs * (len(indices) // self.mini_batch_size)
        for key in total_metrics:
            total_metrics[key] /= num_updates
            
        return total_metrics

    def _prepare_batch_data(
        self,
        tokenizer,
        prompts: List[str],
        responses: List[Dict],
        rewards: List[Dict],
    ) -> Dict[str, torch.Tensor]:
        """Prepare and tokenize batch data"""
        all_input_ids = []
        all_attention_masks = []
        all_rewards = []
        all_action_masks = []

        for prompt, response_data, reward_list in zip(prompts, responses, rewards):
            # Handle different reward formats
            if isinstance(reward_list, list):
                # Multiple responses per prompt
                outputs = response_data["outputs"]
                rewards_for_prompt = reward_list
            else:
                # Single response
                outputs = [response_data["outputs"]]
                rewards_for_prompt = [reward_list]
                
            for output, reward in zip(outputs, rewards_for_prompt):
                # Tokenize prompt + response
                full_text = prompt + output
                encoded = tokenizer(
                    full_text,
                    return_tensors="pt",
                    padding="max_length",
                    truncation=True,
                    max_length=512,
                )

                input_ids = encoded["input_ids"]
                attention_mask = encoded["attention_mask"]

                # Calculate action mask
                prompt_encoded = tokenizer(
                    prompt,
                    return_tensors="pt",
                    padding=False,
                    truncation=True,
                )
                prompt_length = prompt_encoded["input_ids"].shape[1]
                
                action_mask = torch.zeros_like(input_ids)
                action_mask[:, prompt_length:] = attention_mask[:, prompt_length:]

                all_input_ids.append(input_ids)
                all_attention_masks.append(attention_mask)
                all_action_masks.append(action_mask)
                
                # Extract reward value
                if isinstance(reward, dict):
                    reward_value = reward.get("total_reward", 0.0)
                else:
                    reward_value = float(reward)
                all_rewards.append(reward_value)

        # Stack all tensors
        return {
            'input_ids': torch.cat(all_input_ids, dim=0),
            'attention_masks': torch.cat(all_attention_masks, dim=0),
            'action_masks': torch.cat(all_action_masks, dim=0),
            'rewards': torch.tensor(all_rewards),
        }

    def _get_logprobs_and_entropy(
        self, 
        logits: torch.Tensor, 
        input_ids: torch.Tensor, 
        action_masks: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Calculate log probabilities and entropy of actions taken"""
        # Shift for autoregressive
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = input_ids[:, 1:].contiguous()
        shift_masks = action_masks[:, 1:].contiguous()
        
        # Get probabilities
        logprobs = F.log_softmax(shift_logits, dim=-1)
        probs = F.softmax(shift_logits, dim=-1)
        
        # Entropy
        entropy = -(probs * logprobs).sum(dim=-1)
        
        # Get specific action log probs
        batch_size, seq_len = shift_labels.shape
        gathered_logprobs = logprobs.gather(
            dim=2, 
            index=shift_labels.unsqueeze(2)
        ).squeeze(2)
        
        # Apply masking and sum
        masked_logprobs = gathered_logprobs * shift_masks
        masked_entropy = entropy * shift_masks
        
        # Sum over sequence
        total_logprobs = masked_logprobs.sum(dim=1)
        total_entropy = masked_entropy.sum(dim=1)
        
        # Normalize by number of actions
        num_actions = shift_masks.sum(dim=1).clamp(min=1)
        avg_entropy = total_entropy / num_actions
        
        return total_logprobs, avg_entropy


class GRPO(BaseRLPolicy):
    """
    This class implements Group Relative Policy Optimisation and was invented by researchers at Deepseek.
    Refer paper for more info: https://arxiv.org/pdf/2402.03300
    """

    def __init__(
        self,
        group_size: int = 4,
        baseline_momentum: float = 0.9,
        kl_coef: float = 0.1,
        max_grad_norm: float = 1.0,
    ):
        self.group_size = group_size
        self.baseline_momentum = baseline_momentum
        self.kl_coef = kl_coef
        self.max_grad_norm = max_grad_norm
        self.baseline = {}  # Store baseline per prompt

    def compute_group_relative_rewards(
        self, rewards: List[float], group_size: int
    ) -> List[float]:
        """
        Compute relative rewards within groups
        """
        relative_rewards = []

        for i in range(0, len(rewards), group_size):
            group = rewards[i : i + group_size]
            group_mean = np.mean(group)
            group_std = np.std(group) + 1e-8

            # Normalize within group
            for reward in group:
                relative_reward = (reward - group_mean) / group_std
                relative_rewards.append(relative_reward)

        return relative_rewards

    def update_baseline(self, prompt: str, new_reward: float) -> float:
        """
        Update baseline with exponential moving average
        """
        if prompt not in self.baseline:
            self.baseline[prompt] = new_reward
            return 0.0

        old_baseline = self.baseline[prompt]
        self.baseline[prompt] = (
            self.baseline_momentum * old_baseline
            + (1 - self.baseline_momentum) * new_reward
        )

        return new_reward - old_baseline

    def compute_loss(
        self,
        logprobs: torch.Tensor,
        ref_logprobs: torch.Tensor,
        rewards: torch.Tensor,
        action_masks: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute GRPO loss with KL penalty
        """
        # KL divergence penalty
        kl_div = logprobs - ref_logprobs

        # Policy gradient loss
        pg_loss = -(logprobs * rewards)

        # Total loss with KL penalty
        loss = pg_loss + self.kl_coef * kl_div

        if action_masks is not None:
            loss = loss * action_masks
            kl_div = kl_div * action_masks

        return {
            "loss": loss.mean(),
            "pg_loss": pg_loss.mean(),
            "kl_div": kl_div.mean(),
        }

    def update(
        self,
        model_engine,
        optimizer,
        prompts: List[str],
        responses: List[Dict],
        rewards: List[Dict],
    ) -> float:
        """
        Update policy using GRPO algorithm
        """
        model = model_engine.module
        tokenizer = model_engine.tokenizer

        # Group responses by prompt
        prompt_groups = {}
        for prompt, response_data, reward_data in zip(prompts, responses, rewards):
            if prompt not in prompt_groups:
                prompt_groups[prompt] = {"responses": [], "rewards": []}

            for output, reward in zip(response_data["outputs"], reward_data):
                prompt_groups[prompt]["responses"].append(output)
                prompt_groups[prompt]["rewards"].append(reward["total_reward"])

        total_loss = 0.0
        num_updates = 0

        # Process each prompt group
        for prompt, group_data in prompt_groups.items():
            group_responses = group_data["responses"]
            group_rewards = group_data["rewards"]

            # Compute relative rewards
            relative_rewards = self.compute_group_relative_rewards(
                group_rewards, self.group_size
            )

            # Update baseline for this prompt
            avg_reward = np.mean(group_rewards)
            baseline_advantage = self.update_baseline(prompt, avg_reward)

            # Prepare batch for this group
            all_input_ids = []
            all_attention_masks = []
            all_action_masks = []
            all_advantages = []

            for response, rel_reward, raw_reward in zip(
                group_responses, relative_rewards, group_rewards
            ):
                # Tokenize
                full_text = prompt + response
                encoded = tokenizer(
                    full_text,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=512,
                )

                input_ids = encoded["input_ids"]
                attention_mask = encoded["attention_mask"]

                # Action mask
                prompt_encoded = tokenizer(prompt, return_tensors="pt")
                prompt_length = prompt_encoded["input_ids"].shape[1]
                action_mask = torch.zeros_like(input_ids)
                action_mask[:, prompt_length:] = 1

                # Advantage = relative reward + baseline advantage
                advantage = rel_reward + baseline_advantage

                all_input_ids.append(input_ids)
                all_attention_masks.append(attention_mask)
                all_action_masks.append(action_mask)
                all_advantages.append(advantage)

            if not all_input_ids:
                continue

            # Stack tensors
            input_ids = torch.cat(all_input_ids, dim=0).to(model_engine.device)
            attention_masks = torch.cat(all_attention_masks, dim=0).to(
                model_engine.device
            )
            action_masks = torch.cat(all_action_masks, dim=0).to(model_engine.device)
            advantages = torch.tensor(all_advantages, device=model_engine.device)

            # Get reference log probs (from original model)
            with torch.no_grad():
                ref_outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_masks,
                    return_dict=True,
                )
                ref_logits = ref_outputs.logits
                ref_logprobs = self._get_logprobs(ref_logits, input_ids, action_masks)

            # Forward pass
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_masks,
                return_dict=True,
            )
            logits = outputs.logits
            logprobs = self._get_logprobs(logits, input_ids, action_masks)

            # Compute loss
            loss_dict = self.compute_loss(
                logprobs=logprobs,
                ref_logprobs=ref_logprobs,
                rewards=advantages,
                action_masks=action_masks,
            )

            loss = loss_dict["loss"]
            total_loss += loss.item()
            num_updates += 1

            # Backward pass
            model_engine.backward(loss)

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), self.max_grad_norm)

            # Optimizer step
            model_engine.step()

        return total_loss / max(num_updates, 1)

    def _get_logprobs(
        self, logits: torch.Tensor, input_ids: torch.Tensor, action_masks: torch.Tensor
    ) -> torch.Tensor:
        """Calculate log probabilities of actions taken"""
        logprobs = F.log_softmax(logits, dim=-1)

        # Get log probs of actual tokens
        batch_size, seq_len = input_ids.shape
        selected_logprobs = logprobs[
            torch.arange(batch_size).unsqueeze(1),
            torch.arange(seq_len).unsqueeze(0),
            input_ids,
        ]

        # Apply action mask and sum
        masked_logprobs = selected_logprobs * action_masks
        return masked_logprobs.sum(dim=1)


class Hybrid:
    """
    This policy is an extension of both PPO and GRPO. Combining the best out of both algorithms.
    Refer paper for more info: https://arxiv.org/abs/2502.01652
    """


class HGR:
    """
    This is the novel approach developed by the author of this code.

    What it answers?
        1. VeriGen [1] solved:          "Can LLMs generate Verilog?"
        2. VeriReason [2] solved:       "Can LLMs reason about hardware?"
        3. DeepseekCoder+HGR solves:    "Can LLMs learn to design hardware like engineers do?"

    What does it stand for?
        Hybrid (PPO + GRPO) with Absolute Gating and Reward Decomposition

    How does it work?
        Hybrid reinforcement learning method designed for fine-tuning large language models on structured code generation tasks, such as Verilog. HGR combines PPO with Group Relative Policy Optimization (GRPO) to leverage relative ranking among top-k completions, while Absolute Gating ensures structurally invalid outputs are penalized and excluded from relative comparisons. The method incorporates reward decomposition (functional correctness, structural validity, semantic priors, shortcut penalties, relative ranking, and exploration bonuses), along with anchor replay and KL regularization to preserve prior knowledge and prevent catastrophic forgetting.

        Additionally, HGR introduces a Convergence Rescue (CR) workflow that triggers when training stagnates. CR enhances exploration through entropy and novelty bonuses, assigns partial credit to partially correct outputs, resamples multiple candidate completions, injects anchor solutions, and applies a staged curriculum of evaluation difficulty. This allows the policy to escape local optima and converge to fully correct, generalizable code completions.

        HGR is particularly effective for sparse, structured reward environments, achieving robust convergence to correct solutions while maintaining sample efficiency and stability. In practice, HGR ensures partially correct candidates are preserved early in training, fully exploits high-performing candidates, and gradually drives the model toward full functional correctness on complex hardware description tasks.

    Can it generalise to other domains?
        Can't confirm yet, tests running at the moment.

    Reference:
        [1] https://dl.acm.org/doi/full/10.1145/3643681
        [2] https://arxiv.org/abs/2505.11849
    """
