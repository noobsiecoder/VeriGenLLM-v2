"""
All constant data stored here
Note: This prevents circular imports

Author: Abhishek Sriram <noobsiecoder@gmail.com>
Date:   Aug 27th, 2025
Place:  Boston, MA
"""

from dataclasses import dataclass
from enum import Enum


class Creator(Enum):
    """
    Enum class for (Code) Creator
    Use to choose the creator of the code
    """

    Human = "Human"
    LLM = "LLM"


class RLPolicy(Enum):
    """
    Enum class for RL Policy
    Use to choose a policy and fine-tune model
    """

    PPO = "PPO"
    GRPO = "GRPO"
    HYBRID = "HYBRID"
    HGAR = "HGAR"


@dataclass
class RewardScores:
    """Container for all reward components"""

    code_quality: float = 0.0
    compilation: float = 0.0
    functional_correctness: float = 0.0
    reasoning: float = 0.0
    similarity: float = 0.0
    synthesis: float = 0.0

    @property
    def total_reward(self) -> float:
        """Calculate weighted total reward"""
        # Weights can be adjusted based on importance
        weights = {
            "code_quality": 0.15,
            "compilation": 0.20,
            "functional_correctness": 0.25,
            "reasoning": 0.15,
            "similarity": 0.10,
            "synthesis": 0.15,
        }

        return (
            weights["code_quality"] * self.code_quality
            + weights["compilation"] * self.compilation
            + weights["functional_correctness"] * self.functional_correctness
            + weights["reasoning"] * self.reasoning
            # + weights["similarity"] * self.similarity
            + weights["synthesis"] * self.synthesis
        )


# Training Hyperparameters
TRAINING_CONFIG = {
    # LoRA parameters
    "lora_r": 16,
    "lora_alpha": 32,
    "lora_dropout": 0.1,
    # DeepSpeed parameters
    "gradient_accumulation_steps": 4,
    "zero_stage": 2,
    # Training parameters
    "learning_rate": 2e-5,
    "batch_size": 2,
    "epochs": 5,
    "max_tokens": 300,
    "temperature": 0.7,
    "n_samples": 4,
    # PPO specific
    "ppo_clip_epsilon": 0.2,
    "ppo_value_coef": 0.5,
    "ppo_entropy_coef": 0.01,
    "ppo_epochs": 4,
    # GRPO specific
    "grpo_group_size": 4,
    "grpo_baseline_momentum": 0.9,
}

# Model save configuration
SAVE_CONFIG = {
    "save_interval": 100,  # Save every N batches
    "save_path": "checkpoints/",
    "hf_repo_name": None,  # Set to your HF repo name
    "hf_token": None,  # Set via environment variable
}
