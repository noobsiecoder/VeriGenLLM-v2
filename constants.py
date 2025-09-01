"""
All constant data stored here
Prevents circular imports and promotes having a unified space to hold all constants

Author: Abhishek Sriram <noobsiecoder@gmail.com>
Date:   Aug 28th, 2025
Place:  Boston, MA
"""

import sys
import os
from enum import Enum
from abc import ABC, abstractmethod
from peft import TaskType
from src.logger import Logger
from dotenv import load_dotenv
import torch


class ENVLoader:
    """
    Loads all the ENV required in the local machine
    """

    def __init__(self):
        """
        Loads when object is instantiated
        """
        self.log = Logger("env_loader").get_logger()
        self._run()  # run when object is instantiated

    def _run(self):
        """
        Private method to load ENV
        """
        try:
            load_dotenv("secrets/models-api.env")
            self.log.info("ENV loaded successfully")
        except Exception as err:
            self.log.critical(f"Fatal Error on ENV loading: {err}")
            self.log.info("Exiting ...")
            sys.exit(-1)


class Creator(Enum):
    """
    Enum class for (Code) Creator
    Use to choose the creator of the code
    """

    HUMAN = "Human"
    LLM = "LLM"


class RLPolicy(Enum):
    """
    Enum class for RL Policy
    Use to choose a policy and fine-tune model
    """

    GRPO = "GRPO"
    PPO = "PPO"


class BaseRL(ABC):
    """
    Base class/abstract class for all RL algorithms
    """

    @abstractmethod
    def compute_loss():
        """
        Injects RL policies
        Includes the inner mechanism of RL algorithm's policy update
        """

    @abstractmethod
    def update():
        """
        Update old policy -> new policy
        Apply Backpropagation + include optimizer (AdamW)
        """


TEST_DIR = "dataset/testbench/hdlbits/vectors5/"
test_prompt = ""
test_tb_code = ""
with open(os.path.join(TEST_DIR, "prompt2_vector-gates.v"), "r") as fs:
    test_prompt = fs.read()
with open(os.path.join(TEST_DIR, "tb_vector-gates.v"), "r") as fs:
    test_tb_code = fs.read()


# Store all constants to run Fine-tuning guided by Reinforcement Learning
RLFT_TRAIN_CONFIG = {
    "rl_algorithm": RLPolicy.GRPO,
    "unique_id": "deepseek-ai/deepseek-coder-7b-instruct-v1.5",
    "model_name": "deepseek-coder7b",
    # "unique_id": "Salesforce/codegen-350M-mono",
    # "model_name": "salesforce-codegen",
    # "unique_id": "deepseek-ai/deepseek-coder-1.3b-instruct",
    # "model_name": "deepseek-coder1.3b",
    "apply_lora": True,
    # "apply_lora": False,
    "precision": torch.bfloat16,
    "epochs": 30,
    "batch_size": 2,
    "sample_size": 5,
    "padding": True,
    "temperature": 0.3,
    "top_p": 0.9,
    "max_tokens": 512,
    "system_prompt": "You are a Verilog Expert.",
    "update_ref_policy": 5,  # updates per X batches
    "test_data": {
        "prompt": test_prompt,
        "tb_code": test_tb_code,
    },  # TODO: for-now, only one allowed!
}

# Store all constants for weights to calculate the reward function
REWARD_WEIGHTS_CONFIG = {
    "compilation": 0.4,
    "functional_correctness": 0.5,
    "synthesise": 0.3,
    "code_quality": 0.5,
    "reasoning": 0.6,
}

# Store all constants for running PPO RLFT
PPO_CONFIG = {
    "clip_epsilon": 0.2,
    "value_coefficient": 0.5,
    "entropy_coefficient": 0.01,
    "max_grad_norm": 0.5,
}

# Store all constants for running GRPO RLFT
GRPO_CONFIG = {
    "epsilon": 0.2,
    "max_grad_norm": 0.5,
}

# Store all constants pertaining to LoRA adapters
LORA_CONFIG = {
    "rank": 16,
    "alpha": 32,
    "dropout": 0.1,
    "bias": "none",
    "task_type": TaskType.CAUSAL_LM,
    "target_modules": [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],
}

WANDB_CONFIG = {
    "model": RLFT_TRAIN_CONFIG["unique_id"],
    "batch_size": RLFT_TRAIN_CONFIG["batch_size"],
    # "clip_epsilon": PPO_CONFIG["clip_epsilon"],
    "epsilon": GRPO_CONFIG["epsilon"],
}
