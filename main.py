"""
Main driver for all python scripts

Author: Abhishek Sriram <noobsiecoder@gmail.com>
Date:   Aug 14th, 2025
Place:  Boston, MA
"""

import glob
import json
import os
import random
import torch
from src.logger import Logger, WeightsAndBiases
from src.models import Policy
from src.ppo import PPO
from src.rewards import RewardFunction, RewardScores
from constants import Creator, ENVLoader, RLPolicy, RLFT_TRAIN_CONFIG, WANDB_CONFIG


ENVLoader()  # load ENV globally


class Trainer:
    """
    Perform RLFT with X RL algorithm
    """

    def __init__(self):
        """ """
        project_name = f"llm-rlft-{RLFT_TRAIN_CONFIG['rl_algorithm'].value}"
        wandb_config = WANDB_CONFIG
        self.log = Logger("rlft-trainer").get_logger()
        self.wandb_logger = WeightsAndBiases(
            project_name=project_name, config=wandb_config
        )
        self.dataset_path = "dataset/testbench/hdlbits"
        self.filename = "orders.txt"
        self.epochs = RLFT_TRAIN_CONFIG.get("epochs", 0.5)
        self.batch_size = RLFT_TRAIN_CONFIG.get("batch_size", 2)
        self.sample_size = RLFT_TRAIN_CONFIG.get("sample_size", 4)
        self.update_ref_policy = RLFT_TRAIN_CONFIG.get("update_ref_policy", 5)
        self.unique_id = RLFT_TRAIN_CONFIG.get("unique_id", None)
        self.name = RLFT_TRAIN_CONFIG.get("model_name", None)
        self.reward_func = RewardFunction()
        self.algorithm = None
        self.policy = None
        self.ref_policy = None

    def train(self):
        """
        Runner script of RL
        """
        if self.unique_id is None and self.name is None:
            self.log.critical("Model name and HuggingFace ID not found (None returned)")
            raise ValueError("Model name and HuggingFace ID not found")

        # Load LLM Model
        try:
            # Load policy to be updated
            self.policy = Policy(name=self.name, unique_id=self.unique_id)
            self.policy.load()
            # Load policy for reference
            self.ref_policy = Policy(
                name=self.name, unique_id=self.unique_id, apply_lora=False
            )
            self.ref_policy.load()
            # Freeze params - only for reference
            for param in self.ref_policy.model.parameters():
                param.requires_grad = False
            # Optimizer
            adam_w_optim = torch.optim.AdamW(self.policy.model.parameters(), lr=1e-5)
            rl_algorithm = RLFT_TRAIN_CONFIG.get("rl_algorithm", RLPolicy.PPO)
            # Get hidden dimension
            hidden_dim = self.policy.model.config.hidden_size
            # Set RL algorithm
            if rl_algorithm == RLPolicy.PPO:
                self.algorithm = PPO(
                    policy=self.policy,
                    ref_policy=self.ref_policy,
                    optimizer=adam_w_optim,
                    hidden_dim=hidden_dim,
                )
            else:
                self.log.critical(f"Value of RL algorithm not found: {rl_algorithm}")
                raise ValueError("Value of RL Algorithm not found!")
        except Exception as err:
            self.log.error(
                f"Error while establishing connection with model: {self.unique_id} | Error: {err}"
            )
            raise Exception(
                f"Exception caught while loading MODEL into hardware | Reason: {err}"
            )

        self.dataset_files_order = []
        with open(os.path.join(self.dataset_path, self.filename), "r") as fs:
            content = fs.read()
            self.dataset_files_order = [
                line.strip() for line in content.splitlines() if line.strip()
            ]
        self.dataset_files_order = self.dataset_files_order[:15]

        # Run RLFT batch-wise per epoch
        try:
            for epoch in range(self.epochs):
                self.epoch_rewards = []
                self.epoch_compilation_rates = []
                self.epoch_functional_rates = []
                self._batch_executor(epoch)
        except Exception as err:
            self.log.error(f"Error while running RLFT: {err}")
            raise Exception(f"Exception caught while running RLFT | Reason: {err}")

    def _correct_filename(self, pattern: str) -> str:
        """
        Determine the correct function
        """
        matching_files = glob.glob(pattern)

        if matching_files:
            return matching_files[0]  # Get the first (and presumably only) match
        else:
            # Handle case where no file is found
            raise FileNotFoundError(f"Incorrect path: {pattern}")

    def _compute_log_probs_from_scores(self, scores, actions):
        """
        Calculate the log probability of the token chosen
        """
        log_probs_list = []
        for step_scores, action_id in zip(scores, actions.T):
            log_prob = torch.log_softmax(step_scores, dim=-1)
            token_log_prob = log_prob.gather(-1, action_id.unsqueeze(-1)).squeeze(-1)
            log_probs_list.append(token_log_prob)
        return torch.stack(log_probs_list, dim=0)

    def _batch_executor(self, epoch: int):
        """
        Private function that runs the RL batch(es)
        """
        for batch_idx in range(0, len(self.dataset_files_order), self.batch_size):
            batch_responses = None
            rewards = []  # store all rewards in a batch
            input_ids = []  # store all input ids in a batch
            dataset_paths = self.dataset_files_order[
                batch_idx : self.batch_size + batch_idx
            ]
            # Load batch data
            paths_tb_code = [
                self._correct_filename(
                    os.path.join(self.dataset_path, dirname, "tb_*.v")
                )
                for dirname in dataset_paths
            ]
            paths_pm = [
                self._correct_filename(
                    os.path.join(
                        self.dataset_path,
                        dirname,
                        f"prompt{3 - epoch if epoch < 3 else random.randint(1, 3)}_*.v",
                    )
                )
                for dirname in dataset_paths
            ]
            # Read all prompts
            prompts = [open(prompt, "r").read() for prompt in paths_pm]
            # Generate Responses
            batch_responses = self.policy.generate(prompts)
            for prompt_idx in range(len(prompts)):
                # Get corresponding testbench code
                with open(paths_tb_code[prompt_idx], "r") as fs:
                    tb_code_in_str = fs.read()
                samples = batch_responses["texts"][
                    prompt_idx * self.sample_size : (prompt_idx + 1) * self.sample_size
                ]
                for sample in samples:
                    # Compute reward
                    # cd_code_in_str = self.reward_func.extract_code(
                    #     sample, Creator.LLM
                    # )
                    response_wrapped = "```verilog" + sample + "```"
                    cd_code_in_str = self.reward_func.extract_code(
                        response_wrapped, Creator.LLM
                    )  # TODO: Testing

                    compilation_score = self.reward_func.compilation_score(
                        cd_code_in_str
                    )[0]
                    functional_correctness_score = (
                        self.reward_func.functional_correctness_score(
                            cd_code_in_str, tb_code_in_str
                        )[0]
                    )
                    sythesise_score = self.reward_func.sythesise_score(cd_code_in_str)
                    code_quality_score = self.reward_func.code_quality_score(
                        cd_code_in_str
                    )[0]
                    reasoning_score = 0.0  # TODO: Testing
                    # reasoning_score = self.reward_func.reasoning_score(response_wrapped)["score"]

                    reward_scorer = RewardScores()
                    reward = reward_scorer.total_score(
                        compilation_score,
                        functional_correctness_score,
                        sythesise_score,
                        code_quality_score,
                        reasoning_score,
                    )
                    rewards.append(reward)
            attention_mask = (
                batch_responses["sequences"] != self.policy.tokenizer.pad_token_id
            ).long()
            with torch.no_grad():
                ref_outputs = self.ref_policy.model(
                    input_ids=batch_responses["sequences"],
                    attention_mask=attention_mask,
                    output_hidden_states=True,
                )
            ref_logits = ref_outputs.logits
            outputs = self.policy.model(
                input_ids=batch_responses["sequences"],
                attention_mask=attention_mask,
                output_hidden_states=True,
            )
            logits = outputs.logits
            batches = {
                "total_response": self.sample_size * self.batch_size,
                "prompts_token_length": batch_responses["prompts_token_length"],
                "sequences": batch_responses["sequences"],
                "old_prob_seq_logits": ref_logits,
                "new_prob_seq_logits": logits,
                "hidden_states": outputs.hidden_states,
                "input_ids": input_ids,
            }
            # Compute loss and update per batch
            losses = self.algorithm.compute_loss(batches, rewards)
            self.algorithm.update()
            self.wandb_logger.log_batch(losses, batch_idx, epoch)
            self.log.info(f"Losses: {json.dumps(losses, indent=4)}")

            if batch_idx > 0 and batch_idx % self.update_ref_policy == 0:
                self.ref_policy.load_state_dict(self.policy.state_dict())


if __name__ == "__main__":
    rlft = Trainer()
    rlft.train()
