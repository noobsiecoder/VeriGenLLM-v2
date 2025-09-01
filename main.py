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
from src.algo import GRPO, PPO
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
        self.dataset_path = "dataset/testbench/hdlbits"
        self.filename = "orders.txt"
        self.epochs = RLFT_TRAIN_CONFIG.get("epochs", 0.5)
        self.batch_size = RLFT_TRAIN_CONFIG.get("batch_size", 2)
        self.sample_size = RLFT_TRAIN_CONFIG.get("sample_size", 4)
        self.update_ref_policy = RLFT_TRAIN_CONFIG.get("update_ref_policy", 5)
        self.unique_id = RLFT_TRAIN_CONFIG.get("unique_id", None)
        self.name = RLFT_TRAIN_CONFIG.get("model_name", None)
        self.reward_func = RewardFunction()
        self.wandb_logger = WeightsAndBiases(
            project_name=project_name, model_name=self.name, config=wandb_config
        )
        self.algorithm = None
        self.policy = None
        self.ref_policy = None
        self.rl_algorithm = None

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
            self.policy = Policy(
                name=self.name, unique_id=self.unique_id, grad_check=False
            )
            self.policy.load()
            self.rl_algorithm = RLFT_TRAIN_CONFIG.get("rl_algorithm", RLPolicy.PPO)
            # Optimizer
            adam_w_optim = torch.optim.AdamW(self.policy.model.parameters(), lr=1e-5)
            # Get hidden dimension
            hidden_dim = self.policy.model.config.hidden_size
            # Set RL algorithm
            if self.rl_algorithm == RLPolicy.PPO:
                self.algorithm = PPO(
                    policy=self.policy,
                    ref_policy=self.ref_policy,
                    optimizer=adam_w_optim,
                    hidden_dim=hidden_dim,
                )
                # Load policy for reference (PPO)
                self.ref_policy = Policy(
                    name=self.name,
                    unique_id=self.unique_id,
                    device="cpu",
                )
                self.ref_policy.load()
                self.ref_policy.model.eval()  # EVAL Mode
                # Freeze params - only for reference
                for param in self.ref_policy.model.parameters():
                    param.requires_grad = False
            elif self.rl_algorithm == RLPolicy.GRPO:
                self.algorithm = GRPO(
                    policy=self.policy,
                    optimizer=adam_w_optim,
                    # hidden_dim=hidden_dim,
                )
            else:
                self.log.critical(f"Value of RL algorithm not found: {self.rl_algorithm}")
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
            # Track best reward and save checkpoints
            self.best_reward = float("-inf")
            for epoch in range(self.epochs):
                self.epoch_rewards = 0.0
                self.epoch_compilation_rates = 0.0
                self.epoch_functional_rates = 0.0
                self.epoch_reasoning_rates = 0.0
                self._batch_executor(epoch)
                metrics = {
                    "train/reward_per_epoch": self.epoch_rewards,
                    "train/compilation_rates_per_epoch": self.epoch_compilation_rates,
                    "train/functional_rates_per_epoch": self.epoch_functional_rates,
                    "train/reasoning_rates_per_epoch": self.epoch_reasoning_rates,
                }
                self.wandb_logger.log_batch(metrics)
                self.log.info(f"Epoch Level Stat: {json.dumps(metrics, indent=4)}")

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
            compilation_scores = []  # store all comp scores in a batch
            func_corr_scores = []  # store all functional correctness scores in a batch
            synth_scores = []  # store all synthesise scores in a batch
            code_quality_scores = []  # store all code quality scores in a batch
            reasoning_scores = []  # store all reasoning scores in a batch
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
                    cd_code_in_str = self.reward_func.extract_code(sample, Creator.LLM)
                    compilation_score = self.reward_func.compilation_score(
                        cd_code_in_str
                    )[0]
                    functional_correctness_score = (
                        self.reward_func.functional_correctness_score(
                            cd_code_in_str, tb_code_in_str
                        )[0]
                    )
                    synthesise_score = self.reward_func.synthesise_score(cd_code_in_str)
                    code_quality_score = self.reward_func.code_quality_score(
                        cd_code_in_str
                    )[0]
                    reasoning_res = self.reward_func.reasoning_score(sample)
                    if isinstance(reasoning_res, float):
                        reasoning_score = reasoning_res
                    else:
                        reasoning_score = reasoning_res["score"]

                    reward_scorer = RewardScores()
                    reward = reward_scorer.total_score(
                        compilation_score,
                        functional_correctness_score,
                        synthesise_score,
                        code_quality_score,
                        reasoning_score,
                    )
                    # Store data for analysis per batch
                    compilation_scores.append(compilation_score)
                    func_corr_scores.append(functional_correctness_score)
                    synth_scores.append(synthesise_score)
                    code_quality_scores.append(code_quality_score)
                    reasoning_scores.append(reasoning_score)
                    rewards.append(reward)
                    # Add data for analysis per epoch
                    self.epoch_rewards += reward
                    self.epoch_compilation_rates += compilation_score
                    self.epoch_functional_rates += functional_correctness_score
                    self.epoch_reasoning_rates += reasoning_score
                    # Save if policy is better
                    if reward > self.best_reward:
                        self.best_reward = reward
                        self.wandb_logger.save_checkpoint(self.policy.model, epoch)
            attention_mask = (
                batch_responses["sequences"] != self.policy.tokenizer.pad_token_id
            ).long()
            if self.rl_algorithm == RLPolicy.PPO:
                with torch.no_grad():
                    # Move to GPU for computation
                    self.ref_policy.model.to(self.ref_policy.device)
                    ref_outputs = self.ref_policy.model(
                        input_ids=batch_responses["sequences"],
                        attention_mask=attention_mask,
                        output_hidden_states=True,
                    )
                    ref_logits = ref_outputs.logits.clone()
                    ref_hidden_states = [hs.clone() for hs in ref_outputs.hidden_states]
                    # Move back to CPU
                    self.ref_policy.model.to("cpu")
                    del ref_outputs
                    torch.cuda.empty_cache()
            # Run policy model
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
                "old_prob_seq_logits": ref_logits if self.rl_algorithm == RLPolicy.PPO else None,
                "new_prob_seq_logits": logits,
                "hidden_states": outputs.hidden_states,
                "input_ids": input_ids,
            }
            # Compute loss and update per batch
            losses = self.algorithm.compute_loss(batches, rewards)
            self.algorithm.update()
            # Print key metrics
            if self.rl_algorithm == RLPolicy.PPO:
                self.log.info(
                    f"Epoch {epoch} | Batch {batch_idx} | "
                    f"Policy Loss: {losses.get('policy_loss', 0):.4f} | "
                    f"Mean Reward: {losses.get('mean_reward', 0):.4f} | "
                    f"KL: {losses.get('approx_kl', 0):.4f}"
                )
            elif self.rl_algorithm == RLPolicy.GRPO:
                self.log.info(
                    f"Epoch {epoch} | Batch {batch_idx} | "
                    f"Mean Reward: {losses.get('mean_reward', 0):.4f} | "
                    f"KL: {losses.get('approx_kl', 0):.4f}"
                )
            else:
                return
            # Combine all metrics for WANDB analysis
            if self.rl_algorithm == RLPolicy.PPO:
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
                    "train/mean_comp_reward": sum(compilation_scores)
                    / float(len(compilation_scores)),
                    "train/mean_fcor_reward": sum(func_corr_scores)
                    / float(len(func_corr_scores)),
                    "train/mean_synt_reward": sum(synth_scores) / float(len(synth_scores)),
                    "train/mean_coqu_reward": sum(code_quality_scores)
                    / float(len(code_quality_scores)),
                    "train/mean_reas_reward": sum(reasoning_scores)
                    / float(len(reasoning_scores)),
                }
                self.wandb_logger.log_batch(metrics)
            elif self.rl_algorithm == RLPolicy.GRPO:
                metrics = {
                    "train/total_loss": losses.get("total_loss", 0),
                    "train/mean_reward": losses.get("mean_reward", 0),
                    "train/advantage_mean": losses.get("advantage_mean", 0),
                    "train/advantage_std": losses.get("advantage_std", 0),
                    "train/epoch": epoch,
                    "train/batch": batch_idx,
                    "train/mean_comp_reward": sum(compilation_scores)
                    / float(len(compilation_scores)),
                    "train/mean_fcor_reward": sum(func_corr_scores)
                    / float(len(func_corr_scores)),
                    "train/mean_synt_reward": sum(synth_scores) / float(len(synth_scores)),
                    "train/mean_coqu_reward": sum(code_quality_scores)
                    / float(len(code_quality_scores)),
                    "train/mean_reas_reward": sum(reasoning_scores)
                    / float(len(reasoning_scores)),
                }
                self.wandb_logger.log_batch(metrics)
            else:
                return
            if batch_idx > 0 and batch_idx % self.update_ref_policy == 0:
                # Note: update happens on CPU (more memory efficient)
                if self.rl_algorithm == RLPolicy.PPO:
                    with torch.no_grad():
                        # Copy the entire state dict including LoRA weights
                        policy_state = self.policy.model.state_dict()
                        # Move to CPU and load into ref_policy (which also has LoRA)
                        cpu_state = {k: v.cpu() for k, v in policy_state.items()}
                    self.ref_policy.model.load_state_dict(cpu_state)

                # Record updated policy
                if RLFT_TRAIN_CONFIG.get("test_data", None) is not None:
                    self.log.info("Testing New Policy....")
                    prompt = RLFT_TRAIN_CONFIG["test_data"]["prompt"]
                    tb_code_in_str = RLFT_TRAIN_CONFIG["test_data"]["tb_code"]
                    if self.rl_algorithm == RLPolicy.PPO:
                        # Clear any cached tensors
                        torch.cuda.empty_cache()
                        self.ref_policy.model = self.ref_policy.model.to("cuda")
                        self.ref_policy.device = torch.device(
                            "cuda"
                        )  # Update device attribute
                        self.ref_policy.model.eval()  # EVAL Mode
                        responses = self.ref_policy.generate(
                            prompts=[prompt]
                        )  # Only one prompt sent
                        # Compute reward
                        sample = responses["texts"][
                            0
                        ]  # Only one allowed! else -> IndexError
                        cd_code_in_str = self.reward_func.extract_code(sample, Creator.LLM)
                        compilation_score = self.reward_func.compilation_score(
                            cd_code_in_str
                        )[0]
                        functional_correctness_score = (
                            self.reward_func.functional_correctness_score(
                                cd_code_in_str, tb_code_in_str
                            )[0]
                        )
                        synthesise_score = self.reward_func.synthesise_score(cd_code_in_str)
                        code_quality_score = self.reward_func.code_quality_score(
                            cd_code_in_str
                        )[0]
                        reasoning_res = self.reward_func.reasoning_score(sample)
                        if isinstance(reasoning_res, float):
                            reasoning_score = reasoning_res
                        else:
                            reasoning_score = reasoning_res["score"]

                        reward_scorer = RewardScores()
                        reward = reward_scorer.total_score(
                            compilation_score,
                            functional_correctness_score,
                            synthesise_score,
                            code_quality_score,
                            reasoning_score,
                        )
                        self.wandb_logger.log_examples([prompt], [sample], [reward], epoch)
                        data = {
                            "prompt": prompt,
                            "sample": sample,
                            "reward": reward,
                        }
                        self.wandb_logger.save_json_to_wandb(
                            data,
                            filename=f"policy_check_iter_{epoch}-{batch_idx}",
                            artifact_name="rlft-test-data",
                            artifact_type="results",
                            metadata={"model_name": self.name, "hf_id": self.unique_id},
                        )
                        # Properly move the model to CPU
                        self.ref_policy.model = self.ref_policy.model.to(
                            "cpu"
                        )  # Model moved to CPU
                        self.ref_policy.device = torch.device(
                            "cpu"
                        )  # Also update the device attribute
                        # torch.cuda.empty_cache()
                    elif self.rl_algorithm == RLPolicy.GRPO:
                        self.policy.model.eval()  # EVAL Mode
                        responses = self.policy.generate(
                            prompts=[prompt]
                        )  # Only one prompt sent
                        # Compute reward
                        sample = responses["texts"][
                            0
                        ]  # Only one allowed! else -> IndexError
                        cd_code_in_str = self.reward_func.extract_code(sample, Creator.LLM)
                        compilation_score = self.reward_func.compilation_score(
                            cd_code_in_str
                        )[0]
                        functional_correctness_score = (
                            self.reward_func.functional_correctness_score(
                                cd_code_in_str, tb_code_in_str
                            )[0]
                        )
                        synthesise_score = self.reward_func.synthesise_score(cd_code_in_str)
                        code_quality_score = self.reward_func.code_quality_score(
                            cd_code_in_str
                        )[0]
                        reasoning_res = self.reward_func.reasoning_score(sample)
                        if isinstance(reasoning_res, float):
                            reasoning_score = reasoning_res
                        else:
                            reasoning_score = reasoning_res["score"]

                        reward_scorer = RewardScores()
                        reward = reward_scorer.total_score(
                            compilation_score,
                            functional_correctness_score,
                            synthesise_score,
                            code_quality_score,
                            reasoning_score,
                        )
                        self.wandb_logger.log_examples([prompt], [sample], [reward], epoch)
                        data = {
                            "prompt": prompt,
                            "sample": sample,
                            "reward": reward,
                        }
                        self.wandb_logger.save_json_to_wandb(
                            data,
                            filename=f"policy_check_iter_{epoch}-{batch_idx}",
                            artifact_name="rlft-test-data",
                            artifact_type="results",
                            metadata={"model_name": self.name, "hf_id": self.unique_id},
                        )
                        self.policy.model.train()


if __name__ == "__main__":
    rlft = Trainer()
    rlft.train()
