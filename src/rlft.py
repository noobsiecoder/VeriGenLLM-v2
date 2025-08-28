"""
Main driver for Reinforcement Learning Fine-tuning (RLFT)

Author: Abhishek Sriram <noobsiecoder@gmail.com>
Date:   Aug 26th, 2025
Place:  Boston, MA
"""

import os
import glob
import random
import json
from typing import Dict, Iterator, List, Optional, Tuple
import torch
from huggingface_hub import HfApi
from transformers import AutoTokenizer
from src.constants import Creator, RLPolicy, RewardScores
from src.rl_policy import GRPO, PPO, BaseRLPolicy
from src.logger import Logger
from src.models import OpenSourceLLMClient
from src.reward import (
    CodeSimilarityChecker,
    ReasoningChecker,
    SynthesisChecker,
    VerilogCodeAnalyzer,
    VerilogCompiler,
)
from src.wandb_logger import WandbLogger
from src.constants import TRAINING_CONFIG, SAVE_CONFIG


class RLFineTuner:
    """
    Used for fine-tuning a model using RL
    Different policies in RL can be used to FT a model
    """

    def __init__(self, model_info: Dict, policy: RLPolicy = RLPolicy.PPO):
        """
        Parameters:
        -----------
        model_info: Dict
            Model name and nickname/id
        policy:     RLPolicy
            Type of RL algorithm/policy to be used in RLFT
        """
        self.model_info = model_info
        self.log = self._setup_logger()

        # Initialize policy
        self.policy = self._initialize_policy(policy)

        # Initialize components
        self.analyzer = VerilogCodeAnalyzer(self.log)
        self.compiler = VerilogCompiler(self.log)
        self.synthesizer = SynthesisChecker(self.log)
        self.similarity_checker = CodeSimilarityChecker(self.log)
        self.reasoning_checker = ReasoningChecker(self.log)

        self.log.info(f"RLFineTuner initialized with policy: {policy.value}")

    def _setup_logger(self):
        """Setup logger (placeholder for actual implementation)"""
        return Logger("rlft-trainer").get_logger()

    def _initialize_policy(self, policy: RLPolicy) -> BaseRLPolicy:
        """Initialize the specified RL policy"""
        if policy == RLPolicy.PPO:
            self.log.info(f"Initializing PPO policy")
            return PPO()
        elif policy == RLPolicy.GRPO:
            self.log.info(f"Initializing GRPO policy")
            return GRPO()
        else:
            raise ValueError(f"Unsupported policy: {policy}")

    def calculate_reward(
        self,
        candidate_response: str,
        ground_truth_code: str,
        testbench_code: Optional[str] = None,
    ) -> Tuple[RewardScores, Dict]:
        """
        Calculate comprehensive reward for candidate response

        Returns:
            (reward_scores, detailed_metrics)
        """
        scores = RewardScores()
        metrics = {}

        # Extract code and reasoning
        candidate_code = self.analyzer.extract_code(candidate_response, Creator.LLM)
        if not candidate_code:
            self.log.error("Failed to extract code from response")
            return scores, {"error": "No code found"}

        candidate_reasoning = self.analyzer.extract_reasoning(candidate_response)

        # 1. Code Quality Score
        self.log.info("Evaluating code quality...")
        scores.code_quality, quality_metrics = self.analyzer.analyze_code_quality(
            candidate_code
        )
        metrics["code_quality"] = quality_metrics

        # 2. Compilation Score
        self.log.info("Evaluating compilation...")
        scores.compilation, compile_metrics = self.compiler.compile_and_score(
            candidate_code
        )
        metrics["compilation"] = compile_metrics

        # 3. Functional Correctness Score
        if testbench_code:
            self.log.info("Evaluating functional correctness...")
            scores.functional_correctness, func_metrics = (
                self.compiler.compile_and_score(candidate_code, testbench_code)
            )
            metrics["functional_correctness"] = func_metrics

        # 4. Synthesis Score
        self.log.info("Evaluating synthesis...")
        scores.synthesis = self.synthesizer.check_synthesis(candidate_code)

        # 5. Similarity Score
        self.log.info("Evaluating similarity...")
        scores.similarity = self.similarity_checker.calculate_similarity(
            candidate_code, ground_truth_code
        )

        # 6. Reasoning Score
        if candidate_reasoning:
            self.log.info("Evaluating reasoning alignment...")
            scores.reasoning = self.reasoning_checker.check_reasoning_alignment(
                candidate_reasoning, candidate_code
            )

        # Log summary
        self.log.info(f"Reward calculation complete:")
        self.log.info(f"  - Code Quality: {scores.code_quality:.2f}")
        self.log.info(f"  - Compilation: {scores.compilation:.2f}")
        self.log.info(
            f"  - Functional Correctness: {scores.functional_correctness:.2f}"
        )
        self.log.info(f"  - Synthesis: {scores.synthesis:.2f}")
        self.log.info(f"  - Similarity: {scores.similarity:.2f}")
        self.log.info(f"  - Reasoning: {scores.reasoning:.2f}")
        self.log.info(f"  - Total Reward: {scores.total_reward:.2f}")

        return scores, metrics

    def train_step(self, batch_data: List[Dict]) -> Dict:
        """
        Perform one training step with a batch of data

        Args:
            batch_data: List of dictionaries containing:
                - 'prompt': The input prompt
                - 'response': The generated response
                - 'ground_truth': The ground truth code
                - 'testbench': Optional testbench code

        Returns:
            Training metrics
        """
        rewards = []
        all_metrics = []

        for data in batch_data:
            scores, metrics = self.calculate_reward(
                data["response"], data["ground_truth"], data.get("testbench")
            )
            rewards.append(scores.total_reward)
            all_metrics.append(metrics)

        # Update policy (placeholder)
        # self.policy.update(states, actions, rewards)

        return {
            "mean_reward": sum(rewards) / len(rewards),
            "individual_metrics": all_metrics,
        }

    def _create_batches(self, data, size) -> Iterator[str]:
        """
        Create batches

        Parameters:
        -----------
        data: List
            List of data
        size: int
            Size to pair the values into

        Returns:
        --------
        List of values in {size} number of pairs
        """
        for idx in range(0, len(data), size):
            yield data[idx : idx + size]

    def _correct_filename(self, pattern: str) -> str:
        """
        Determine the correct function

        Parameters:
        -----------
        pattern:   str
            Regex like pattern (wildcard) to find the correct filename (incl. of path)

        Returns:
        --------
        Correct path is found

        Raises:
        -------
        Raises FileNotFoundError
        """
        matching_files = glob.glob(pattern)

        if matching_files:
            return matching_files[0]  # Get the first (and presumably only) match
        else:
            # Handle case where no file is found
            raise FileNotFoundError(f"Incorrect path: {pattern}")

    def fine_tune(
        self,
        model_id: str,
        model_name: str,
        batch_size: int = 2,
        epochs: int = 5,
        policy: RLPolicy = RLPolicy.PPO,
        save_to_hub: bool = True,
        wandb_project: str = "verilog-rlft",
    ):
        """
        Main function of the class, All fine-tuning starts from here
        """

        # Initialize W&B logger
        wandb_config = {
            "model_id": model_id,
            "model_name": model_name,
            "batch_size": batch_size,
            "epochs": epochs,
            "policy": policy.value,
            **TRAINING_CONFIG,
        }

        wandb_logger = WandbLogger(
            project_name=wandb_project,
            run_name=f"{model_id}_{policy.value}",
            config=wandb_config,
            tags=[model_id, policy.value, "rlft"],
        )
        wandb_logger.init()

        # Load Model
        llm_client = OpenSourceLLMClient(
            model_id=model_id, model_name=model_name, device="cuda", training_mode=True
        )

        # Connect to the model
        if not llm_client.connect():
            self.log.critical(f"Failed to connect to model: {model_name}")
            return

        # Prepare model for RLFT
        try:
            # Use actor-critic for PPO
            use_actor_critic = policy == RLPolicy.PPO

            model_engine, optimizer, lr_scheduler = llm_client.prepare_for_rlft(
                lora_r=TRAINING_CONFIG["lora_r"],
                lora_alpha=TRAINING_CONFIG["lora_alpha"],
                lora_dropout=TRAINING_CONFIG["lora_dropout"],
                gradient_accumulation_steps=TRAINING_CONFIG[
                    "gradient_accumulation_steps"
                ],
                train_batch_size=batch_size,
                zero_stage=TRAINING_CONFIG["zero_stage"],
                lr=TRAINING_CONFIG["learning_rate"],
                use_actor_critic=use_actor_critic,
            )

            # Set tokenizer on model engine
            model_engine.tokenizer = llm_client.tokenizer
        except Exception as e:
            self.log.critical(f"Failed to prepare model for RLFT: {e}")
            return

        # Dataset setup
        dataset_path = "dataset/testbench/hdlbits/"
        dataset_files_order = []
        with open(os.path.join(dataset_path, "orders.txt"), "r") as fs:
            content = fs.read()
            dataset_files_order = [
                line.strip() for line in content.splitlines() if line.strip()
            ]

        # Training metrics tracking
        best_avg_reward = -float("inf")
        total_steps = 0

        try:
            for epoch in range(epochs):
                epoch_rewards = []
                epoch_compilation_rates = []
                epoch_functional_rates = []

                for batch_idx, batch in enumerate(
                    self._create_batches(dataset_files_order, size=batch_size)
                ):
                    # Load batch data
                    paths_tb_code = [
                        self._correct_filename(
                            os.path.join(dataset_path, dirname, "tb_*.v")
                        )
                        for dirname in batch
                    ]
                    paths_gt_code = [
                        self._correct_filename(
                            os.path.join(dataset_path, dirname, "answer_*.v")
                        )
                        for dirname in batch
                    ]
                    paths_pm = [
                        self._correct_filename(
                            os.path.join(
                                dataset_path,
                                dirname,
                                f"prompt{3 - epoch if epoch < 3 else random.randint(1, 3)}_*.v",
                            )
                        )
                        for dirname in batch
                    ]

                    # Read prompts
                    prompts = []
                    for prompt_path in paths_pm:
                        with open(prompt_path, "r") as fs:
                            prompts.append(fs.read())

                    # Generate responses
                    batch_responses = llm_client.batch_generate(
                        prompts=prompts,
                        temperature=TRAINING_CONFIG["temperature"],
                        max_tokens=TRAINING_CONFIG["max_tokens"],
                        n_samples=TRAINING_CONFIG["n_samples"],
                        training_mode=True
                    )

                    # Calculate rewards
                    all_rewards = []
                    batch_reward_values = []
                    compilation_success = 0
                    functional_success = 0
                    total_samples = 0

                    for idx, response_data in enumerate(batch_responses):
                        with open(paths_tb_code[idx], "r") as fs:
                            tb_code = fs.read()
                        with open(paths_gt_code[idx], "r") as fs:
                            gt_code = fs.read()

                        rewards_for_prompt = []
                        for output in response_data["outputs"]:
                            scores, metrics = self.calculate_reward(
                                candidate_response=output,
                                ground_truth_code=gt_code,
                                testbench_code=tb_code,
                            )

                            reward_dict = {
                                "total_reward": scores.total_reward,
                                "code_quality": scores.code_quality,
                                "compilation": scores.compilation,
                                "functional_correctness": scores.functional_correctness,
                                "synthesis": scores.synthesis,
                                "similarity": scores.similarity,
                                "reasoning": scores.reasoning,
                                "metrics": metrics,
                            }

                            rewards_for_prompt.append(reward_dict)
                            batch_reward_values.append(scores.total_reward)

                            # Track success rates
                            if scores.compilation > 0:
                                compilation_success += 1
                            if scores.functional_correctness > 0:
                                functional_success += 1
                            total_samples += 1

                        all_rewards.append(rewards_for_prompt)

                    # Update policy
                    update_metrics = self.policy.update(
                        model_engine=model_engine,
                        optimizer=optimizer,
                        prompts=prompts,
                        responses=batch_responses,
                        rewards=all_rewards,
                    )

                    # Update learning rate
                    lr_scheduler.step()

                    # Calculate batch metrics
                    avg_batch_reward = sum(batch_reward_values) / len(
                        batch_reward_values
                    )
                    compilation_rate = compilation_success / total_samples
                    functional_rate = functional_success / total_samples

                    # Log to W&B
                    wandb_logger.log_batch_metrics(
                        epoch=epoch,
                        batch_idx=batch_idx,
                        avg_reward=avg_batch_reward,
                        policy_loss=update_metrics.get("policy_loss", 0),
                        value_loss=update_metrics.get("value_loss", 0),
                        rewards_dist=batch_reward_values,
                    )

                    # Track epoch metrics
                    epoch_rewards.extend(batch_reward_values)
                    epoch_compilation_rates.append(compilation_rate)
                    epoch_functional_rates.append(functional_rate)

                    # Log progress
                    self.log.info(
                        f"Epoch {epoch + 1}/{epochs}, Batch {batch_idx + 1}: "
                        f"Avg Reward: {avg_batch_reward:.4f}, "
                        f"Compilation Rate: {compilation_rate:.2%}, "
                        f"Functional Rate: {functional_rate:.2%}"
                    )

                    total_steps += 1

                    # Save checkpoint periodically
                    if total_steps % SAVE_CONFIG["save_interval"] == 0:
                        self._save_checkpoint(
                            model_engine,
                            llm_client,
                            epoch,
                            total_steps,
                            avg_batch_reward,
                            wandb_logger,
                            save_to_hub,
                        )

                # Epoch-level logging
                avg_epoch_reward = sum(epoch_rewards) / len(epoch_rewards)
                avg_compilation_rate = sum(epoch_compilation_rates) / len(
                    epoch_compilation_rates
                )
                avg_functional_rate = sum(epoch_functional_rates) / len(
                    epoch_functional_rates
                )

                wandb_logger.log_epoch_metrics(
                    epoch=epoch,
                    avg_epoch_reward=avg_epoch_reward,
                    compilation_rate=avg_compilation_rate,
                    functional_rate=avg_functional_rate,
                )

                # Save best model
                if avg_epoch_reward > best_avg_reward:
                    best_avg_reward = avg_epoch_reward
                    self._save_checkpoint(
                        model_engine,
                        llm_client,
                        epoch,
                        total_steps,
                        avg_epoch_reward,
                        wandb_logger,
                        save_to_hub,
                        is_best=True,
                    )
                    self.log.info(
                        f"New best model saved with avg reward: {best_avg_reward:.4f}"
                    )

        except Exception as err:
            self.log.critical(f"Error during fine-tuning: {err}")
            raise
        finally:
            # Final save
            self._save_checkpoint(
                model_engine,
                llm_client,
                epochs - 1,
                total_steps,
                best_avg_reward,
                wandb_logger,
                save_to_hub,
                is_final=True,
            )

            wandb_logger.finish()
            self.log.info("RLFT completed")

    def _save_checkpoint(
        self,
        model_engine,
        llm_client,
        epoch: int,
        steps: int,
        reward: float,
        wandb_logger,
        save_to_hub: bool,
        is_best: bool = False,
        is_final: bool = False,
    ):
        """Save model checkpoint"""

        # Create save directory
        checkpoint_name = f"checkpoint_epoch{epoch}_steps{steps}"
        if is_best:
            checkpoint_name = "best_model"
        elif is_final:
            checkpoint_name = "final_model"

        save_path = os.path.join(SAVE_CONFIG["save_path"], checkpoint_name)
        os.makedirs(save_path, exist_ok=True)

        # Save model and tokenizer
        model = model_engine.module
        if hasattr(model, "base_model"):
            # Save base model if using ActorCriticModel
            model.base_model.save_pretrained(save_path)
            # Also save value head
            torch.save(
                model.value_head.state_dict(), os.path.join(save_path, "value_head.pt")
            )
        else:
            model.save_pretrained(save_path)

        # Save tokenizer
        llm_client.tokenizer.save_pretrained(save_path)

        # Save training info
        info = {
            "epoch": epoch,
            "steps": steps,
            "reward": reward,
            "policy": self.policy.__class__.__name__,
        }
        with open(os.path.join(save_path, "training_info.json"), "w") as f:
            json.dump(info, f)

        self.log.info(f"Model saved to {save_path}")

        # Log to W&B
        aliases = []
        if is_best:
            aliases.append("best")
        if is_final:
            aliases.append("final")
        wandb_logger.save_model(save_path, aliases=aliases)

        # Push to HuggingFace Hub
        if save_to_hub and SAVE_CONFIG["hf_repo_name"]:
            try:
                api = HfApi()

                # Get token from environment
                hf_token = os.getenv("HUGGINGFACE_TOKEN", SAVE_CONFIG["hf_token"])

                if hf_token:
                    api.upload_folder(
                        folder_path=save_path,
                        repo_id=SAVE_CONFIG["hf_repo_name"],
                        repo_type="model",
                        token=hf_token,
                        commit_message=f"Upload {checkpoint_name} - Reward: {reward:.4f}",
                    )
                    self.log.info(
                        f"Model uploaded to HuggingFace: {SAVE_CONFIG['hf_repo_name']}"
                    )
                else:
                    self.log.warning("No HuggingFace token found, skipping upload")
            except Exception as e:
                self.log.error(f"Failed to upload to HuggingFace: {e}")
