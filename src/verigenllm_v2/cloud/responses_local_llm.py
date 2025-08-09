"""
Run OSS LLM models to get all responses.

This file targets four LLMs:
1. [CodeLlama-7b-instruct](https://huggingface.co/codellama/CodeLlama-7b-Instruct-hf)
2. [Deepseek-coder-7b-instruct-v1.5](https://huggingface.co/deepseek-ai/deepseek-coder-7b-instruct-v1.5)
3. [Codegen-6B-nl](https://huggingface.co/Salesforce/codegen-6B-nl)
4. [Qwen2.5-Coder-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-Coder-7B-Instruct)

Author: Abhishek Sriram <noobsiecoder@gmail.com>
Date:   Aug 7th, 2025

Note: LLM was used to generate comments
"""

import json
import re
import os
import time
import torch
from dotenv import load_dotenv
from huggingface_hub import login
from verigenllm_v2.utils.logger import Logger
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import Dict, List
from verigenllm_v2.cloud.gcp_storage import upload_file_to_gcs


class LLMData:
    """
    This class is responsible to prepare instruction/prompts for the LLMs

    Handles loading and managing the evaluation prompts that will be
    sent to various LLMs for Verilog code generation.
    """

    def __init__(self):
        """
        Initialize the LLMData class with empty prompts list and dataset path
        """
        self.prompts = []  # Container for all Verilog generation prompts
        self.dataset_path = "dataset"  # Base path for dataset files

    def set_prompts(self) -> None:
        """
        Load prompts from the selected-prompts.json file

        This file contains the curated list of Verilog problems/prompts
        extracted from evaluation datasets.

        Raises:
        -------
        FileNotFoundError
            If selected-prompts.json doesn't exist
        json.JSONDecodeError
            If the JSON file is malformed
        """
        # Construct the full path to the prompts file
        prompts_file = os.path.join(self.dataset_path, "selected-prompts.json")

        # Load prompts from JSON file
        with open(prompts_file, "r") as fs:
            self.prompts = json.load(fs)

        # Verify prompts were loaded successfully
        if len(self.prompts) != 0:
            log.info(f"All prompts loaded - size: {len(self.prompts)}")
        else:
            log.warning("Prompts not loaded")

    def get_prompts(self) -> List[str]:
        """
        Return the loaded prompts

        Returns:
        --------
        List[str]
            List of Verilog generation prompts
        """
        return self.prompts


class LLMs:
    """
    Wrapper for all models.
    Allows generating responses to prompts.

    This class handles model initialization, tokenization, and generation
    for various open-source LLMs from HuggingFace.
    """

    def __init__(
        self,
        model_id: str = "codellama",
        model_name: str = "meta-llama/CodeLlama-7b-Instruct-hf",
    ):
        """
        Initializes the tokenizer and model once for efficiency.

        Parameters:
        -----------
        model_id : str
            Short identifier for the model (used for logging)
        model_name : str
            Full HuggingFace model path/name

        Notes:
        ------
        - Models are loaded in float16 precision to save memory
        - Uses device_map="auto" for automatic GPU/CPU distribution
        - First run will download the model (~13GB)
        """
        self.model_id = model_id
        self.model_name = model_name

        # Initialize logger with model-specific name
        self.log = Logger(model_id).get_logger()

        # Load HuggingFace API token from environment file
        # Required for accessing gated models like CodeLlama
        load_dotenv("secrets/models-api.env")
        token = os.environ.get("HUGGINGFACE_TOKEN")
        if token:
            login(token=token)
            self.log.info("Successfully logged in to HuggingFace")

        # Load tokenizer - converts text to tokens the model understands
        self.log.info(f"Loading tokenizer from {model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
        )

        # Load the actual model weights
        self.log.info(f"Loading model from {model_name}...")
        self.log.info(
            "This may take several minutes on first run as it downloads ~13GB"
        )

        # Initialize model with memory-efficient settings
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,  # Use half precision to save memory
            device_map="auto",  # Automatically distribute across available devices
            low_cpu_mem_usage=True,  # Optimize CPU memory usage during loading
        )

        self.log.info("Model loaded successfully!")
        # Set model to evaluation mode (disables dropout, etc.)
        self.model.eval()

    def generate(
        self,
        prompt: str,
        temperature: float = 0.2,
        max_tokens: int = 1024,
        n_samples: int = 2,
    ) -> Dict:
        """
        Generates responses for a given prompt using the loaded LLM.

        Parameters:
        -----------
        prompt : str
            Input prompt describing the Verilog module to generate
        temperature : float
            Sampling temperature (lower = more deterministic, higher = more creative)
        max_tokens : int
            Maximum number of new tokens to generate
        n_samples : int
            Number of different outputs to generate for the same prompt

        Returns:
        --------
        Dict
            Dictionary containing:
            - question: The input prompt
            - outputs: List of generated Verilog code samples
            - config: Generation configuration used
        """
        # System prompt to guide the model's behavior
        # This helps ensure the model generates only Verilog code
        system_prompt = (
            "You are a Verilog code generator. Output only synthesizable Verilog code."
        )

        # Format messages in the chat template expected by the model
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ]

        # Convert messages to model input format
        # apply_chat_template formats the conversation according to model's expectations
        inputs = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,  # Add the prompt that signals the model to generate
            tokenize=True,  # Convert to tokens
            return_dict=True,  # Return as dictionary
            return_tensors="pt",  # Return PyTorch tensors
        ).to(self.model.device)  # Move to same device as model

        # Log generation start
        self.log.info(f"Running chat: {self.model_id} ...")
        start_time = time.time()

        # Generate responses
        # Uses sampling with temperature to create diverse outputs
        outputs = self.model.generate(
            **inputs,  # Unpack input tensors
            do_sample=True,  # Enable sampling (vs greedy decoding)
            temperature=temperature,  # Control randomness
            max_new_tokens=max_tokens,  # Limit response length
            num_return_sequences=n_samples,  # Generate multiple samples
        )

        end_time = time.time()
        generation_time = end_time - start_time

        # Log generation completion and timing
        self.log.info(
            f"Time taken to generate for '{prompt}': {generation_time:.2f} sec"
        )
        self.log.info(f"Generating response: {self.model_id} ...")

        # Decode generated tokens back to text
        # Only decode the newly generated tokens (skip the input prompt)
        responses = {
            "question": prompt,
            "outputs": [
                self.tokenizer.decode(
                    output[inputs["input_ids"].shape[-1] :],  # Skip input tokens
                    skip_special_tokens=True,  # Remove special tokens like <s>, </s>
                )
                for output in outputs
            ],
            "config": {
                "model": self.model_name,
                "system_instruction": system_prompt,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "samples": n_samples,
            },
        }

        return responses


class CodeGenModel:
    """
    Placeholder for CodeGen model implementation

    This class would handle the Salesforce CodeGen models
    which have a different architecture and generation approach.
    """

    pass


if __name__ == "__main__":
    # Initialize global logger for main execution
    log = Logger("local_llm").get_logger()

    # Load evaluation prompts
    llm_data = LLMData()
    llm_data.set_prompts()
    prompts = llm_data.get_prompts()

    # Define models to evaluate
    # Each model has an ID (for file naming) and full HuggingFace path
    models = [
        {"id": "code-llama", "name": "meta-llama/CodeLlama-7b-Instruct-hf"},
        {"id": "deepseek-coder", "name": "deepseek-ai/deepseek-coder-7b-instruct-v1.5"},
        {"id": "qwen-coder", "name": "Qwen/Qwen2.5-Coder-7B-Instruct"},
    ]

    # Process each model
    for model_info in models:
        # Number of samples to generate per prompt
        samples = 10

        # Initialize the model
        model = LLMs(model_id=model_info["id"], model_name=model_info["name"])

        # Container for all responses from this model
        responses = []

        print(f"For model: {model_info['name']}")
        start_time = time.time()

        # Generate responses for each prompt
        for prompt in prompts:
            log.info(
                f"Model: {model_info['name']} | Question: {prompt} | Sample: {samples}"
            )
            # Generate n_samples responses for this prompt
            responses.append(model.generate(prompt, n_samples=samples))

        end_time = time.time()
        total_time = end_time - start_time

        # Log total generation time for all prompts
        log.info(f"Total time for {model_info['name']}: {total_time:.2f} sec")
        print(f"Total time for {model_info['name']}: {total_time:.2f} sec")

        # Save responses locally
        output_filename = f"{model_info['id']}-response-n{samples}.json"
        with open(output_filename, "w") as fs:
            json.dump(responses, fs, indent=4, ensure_ascii=False)
            log.info(f"Dumped data into: {output_filename}")

        # Upload results to Google Cloud Storage for backup/sharing
        upload_file_to_gcs(
            local_file_path=output_filename,
            bucket_name="verilog-llm-evals",
            blob_name=f"results/{output_filename}",
        )
        print(f"Uploaded {model_info['id']} results to Storage Bucket")

        # Upload associated log files
        # Find all log files for this specific model
        log_files = os.listdir("logs/")
        for log_filename in log_files:
            # Match log files containing the model ID
            match = re.search(
                r"{}(.*)".format(model_info["id"]), log_filename, re.DOTALL
            )
            if match:
                # Upload matching log file to GCS
                upload_file_to_gcs(
                    local_file_path=f"logs/{match.group(0)}",
                    bucket_name="verilog-llm-evals",
                    blob_name=f"logs/{match.group(0)}",
                )
                print(f"Uploaded {match.group(0)} log to Storage Bucket")
