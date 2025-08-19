"""
This file is used for evaluating closed and open source LLMs.

Proprietary LLMs:
1. [OpenAI GPT-4o](https://platform.openai.com/docs/models/gpt-4o)
2. [Claude Sonnet-4](https://www.anthropic.com/claude/sonnet)
3. [Gemini 2.5-Flash](https://deepmind.google/models/gemini/flash/)

Open-source LLMs:
1. [CodeLlama-7b-Instruct](https://huggingface.co/codellama/CodeLlama-7b-Instruct-hf)
2. [Deepseek-Coder-7b-Instruct-v1.5](https://huggingface.co/deepseek-ai/deepseek-coder-7b-instruct-v1.5)
3. [Fine-tuned-Codegen-6B-Verilog](https://huggingface.co/shailja/fine-tuned-codegen-6B-Verilog)
4. [Qwen2.5-Coder-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-Coder-7B-Instruct)

Author: Abhishek Sriram <noobsiecoder@gmail.com>
Date:   Aug 14th, 2025
Place:  Boston, MA
"""

import os
import time
import torch
from abc import ABC, abstractmethod
from typing import Dict
from src.logger import Logger
import anthropic
from google import genai
from google.genai import types
from huggingface_hub import login
from openai import OpenAI
from transformers import AutoTokenizer, AutoModelForCausalLM


SYSTEM_PROMPT = "You are a Verilog code generator. Output only Verilog code."


class ProprietaryLLM(ABC):
    """
    Base class for Proprietary LLMs like:
    1. Claude
    2. Gemini
    3. OpenAI
    """

    @abstractmethod
    def __init__(self, model: str = "claude-sonnet-4-20250514"):
        """
        Parameters:
        -----------
        model:  str
            Name of the model
            Default: Claude Sonnet 4 - Balanced performance and cost

        Returns:
        --------
        None
        """
        self.log = Logger("base_class").get_logger()
        self.model = model
        self.client = None

    @abstractmethod
    def connect(self) -> bool:
        """
        Establish connection with Claude API

        Returns:
        --------
        Bool value to confirm connection
        """
        pass

    @abstractmethod
    def generate(
        self,
        prompt: str,
        temperature: float = 0.2,
        max_tokens: int = 300,
        n_samples: int = 2,
    ) -> Dict:
        """
        Generate answer via Claude API
        prompt:         str
            Natural language description of the Verilog module to generate
        temperature:    float, default=0.2
            Controls randomness in generation (0=deterministic, 1=very random)
            Lower values produce more consistent, focused code
        max_tokens:     int, default=300
            Maximum number of tokens to generate per response
            Sufficient for most Verilog modules
        n_samples:      int, default=2
            Number of different code samples to generate
        """
        pass


class ClaudeAPIClient(ProprietaryLLM):
    """
    SDK client for Claude LLM
    """

    def __init__(self, model: str = "claude-sonnet-4-20250514"):
        """
        Parameters:
        -----------
        model:  str
            Name of the model
            Default: Claude Sonnet 4 - Balanced performance and cost

        Returns:
        --------
        None
        """
        self.log = Logger("claude_api_client").get_logger()
        self.model = model
        self.client = None

    def connect(self) -> bool:
        """
        Establish connection with Claude API

        Returns:
        --------
        Bool value to confirm connection
        """
        try:
            api_key = os.getenv("ANTHROPIC_API_KEY")
            if not api_key:
                raise ValueError("ANTHROPIC_API_KEY not found in environment")

            self.client = anthropic.Anthropic(api_key=api_key)
            self.log.info(f"Model chosen for Claude: {self.model}")
            return True
        except Exception as err:
            self.log.error(
                f"Error while establishing connection with model: {self.model} | Error: {err}"
            )
            return False

    def generate(
        self,
        prompt: str,
        temperature: float = 0.2,
        max_tokens: int = 300,
        n_samples: int = 2,
    ) -> Dict:
        """
        Generate answer via Claude API

        Parameters:
        -----------
        prompt:         str
            Natural language description of the Verilog module to generate
        temperature:    float, default=0.2
            Controls randomness in generation (0=deterministic, 1=very random)
            Lower values produce more consistent, focused code
        max_tokens:     int, default=300
            Maximum number of tokens to generate per response
            Sufficient for most Verilog modules
        n_samples:      int, default=2
            Number of different code samples to generate

        Returns:
        --------
        Dict of all responses

        Raises:
        -------
        anthropic.OpenAIError
            If API call fails (rate limits, invalid API key, etc.)
        """
        self.log.info("Running chat")

        # Container for all generated samples
        outputs = []
        start_time = time.time()

        # Generate n_samples by making multiple API calls
        # Claude API doesn't support n>1 in a single call like some other APIs
        for _ in range(n_samples):
            # Make API call to Claude
            response = self.client.messages.create(
                model=self.model,
                # System prompt specifically instructs Claude to generate Verilog
                system=SYSTEM_PROMPT,
                messages=[
                    {"role": "user", "content": [{"type": "text", "text": prompt}]},
                ],
                temperature=temperature,  # Control randomness
                max_tokens=max_tokens,  # Limit response length
            )

            # Extract text from Claude's response
            # Claude returns content as a list, we take the first text block
            outputs.append(response.content[0].text)

        end_time = time.time()
        generation_time = end_time - start_time
        # Log generation completion and timing
        self.log.info(
            f"Time taken to generate for '{prompt}': {generation_time:.2f} sec"
        )
        self.log.info("Successfully ran chat")

        # Format response to match expected structure for evaluation
        # This ensures consistency across different LLM providers
        response_data = {
            "question": prompt,  # Original prompt for reference
            "outputs": outputs,  # List of generated code samples
            "config": {  # Configuration used for generation
                "model": self.model,
                "system_instruction": SYSTEM_PROMPT,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "samples": n_samples,
            },
        }

        return response_data


class GeminiAPIClient:
    """
    SDK client for Gemini LLM
    """

    def __init__(self, model: str = "gemini-2.5-flash"):
        """
        Parameters:
        -----------
        model:  str
            Name of the model
            Default: Gemini Flash 2.5 - Optimized for speed with good performance

        Returns:
        --------
        None
        """
        self.log = Logger("gemini_api_client").get_logger()
        self.model = model
        self.client = None

    def connect(self) -> bool:
        """
        Establish connection with Gemini API

        Returns:
        --------
        Bool value to confirm connection
        """
        try:
            api_key = os.getenv("GEMINI_API_KEY")
            if not api_key:
                raise ValueError("GEMINI_API_KEY not found in environment")

            self.client = genai.Client(api_key=api_key)
            self.log.info(f"Model chosen for Gemini: {self.model}")
            return True
        except Exception as err:
            self.log.error(
                f"Error while establishing connection with model: {self.model} | Error: {err}"
            )
            return False

    def generate(
        self,
        prompt: str,
        temperature: float = 0.2,
        max_tokens: int = 300,
        n_samples: int = 2,
    ) -> Dict:
        """
        Generate answer via Gemini API

        Parameters:
        -----------
        prompt:         str
            Natural language description of the Verilog module to generate
        temperature:    float, default=0.2
            Controls randomness in generation (0=deterministic, 1=very random)
            Lower values produce more consistent, focused code
        max_tokens:     int, default=300
            Maximum number of tokens to generate per response
            Sufficient for most Verilog modules
        n_samples:      int, default=2
            Number of different code samples to generate

        Returns:
        --------
        Dict of all responses

        Raises:
        -------
        openai.OpenAIError
            If API call fails (rate limits, invalid API key, etc.)
        """
        self.log.info("Running chat")

        start_time = time.time()

        # Make single API call for generation
        response = self.client.models.generate_content(
            model=self.model,
            contents=prompt,
            config=types.GenerateContentConfig(
                # System instruction for Verilog generation
                system_instruction=SYSTEM_PROMPT,
                temperature=temperature,
                maxOutputTokens=max_tokens,  # Note: camelCase for Gemini API
                candidateCount=n_samples,  # Generate multiple candidates at once
            ),
        )

        end_time = time.time()
        generation_time = end_time - start_time
        # Log generation completion and timing
        self.log.info(
            f"Time taken to generate for '{prompt}': {generation_time:.2f} sec"
        )
        self.log.info("Successfully ran chat")

        # Format response to match expected structure
        response_data = {
            "question": prompt,
            # Extract text from each candidate response
            "outputs": [
                candidate.content.parts[0].text for candidate in response.candidates
            ],
            "config": {
                "model": self.model,
                "system_instruction": SYSTEM_PROMPT,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "samples": n_samples,
            },
        }

        return response_data


class OpenAIAPIClient:
    """
    SDK client for OpenAI LLM
    """

    def __init__(self, model: str = "gpt-4o"):
        """
        Parameters:
        -----------
        model:  str
            Name of the model
            Default: OpenAI GPT 4o

        Returns:
        --------
        None
        """
        self.log = Logger("gpt_api_client").get_logger()
        self.model = model
        self.client = None

    def connect(self) -> bool:
        """
        Establish connection with OpenAI API

        Returns:
        --------
        Bool value to confirm connection
        """
        try:
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OPENAI_API_KEY not found in environment")

            self.client = OpenAI(api_key=api_key)
            self.log.info(f"Model chosen for OpenAI: {self.model}")
            return True
        except Exception as err:
            self.log.error(
                f"Error while establishing connection with model: {self.model} | Error: {err}"
            )
            return False

    def generate(
        self,
        prompt: str,
        temperature: float = 0.2,
        max_tokens: int = 300,
        n_samples: int = 2,
    ) -> Dict:
        """
        Generate answer via OpenAI API

        Parameters:
        -----------
        prompt:         str
            Natural language description of the Verilog module to generate
        temperature:    float, default=0.2
            Controls randomness in generation (0=deterministic, 1=very random)
            Lower values produce more consistent, focused code
        max_tokens:     int, default=300
            Maximum number of tokens to generate per response
            Sufficient for most Verilog modules
        n_samples:      int, default=2
            Number of different code samples to generate

        Returns:
        --------
        Dict of all responses

        Raises:
        -------
        openai.OpenAIError
            If API call fails (rate limits, invalid API key, etc.)
        """
        self.log.info("Running chat")

        start_time = time.time()

        # Make API call to OpenAI
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    # System message instructs the model to behave as a Verilog generator
                    "content": SYSTEM_PROMPT,
                },
                {
                    "role": "user",
                    "content": prompt,  # The actual Verilog generation request
                },
            ],
            temperature=temperature,  # Control randomness
            max_tokens=max_tokens,  # Limit response length
            n=n_samples,  # Generate multiple samples at once
        )

        end_time = time.time()
        generation_time = end_time - start_time
        # Log generation completion and timing
        self.log.info(
            f"Time taken to generate for '{prompt}': {generation_time:.2f} sec"
        )

        self.log.info("Successfully ran chat")

        # Format response to match expected structure for evaluation
        # This ensures consistency across different LLM providers
        response_data = {
            "question": prompt,  # Original prompt for reference
            # Extract generated text from each completion choice
            "outputs": [choice.message.content for choice in response.choices],
            "config": {  # Configuration used for generation
                "model": self.model,
                "system_instruction": SYSTEM_PROMPT,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "samples": n_samples,
            },
        }

        return response_data


class OpenSourceLLMClient:
    """
    SDK client for all OSS LLM
    """

    def __init__(
        self,
        model_id: str = "codellama",
        model_name: str = "meta-llama/CodeLlama-7b-Instruct-hf",
        device: str = "cuda",
    ):
        """
        Initializes the tokenizer and model once for efficiency.

        Parameters:
        -----------
        model_id:   str
            Short identifier for the model (used for logging)
        model_name: str
            Full HuggingFace model path/name
        device:     str
            Device for GPU acceleration (if available)

        Notes:
        ------
        - Models are loaded in float16 precision to save memory
        - Uses device_map="auto" for automatic GPU/CPU distribution
        - First run will download the model (~13GB)
        """
        self.model_id = model_id
        self.model_name = model_name
        self.device = device

        # Initialize logger with model-specific name
        self.log = Logger(model_id).get_logger()

    def connect(self) -> bool:
        """
        Establish connection with LLM

        Returns:
        --------
        Bool value to confirm connection
        """
        # Load HuggingFace API token from environment file
        # Required for accessing gated models like CodeLlama
        try:
            api_key = os.getenv("HUGGINGFACE_API_KEY")
            if not api_key:
                raise ValueError("HUGGINGFACE_API_KEY not found in environment")

            login(token=api_key)
            self.log.info(f"Model chosen for HUGGING_FACE: {self.model_name}")
            # Load tokenizer - converts text to tokens the model understands
            self.log.info(f"Loading tokenizer from {self.model_name}...")
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
            )

            # Load the actual model weights
            self.log.info(f"Loading model from {self.model_name}...")
            self.log.info(
                "This may take several minutes on first run as it is downloading"
            )

            # Initialize model with memory-efficient settings
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16,  # Use half precision to save memory
                device_map="auto",  # Automatically distribute across available devices
                low_cpu_mem_usage=True,  # Optimize CPU memory usage during loading
            )

            self.log.info("Model loaded successfully!")
            # Set model to evaluation mode (disables dropout, etc.)
            self.model.eval()
            return True
        except Exception as err:
            self.log.error(
                f"Error while establishing connection with model: {self.model_name} | Error: {err}"
            )
            return False

    def generate(
        self,
        prompt: str,
        temperature: float = 0.2,
        max_tokens: int = 300,
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

        # Format messages in the chat template expected by the model
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]

        # Convert messages to model input format
        # apply_chat_template formats the conversation according to model's expectations
        if self.model_id == "verigen-finetuned":
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
            ).to(self.device)
        else:
            inputs = self.tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,  # Add the prompt that signals the model to generate
                tokenize=True,  # Convert to tokens
                return_dict=True,  # Return as dictionary
                return_tensors="pt",  # Return PyTorch tensors
            ).to(self.device)  # Move to same device as model

        # Count input tokens
        input_token_count = inputs["input_ids"].shape[-1]

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

        # Calculate output tokens for each sample
        tokens_per_sample = []
        total_output_tokens = 0

        for output in outputs:
            # Count tokens in this sample (excluding input tokens)
            output_tokens_in_sample = len(output) - input_token_count
            tokens_per_sample.append(output_tokens_in_sample)
            total_output_tokens += output_tokens_in_sample

        # Log generation completion and timing
        self.log.info(
            f"Time taken to generate for '{prompt}': {generation_time:.2f} sec"
        )
        self.log.info(f"Generating response: {self.model_id} ...")

        # Decode generated tokens back to text
        responses = {
            "question": prompt,
            "outputs": [
                self.tokenizer.decode(
                    output[inputs["input_ids"].shape[-1] :],
                    skip_special_tokens=True,
                )
                for output in outputs
            ],
            "config": {
                "model": self.model_name,
                "system_instruction": SYSTEM_PROMPT,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "samples": n_samples,
            },
            "time": generation_time,
            "input_tokens": input_token_count,
            "output_tokens": total_output_tokens,
            "tokens_per_sample": tokens_per_sample,
            "avg_tokens_per_sample": total_output_tokens / n_samples
            if n_samples > 0
            else 0,
        }

        return responses
