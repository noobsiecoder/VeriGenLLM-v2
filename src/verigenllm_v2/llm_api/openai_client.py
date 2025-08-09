"""
OpenAI Client SDK

This module provides a wrapper around OpenAI's API for generating
Verilog code. It supports both single-prompt and batch generation
using OpenAI's chat completion models.

Author: Abhishek Sriram <noobsiecoder@gmail.com>
Date:   Jul 31st, 2025

Note: LLM was used to generate comments
"""

from enum import Enum
from typing import Dict, List
from verigenllm_v2.utils.logger import Logger
from openai import OpenAI


class OpenAIModels(Enum):
    """
    Contains necessary OpenAI model names

    OpenAI offers different GPT-4 variants with varying capabilities:
    - GPT-4.1: Latest flagship model with best performance
    - GPT-4o: Optimized version with improved efficiency
    - GPT-4o-mini: Smaller, faster variant for simpler tasks
    """

    # GPT-4.1 - Latest and most capable model
    GPT_4_1 = "gpt-4.1"

    # GPT-4o - Optimized version balancing performance and speed
    GPT_4o = "gpt-4o"

    # GPT-4o-mini - Lightweight version for faster, simpler tasks
    GPT_4o_mini = "gpt-4o-mini"


class OpenAIClient:
    """
    OpenAI client SDK connector class

    This class provides a unified interface for interacting with OpenAI's API,
    specifically tailored for Verilog code generation tasks. It handles:
    - API authentication
    - Chat completion formatting
    - Multiple sample generation
    - Batch processing of multiple prompts
    """

    def __init__(self, api_key: str, model: str) -> None:
        """
        Initialize OpenAI client and establish connection and set model name

        Parameters:
        -----------
        api_key : str
            OpenAI API key for authentication
        model : str
            Model identifier (e.g., OpenAIModels.GPT_4_1.value)

        Returns:
        --------
        None

        Notes:
        ------
        The API key should be stored securely in environment variables
        or a secrets management system, not hardcoded.
        """
        # Initialize the OpenAI client with authentication
        self.client = OpenAI(api_key=api_key)

        # Store model identifier for API calls
        self.model = model

        # Initialize logger for tracking operations
        self.log = Logger("open_ai_client").get_logger()

    def batch_generate(
        self,
        prompts: List[str],
        temperature: float = 0.2,
        max_tokens: int = 1024,
        n_samples: int = 2,
    ) -> List[Dict]:
        """
        Generate outputs from a list of prompts

        This method processes multiple prompts sequentially. Unlike Gemini's
        batch API, OpenAI doesn't have a native batch endpoint, so this
        method simply loops through prompts calling generate() for each.

        Parameters:
        -----------
        prompts : List[str]
            List of Verilog module descriptions to generate code for
        temperature : float, default=0.2
            Controls randomness (0=deterministic, 1=very random)
            Lower values produce more consistent Verilog code
        max_tokens : int, default=1024
            Maximum tokens per response
        n_samples : int, default=2
            Number of different code samples per prompt

        Returns:
        --------
        List[Dict]
            List of response dictionaries, one per prompt

        Notes:
        ------
        For large batches, consider implementing rate limiting to avoid
        hitting OpenAI's API rate limits.
        """
        results = []

        # Process each prompt individually
        for prompt in prompts:
            # Generate response for current prompt
            result = self.generate(prompt, temperature, max_tokens, n_samples)
            results.append(result)

        return results

    def generate(
        self,
        prompt: str,
        temperature: float = 0.2,
        max_tokens: int = 1024,
        n_samples: int = 2,
    ) -> Dict:
        """
        Generate output from prompt

        This method generates Verilog code for a single prompt using
        OpenAI's chat completion API. It can generate multiple samples
        in a single API call using the 'n' parameter.

        Parameters:
        -----------
        prompt : str
            Natural language description of the Verilog module to generate
        temperature : float, default=0.2
            Controls randomness in generation
            - 0.0: Deterministic (same input → same output)
            - 0.2: Slight variation while maintaining coherence
            - 1.0: Maximum creativity/randomness
        max_tokens : int, default=1024
            Maximum number of tokens to generate
            Sufficient for most Verilog modules
        n_samples : int, default=2
            Number of different completions to generate
            OpenAI can generate multiple samples in one API call

        Returns:
        --------
        Dict
            Response dictionary containing:
            - question: The original prompt
            - outputs: List of generated Verilog code samples
            - config: Generation configuration used

        Raises:
        -------
        openai.OpenAIError
            If API call fails (rate limits, invalid API key, etc.)
        """
        self.log.info("Running chat")

        # Make API call to OpenAI
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    # System message instructs the model to behave as a Verilog generator
                    # "synthesizable" ensures the generated code can be converted to hardware
                    "content": "You are a Verilog code generator. Output only synthesizable Verilog code.",
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

        self.log.info("Successfully ran chat")

        # Format response to match expected structure for evaluation
        # This ensures consistency across different LLM providers
        response_data = {
            "question": prompt,  # Original prompt for reference
            # Extract generated text from each completion choice
            "outputs": [choice.message.content for choice in response.choices],
            "config": {  # Configuration used for generation
                "model": self.model,
                "system_instruction": "You are a Verilog code generator. Output only synthesizable Verilog code.",
                "temperature": temperature,
                "max_tokens": max_tokens,
                "samples": n_samples,
            },
        }

        return response_data
