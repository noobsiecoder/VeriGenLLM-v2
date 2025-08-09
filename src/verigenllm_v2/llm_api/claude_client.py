"""
Claude Client SDK

This module provides a wrapper around Anthropic's Claude API for generating
Verilog code. It handles API communication and formats responses consistently
with other LLM clients in the evaluation framework.

Author: Abhishek Sriram <noobsiecoder@gmail.com>
Date:   Aug 1st, 2025

Note: LLM was used to generate comments
"""

from enum import Enum
from typing import Dict
from verigenllm_v2.utils.logger import Logger
import anthropic


class ClaudeModels(Enum):
    """
    Contains necessary Claude model names

    Enum provides a clean way to manage model versions and ensures
    consistent model naming across the codebase. New models can be
    added here as they become available.
    """

    # Claude Opus 4 - Most capable model for complex tasks
    Opus_4 = "claude-opus-4-20250514"

    # Claude Sonnet 4 - Balanced performance and cost
    Sonnet_4 = "claude-sonnet-4-20250514"


class ClaudeClient:
    """
    Claude client SDK connector class

    This class provides a unified interface for interacting with Claude's API,
    specifically tailored for Verilog code generation tasks. It handles:
    - API authentication
    - Request formatting
    - Multiple sample generation
    - Response standardization
    """

    def __init__(self, api_key: str, model: str) -> None:
        """
        Initialize Claude client and establish connection and set model name

        Parameters:
        -----------
        api_key : str
            Anthropic API key for authentication
        model : str
            Model identifier (e.g., ClaudeModels.Opus_4.value)

        Returns:
        --------
        None

        Notes:
        ------
        The API key should be kept secure and not hardcoded in the source.
        """
        # Initialize the Anthropic client with authentication
        self.client = anthropic.Anthropic(api_key=api_key)

        # Store the model identifier for use in API calls
        self.model = model

        # Initialize logger with client-specific name for tracking
        self.log = Logger("claude_client").get_logger()

    def generate(
        self,
        prompt: str,
        temperature: float = 0.2,
        max_tokens: int = 1024,
        n_samples: int = 2,
    ) -> Dict:
        """
        Generate output from prompt

        This method generates multiple Verilog code samples for a given prompt
        by making multiple API calls to Claude. Each call produces one sample.

        Parameters:
        -----------
        prompt : str
            Natural language description of the Verilog module to generate
        temperature : float, default=0.2
            Controls randomness in generation (0=deterministic, 1=very random)
            Lower values produce more consistent, focused code
        max_tokens : int, default=1024
            Maximum number of tokens to generate per response
            Sufficient for most Verilog modules
        n_samples : int, default=2
            Number of different code samples to generate

        Returns:
        --------
        Dict
            Response dictionary containing:
            - question: The original prompt
            - outputs: List of generated Verilog code samples
            - config: Generation configuration used

        Raises:
        -------
        anthropic.APIError
            If API call fails (rate limits, network issues, etc.)
        """
        self.log.info("Running chat")

        # Container for all generated samples
        outputs = []

        # Generate n_samples by making multiple API calls
        # Claude API doesn't support n>1 in a single call like some other APIs
        for _ in range(n_samples):
            # Make API call to Claude
            response = self.client.messages.create(
                model=self.model,
                # System prompt specifically instructs Claude to generate Verilog
                # "synthesizable" ensures the code can be converted to hardware
                system="You are a Verilog code generator. Output only synthesizable Verilog code.",
                messages=[
                    {"role": "user", "content": [{"type": "text", "text": prompt}]},
                ],
                temperature=temperature,  # Control randomness
                max_tokens=max_tokens,  # Limit response length
            )

            # Extract text from Claude's response
            # Claude returns content as a list, we take the first text block
            outputs.append(response.content[0].text)

        self.log.info("Successfully ran chat")

        # Format response to match expected structure for evaluation
        # This ensures consistency across different LLM providers
        response_data = {
            "question": prompt,  # Original prompt for reference
            "outputs": outputs,  # List of generated code samples
            "config": {  # Configuration used for generation
                "model": self.model,
                "system_instruction": "You are a Verilog code generator. Output only synthesizable Verilog code.",
                "temperature": temperature,
                "max_tokens": max_tokens,
                "samples": n_samples,
            },
        }

        return response_data
