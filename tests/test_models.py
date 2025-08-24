"""
Testbench to check all Model Client SDK

Author: Abhishek Sriram <noobsiecoder@gmail.com>
Date:   Aug 14th, 2025
Place:  Boston, MA
"""

import re
from dotenv import load_dotenv
import pytest
from src.logger import Logger
from src.models import (
    ClaudeAPIClient,
    GeminiAPIClient,
    OpenAIAPIClient,
    OpenSourceLLMClient,
)

log = Logger("test_models").get_logger()

claude_api = ClaudeAPIClient()
gemini_api = GeminiAPIClient()
openai_api = OpenAIAPIClient()


@pytest.mark.skip(reason="Already tested")
def test_api_connection_with_no_env_load():
    """
    Check connection status
    """
    res = claude_api.connect()
    assert not res  # returns false

    # res = gemini_api.connect()
    # assert not res  # returns false

    res = openai_api.connect()
    assert not res  # returns false


@pytest.mark.skip(reason="Already tested")
def test_api_connection_with_env_load():
    """
    Check connection status
    """
    load_dotenv("secrets/models-api.env")

    res = claude_api.connect()
    assert res  # should connect now and return true

    # res = gemini_api.connect()
    # assert res  # should connect now and return true

    res = openai_api.connect()
    assert res  # should connect now and return true


def test_prompting(skip_non_oss_llm: bool = True):
    """
    Checking answers
    """
    load_dotenv("secrets/models-api.env")  # Load ENV
    prompt = """
    // This is a module that assigns the output to the input
    module wire_assign( input in, output out );
    // implement simple wire
    // insert code here
    """  # Prompt
    samples = 4  # Number of samples to generate per prompt

    def is_only_code_block(s: str, lang: str = "verilog") -> bool:
        """
        Check if string is only a code block of a given language.
        If lang is None, any language is allowed.
        """
        if lang:
            pattern = rf"^```{lang}\n[\s\S]*\n```$"
        else:
            pattern = r"^```[a-zA-Z0-9]*\n[\s\S]*\n```$"

        return bool(re.match(pattern, s.strip()))

    # Skips Proprietary LLMs
    if not skip_non_oss_llm:
        log.info("Skipping OPEN AI LLM")
        res = openai_api.connect()
        if res:
            responses = openai_api.generate(
                prompt, temperature=0.2, max_tokens=300, n_samples=samples
            )
            assert len(responses["outputs"]) == samples
            for output in responses["outputs"]:
                assert len(output) != 0
                assert is_only_code_block(output)

    # Test with all open-source LLM
    # Define models to evaluate
    # Each model has an ID (for file naming) and full HuggingFace path
    models = [
        {"id": "deepseek-coder", "name": "deepseek-ai/deepseek-coder-7b-instruct-v1.5"},
        {"id": "qwen-coder", "name": "Qwen/Qwen2.5-Coder-7B-Instruct"},
    ]

    # Process each model
    for model_info in models:
        # Initialize the model
        oss_llm_api = OpenSourceLLMClient(
            model_id=model_info["id"], model_name=model_info["name"]
        )
        res = oss_llm_api.connect()
        if res:
            responses = oss_llm_api.generate(
                prompt, temperature=0.2, max_tokens=300, n_samples=samples
            )
            assert len(responses["outputs"]) == samples
            for output in responses["outputs"]:
                assert len(output) != 0
                assert is_only_code_block(output)
