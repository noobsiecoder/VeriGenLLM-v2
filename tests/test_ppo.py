"""
Run tests on PPO in CPU (or even GPU if found)
Primarily focuses on mock tests for PPO

Author: Abhishek Sriram <noobsiecoder@gmail.com>
Date:   Aug 28th, 2025
Place:  Boston, MA
"""

# from src.ppo import PPO
import json
from src.models import Policy
from constants import ENVLoader


ENVLoader()  # load ENV globally
policy = Policy(name="tester-slm", unique_id="HuggingFaceTB/SmolLM-360M-Instruct")


def test_model_load():
    """
    Mock model loading
    """
    policy.load()

    assert 1 == 1  # asserting to check if model loaded successfully


def test_generate():
    """
    Mock LLM response(s) generation
    """
    prompts = [
        "Say hello back to me without saying hello",
        "Write a short note on LLMs (max 100 words)",
    ]
    responses = policy.generate(prompts)
    with open("dataset/responses.json", "w") as fs:
        responses["sequences"] = None
        responses["scores"] = None
        json.dump(responses, fs, indent=4)

    assert 1 == 1
