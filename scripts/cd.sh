#!/bin/bash
# This script aids in:
#   2) Running more tests from tests/ for LLM (loaded in GPU)
#   3) Runner for RLFT
#
# Author: Abhishek Sriram <noobsiecoder@gmail.com>
# Date:   Aug 20th, 2025
# Place:  Boston, MA
set -e

# Step 1: Run few more tests to check if LLM loads in cloud instance
echo "=== Running LLM Sepecific Tests ==="
uv run pytest -v \
    tests/test_llm_prompts.py::test_prompt_extraction \
    tests/test_models.py

# Step 2: Run RLFT
# TODO: insert command(s) here ...
echo "=== Running RLFT ==="
