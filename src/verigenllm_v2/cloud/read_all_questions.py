"""
Read all questions used by Claude and OpenAI models

This script extracts question prompts from the Claude model's response file
and saves them to a separate JSON file for further analysis or reuse.

Author: Abhishek Sriram <noobsiecoder@gmail.com>
Date:   Aug 7th, 2025

Note: LLM was used to generate comments
"""

import json
from typing import List


def read_all_questions() -> List[str]:
    """
    Extract all question prompts from Claude's response file

    This function reads the Claude model's response JSON file and extracts
    all the question/prompt fields from each response entry.

    Returns:
    --------
    List[str]
        A list containing all question prompts used in the Claude evaluation

    Raises:
    -------
    FileNotFoundError
        If the Claude response file doesn't exist
    json.JSONDecodeError
        If the JSON file is malformed
    KeyError
        If the expected 'question' field is missing from any entry
    """
    # Initialize empty list to store all prompts
    prompts = []

    # Open and read the Claude model's response file (same throughout all LLM response data)
    # This file contains all responses from Claude for the evaluation dataset
    with open("dataset/models/claude/claude-response-n10.json", "r") as fs:
        # Parse the JSON file into a Python data structure
        # Expected format: List of dictionaries, each containing a 'question' field
        metadata = json.load(fs)

        # Iterate through each response entry in the metadata
        for info in metadata:
            # Extract the question/prompt that was sent to Claude
            # Each entry should have a 'question' field containing the Verilog problem prompt
            prompt = info["question"]

            # Add the prompt to our collection
            prompts.append(prompt)

    return prompts


def write_to_json_file(prompts: List[str]):
    """
    Save the extracted prompts to a new JSON file

    This function writes all collected prompts to a separate JSON file
    for easy access and potential reuse with other models.

    Parameters:
    -----------
    prompts : List[str]
        List of question prompts to save

    Notes:
    ------
    - The output file is saved with proper indentation for readability
    - Overwrites any existing file at the destination path
    """
    # Save the prompts to a new JSON file in the models directory
    # This creates a consolidated list of all prompts used in the evaluation
    with open("dataset/models/selected-prompts.json", "w") as fs:
        # Write with indentation for human readability
        # indent=4 creates nicely formatted JSON with 4-space indentation
        json.dump(prompts, fs, indent=4)


# Main execution block - runs only when script is executed directly
if __name__ == "__main__":
    # Step 1: Read all questions from Claude's response file
    prompts = read_all_questions()

    # Step 2: Save the extracted prompts to a separate file
    # This creates a reusable prompt dataset that can be used for:
    # - Evaluating other models with the same prompts
    # - Analysis of prompt characteristics
    # - Ensuring consistency across model evaluations
    write_to_json_file(prompts)
