"""
Gemini Client SDK

This module provides a wrapper around Google's Gemini API for generating
Verilog code. It supports both single-request generation and batch processing
for handling multiple prompts efficiently.

Author: Abhishek Sriram <noobsiecoder@gmail.com>
Date:   Jul 31st, 2025

Note: LLM was used to generate comments
"""

import json
import tempfile
import time
from enum import Enum
from typing import Dict, List
from verigenllm_v2.utils.logger import Logger
from google import genai
from google.genai import types
from google.api_core.exceptions import GoogleAPICallError


class GeminiModels(Enum):
    """
    Contains necessary Gemini model names

    Google offers different tiers of Gemini models with varying
    capabilities and costs:
    - Pro: Most capable, highest cost
    - Flash: Balanced performance and speed
    - Flash Lite: Fastest, most economical
    """

    # Gemini 2.5 Pro - Most capable model for complex tasks
    GEMINI_2_5_PRO = "gemini-2.5-pro"

    # Gemini 2.5 Flash - Optimized for speed with good performance
    GEMINI_2_5_FLASH = "gemini-2.5-flash"

    # Gemini 2.5 Flash Lite - Lightweight, fast, cost-effective
    GEMINI_2_5_FLASH_LITE = "gemini-2.5-flash-lite"


class GeminiClient:
    """
    Gemini client SDK connector class

    This class provides interfaces for both single-request and batch
    processing with Google's Gemini API. Batch processing is particularly
    useful for evaluating multiple prompts efficiently.
    """

    def __init__(self, api_key: str, model: str) -> None:
        """
        Initialize Gemini client and establish connection and set model name

        Parameters:
        -----------
        api_key : str
            Google Cloud API key for authentication
        model : str
            Model identifier (e.g., GeminiModels.GEMINI_2_5_PRO.value)

        Returns:
        --------
        None
        """
        # Initialize the Google GenAI client with authentication
        self.client = genai.Client(api_key=api_key)

        # Store model identifier for API calls
        self.model = model

        # Initialize logger for tracking operations
        self.log = Logger("gemini_client").get_logger()

    def submit_batch_job(
        self,
        prompts: List[str],
        job_name: str = "verilog-batch-job",
        poll_interval: int = 5,
        temperature: float = 0.2,
        max_tokens: int = 1024,
        n_samples: int = 2,
    ) -> List[Dict]:
        """
        Submit and retrieve results from a Gemini batch job

        Batch processing is more efficient for multiple prompts as it:
        - Reduces API call overhead
        - Allows parallel processing on Google's servers
        - Provides better rate limit management

        Parameters:
        -----------
        prompts : List[str]
            List of Verilog generation prompts to process
        job_name : str, default="verilog-batch-job"
            Display name for tracking the batch job
        poll_interval : int, default=5
            Seconds between status checks while job is running
        temperature : float, default=0.2
            Controls randomness (0=deterministic, 1=very random)
        max_tokens : int, default=1024
            Maximum tokens per generated response
        n_samples : int, default=2
            Number of different outputs per prompt

        Returns:
        --------
        List[Dict]
            List of response dictionaries, one per prompt

        Raises:
        -------
        RuntimeError
            If batch job creation fails or job doesn't complete successfully
        """

        # Step 1: Create a JSONL file with the batch requests
        # Gemini's batch API requires requests in JSONL format (one JSON object per line)
        batch_requests = []

        # System instruction to guide Verilog generation
        system_instruction = (
            "You are a Verilog code generator. Output only synthesizable Verilog code."
        )

        # Format each prompt as a batch request
        for i, prompt in enumerate(prompts):
            # Create request object following Gemini's batch API format
            request_obj = {
                "key": f"request_{i}",  # Unique identifier to match responses
                "request": {
                    "contents": [{"parts": [{"text": prompt}], "role": "user"}],
                    "system_instruction": {"parts": [{"text": system_instruction}]},
                    "generation_config": {
                        "temperature": temperature,
                        "max_output_tokens": max_tokens,
                        "candidate_count": n_samples,  # Multiple outputs per prompt
                    },
                },
            }
            # Convert to JSON string for JSONL format
            batch_requests.append(json.dumps(request_obj))

        # Write requests to a temporary JSONL file
        # Using tempfile ensures proper cleanup and avoids file conflicts
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            for request in batch_requests:
                f.write(request + "\n")  # One JSON object per line
            temp_filename = f.name

        self.log.info(f"Created temporary JSONL file: {temp_filename}")

        try:
            # Step 2: Upload the JSONL file to Google's servers
            # Files must be uploaded before they can be used in batch jobs
            uploaded_file = self.client.files.upload(
                file=temp_filename,
                config=types.UploadFileConfig(
                    display_name=f"{job_name}_input",
                    mime_type="application/json",  # JSONL uses JSON mime type
                ),
            )
            self.log.info(f"Uploaded file: {uploaded_file.name}")

            # Step 3: Create the batch job with the uploaded file
            batch_job = self.client.batches.create(
                model=f"models/{self.model}",  # Format model name for API
                src=uploaded_file.name,  # Reference uploaded file
                config={
                    "display_name": job_name,  # Human-readable job name
                },
            )
            job_name = batch_job.name
            self.log.info(f"Batch job submitted: {job_name}")

        except Exception as e:
            # Log detailed error information for debugging
            self.log.error(f"Failed to create batch job: {e}")
            import traceback

            traceback.print_exc()
            raise RuntimeError(f"Failed to create batch job: {e}")
        finally:
            # Always clean up temporary file, even if errors occur
            import os

            if os.path.exists(temp_filename):
                os.remove(temp_filename)

        # Step 4: Poll until job finishes
        # Batch jobs are asynchronous, so we need to check status periodically
        completed_states = {
            "JOB_STATE_SUCCEEDED",  # Job completed successfully
            "JOB_STATE_FAILED",  # Job failed with error
            "JOB_STATE_CANCELLED",  # Job was cancelled
            "JOB_STATE_PAUSED",  # Job was paused (unusual)
        }

        while True:
            # Get current job status
            job_status = self.client.batches.get(name=job_name)

            # Handle both string states and enum states for compatibility
            state = job_status.state
            if hasattr(state, "name"):
                state = state.name

            self.log.info(f"Batch job status: {state}")

            # Check if job has reached a terminal state
            if state in completed_states:
                if state != "JOB_STATE_SUCCEEDED":
                    # Job failed - raise error with details
                    error_msg = f"Batch job {job_name} failed with state: {state}"
                    self.log.error(error_msg)
                    raise RuntimeError(error_msg)
                break  # Job succeeded

            # Wait before checking again to avoid excessive API calls
            time.sleep(poll_interval)

        # Step 5: Retrieve and process results
        final_outputs = []

        # Get the output file containing results
        if hasattr(job_status, "dest") and hasattr(job_status.dest, "file_name"):
            output_file_name = job_status.dest.file_name
            self.log.info(f"Downloading results from: {output_file_name}")

            # Download the results file from Google's servers
            file_content_bytes = self.client.files.download(file=output_file_name)
            file_content = file_content_bytes.decode("utf-8")

            # Parse the JSONL results (one JSON object per line)
            for line in file_content.splitlines():
                if line.strip():  # Skip empty lines
                    result_obj = json.loads(line)
                    key = result_obj.get("key", "")
                    response = result_obj.get("response", {})

                    # Extract the prompt index from the key to match with original prompt
                    prompt_idx = int(key.split("_")[1]) if "_" in key else 0

                    # Extract generated text from all candidates
                    outputs = []
                    if "candidates" in response:
                        for candidate in response["candidates"]:
                            # Navigate nested response structure
                            if (
                                "content" in candidate
                                and "parts" in candidate["content"]
                            ):
                                for part in candidate["content"]["parts"]:
                                    if "text" in part:
                                        outputs.append(part["text"])

                    # Format response to match expected structure
                    final_outputs.append(
                        {
                            "question": prompts[prompt_idx]
                            if prompt_idx < len(prompts)
                            else f"Prompt {prompt_idx}",
                            "outputs": outputs,
                            "config": {
                                "model": self.model,
                                "system_instruction": system_instruction,
                                "temperature": temperature,
                                "max_tokens": max_tokens,
                                "samples": n_samples,
                            },
                        }
                    )
        else:
            # No output file found - this shouldn't happen for successful jobs
            raise RuntimeError(
                "Unable to retrieve batch job results - no output file found"
            )

        return final_outputs

    def generate(
        self,
        prompt: str,
        temperature: float = 0.2,
        max_tokens: int = 1024,
        n_samples: int = 2,
    ) -> Dict:
        """
        Generate output from a single prompt

        This method is simpler than batch processing but less efficient
        for multiple prompts. Use this for:
        - Single prompt generation
        - Interactive/real-time use cases
        - Testing individual prompts

        Parameters:
        -----------
        prompt : str
            Verilog module description to generate code for
        temperature : float, default=0.2
            Controls randomness in generation
        max_tokens : int, default=1024
            Maximum tokens in response
        n_samples : int, default=2
            Number of different code samples to generate

        Returns:
        --------
        Dict
            Response dictionary with question, outputs, and config

        Notes:
        ------
        Unlike Claude, Gemini can generate multiple candidates in a single
        API call using the candidateCount parameter.
        """
        self.log.info("Running chat")

        # Make single API call for generation
        response = self.client.models.generate_content(
            model=self.model,
            contents=prompt,
            config=types.GenerateContentConfig(
                # System instruction for Verilog generation
                system_instruction="You are a Verilog code generator. Output only synthesizable Verilog code.",
                temperature=temperature,
                maxOutputTokens=max_tokens,  # Note: camelCase for Gemini API
                candidateCount=n_samples,  # Generate multiple candidates at once
            ),
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
                "system_instruction": "You are a Verilog code generator. Output only synthesizable Verilog code.",
                "temperature": temperature,
                "max_tokens": max_tokens,
                "samples": n_samples,
            },
        }

        return response_data
