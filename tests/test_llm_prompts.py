"""
Testbench to check prompt extraction

Author: Abhishek Sriram <noobsiecoder@gmail.com>
Date:   Aug 14th, 2025
Place:  Boston, MA
"""

from src.logger import Logger
from main import RunLLMPrompts
from google.cloud import storage
from google.oauth2 import service_account
from google.api_core.exceptions import NotFound

log = Logger("test_llm_prompts").get_logger()


def test_prompt_extraction():
    """
    Check extraction from each questions
    """
    llm_prompts = RunLLMPrompts()
    prompt = """
// This is a signed adder that adds two 8-bit 2's complement numbers. It also captures a signed overflow. 
module signed_adder(input [7:0] a, input [7:0] b, output [7:0] s, output overflow ); 
    """
    prompts = llm_prompts.collect_prompts()
    assert len(prompts) == 17
    assert prompts[0]["prompt"].strip() == prompt.strip()


def test_gen_for_all_models_lite():
    """
    Check if all models generate values and upload to GCP Storage
    """

    def blob_exists(blob: str) -> bool:
        service_filepath = "secrets/gcp-storage.json"
        credentials = service_account.Credentials.from_service_account_file(
            service_filepath
        )
        # Create a storage client using the loaded credentials
        client = storage.Client(credentials=credentials)
        bucket = client.bucket("verilog-llm-eval")
        blob = bucket.blob()

        try:
            blob.reload()  # Raises NotFound if blob doesn't exist
            return True
        except NotFound:
            return False

    llm_prompts = RunLLMPrompts()
    llm_prompts.collect_prompts()
    _, filenames = llm_prompts.run_all_models(prompt_size=2, n_samples=1)

    for filename in filenames:
        assert blob_exists(blob=filename)
