"""
Testbench to check prompt extraction

Author: Abhishek Sriram <noobsiecoder@gmail.com>
Date:   Aug 14th, 2025
Place:  Boston, MA
"""

from main import ENVLoader, RunLLMPrompts
from src.logger import Logger
from src.gcp import GoogleStorageClient
from google.cloud import storage
from google.oauth2 import service_account
from google.api_core.exceptions import NotFound

log = Logger("test_llm_prompts").get_logger()
ENVLoader()  # load ENV


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


def test_gen_for_all_models_lite(bucket: str):
    """
    Check if all models generate values and upload to GCP Storage

    Parameters:
    -----------
    bucket: str
        Name of the GCP storage bucket
    """
    bucket_name = bucket

    def blob_exists(blob: str) -> bool:
        service_filepath = "secrets/gcp-storage.json"
        credentials = service_account.Credentials.from_service_account_file(
            service_filepath
        )
        # Create a storage client using the loaded credentials
        client = storage.Client(credentials=credentials)
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(blob)

        try:
            blob.reload()  # Raises NotFound if blob doesn't exist
            return True
        except NotFound:
            return False

    llm_prompts = RunLLMPrompts()
    llm_prompts.collect_prompts()
    _, filenames = llm_prompts.run_all_models(prompt_size=2, n_samples=1)
    gcp_storage = GoogleStorageClient()
    gcp_storage.connect()

    for filename in filenames:
        try:
            gcp_storage.upload_file(
                local_file_path=filename,
                bucket_name=bucket_name,
                blob_name=f"results/{filename}",
            )
            log.info(f"Copied file {filename} to GCP storage: results/")
        except Exception as err:
            log.error(f"Error in uploading file: {err}")

    for filename in filenames:
        assert blob_exists(blob=filename)
