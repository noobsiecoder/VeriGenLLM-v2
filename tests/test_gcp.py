"""
Testbench to check all GCP client SDK

Author: Abhishek Sriram <noobsiecoder@gmail.com>
Date:   Aug 14th, 2025
Place:  Boston, MA
"""

from src.logger import Logger
from src.bucket import GoogleStorageClient
from google.cloud import storage
from google.oauth2 import service_account
from google.api_core.exceptions import NotFound

log = Logger("test_gcp").get_logger()


gcp_storage = GoogleStorageClient()


def test_connection():
    """
    Test connection (if alive)
    """
    res = gcp_storage.connect()
    assert res


def test_upload(bucket: str):
    """
    Check if file upload works

    Parameters:
    -----------
    bucket: str
        Name of the GCP storage bucket
    """
    bucket_name = bucket

    def blob_exists() -> bool:
        service_filepath = "secrets/gcp-storage.json"
        credentials = service_account.Credentials.from_service_account_file(
            service_filepath
        )
        # Create a storage client using the loaded credentials
        client = storage.Client(credentials=credentials)
        bucket = client.bucket(bucket_name)
        blob = bucket.blob("test/LICENSE")  # Test file

        try:
            blob.reload()  # Raises NotFound if blob doesn't exist
            return True
        except NotFound:
            return False

    gcp_storage.upload_file(
        local_file_path="LICENSE",
        bucket_name="verilog-llm-eval",
        blob_name="test/LICENSE",
    )
    assert blob_exists()
