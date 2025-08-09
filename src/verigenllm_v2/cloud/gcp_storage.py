"""
Used for copying files to the GCP bucket

Author: Abhishek Sriram <noobsiecoder@gmail.com>
Date:   Aug 7th, 2025

Note: LLM was used to generate comments
"""

import os
from verigenllm_v2.utils.logger import Logger
from google.cloud import storage
from google.oauth2 import service_account


# Initialize logger instance for tracking GCP storage operations
log = Logger("gcp_storage").get_logger()


def upload_file_to_gcs(local_file_path, bucket_name, blob_name=None):
    """
    Upload a file to Google Cloud Storage

    Parameters:
    -----------
    local_file_path : str
        Path to the local file that needs to be uploaded
    bucket_name : str
        Name of the GCS bucket where the file will be uploaded
    blob_name : str, optional
        Custom name for the file in GCS. If None, uses the original filename

    Returns:
    --------
    None

    Raises:
    -------
    FileNotFoundError
        If the local file doesn't exist
    google.cloud.exceptions
        Various GCS-related exceptions for authentication or upload failures
    """
    # Initialize GCS client with service account credentials
    # The credentials file should contain the service account key in JSON format
    credentials = service_account.Credentials.from_service_account_file(
        "secrets/gcp-storage.json"
    )

    # Create a storage client using the loaded credentials
    client = storage.Client(credentials=credentials)

    # Get a reference to the target bucket
    # Note: This doesn't verify if the bucket exists
    bucket = client.bucket(bucket_name)

    # If no custom blob name is provided, extract the filename from the local path
    # For example: "/path/to/file.txt" -> "file.txt"
    if blob_name is None:
        blob_name = os.path.basename(local_file_path)

    # Create a blob object representing the file in GCS
    # A blob is the basic storage unit in GCS (similar to a file)
    blob = bucket.blob(blob_name)

    # Upload the file content to GCS
    # This method handles the actual file transfer
    blob.upload_from_filename(local_file_path)

    # Log successful upload with full GCS path for reference
    log.info(f"File {local_file_path} uploaded to gs://{bucket_name}/{blob_name}")
