"""
Client script to cloud storage bucket (Azure, AWS, GCP)

Author: Abhishek Sriram <noobsiecoder@gmail.com>
Date:   Aug 22nd, 2025
Place:  Boston, MA
"""

import os
from src.logger import Logger
from google.cloud import storage
from google.oauth2 import service_account


# TODO: Handle storage in Azure Storage
class AzureStorageClient:
    """
    Handles all operations in Azure Storage bucket
    """


# TODO: Handle storage in AWS S3 bucket
class AWSStorageClient:
    """
    Handles all operations in AWS S3 bucket
    """


class GoogleStorageClient:
    """
    Handles all operations in GCP Storage bucket
    """

    def __init__(self):
        """
        Loads when object is instantiated
        """
        self.client = None
        self.log = Logger(
            "gcp_storage"
        ).get_logger()  # Initialize logger instance for tracking GCP storage operations

    def connect(self) -> bool:
        """
        Establish connection

        Returns:
        --------
        Bool value to confirm connection
        """
        service_filepath = "secrets/gcp-storage.json"
        try:
            if not os.path.exists(service_filepath):
                raise FileNotFoundError(f"File not found in path: {service_filepath}")

            # Initialize GCS client with service account credentials
            # The credentials file should contain the service account key in JSON format
            credentials = service_account.Credentials.from_service_account_file(
                service_filepath
            )
            # Create a storage client using the loaded credentials
            self.client = storage.Client(credentials=credentials)
            return True
        except Exception as err:
            self.log.error(
                f"Error while trying to read secret file for GCP storage: {err}"
            )
            return False

    def upload_file(self, local_file_path, bucket_name, blob_name=None):
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
        """
        # Get a reference to the target bucket
        # Note: This doesn't verify if the bucket exists
        bucket = self.client.bucket(bucket_name)

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
        self.log.info(
            f"File {local_file_path} uploaded to gs://{bucket_name}/{blob_name}"
        )
