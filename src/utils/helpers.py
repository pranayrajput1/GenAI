import time
import logging
from google.cloud import storage


def setup_logger():
    """
    Initializing logger basic configuration
    """
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    return logging


logger = setup_logger()


def get_time(start_time_input, end_time_input):
    elapsed_time_minutes = (start_time_input - end_time_input) / 60

    logger.info(f"Start Time: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time_input))}")
    logger.info(f"End Time: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(end_time_input))}")
    logger.info(f"Elapsed Time: {elapsed_time_minutes:.2f}")


def create_directory_if_not(directory_path):
    if not directory_path.exists():
        directory_path.mkdir(parents=True, exist_ok=True)
        logger.info(f"Directory '{directory_path}' created.")
    else:
        logger.info(f"Directory '{directory_path}' already exists.")


def download_files_from_bucket(bucket_name, destination_folder):
    try:
        client = storage.Client()
        bucket = client.get_bucket(bucket_name)
        blobs = bucket.list_blobs()

        files_downloaded = 0

        for blob in blobs:
            if blob.name.lower().endswith(('.pdf', '.doc', '.docx')):
                destination_path = f"{destination_folder}/{blob.name}"
                blob.download_to_filename(destination_path)
                files_downloaded += 1
                logger.info(f"Downloaded {blob.name} to {destination_path}")

        if files_downloaded > 0:
            return True, 200
        else:
            return f"No files found in the bucket: {bucket_name}", 404

    except Exception as e:
        logger.error(f"Some error occurred in downloading resume from bucket, error: {str(e)}")
        raise
