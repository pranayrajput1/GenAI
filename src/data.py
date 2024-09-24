import logging
import os
import pandas as pd
from utils import preprocessing
from google.cloud import storage
import json


def get_logger():
    logger = logging.getLogger('tipper')
    logger.setLevel(logging.DEBUG)
    logger.addHandler(logging.StreamHandler())
    return logger


def make_gcs_connection():
    logging.info("Making GCS Connection")
    client = storage.Client()
    return client


def download_file_from_gcs(client, bucket_name, file_name):
    bucket = client.get_bucket(bucket_name)
    blob = bucket.blob(file_name)

    logging.info(f"Downloading file: {file_name} from GCS")
    blob.download_to_filename(file_name)
    logging.info(f"File {file_name} downloaded")


def upload_file_to_gcs(client, bucket_name, file_name):
    blob = client.get_bucket(bucket_name).blob(file_name)
    blob.upload_from_filename(file_name)
    logging.info(f"File {file_name} uploaded to GCS")


def process_pipeline_image_details(bucket_name: str, file_name: str, key=None, new_entry=None):
    try:
        client = make_gcs_connection()
        download_file_from_gcs(client, bucket_name, file_name)

        logging.info(f"Opening the {file_name}")
        with open(file_name, 'r') as file:
            pipeline_config = json.load(file)

        if new_entry is not None:
            logging.info("Appending new entries to the pipeline configuration")
            pipeline_run_data = pipeline_config["pipeline_run_configuration"]
            pipeline_run_data.update(new_entry)

            logging.info(f"Saving updated JSON to {file_name}")
            with open(file_name, 'w') as file:
                json.dump(pipeline_config, file, indent=4)

            logging.info(f"Uploading the updated file: {file_name} to GCS")
            upload_file_to_gcs(client, bucket_name, file_name)
        else:
            return pipeline_config["pipeline_run_configuration"][key]

    except Exception as e:
        logging.error(f"An error occurred during processing pipeline image details: {str(e)}")
        raise e

    finally:
        logging.info("Removing the downloaded file")
        os.remove(file_name)
