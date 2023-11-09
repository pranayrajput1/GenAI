import logging
import os
import pandas as pd
from utils import preprocessing
from google.cloud import storage
import json

logger = logging.getLogger('tipper')
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler())


def dataset_processing(
        dataset_path: str,
        batch_size: int):
    logging.info(f"Reading process data from: {dataset_path}")
    data_frame = pd.read_parquet(dataset_path)

    logging.info(f"Dataset Features: {data_frame.columns} in model training")

    logging.info('Task: performing preprocessing of data for model training')
    household_train, household_test = preprocessing.processing_data(data_frame)

    logging.info('Task: creating training batches for model fitting')
    train_batches = preprocessing.create_batches(household_train, batch_size)

    logging.info('Task: Getting initial batches from the dataset')
    initial_data = train_batches[0]

    return initial_data


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

        '''Removing extension from the file name'''
        if file_name.endswith(".json"):
            parent_key_name = file_name[:-5]

        if new_entry is not None:
            logging.info("Appending new entries to the pipeline configuration")
            pipeline_run_data = pipeline_config[parent_key_name]
            pipeline_run_data.update(new_entry)

            logging.info(f"Saving updated JSON to {file_name}")
            with open(file_name, 'w') as file:
                json.dump(pipeline_config, file, indent=4)

            logging.info(f"Uploading the updated file: {file_name} to GCS")
            upload_file_to_gcs(client, bucket_name, file_name)
        else:
            return pipeline_config[parent_key_name][key]

    except Exception as e:
        logging.error(f"An error occurred during processing pipeline image details: {str(e)}")
        raise e

    finally:
        logging.info("Removing the downloaded file")
        os.remove(file_name)


# model_details = {
#     "deployed_display_name": "deployed_display_name",
#     "endpoint_id": "endpoint_id",
#     "deployed_model_id": "deployed_model_id",
#     "machine_type": "machine_type",
#     "dataset": "dataset_name",
#     "evaluation_score": "evaluation_score",
# }
# process_pipeline_image_details(bucket_name="clustering-pipeline-artifact",
#                                file_name="pipeline_configuration_0.0.1.json",
#                                key=None,
#                                new_entry=model_details)
