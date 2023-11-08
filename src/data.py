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


def process_pipeline_image_details(bucket_name: str, file_name: str, key=None, new_entry=None):
    try:
        logging.info("Making GCS Connection")
        client = storage.Client()
        bucket = client.get_bucket(bucket_name)
        blob = bucket.blob(file_name)

        logging.info(f"Downloading file: {file_name} from GCS")
        blob.download_to_filename(file_name)
        logging.info(f"File {file_name} downloaded")

        logging.info(f"Opening the {file_name}")
        with open(file_name, 'r') as file:
            pipeline_config = json.load(file)

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
            blob.upload_from_filename(file_name)
            logging.info(f"File {file_name} uploaded to GCS")

        else:
            return pipeline_config[key]

    except Exception as e:
        logging.error(f"An error occurred during processing pipeline image details: {str(e)}")
        raise e

    finally:
        logging.info("Removing the downloaded file")
        os.remove(file_name)
