import logging
import pandas as pd
from utils import preprocessing

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
