from sklearn.cluster import DBSCAN
from google.cloud import storage
import joblib
import pickle
import pandas as pd
import os
import logging
from kfp.v2 import dsl
from utils import preprocessing


logger = logging.getLogger('tipper')
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler())


def fit_db_scan_model(dataset_path: str,
                      batch_size: int,
                      model_artifact_path: dsl.Output[dsl.Model],
                      model_name: str
                      ):
    """
    Function to get dataset path and call helper function used to process data,
    then fit model and save to artifact and giver artifact path as output.
    @dataset_path: dataset parquet file path
    @batch_size: batch size used to create batches of data.
    @model_artifact_path: model path saved at artifact registry and given path as output.
    @model_name: model name used to save trained model.
    """
    logging.info(f"Reading process data from: {dataset_path}")
    data_frame = pd.read_parquet(dataset_path)

    logging.info('Task: performing preprocessing of data for model training')
    household_train, household_test = preprocessing.processing_data(data_frame)

    logging.info('Task: creating training batches for model fitting')
    train_batches = preprocessing.create_batches(household_train, batch_size)

    dbscan = DBSCAN(eps=6.5, min_samples=1000, leaf_size=30, p=2)
    model = dbscan

    logging.info("Fitting model")
    initial_data = train_batches[0]
    model = model.fit(initial_data)

    logging.info(f"Writing Model to pickle file to: {model_artifact_path}")

    file_name = f"{model_artifact_path}.pkl"
    with open(file_name, 'wb') as file:
        pickle.dump(model, file)

    logging.info("Saving Model To Bucket: 'gs://dbscan-model/'")
    local_path = f"{model_name}.joblib"
    joblib.dump(model, local_path)

    model_directory = os.environ['AIP_MODEL_DIR']
    storage_path = os.path.join(model_directory, local_path)
    blob = storage.blob.Blob.from_string(storage_path, client=storage.Client())
    blob.upload_from_filename(local_path)

    logging.info("Model Saved to Bucket Successfully")
