import pandas as pd
from sklearn.cluster import DBSCAN
from google.cloud import storage
import joblib
import pickle
import os
import logging
from kfp.v2 import dsl

logger = logging.getLogger('tipper')
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler())


def fit_db_scan_model(
        model_name: str,
        train_dataset: str,
        model_artifact_path: dsl.Output[dsl.Model],
):
    """
    Function to get dataset path,
    then fit model and save to artifact and giver artifact path as output.
    @dataset_path: dataset parquet file path
    @batch_size: batch size used to create batches of data.
    @model_artifact_path: model path saved at artifact registry and given path as output.
    @model_name: model name used to save trained model.
    """

    dbscan = DBSCAN(eps=6.5, min_samples=1000, leaf_size=30, p=2)
    model = dbscan

    logging.info(f"Reading processed train data from: {train_dataset}")
    train_data_batch = pd.read_parquet(train_dataset)

    logging.info("Fitting model")
    model = model.fit(train_data_batch)

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
