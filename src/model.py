import pandas as pd
from sklearn.cluster import DBSCAN, KMeans
from google.cloud import storage
import joblib
import pickle
import os
import logging
from kfp.v2 import dsl
from google.cloud.devtools import cloudbuild_v1
from google.cloud import storage
import json

logger = logging.getLogger('tipper')
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler())


def get_validated_model(score_one, score_two):
    return "db_scan" if score_one > score_two else "k_means"


def upload_model(project, trigger):
    logging.info("Making Client Connection: ")
    cloud_build_client = cloudbuild_v1.CloudBuildClient()

    logging.info("Triggering Cloud Build For DB Scan Serving Container")
    response = cloud_build_client.run_build_trigger(project_id=project, trigger_id=trigger)

    if response.result():
        logging.info("Cloud Build Successful")
        return True
    else:
        logging.info("Cloud Build Failed !")
        raise RuntimeError


def get_model(model_type):
    model_mapping = {
        "db_scan": DBSCAN(eps=6.5, min_samples=1000, leaf_size=30, p=2),
        "k_means": KMeans(n_clusters=2)
    }

    return model_mapping.get(model_type, None)


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

    logging.info(f"Reading processed train data from: {train_dataset}")
    train_data_batch = pd.read_parquet(train_dataset)

    logging.info(f"Task: Fitting {model_name} model")
    model = get_model(model_name)

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


def save_model_details(dict_data, file_name, bucket_name):
    logging.info(f"Task: Dumping model details to a json file as: {file_name}")
    with open(file_name, "w") as file:
        json.dump(dict_data, file)

    logging.info("Task: Making client connection to save model details to bucket")
    client = storage.Client()
    bucket = client.get_bucket(bucket_name)

    blob = bucket.blob(file_name)

    logging.info(f"Task: Uploading model details to GCS Bucket: {bucket_name}")
    blob.upload_from_filename(file_name)

    logging.info("Task: Removing model details files from local environment")
    os.remove(file_name)