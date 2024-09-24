import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from utils import preprocessing
from utils import silhouette_score
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


def upload_model(project, trigger):
    logging.info("Making Client Connection: ")
    cloud_build_client = cloudbuild_v1.CloudBuildClient()

    logging.info("Triggering Cloud Build For Serving Container")
    response = cloud_build_client.run_build_trigger(project_id=project, trigger_id=trigger)

    if response.result():
        logging.info("Cloud Build Successful")
        return True
    else:
        logging.info("Cloud Build Failed !")
        raise RuntimeError


# def get_model(model_type):
#     model_mapping = {
#         "db_scan": DBSCAN(eps=6.5, min_samples=1000, leaf_size=30, p=2),
#         "k_means": KMeans(n_clusters=3)
#     }
#
#     return model_mapping.get(model_type, None)


def fit_model(
        x_train_path: str,
        y_train_path: str,
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

    # Load x_train
    x_train = pd.read_parquet(x_train_path)
    logger.info(f"Loaded x_train from: {x_train_path}")

    # Load y_train
    y_train = pd.read_parquet(y_train_path)
    logger.info(f"Loaded dataset from: {y_train_path}")
    model_name = "GradientBoostingRegressor"
    logging.info(f"Task: Fitting {model_name} model")
    model = GradientBoostingRegressor()

    logging.info("Fitting model")
    model.fit(x_train, y_train)

    logging.info(f"Writing Model to pickle file to: {model_artifact_path}")

    file_name = f"{model_artifact_path}.pkl"
    with open(file_name, 'wb') as file:
        pickle.dump(model, file)

    logging.info("Saving Model To Bucket: 'gs://nashtech_vertex_ai_artifact/'")
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


def evaluation_score(batch,
                     bucket_name,
                     dataset_path,
                     model_path,
                     image_path):

    logging.info(f"Reading model from: {model_path.path}")
    file_name = model_path.path + ".pkl"

    try:
        with open(file_name, 'rb') as model_file:
            trained_model = pickle.load(model_file)

        logging.info(f"Reading process data from: {dataset_path.path}")
        data_frame = pd.read_parquet(dataset_path.path)
        household_train, household_test = preprocessing.processing_data(data_frame)

        logging.info("calculating average silhouette_scores")
        average_silhouette_score = silhouette_score.get_silhouette_score_and_cluster_image(
            household_train,
            batch,
            trained_model,
            image_path)

        logging.info("Setting client connection using storage client API'")
        client = storage.Client()

        logging.info(f"Getting bucket: {bucket_name} from GCS")
        bucket = client.get_bucket(bucket_name)
        blob = bucket.blob(image_path)

        logging.info(f"Uploading Image to Bucket: 'gs://{bucket_name}/'")
        with open(image_path, 'rb') as file:
            blob.upload_from_file(file, content_type='image/png')

        logging.info(f"Uploaded Image to Bucket: 'gs://{bucket_name}/' successfully'")
        image_url = f"https://storage.cloud.google.com/{bucket_name}/{image_path}"

        return average_silhouette_score, image_url

    except Exception as exc:
        logging.info("Failed To Save Image to Bucket")
        raise exc
