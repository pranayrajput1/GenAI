from pathlib import Path

from src.data import process_pipeline_image_details

path = Path(__file__).resolve().parent

PROJECT_ID = "nashtech-ai-dev-389315"
REGION = "us-central1"
SERVICE_ACCOUNT_ML = "nashtech-ai-dev-app-sa@nashtech-ai-dev-389315.iam.gserviceaccount.com"

MODEL_DISPLAY_NAME = "db_scan_model"

PIPELINE_NAME = "clustering-kubeflow"
PIPELINE_DESCRIPTION = "Kubeflow pipeline tutorial."

PIPELINE_ROOT_GCS = f"gs://{PROJECT_ID}-kubeflow-pipeline"

BASE_IMAGE_QUALIFIER = "db-scan-image"
SERVE_IMAGE_QUALIFIER = "dbscan-serve-image"

'''Pipeline Base & Serve Image Constants'''

SHA_GET_KEY = "pipeline_commit"
PIPELINE_CONFIG_FILE = "pipeline_configuration.json"
SAVE_MODEL_DETAILS_BUCKET = "clustering-pipeline-artifact"
PIPELINE_ARTIFACT = "nashtech_vertex_ai_artifact"

IMAGE_TAG = process_pipeline_image_details(PIPELINE_ARTIFACT, PIPELINE_CONFIG_FILE,
                                           key=SHA_GET_KEY, new_entry=None)

BASE_IMAGE = f"{REGION}-docker.pkg.dev/{PROJECT_ID}/clustering-pipeline/{BASE_IMAGE_QUALIFIER}:{IMAGE_TAG}"
SERVING_IMAGE = f"{REGION}-docker.pkg.dev/{PROJECT_ID}/clustering-pipeline/{SERVE_IMAGE_QUALIFIER}:{IMAGE_TAG}"

STAGING_BUCKET = "gs://dbscan-model/"
BATCH_SIZE = 10000

SAVE_MODEL_DETAILS_FILE = "model_details.json"
PIPELINE_JSON = "dbscan_pipeline.json"


TRIGGER_ID = "8ecef415-9458-48aa-a848-730f41924d9b"

dataset_bucket = "nashtech_vertex_ai_artifact"
dataset_name = "household_power_consumption.txt"

fit_db_model_name = "db_scan"
fit_k_means_model_name = "k_means"

cluster_image_bucket = "clustering-pipeline-artifact"

model_details_file_name = "model_details.json"
validated_file_name = "validated_model.json"

experiment_pipeline = "experiment_pipeline.json"

models_list = ["db_scan", "k_means"]
