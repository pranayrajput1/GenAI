import os
from pathlib import Path
from src.data import process_pipeline_image_details

path = Path(__file__).resolve().parent

PROJECT_ID = "nashtech-ai-dev-389315"
REGION = "us-central1"
SERVICE_ACCOUNT_ML = "nashtech-ai-dev-app-sa@nashtech-ai-dev-389315.iam.gserviceaccount.com"

os.environ['GOOGLE_CLOUD_PROJECT'] = PROJECT_ID

MODEL_DISPLAY_NAME = "db_scan_model"

PIPELINE_NAME = "clustering-kubeflow"
PIPELINE_DESCRIPTION = "Kubeflow pipeline"

PIPELINE_ROOT_GCS = f"gs://{PROJECT_ID}-kubeflow-pipeline"

BASE_IMAGE_QUALIFIER = "db-scan-image"
SERVE_IMAGE_QUALIFIER = "dbscan-serve-image"

'''Pipeline Base & Serve Image Constants'''
SHA_GET_KEY = "pipeline_commit"
PIPELINE_CONFIG_FILE = "pipeline_configuration.json"
SAVE_MODEL_DETAILS_BUCKET = "clustering-pipeline-artifact"

'''Uncomment this below line whenever you submit pipeline using cloud build trigger'''
IMAGE_TAG = process_pipeline_image_details(SAVE_MODEL_DETAILS_BUCKET, PIPELINE_CONFIG_FILE,
                                           key=SHA_GET_KEY, new_entry=None)

'''Uncomment this below line whenever you submit pipeline locally'''
# IMAGE_TAG = "0.0.1"

BASE_IMAGE = f"{REGION}-docker.pkg.dev/{PROJECT_ID}/clustering-pipeline/{BASE_IMAGE_QUALIFIER}:{IMAGE_TAG}"
SERVING_IMAGE = f"{REGION}-docker.pkg.dev/{PROJECT_ID}/clustering-pipeline/{SERVE_IMAGE_QUALIFIER}:{IMAGE_TAG}"

RESOURCE_BUCKET = "nashtech_vertex_ai_artifact"

STAGING_BUCKET = f"gs://{RESOURCE_BUCKET}/"
BATCH_SIZE = 10000

SAVE_MODEL_DETAILS_FILE = "model_details.json"
PIPELINE_JSON = "pipeline_configuration.json"

TRIGGER_ID = "26a3629d-793e-4ab2-a2b3-9b4c0966b20d"

dataset_bucket = "nashtech_vertex_ai_artifact"
dataset_name = "household_power_consumption.txt"

fit_db_model_name = "db_scan"

cluster_image_bucket = "clustering-pipeline-artifact"

model_details_file_name = "model_details.json"
