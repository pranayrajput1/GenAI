import os
from pathlib import Path
from src.data import process_pipeline_image_details

path = Path(__file__).resolve().parent

PROJECT_ID = "nashtech-ai-dev-389315"
REGION = "us-central1"
SERVICE_ACCOUNT_ML = "nashtech-ai-dev-app-sa@nashtech-ai-dev-389315.iam.gserviceaccount.com"

os.environ['GOOGLE_CLOUD_PROJECT'] = PROJECT_ID

MODEL_DISPLAY_NAME = "db_scan_model"

DATASET_ID = "household_data"
DATASET_LOCATION = "US"

TABLE_ID = "house_data"
RESOURCE_BUCKET = "nashtech_vertex_ai_artifact"
CSV_FILE_NAME = "Housing.csv"

FEATURE_STORE_ID = "house_featurestore"
FEATURE_STORE_VIEW_ID = "house_featureview"

PIPELINE_NAME = "feature-store-kubeflow"
PIPELINE_DESCRIPTION = "Feature Store Pipeline"

PIPELINE_ROOT_GCS = f"gs://{PROJECT_ID}-kubeflow-pipeline"

BASE_IMAGE_QUALIFIER = "house_base_image"
SERVE_IMAGE_QUALIFIER = "gb-serve-image"

'''Pipeline Base & Serve Image Constants'''
SHA_GET_KEY = "pipeline_commit"
PIPELINE_CONFIG_FILE = "pipeline_configuration.json"

'''Uncomment this below line whenever you submit pipeline using cloud build trigger'''
# IMAGE_TAG = process_pipeline_image_details(RESOURCE_BUCKET, PIPELINE_CONFIG_FILE,
#                                            key=SHA_GET_KEY, new_entry=None)
IMAGE_TAG = 0.1
BASE_IMAGE = f"{REGION}-docker.pkg.dev/{PROJECT_ID}/feature-store-pipeline/{BASE_IMAGE_QUALIFIER}:{IMAGE_TAG}"
SERVING_IMAGE = f"{REGION}-docker.pkg.dev/{PROJECT_ID}/feature-store-pipeline/{SERVE_IMAGE_QUALIFIER}:{IMAGE_TAG}"

STAGING_BUCKET = f"gs://{RESOURCE_BUCKET}/"
BATCH_SIZE = 10000

SAVE_MODEL_DETAILS_FILE = "model_details.json"
COMPILE_PIPELINE_JSON = "gb_pipeline.json"

# TRIGGER_ID = "c8b3bb25-d54f-41f5-bee7-96cc05a107fb"

dataset_name = "household_power_consumption.txt"

fit_db_model_name = "gb_regressor"
"""Test Pipeline Changes"""
