from pathlib import Path

path = Path(__file__).resolve().parent

PROJECT_ID = "nashtech-ai-dev-389315"
REGION = "us-central1"
SERVICE_ACCOUNT_ML = "nashtech-ai-dev-app-sa@nashtech-ai-dev-389315.iam.gserviceaccount.com"

DATASET_BUCKET = 'gs://nashtech_vertex_ai_artifact/household_power_consumption.csv'

MODEL_DISPLAY_NAME = "db_scan_model"

PIPELINE_NAME = "clustering-kubeflow"
PIPELINE_DESCRIPTION = "Kubeflow pipeline tutorial."

PIPELINE_ROOT_GCS = f"gs://{PROJECT_ID}-kubeflow-pipeline"

BASE_IMAGE_QUALIFIER = "db-scan-image"
SERVE_IMAGE_QUALIFIER = "dbscan-serve-image"

BASE_IMAGE_TAG = "0.0.1"
SERVING_IMAGE_TAG = "0.0.1"

BASE_IMAGE = f"{REGION}-docker.pkg.dev/{PROJECT_ID}/db-scan-pipeline/{BASE_IMAGE_QUALIFIER}:{BASE_IMAGE_TAG}"
SERVING_IMAGE = f"{REGION}-docker.pkg.dev/{PROJECT_ID}/model-serving/{SERVE_IMAGE_QUALIFIER}:{BASE_IMAGE_TAG}"

CLUSTER_IMAGE_BUCKET = "nashtech_vertex_ai_artifact"

STAGING_BUCKET = "gs://dbscan-model/"
BATCH_SIZE = 10000

TRIGGER_ID = "00c14313-1ad2-4200-a4e0-57adae910784"

# PIPELINE_IMAGE = "us-central1-docker.pkg.dev/nashtech-ai-dev-389315/clustering-pipeline/db-scan-image:0.0.1"
# SERVING_IMAGE = "us-central1-docker.pkg.dev/nashtech-ai-dev-389315/clustering-pipeline/dbscan-serve-image:0.0.1"

pipeline = "86f32f92-c98f-4c7a-a912-fc2ec474248f"