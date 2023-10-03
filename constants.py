from pathlib import Path

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

BASE_IMAGE_TAG = "0.0.1"
SERVING_IMAGE_TAG = "0.0.1"

BASE_IMAGE = f"{REGION}-docker.pkg.dev/{PROJECT_ID}/clustering-pipeline/{BASE_IMAGE_QUALIFIER}:{BASE_IMAGE_TAG}"
SERVING_IMAGE = f"{REGION}-docker.pkg.dev/{PROJECT_ID}/clustering-pipeline/{SERVE_IMAGE_QUALIFIER}:{BASE_IMAGE_TAG}"

CLUSTER_IMAGE_BUCKET = "nashtech_vertex_ai_artifact"

STAGING_BUCKET = "gs://dbscan-model/"
BATCH_SIZE = 10000

TRIGGER_ID = "00c14313-1ad2-4200-a4e0-57adae910784"

dataset_bucket = "nashtech_vertex_ai_artifact"
dataset_name = "household_power_consumption.txt"

fit_model_name = "db_scan"

model_details_file_name = "model_details.json"

# PIPELINE_IMAGE = "us-central1-docker.pkg.dev/nashtech-ai-dev-389315/clustering-pipeline/db-scan-image:0.0.1"
# SERVING_IMAGE = "us-central1-docker.pkg.dev/nashtech-ai-dev-389315/clustering-pipeline/dbscan-serve-image:0.0.1"

pipeline = "7c74cb60-8a30-44b3-a5cf-3637962ce85a"

' gcloud builds triggers create github --name="clustering-pipeline" --service-account="projects/nashtech-ai-dev-389315/serviceAccounts/nashtech-ai-dev-app-sa@nashtech-ai-dev-389315.iam.gserviceaccount.com" --repo-owner="pranayrajput1" --repo-name="GenAI" --branch-pattern="clustering-pipeline" --build-config="cloudbuild.yaml" --require-approval '


' gcloud builds triggers create manual --name="serve-trigger" --build-config="serving_container/serve_model_build.yaml" --repo="https://github.com/pranayrajput1/GenAI.git" --repo-type=GITHUB --branch="clustering-pipeline" –service-account=”nashtech-ai-dev-app-sa@nashtech-ai-dev-389315.iam.gserviceaccount.com” '