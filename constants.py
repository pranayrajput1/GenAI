from pathlib import Path

path = Path(__file__).resolve().parent

dataset_dir = path / "dataset"
dataset_path = dataset_dir / "query_train.csv"

dataset_name = "query_train.json"
save_model_bucket_name = "llm_dolly_model"
original_model_name = "databricks/dolly-v2-3b"

dataset_bucket = "nashtech_vertex_ai_artifact"

service_account = "nashtech-ai-dev-app-sa@nashtech-ai-dev-389315.iam.gserviceaccount.com"

project_region = "us-central1"
project_id = "nashtech-ai-dev-389315"

model_display_name = "dolly_v2_3b"
staging_bucket = "gs://llm_dolly_model/"

pipeline_description = "This is the dolly_v2_3b model fine tuning pipeline"
pipeline_name = "dolly-llm-pipeline"
pipeline_root_gcs = f"gs://{project_id}/llm-dolly-pipeline"

base_image_tag = "0.1"

base_image_qualifier = "llm-dolly-image"
serve_image_qualifier = "llm-dolly-serve-image"

docker_artifact_registry = "nashtech-ai-docker-registry"

base_image = f"{project_region}-docker.pkg.dev/{project_id}/{docker_artifact_registry}/{base_image_qualifier}:{base_image_tag}"

serving_image = f"{project_region}-docker.pkg.dev/{project_id}/{docker_artifact_registry}/{serve_image_qualifier}:{base_image_tag}"

trigger_id = ""

component_execution = False

"us-central1-docker.pkg.dev/nashtech-ai-dev-389315/nashtech-ai-docker-registry/llm-dolly-image:0.1"
