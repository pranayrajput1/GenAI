from pathlib import Path

path = Path(__file__).resolve().parent

dataset_bucket = "llm-bucket-dolly"
save_model_bucket_name = "fine_tuned_model"
original_model_name = "databricks/dolly-v2-3b"

service_account = "1053338264064-compute@developer.gserviceaccount.com"

project_region = "us-central1"
project_id = "llm-dolly"

trigger_id = "889113a9-dada-49b0-beb7-ada0efba4bc2"
serve_image_qualifier = "llm-model-serve-image"
serve_model_tag = "0.1"
serving_image = f"{project_region}-docker.pkg.dev/{project_id}/model-serving/{serve_image_qualifier}:{serve_model_tag}"


model_display_name = "dolly_v2_3b"
staging_bucket = "gs://llm-bucket-dolly/"

pipeline_description = "dolly-model-fine-tuning-pipeline"
pipeline_name = "dolly-llm-kubeflow"
pipeline_root_gcs = f"gs://{project_id}-kubeflow-pipeline"

base_image_tag = "0.1"
base_image_qualifier = "llm-dolly-image"
base_image = f"{project_region}-docker.pkg.dev/{project_id}/train-dolly-pipeline/{base_image_qualifier}:{base_image_tag}"
