from pathlib import Path

path = Path(__file__).resolve().parent

dataset_name = "query_train.csv"
original_model_name = "databricks/dolly-v2-3b"
model_path = "./model_dir"

service_account = "1053338264064-compute@developer.gserviceaccount.com"

project_region = "us-central1"
project_id = "llm-dolly"

trigger_id = ""
serving_image = ""
model_display_name = "dolly_v2_3b"
staging_bucket = "gs://llm-bucket-dolly/"

pipeline_description = "dolly-model-fine-tuning-pipeline"
pipeline_name = "dolly-llm-kubeflow"
pipeline_root_gcs = f"gs://{project_id}-kubeflow-pipeline"

base_image_tag = "0.1"
base_image_qualifier = "llm-dolly-image"
base_image = f"{project_region}-docker.pkg.dev/{project_id}/train-dolly-pipeline/{base_image_qualifier}:{base_image_tag}"

"us-central1-docker.pkg.dev/llm-dolly/train-dolly-pipeline/llm-dolly-image:0.1"
