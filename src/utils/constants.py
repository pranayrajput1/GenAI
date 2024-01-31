from pathlib import Path
import os

root = Path(__file__).resolve().parent.parent

model_id = "mistralai/Mistral-7B-Instruct-v0.1"
auth_token = "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"
image_name = "us-central1-docker.pkg.dev/nashtech-ai-dev-389315/nashtech-ai-docker-registry/mistral-image"
image_tag = "0.1"
PROJECT_ID = "nashtech-ai-dev-389315"
REGION = "us-central1"
SERVICE_ACCOUNT_ML = "nashtech-ai-dev-app-sa@nashtech-ai-dev-389315.iam.gserviceaccount.com"

STAGING_BUCKET = "gs://dbscan-model/"

SERVING_IMAGE = "us-central1-docker.pkg.dev/nashtech-ai-dev-389315/nashtech-ai-docker-registry/mistral-image:0.1"
MODEL_DISPLAY_NAME = "mistral"

decimal_model_name = "Deci/DeciLM-7B-instruct"

system_prompt = "You are an AI assistant that follows instruction extremely well. Help as much as you can."

resume_bucket_path = "dbscan-model"

data_dir = root / "data"
resume_path = data_dir / "resumes"

structured_text_dir = data_dir / "structured_text"
sentence_transformer_model = 'sentence-transformers/all-mpnet-base-v2'

embeddings_model = 'all-MiniLM-L6-v2'

persistence_directory = "test_db"

os.environ['GOOGLE_CLOUD_PROJECT'] = PROJECT_ID
reload_model_state = False

port = "5050"

local_ip = '127.0.0.1'
local_instance_endpoint_url = f'http://{local_ip}:{port}/predict'
