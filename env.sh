export PROJECT_ID=$(gcloud config list project --format "value(core.project)")
export REPO_NAME=customrepo
export IMAGE_NAME="my-customhouse-app"
export IMAGE_TAG=latest
export IMAGE_URI="gcr.io/${PROJECT_ID}/${IMAGE_NAME}:${IMAGE_TAG}"
export AIP_MODEL_DIR=us-central1-docker.pkg.dev/${PROJECT_ID}/${REPO_NAME}
