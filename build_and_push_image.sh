if [ -z "$2" ]; then
  version=`cat version.txt`
  echo "usage: $0 <pipeline_qualifier> <version_tag>"
  echo "example: $0 kubeflow-tutorial-image ${version}"
  exit 1
fi
PROJECT_ID="nashtech-ai-dev-389315"
REGION="us-central1"
PIPELINE_QUALIFIER="db-scan-image"
VERSION_TAG="0.1"

IMAGE="${REGION}-docker.pkg.dev/${PROJECT_ID}/clustering-pipeline/${PIPELINE_QUALIFIER}:${VERSION_TAG}"

echo "Building and pushing image: ${IMAGE}"
docker buildx build --platform=linux/x86_64 -t "$IMAGE" .
docker push "$IMAGE"
