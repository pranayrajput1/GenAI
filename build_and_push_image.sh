if [ -z "$2" ]; then
  version=`cat version.txt`
  echo "usage: $0 <pipeline_qualifier> <version_tag>"
  echo "example: $0 kubeflow-tutorial-image ${version}"
  exit 1
fi

PIPELINE_QUALIFIER="$1"
VERSION_TAG="$2"
IMAGE="gcr.io/nashtech-ai-dev-389315/${PIPELINE_QUALIFIER}:${VERSION_TAG}"

echo "Building and pushing image: ${IMAGE}"
docker buildx build --platform=linux/x86_64 -t "$IMAGE" .
docker push "$IMAGE"