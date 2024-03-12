# Custom Model Training with Google Vertex AI

This repository contains code and instructions for training a custom machine learning model using Google Vertex AI. The model is trained on tabular data using a custom TensorFlow Keras model and then deployed to Google Vertex AI for inference.

## Prerequisites

Before getting started, make sure you have the following:

- Google Cloud Platform (GCP) account with appropriate permissions.
- Docker installed on your local machine.
- Google Cloud SDK (`gcloud`) installed and configured.
- Python environment with necessary packages (`tensorflow`, `google-cloud-aiplatform`, etc.).
- A Google Cloud Storage (GCS) bucket for storing training artifacts.

### 1. Building and Pushing Docker Image

1. Set the necessary environment variables:
   ```bash
   export PROJECT_ID="your-project-id"
   export REPO_NAME="your-repo-name"
   export IMAGE_NAME="models/model"
   export IMAGE_TAG="latest"
   export IMAGE_URI="gcr.io/${PROJECT_ID}/${IMAGE_NAME}:${IMAGE_TAG}"

    Build your Docker image:

### Build the Docker Image 
    docker build -t $IMAGE_URI .



### Push the Docker image to GCR:
    docker push $IMAGE_URI

### 2. Running Custom Training Job in Vertex AI

    Install required Python libraries:

    bash

    pip install google-cloud-aiplatform

    Create a Python script to run the custom training job.

    Initialize the Vertex AI client with your project and region.

    Specify the container URI for your Docker image.

    Create a custom training job in Vertex AI using the CustomContainerTrainingJob class.

    Run the training job with desired configurations such as replica count and machine type.

