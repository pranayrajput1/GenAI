# import vertex_ray
# from vertex_ray import Resources
# from datetime import datetime
# from google.cloud import aiplatform as vertex_ai
# from constants import PROJECT_ID, REGION
#
# vertex_ai.init(project=PROJECT_ID, location=REGION)
#
# # TIMESTAMP = datetime.now().strftime("%Y%m%d%H%M%S")
# # head_node_type = Resources(
# #     machine_type="n1-standard-16",
# #     node_count=1,
# # )
# #
# # worker_node_types = [
# #     Resources(
# #         machine_type="n1-standard-4",
# #         node_count=1,
# #     )
# # ]
# # cluster_name = f"ray-cluster-{TIMESTAMP}"
# # ray_cluster_name = vertex_ray.create_ray_cluster(
# #     head_node_type=head_node_type,
# #     worker_node_types=worker_node_types,
# #     cluster_name=cluster_name,
# # )
# # print(ray_cluster_name)
# # ray_cluster = vertex_ray.get_ray_cluster("projects/24562761082/locations/us-central1/persistentResources/ray-cluster-20240920114552")
# # print("Ray cluster on Vertex AI:", "ray-cluster-20240919213429")
# #
# CLUSTER_NAME = "ray-cluster-20240920114552"
# CLUSTER_RESOURCE_NAME = 'projects/{}/locations/{}/persistentResources/{}'.format(PROJECT_ID, REGION, CLUSTER_NAME)
#
# vertex_ray.delete_ray_cluster(CLUSTER_RESOURCE_NAME)
#
#
# # import ray
# # import vertex_ray
# # from ray.job_submission import JobSubmissionClient
# # from google.cloud import aiplatform  # Necessary even if aiplatform.* symbol is not directly used in your program.
# # CLUSTER_NAME = "ray-cluster-20240920114552"
# # CLUSTER_RESOURCE_NAME='projects/{}/locations/REGION/persistentResources/{}'.format(PROJECT_ID, CLUSTER_NAME)
# # #
# # client = JobSubmissionClient("vertex_ray://{}".format(CLUSTER_RESOURCE_NAME))
# #
# # job_id = client.submit_job(
# #   # Entrypoint shell command to execute
# #   entrypoint="python my_script.py",
# #   # Path to the local directory that contains the my_script.py file.
# #   runtime_env={
# #     "working_dir": "./ray_scripts",
# #     "pip": ["numpy",
# #             "xgboost",
# #             "ray==2.33.0", # pin the Ray version to prevent it from being overwritten
# #            ]
# #   }
# # )
# # #
# # # # Ensure that the Ray job has been created.
# # print(job_id)

import os
import ray
import numpy as np
import pandas as pd
from google.cloud import storage
from google.cloud import aiplatform as vertex_ai
import vertex_ray
from utils.preprocessing import create_ray_cluster, handle_missing_values, encode_categorical_variables, feature_engineering, \
    prepare_features_and_target, create_preprocessor


@ray.remote
class DataProcessor:
    def __init__(self):
        self.preprocessor = create_preprocessor()

    def process_batch(self, batch):
        batch = handle_missing_values(batch)
        batch = encode_categorical_variables(batch)
        batch = feature_engineering(batch)
        X, y = prepare_features_and_target(batch)
        X_processed = self.preprocessor.fit_transform(X)
        return X_processed, y


def test_ray_data_processing(
        project_id: str,
        region: str,
        input_data_path: str,
        output_bucket: str,
        output_prefix: str,
        num_workers: int = 4
):
    # Set up Google Cloud credentials
    # os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/path/to/your/service-account-key.json"

    # Initialize Vertex AI
    vertex_ai.init(project=project_id, location=region)

    cluster_name = "ray-cluster-20240924124853-62497a"
    # Create Ray cluster
    # cluster_name = create_ray_cluster(project_id, region)
    print(f"Created Ray cluster: {cluster_name}")

    try:
        # Initialize Ray
        CLUSTER_RESOURCE_NAME = 'projects/{}/locations/{}/persistentResources/{}'.format(project_id, region,
                                                                                         cluster_name)
        ray.init(address=CLUSTER_RESOURCE_NAME)

        # Load data
        df = pd.read_csv(input_data_path)

        # Split data into batches
        batches = np.array_split(df, num_workers)

        # Create DataProcessor actors
        processors = [DataProcessor.remote() for _ in range(num_workers)]

        # Submit jobs to the cluster
        futures = [processor.process_batch.remote(batch) for processor, batch in zip(processors, batches)]

        # Get results
        results = ray.get(futures)

        # Combine results
        X_processed = np.concatenate([result[0] for result in results])
        y = np.concatenate([result[1] for result in results])

        # Split into train and test sets
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(X_processed, y, test_size=0.2, random_state=42)

        # Save processed data to GCS
        storage_client = storage.Client()
        bucket = storage_client.bucket(output_bucket)

        datasets = {
            "X_train": X_train,
            "X_test": X_test,
            "y_train": y_train,
            "y_test": y_test
        }

        for name, data in datasets.items():
            blob = bucket.blob(f"{output_prefix}/{name}.npy")
            with blob.open("wb") as f:
                np.save(f, data)
            print(f"Saved {name} to gs://{output_bucket}/{output_prefix}/{name}.npy")

        print("Data processing completed successfully.")

    finally:
        # Clean up Ray cluster
        ray.shutdown()
        # vertex_ray.delete_ray_cluster(cluster_name)
        print(f"Deleted Ray cluster: {cluster_name}")


# if __name__ == "__main__":
#     # Configuration
#     PROJECT_ID = "nashtech-ai-dev-389315"
#     REGION = "us-central1"
#     INPUT_DATA_PATH = "/home/nashtech/PycharmProjects/Mlops/Housing.csv"
#     OUTPUT_BUCKET = "nashtech_vertex_ai_artifact"
#     OUTPUT_PREFIX = "processed_data"
#     NUM_WORKERS = 4
#
#     test_ray_data_processing(
#         PROJECT_ID,
#         REGION,
#         INPUT_DATA_PATH,
#         OUTPUT_BUCKET,
#         OUTPUT_PREFIX,
#         NUM_WORKERS
#     )

from google.cloud import aiplatform


def get_ray_cluster_endpoint(project_id, region, cluster_name):
    # Initialize Vertex AI
    aiplatform.init(project=project_id, location=region)

    # Get the cluster resource
    cluster_resource = aiplatform.gapic.VertexAIClient().get_persistent_resource(
        name=f"projects/{project_id}/locations/{region}/persistentResources/{cluster_name}"
    )

    # Extract endpoint information
    endpoint = cluster_resource.endpoint  # This should contain IP:PORT
    return endpoint

get_ray_cluster_endpoint(project_id="nashtech-ai-dev-389315",region="us-central1",cluster_name="ray-cluster-20240924124853-62497a")