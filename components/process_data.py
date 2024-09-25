from kfp import dsl
from kfp.dsl import component
from constants import BASE_IMAGE
from components.dependencies import resolve_dependencies



@component(
    base_image=BASE_IMAGE,
    packages_to_install=resolve_dependencies(
        "google-cloud-aiplatform",
        "google-cloud-storage",
        "google-cloud-aiplatform[ray]",
        "pandas",
        "ray",
    )
)
def pre_process_data(
        project_id: str,
        region: str,
        # input_data: dsl.Input[dsl.Dataset],
        x_train_path: dsl.Output[dsl.Dataset],
        x_test_path: dsl.Output[dsl.Dataset],
        y_train_path: dsl.Output[dsl.Dataset],
        y_test_path: dsl.Output[dsl.Dataset],
        cluster_name: str = None,
) :
    import ray
    import logging
    import pandas as pd
    import vertex_ray
    from utils.preprocessing import create_ray_cluster, data_processing_pipeline
    from src.data import get_logger

    logger = get_logger()


    try:
        # Create Ray cluster if not provided
        if not cluster_name:
            cluster_name = create_ray_cluster(project_id, region)
        logger.info("Initializing Cluster")
        # Initialize Ray
        ray.init(
            address=f'vertex_ray://projects/24562761082/locations/us-central1/persistentResources/{cluster_name}')

        logger.info("Processing Dataset")

        # Process data
        X_train, X_test, y_train, y_test = data_processing_pipeline()

        logger.info("Processing Dataset Completed")

        y_test_data_frame = pd.DataFrame(y_test)
        y_train_data_frame = pd.DataFrame(y_train)

        # Saving the splits of data in parquet format.
        X_train.to_parquet(x_train_path.path)
        X_test.to_parquet(x_test_path.path)
        y_train_data_frame.to_parquet(y_train_path.path)
        y_test_data_frame.to_parquet(y_test_path.path)
        logger.info('Processed data splits are saved into the intermediate artifact in parquet format')

        # Clean up Ray cluster
        # vertex_ray.delete_ray_cluster(cluster_name)
        # logger.info(f"Deleted Ray cluster: {cluster_name}")
    except Exception as e:
        logging.error("Failed to Pre-Process Dataset")
        raise e