from kfp.v2 import dsl
from kfp.v2.components.component_decorator import component
from components.dependencies import resolve_dependencies
from constants import BASE_IMAGE


@component(
    base_image=BASE_IMAGE,
    packages_to_install=resolve_dependencies(
        'pandas',
        'numpy',
        'google-cloud-storage',
        'gcsfs',
        'pyarrow',
        'fsspec'
    )
)
def processed_data(
        dataset_path: dsl.Input[dsl.Dataset],
        x_train_path: dsl.Output[dsl.Dataset],
        x_test_path: dsl.Output[dsl.Dataset],
        y_train_path: dsl.Output[dsl.Dataset],
        y_test_path: dsl.Output[dsl.Dataset]
) -> None:
    """
    @param dataset_path: Input path of dataset to be processed.
    @param x_train_path: Output path of x_train.
    @param x_test_path: Output path of x_test.
    @param y_train_path: Output path of y_train.
    @param y_test_path: Output path of y_test.
    """
    import logging
    from src import data
    import pandas as pd

    logger = logging.getLogger('tipper')
    logger.setLevel(logging.DEBUG)
    logger.addHandler(logging.StreamHandler())
    try:
        x_train, x_test, y_train, y_test = data.dataset_processing(dataset_path.path)
        y_test_data_frame = pd.DataFrame(y_test)
        y_train_data_frame = pd.DataFrame(y_train)

        # Saving the splits of data in parquet format.
        x_train.to_parquet(x_train_path.path)
        x_test.to_parquet(x_test_path.path)
        y_train_data_frame.to_parquet(y_train_path.path)
        y_test_data_frame.to_parquet(y_test_path.path)
        logger.info('Processed data splits are saved into the intermediate artifact in parquet format')
    except Exception as e:
        logging.error("Failed to Pre-Process Dataset")
        raise e


from kfp.v2.dsl import (
    component,
)
from google.cloud import storage
import tempfile
import os
from utils.preprocessing import create_ray_cluster, data_processing_pipeline


@component(
    base_image="gcr.io/your-project/ray-data-processing:latest",
    packages_to_install=[
        "google-cloud-aiplatform",
        "google-cloud-storage",
        "vertex-ray",
        "pandas",
        "numpy",
        "scikit-learn",
        "ray"
    ],
)
def ray_data_processing_component(
        project_id: str,
        region: str,
        input_data: dsl.Input[dsl.Dataset],
        x_train_path: dsl.Output[dsl.Dataset],
        x_test_path: dsl.Output[dsl.Dataset],
        y_train_path: dsl.Output[dsl.Dataset],
        y_test_path: dsl.Output[dsl.Dataset],
        cluster_name: str = None,
) -> None:
    import ray
    import logging
    import numpy as np
    import vertex_ray
    import pandas as pd

    logger = logging.getLogger('tipper')
    logger.setLevel(logging.DEBUG)
    logger.addHandler(logging.StreamHandler())

    try:
        # Create Ray cluster if not provided
        if not cluster_name:
            cluster_name = create_ray_cluster(project_id, region)

        # Initialize Ray
        ray.init(address='auto')

        # Process data
        X_train, X_test, y_train, y_test = data_processing_pipeline(input_data.uri)

        y_test_data_frame = pd.DataFrame(y_test)
        y_train_data_frame = pd.DataFrame(y_train)

        # Saving the splits of data in parquet format.
        X_train.to_parquet(x_train_path.path)
        X_test.to_parquet(x_test_path.path)
        y_train_data_frame.to_parquet(y_train_path.path)
        y_test_data_frame.to_parquet(y_test_path.path)
        logger.info('Processed data splits are saved into the intermediate artifact in parquet format')

        # Clean up Ray cluster
        vertex_ray.delete_ray_cluster(cluster_name)
        print(f"Deleted Ray cluster: {cluster_name}")
    except Exception as e:
        logging.error("Failed to Pre-Process Dataset")
        raise e