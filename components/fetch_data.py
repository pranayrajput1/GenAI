from kfp.v2 import dsl
from kfp.v2.components.component_decorator import component
from components.dependencies import resolve_dependencies


@component(
    base_image="python:3.8",
    packages_to_install=resolve_dependencies(
        'pandas',
        'numpy',
        'google-cloud-storage',
        'gcsfs',
        'pyarrow',
        'fsspec',
        'mlflow'
    )
)
def fetch_dataset(
        dataset_bucket: str,
        dataset_name: str,
        dataset: dsl.Output[dsl.Dataset],
):
    """
    Function to load dataset from gcs bucket and save it
    in parquet file then pass the output path of dataset.
    @dataset: dataset parquet file path given as output
    """
    import logging
    import pandas as pd
    import mlflow

    logger = logging.getLogger('tipper')
    logger.setLevel(logging.DEBUG)
    logger.addHandler(logging.StreamHandler())
    try:
        logging.info('Task: loading data from gcs bucket"')
        dataset_source_url = f'gs://{dataset_bucket}/{dataset_name}'
        house_hold_data = pd.read_csv(dataset_source_url, delimiter=";", low_memory=False)
        house_hold_df = pd.DataFrame(house_hold_data)

        logging.info(f"Dataset Features: {house_hold_data.columns}")
        logging.info(f'Task: saving data to parquet file at path: {dataset.uri}')
        house_hold_df.to_parquet(dataset.path)

    except Exception as e:
        logging.error("Failed to Save Model to Bucket")
        raise e
