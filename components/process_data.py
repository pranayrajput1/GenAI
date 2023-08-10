from kfp.v2 import dsl
from kfp.v2.components.component_decorator import component
from components.dependencies import resolve_dependencies


@component(
    base_image="python:3.10.6",
    packages_to_install=resolve_dependencies(
        'pandas',
        'google-cloud-storage',
        'gcsfs',
        'pyarrow',
        'fsspec'
    )
)
def process_data(
        dataset_bucket: str,
        dataset: dsl.Output[dsl.Dataset]
):
    """
    Function to load dataset from gcs bucket and pass to next component as a parquet file path.
    @param dataset_bucket: bucket name where dataset is stored
    @param dataset: parquet file dataset path
    """
    import logging
    import pandas as pd

    logger = logging.getLogger('tipper')
    logger.setLevel(logging.DEBUG)
    logger.addHandler(logging.StreamHandler())

    try:
        logging.debug("Task: Getting dataset from gcs bucket: 'llm-bucket-dolly'")
        read_file = pd.read_csv(f"gs://{dataset_bucket}/query_train.csv")
        train_df = pd.DataFrame(read_file)

        logging.debug("Task: Saving dataset to parquet file")
        train_df.to_parquet(dataset.path)

    except Exception as e:
        logging.error("Failed to process data!")
        raise e


