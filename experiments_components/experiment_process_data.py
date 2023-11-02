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
def pre_process_data(
        dataset_path: dsl.Input[dsl.Dataset],
        batch_size: int,
        train_dataset: dsl.Output[dsl.Dataset]
):
    """
    Function to load dataset from gcs bucket and save it
    in parquet file then pass the output path of dataset.
    @dataset: dataset parquet file path given as output
    """
    import logging
    from src import data

    logger = logging.getLogger('tipper')
    logger.setLevel(logging.DEBUG)
    logger.addHandler(logging.StreamHandler())
    try:
        train_batches = data.dataset_processing(dataset_path.path, batch_size)

        logger.info('Task: Setting processed training batches')
        train_batches.to_parquet(train_dataset.path)

    except Exception as e:
        logging.error("Failed to Pre-Process Dataset")
        raise e
