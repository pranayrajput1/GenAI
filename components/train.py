from kfp.v2 import dsl
from kfp.v2.components.component_decorator import component
from components.dependencies import resolve_dependencies
from constants import BASE_IMAGE


@component(
    base_image=BASE_IMAGE,
    packages_to_install=resolve_dependencies(
        'pandas',
        'kfp',
        'numpy',
        'scikit-learn',
        'fsspec',
        'pyarrow',
        'gcsfs',
        'scikit-learn',
        'google-cloud-storage',
    )
)
def fit_model(
        batch_size: int,
        dataset_path: dsl.Input[dsl.Dataset],
        model_artifact_path: dsl.Output[dsl.Model],
        model_name: str = 'db_scan',
):
    """
    Function to train model using household dataset,
    save trained model over GCS Bucket
    @batch_size: batch size used to create batches of data.
    @dataset_path: dataset parquet file path.
    @model_artifact_path: model path saved at artifact registry and given path as output.
    @model_name: model name used to save trained model.
    """
    import logging
    from src import model

    logger = logging.getLogger('tipper')
    logger.setLevel(logging.DEBUG)
    logger.addHandler(logging.StreamHandler())

    try:
        logging.info("fitting db scan model on processed data")
        model.fit_db_scan_model(dataset_path.path, batch_size, model_artifact_path.path, model_name)

    except Exception as e:
        logging.error("Failed to Save Model to Bucket")
        raise e
