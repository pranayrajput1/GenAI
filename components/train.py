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
        model_name: str,
        dataset_path: dsl.Input[dsl.Dataset],
        model_artifact_path: dsl.Output[dsl.Model],
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
        logging.info("fitting model on processed data")
        model.fit_model(model_name, dataset_path.path, model_artifact_path.path)

    except Exception as e:
        logging.error("Failed to Save Model to Bucket")
        raise e