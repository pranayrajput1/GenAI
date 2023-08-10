from kfp.v2 import dsl
from kfp.v2.components.component_decorator import component
from components.dependencies import resolve_dependencies
from constants import base_image


@component(
    base_image=base_image,
    packages_to_install=resolve_dependencies(
        'pandas',
        'kfp',
        'numpy',
        'fsspec',
        'pyarrow',
        'gcsfs',
        'google-cloud-storage',
    )
)
def fine_tune_model(dataset_path: dsl.Input[dsl.Dataset],
                    model_name: str,
                    save_model_bucket_name: str,
                    model_artifact_path: dsl.OutputPath()
                    ):
    import logging
    from src import model

    logger = logging.getLogger('tipper')
    logger.setLevel(logging.DEBUG)
    logger.addHandler(logging.StreamHandler())

    try:
        model.fine_tune_model(dataset_path.path, model_name, save_model_bucket_name, model_artifact_path)

    except Exception as e:
        logging.error("Failed to train model!")
        raise e
