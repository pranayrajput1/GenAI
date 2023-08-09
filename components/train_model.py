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
        'scikit-learn',
        'fsspec',
        'pyarrow',
        'gcsfs',
        'scikit-learn',
        'google-cloud-storage',
    )
)
def fine_tune_model(dataset_path: dsl.Input[dsl.Dataset],
                    model_artifact_path: dsl.Output[dsl.Model],
                    model_name: str):
    import logging
    from src import model

    logger = logging.getLogger('tipper')
    logger.setLevel(logging.DEBUG)
    logger.addHandler(logging.StreamHandler())

    try:
        model.fine_tune_model(dataset_path.path, model_name, model_artifact_path.path)

    except Exception as e:
        logging.error("Failed to train model!")
        raise e
