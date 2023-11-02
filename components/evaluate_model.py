from kfp.v2.components.component_decorator import component
from kfp.v2 import dsl
from components.dependencies import resolve_dependencies
from constants import BASE_IMAGE


@component(
    base_image=BASE_IMAGE,
    packages_to_install=resolve_dependencies(
        'pandas',
        'fsspec',
        'google-cloud-storage',
        'gcsfs',
        'pyarrow',
        'matplotlib',
        'numpy',
        'scikit-learn'
    )
)
def evaluate_model(
        batch_size: int,
        bucket_name: str,
        dataset_path: dsl.Input[dsl.Dataset],
        trained_model: dsl.Input[dsl.Model],
        avg_score: dsl.Output[dsl.Metrics],
        cluster_image: dsl.Output[dsl.Artifact],
):
    import logging
    from src.model import evaluation_score
    from constants import models_list

    logger = logging.getLogger('tipper')
    logger.setLevel(logging.DEBUG)
    logger.addHandler(logging.StreamHandler())

    try:
        """Evaluating Db-Scan Model Performance"""
        logging.info(f"Task: Evaluating Model Performance of {models_list[0]} model")
        image_name = "formed_cluster_image.png"
        silhouette_score, formed_cluster_image = evaluation_score(batch_size,
                                                                  bucket_name,
                                                                  dataset_path,
                                                                  trained_model,
                                                                  image_name)
        logging.info(f"Task: db_scan_silhouette_score: {silhouette_score}")

        logging.info(f"Setting Average Silhouette Score: {silhouette_score}")
        avg_score.log_metric("score", silhouette_score)

        logging.info("Task: Setting db_scan cluster image")
        cluster_image.uri = formed_cluster_image

    except Exception as e:
        logging.info("Failed To Execute Model validation")
        raise e
