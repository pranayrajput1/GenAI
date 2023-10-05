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
        validated_file_name: str,
        dataset_path: dsl.Input[dsl.Dataset],

        db_scan_model: dsl.Input[dsl.Model],
        k_means_model: dsl.Input[dsl.Model],

        db_scan_avg_score: dsl.Output[dsl.Metrics],
        db_scan_cluster_image: dsl.Output[dsl.Artifact],

        k_means_avg_score: dsl.Output[dsl.Metrics],
        k_means_cluster_image: dsl.Output[dsl.Artifact],

        validated_model: dsl.Output[dsl.Metrics],
):
    import logging
    from src.model import evaluation_score
    from constants import models_list
    from src.model import get_validated_model, save_model_details

    logger = logging.getLogger('tipper')
    logger.setLevel(logging.DEBUG)
    logger.addHandler(logging.StreamHandler())

    try:
        """Evaluating Db-Scan Model Performance"""
        logging.info(f"Task: Evaluating Model Performance of {models_list[0]} model")
        db_scan_image = f"{models_list[0]}_cluster_image.png"
        db_scan_silhouette_score, db_scan_cluster_image = evaluation_score(batch_size,
                                                                           bucket_name,
                                                                           dataset_path,
                                                                           db_scan_model,
                                                                           db_scan_image)
        logging.info(f"Task: db_scan_silhouette_score: {db_scan_silhouette_score}")

        logging.info(f"Setting Average Silhouette Score: {db_scan_silhouette_score}")
        db_scan_avg_score.log_metric("score", db_scan_silhouette_score)

        logging.info("Task: Setting db_scan cluster image")
        db_scan_cluster_image.uri = db_scan_cluster_image

        """Evaluating K-Means Model Performance"""
        logging.info(f"Task: Evaluating Model Performance of {models_list[1]} model")
        k_means_image = f"{models_list[1]}_cluster_image.png"
        k_means_silhouette_score, k_means_cluster_image = evaluation_score(batch_size,
                                                                           bucket_name,
                                                                           dataset_path,
                                                                           k_means_model,
                                                                           k_means_image)
        logging.info(f"Task: k_means_silhouette_score: {k_means_silhouette_score}")

        logging.info(f"Setting Average Silhouette Score: {k_means_silhouette_score}")
        k_means_avg_score.log_metric("score", k_means_silhouette_score)

        logging.info("Task: Setting k_means cluster image")
        k_means_cluster_image.uri = k_means_cluster_image

        """Comparing Both Model Scores"""
        logging.info("Task: Setting Validated Model Metric")
        model_metrics = get_validated_model(db_scan_silhouette_score, k_means_silhouette_score)

        validated_model.log_metric("Validated Model:", model_metrics)

        logging.info(f"Task: Creating model name: {model_metrics} dictionary")
        model_details = {
            "validated_model": model_metrics,
        }

        logging.info("Task: Saving Validated Model Details Over GCS Bucket")
        save_model_details(model_details, validated_file_name, bucket_name)

    except Exception as e:
        logging.info("Failed To Execute Model validation")
        raise e
