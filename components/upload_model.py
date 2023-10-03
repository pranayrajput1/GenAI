from kfp.v2.components.component_decorator import component
from components.dependencies import resolve_dependencies
from constants import BASE_IMAGE
from kfp.v2 import dsl


@component(
    base_image=BASE_IMAGE,
    packages_to_install=resolve_dependencies(
        'google-cloud-build',
        'kfp',
        'gcsfs',
        'google-cloud-storage'
    )
)
def upload_container(
        project_id: str,
        trigger_id: str,
        save_model_details_bucket: str,
        file_name: str,
        model_one_score: dsl.Input[dsl.Metrics],
        model_two_score: dsl.Input[dsl.Metrics],
        validated_model: dsl.Output[dsl.Metrics],
):
    """
    Function to trigger cloud build over GCP,
    which create the serve model docker image
    and push to artifact registry.
    @project_id: Project Unique ID
    @trigger_id: cloud build trigger id.
    """
    import logging
    from src.model import get_validated_model, upload_model
    from src.model import save_model_details

    logger = logging.getLogger('tipper')
    logger.setLevel(logging.DEBUG)
    logger.addHandler(logging.StreamHandler())

    try:
        logging.info("Task: Setting Validated Model Metric")
        model_metrics = get_validated_model(model_one_score, model_two_score)
        validated_model.log_metric("Validated Model:", model_metrics)

        logging.info(f"Task: Creating model name: {model_metrics} dictionary")
        model_details = {
            "validated_model": model_metrics,
        }

        logging.info("Task: Saving Validated Model Details Over GCS Bucket")
        save_model_details(model_details, file_name, save_model_details_bucket)

        if upload_model(project_id, trigger_id) is True:
            logging.info("Cloud Build completed successfully passing to next component")
        else:
            logging.error("Cloud Build failed. Cannot proceed to the next component.")

    except Exception as e:
        logging.error("Failed to create serving container and push task: upload_container()")
        raise e


