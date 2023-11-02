from kfp.v2.components.component_decorator import component
from components.dependencies import resolve_dependencies
from constants import BASE_IMAGE


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
        trigger_id: str
):
    """
    Function to trigger cloud build over GCP,
    which create the serve model docker image
    and push to artifact registry.
    @project_id: Project Unique ID
    @trigger_id: cloud build trigger id.
    """
    import logging
    from src.model import upload_model

    logger = logging.getLogger('tipper')
    logger.setLevel(logging.DEBUG)
    logger.addHandler(logging.StreamHandler())

    try:
        if upload_model(project_id, trigger_id) is True:
            logging.info("Cloud Build completed successfully passing to next component")
        else:
            logging.error("Cloud Build failed. Cannot proceed to the next component.")

    except Exception as e:
        logging.error("Failed to create serving container and push task: upload_container()")
        raise e
