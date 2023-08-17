from kfp.v2.components.component_decorator import component
from components.dependencies import resolve_dependencies
from constants import base_image


@component(
    base_image=base_image,
    packages_to_install=resolve_dependencies(
        'google-cloud-build'
    )
)
def upload_container(project_id: str,
                     trigger_id: str,
                     component_execution: bool
                     ):
    """
    Function to trigger cloud build over GCP, which create the serve model docker image.
    and push to artifact registry.
    @project_id: Project ID
    @trigger_id: cloud build trigger ID.
    """
    from google.cloud.devtools import cloudbuild_v1
    import logging

    logger = logging.getLogger('tipper')
    logger.setLevel(logging.DEBUG)
    logger.addHandler(logging.StreamHandler())

    try:
        if not component_execution:
            logging.info("Component execution: upload serving container image is bypassed")
        else:
            def upload_model(get_project_id, get_trigger_id):
                logging.info("Making Client Connection: ")
                cloud_build_client = cloudbuild_v1.CloudBuildClient()

                logging.info("Triggering Cloud Build For DB Scan Serving Container")
                response = cloud_build_client.run_build_trigger(project_id=get_project_id, trigger_id=get_trigger_id)

                if response.result():
                    logging.info("Cloud Build Successful")
                    return True
                else:
                    logging.info("Cloud Build Failed !")
                    raise RuntimeError

            if upload_model(project_id, trigger_id) is True:
                logging.info("Cloud Build completed successfully passing to next component")
                pass

    except Exception as e:
        logging.error("Failed to create serving container and push task")
        raise e
