from kfp.v2.components.component_decorator import component
from components.dependencies import resolve_dependencies
from constants import BASE_IMAGE


@component(
    base_image=BASE_IMAGE,
    packages_to_install=resolve_dependencies(
        'google-cloud-build'
    )
)
def upload_container(project_id: str,
                     trigger_id: str,
                     ):
    """
    Function to trigger cloud build over GCP,
    which create the serve model docker image
    and push to artifact registry.
    @project_id: Project Unique ID
    @trigger_id: cloud build trigger id.
    """
    from google.cloud.devtools import cloudbuild_v1
    import logging

    logger = logging.getLogger('tipper')
    logger.setLevel(logging.DEBUG)
    logger.addHandler(logging.StreamHandler())

    try:
        def upload_model(project, trigger):
            logging.info("Making Client Connection: ")
            cloud_build_client = cloudbuild_v1.CloudBuildClient()

            logging.info("Triggering Cloud Build For DB Scan Serving Container")
            response = cloud_build_client.run_build_trigger(project_id=project, trigger_id=trigger)

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
        logging.error("Failed to create serving container and push task: upload_container()")
        raise e
