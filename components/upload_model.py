from kfp.v2.components.component_decorator import component
from components.dependencies import resolve_dependencies
from constants import base_image, project_id, pipeline_name, serving_trigger_id, component_execution
from utils.email_credentials import email, password, receiver
from utils.send_email import send_cloud_build_success_email


# @component(
#     base_image=base_image,
#     packages_to_install=resolve_dependencies(
#         'google-cloud-build'
#     )
# )
def upload_container(project_id: str,
                     pipeline_name: str,
                     trigger_id: str,
                     component_execution: bool,
                     user_email: str,
                     user_email_password: str,
                     receiver_email: str
                     ):
    """
    Function to trigger cloud build over GCP, which create the serve model docker image.
    and push to artifact registry.
    @project_id: Project ID
    @trigger_id: cloud build trigger ID.
    """
    from google.cloud.devtools import cloudbuild_v1
    from utils.send_email import send_cloud_build_failed_email
    import logging

    logger = logging.getLogger('tipper')
    logger.setLevel(logging.DEBUG)
    logger.addHandler(logging.StreamHandler())

    if not component_execution:
        logging.info("Component execution: upload serving container image is bypassed")
    else:
        def upload_model(get_project_id, get_trigger_id):
            logging.info("Task: Making Client Connection: ")
            cloud_build_client = cloudbuild_v1.CloudBuildClient()

            logging.info("Task: Triggering Cloud Build For Dolly Model Serving Container")
            response = cloud_build_client.run_build_trigger(project_id=get_project_id, trigger_id=get_trigger_id)

            if response.done:
                build = response.result()

                if build.status == cloudbuild_v1.Build.Status.SUCCESS:
                    logging.info("Cloud Build Successful")
                    return True

                elif build.status == cloudbuild_v1.Build.Status.FAILURE:
                    logging.error("Cloud Build Failed")
                    raise RuntimeError

        try:
            if upload_model(project_id, trigger_id) is True:
                logging.info("Cloud Build completed successfully passing to next component")

                logging.error(f"Sending Cloud Build Success Email to: {receiver_email}")
                send_cloud_build_success_email(project_id, pipeline_name, user_email, user_email_password,
                                               receiver_email)
                pass

        except Exception as exc:
            logging.error("Some error occurred in upload model component!")
            logging.error(f"Sending Cloud Build Failure Email to: {receiver_email}")
            send_cloud_build_failed_email(project_id, pipeline_name, user_email, user_email_password,
                                          receiver_email)
            raise exc


upload_container(project_id, pipeline_name, serving_trigger_id, component_execution, email, password, receiver)
