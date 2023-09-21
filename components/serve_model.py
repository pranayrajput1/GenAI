from kfp.v2.components.component_decorator import component
from kfp.v2.dsl import Artifact, Output, Model
from components.dependencies import resolve_dependencies
from constants import BASE_IMAGE


@component(
    base_image=BASE_IMAGE,
    packages_to_install=resolve_dependencies(
        'google-cloud-aiplatform'
    )
)
def serve_model_component(
        project_id: str,
        location: str,
        staging_bucket: str,
        serving_image_uri: str,
        model_display_name: str,
        service_account: str,
        vertex_endpoint: Output[Artifact],
        vertex_model: Output[Model],
        machine_type: str = 'e2-standard-2',

):
    """
    Function to upload model to model registry,
    serve model to an endpoint.
    @project_id: Project Unique ID.
    @location: specific region project located at.
    @staging_bucket: GCS Bucket used for model staging.
    @serving_image_uri: Serving Docker Image Path over GCP Artifact Registry.
    @model_display_name: Model display name
    @vertex_endpoint: created endpoint address
    @vertex_model: Model located at model registry
    """
    from google.cloud import aiplatform
    import logging

    logger = logging.getLogger('tipper')
    logger.setLevel(logging.DEBUG)
    logger.addHandler(logging.StreamHandler())

    try:
        logging.info(f"Task: Initiating aiplatform for project: {project_id} & {location}")
        aiplatform.init(project=project_id, location=location, staging_bucket=staging_bucket)

        logging.info("Task: Uploading model to model registry")

        model = aiplatform.Model.upload(display_name=model_display_name,
                                        location=location,
                                        serving_container_image_uri=serving_image_uri,
                                        serving_container_ports=[8080]
                                        )

        logging.info("Task: Uploaded Model to Model Registry Successfully")

        logging.info("Task: Deploying Model to an endpoint")
        endpoint = model.deploy(machine_type=machine_type,
                                min_replica_count=1,
                                max_replica_count=2,
                                accelerator_type=None,
                                accelerator_count=None,
                                service_account=service_account)
        logging.info(endpoint)

        vertex_endpoint.uri = endpoint.resource_name
        vertex_model.uri = model.resource_name

        logging.info("Task: Uploaded Model to an Endpoint Successfully")

    except Exception as e:
        logging.error("Failed to Deployed Model To an Endpoint! Task: (serve_model_component)")
        raise e
