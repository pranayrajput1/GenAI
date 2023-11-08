from kfp.v2.components.component_decorator import component
from kfp.v2.dsl import Artifact, Output, Model, Input, Metrics
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
        details_bucket: str,
        details_file_name: str,
        dataset_name: str,
        evaluation_score: Input[Metrics],
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
    from src.data import process_pipeline_image_details

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
                                max_replica_count=1,
                                accelerator_type=None,
                                accelerator_count=None,
                                service_account=service_account)
        logging.info(endpoint)

        vertex_endpoint.uri = endpoint.resource_name
        vertex_model.uri = model.resource_name

        logging.info("Task: Uploaded Model to an Endpoint Successfully")

        logging.info("Task: Extracting model id and endpoint id")
        deployed_display_name = f"{model_display_name}_endpoint"
        deployed_model_id = model.resource_name.split("/")[-1]
        endpoint_id = endpoint.resource_name.split("/")[-1]

        logging.info("Task: Appending ID's to the dictionary")
        model_details = {
            "deployed_display_name": deployed_display_name,
            "endpoint_id": endpoint_id,
            "deployed_model_id": deployed_model_id,
            "machine_type": machine_type,
            "dataset": dataset_name,
            "evaluation_score": evaluation_score,
        }

        logging.info("Saving Deployed Model Details Over GCS Bucket")

        try:
            logging.info(f"Saving pipeline details into file: {details_file_name}")
            process_pipeline_image_details(bucket_name=details_bucket,
                                           file_name=details_file_name,
                                           key=None,
                                           new_entry=model_details)
        except Exception as e:
            logging.error(f"Failed to save pipeline details into file: {details_file_name}")
            raise e

        logger.info('Returning deployed model details')

    except Exception as e:
        logging.error("Failed to Deployed Model To an Endpoint! Task: (serve_model_component)")
        raise e
