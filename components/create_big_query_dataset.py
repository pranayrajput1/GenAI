from kfp.v2 import dsl
from kfp.v2.components.component_decorator import component
from components.dependencies import resolve_dependencies


@component(
    base_image="python:3.8",
    packages_to_install=resolve_dependencies(
        'google-cloud-bigquery'
    )
)
def create_bigquery_dataset(
        project_id: str,
        dataset_id: str,
        dataset_location: str,
        output_dataset_id: dsl.Output[str]
):
    """
    Function to create a new BigQuery dataset.

    @output_dataset_id: new dataset ID as output
    """
    import logging
    from google.cloud import bigquery

    logger = logging.getLogger('tipper')
    logger.setLevel(logging.DEBUG)
    logger.addHandler(logging.StreamHandler())

    try:
        # Construct a BigQuery client object.
        client = bigquery.Client(project=project_id)

        # Construct a full Dataset object to send to the API.
        dataset = bigquery.Dataset(f"{project_id}.{dataset_id}")

        # Specify the geographic location where the dataset should reside.
        dataset.location = dataset_location

        # Send the dataset to the API for creation, with an explicit timeout.
        dataset = client.create_dataset(dataset, timeout=30)  # Make an API request.

        logger.info(f"Created dataset {client.project}.{dataset.dataset_id}")

        # Output the new dataset ID
        output_dataset_id.uri = dataset.dataset_id

    except Exception as e:
        logger.error(f"Failed to create BigQuery dataset: {e}")
        raise e