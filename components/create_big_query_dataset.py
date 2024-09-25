from kfp import dsl
from kfp.dsl import component
from components.dependencies import resolve_dependencies
from constants import BASE_IMAGE


@component(
    base_image=BASE_IMAGE,
    packages_to_install=resolve_dependencies(
        'google-cloud-bigquery'
    )
)
def create_bigquery_dataset(
        project_id: str,
        dataset_id: str,
        dataset_location: str,
        output_dataset_id: dsl.Output[dsl.Artifact]
):
    """
    Function to create a new BigQuery dataset.
    @output_dataset_id: new dataset ID as output
    """
    from google.cloud import bigquery
    from src.data import get_logger

    logger = get_logger()

    try:
        client = bigquery.Client(project=project_id)

        dataset = bigquery.Dataset(f"{project_id}.{dataset_id}")

        dataset.location = dataset_location

        dataset = client.create_dataset(dataset, timeout=30)

        logger.info(f"Created dataset {client.project}.{dataset.dataset_id}")

        output_dataset_id.uri = dataset.dataset_id

    except Exception as e:
        logger.error(f"Failed to create BigQuery dataset: {e}")
        raise e