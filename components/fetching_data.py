from kfp import dsl
from kfp.dsl import component
from components.dependencies import resolve_dependencies
from constants import BASE_IMAGE


@component(
    base_image=BASE_IMAGE,
    packages_to_install=resolve_dependencies(
        'google-cloud-aiplatform',
        'pandas',
    )
)
def process_data(
        project: str,
        location: str,
        feature_view_id: dsl.Input[dsl.Artifact],
        feature_online_store_id: dsl.Input[dsl.Artifact],
        output_csv: dsl.Output[dsl.Dataset],
        batch_size: int = 1000
):
    """
    Function to fetch all data from a Vertex AI Feature Store feature view,
    process it, and output as a CSV file.
    @output_csv: Path to the output CSV file containing the processed data
    """
    from google.cloud import aiplatform
    from vertexai.resources.preview import FeatureView
    from utils.helpers import setup_data_client, fetch_all_data, process_feature_store_data
    from src.data import get_logger

    logger = get_logger()

    try:
        aiplatform.init(project=project, location=location)

        logger.info(f"Fetching data from feature view: {feature_view_id}")
        logger.info(f"Using feature online store: {feature_online_store_id}")

        data_client = setup_data_client(location)

        fv = FeatureView(name=feature_view_id, feature_online_store_id=feature_online_store_id)

        logger.info("Starting to fetch all data")
        data = fetch_all_data(data_client, fv.resource_name, batch_size)
        logger.info(f"Fetched {len(data)} batches of data")

        logger.info("Processing fetched data")
        df = process_feature_store_data(data)
        logger.info(f"Processed data into DataFrame with shape: {df.shape}")

        logger.info(f"Dataset Features: {df.columns}")
        logger.info(f'Task: saving data to parquet file at path: {output_csv.uri}')
        df.to_parquet(output_csv.path)

    except Exception as e:
        logger.error(f"Failed to fetch or process feature store data: {e}")
        raise e