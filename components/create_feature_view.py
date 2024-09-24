from kfp.v2 import dsl
from kfp.v2.components.component_decorator import component
from components.dependencies import resolve_dependencies
from typing import List

@component(
    base_image="python:3.8",
    packages_to_install=resolve_dependencies(
        'google-cloud-aiplatform',
        'vertexai',
    )
)
def create_and_sync_feature_view_from_bq_source(
    project: str,
    location: str,
    existing_feature_online_store_id: str,
    feature_view_id: str,
    bq_table_uri: str,
    entity_id_columns: List[str],
    output_feature_view: dsl.Output[str]
):
    """
    Function to create a feature view from a BigQuery source in Vertex AI Feature Store
    and synchronize it.

    @output_feature_view: ID of the created and synchronized feature view as output
    """
    import logging
    import time
    from google.cloud import aiplatform
    from vertexai.resources.preview import feature_store
    from google.cloud.aiplatform_v1beta1 import FeatureOnlineStoreAdminServiceClient

    logger = logging.getLogger('feature_view_creator')
    logger.setLevel(logging.DEBUG)
    logger.addHandler(logging.StreamHandler())

    try:
        # Initialize Vertex AI
        aiplatform.init(project=project, location=location)

        logger.info(f"Creating feature view: {feature_view_id}")
        logger.info(f"Using feature online store: {existing_feature_online_store_id}")
        logger.info(f"BigQuery table URI: {bq_table_uri}")
        logger.info(f"Entity ID columns: {entity_id_columns}")

        # Get the existing feature online store
        fos = feature_store.FeatureOnlineStore(existing_feature_online_store_id)

        # Create the feature view
        fv = fos.create_feature_view(
            name=feature_view_id,
            source=feature_store.utils.FeatureViewBigQuerySource(
                uri=bq_table_uri, entity_id_columns=entity_id_columns
            ),
        )

        logger.info(f"Successfully created feature view: {feature_view_id}")

        # Set up the API endpoint and admin client for synchronization
        api_endpoint = f"{location}-aiplatform.googleapis.com"
        admin_client = FeatureOnlineStoreAdminServiceClient(
            client_options={"api_endpoint": api_endpoint}
        )

        # Initiate feature view synchronization
        sync_response = admin_client.sync_feature_view(
            feature_view=f"projects/{project}/locations/{location}/featureOnlineStores/{existing_feature_online_store_id}/featureViews/{feature_view_id}"
        )

        logger.info("Initiated feature view synchronization")

        # Wait for synchronization to complete
        while True:
            feature_view_sync = admin_client.get_feature_view_sync(
                name=sync_response.feature_view_sync
            )
            if feature_view_sync.run_time.end_time.seconds > 0:
                status = "Succeeded" if feature_view_sync.final_status.code == 0 else "Failed"
                logger.info(f"Sync {status} for {feature_view_sync.name}.")
                break
            else:
                logger.info("Sync ongoing, waiting for 30 seconds.")
            time.sleep(30)

        # Set the output for the pipeline
        output_feature_view.uri = fv.name

    except Exception as e:
        logger.error(f"Failed to create or sync feature view: {e}")
        raise e