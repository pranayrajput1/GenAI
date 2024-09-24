from kfp.v2 import dsl
from kfp.v2.components.component_decorator import component
from components.dependencies import resolve_dependencies
from typing import List
from constants import BASE_IMAGE


@component(
    base_image=BASE_IMAGE,
    packages_to_install=resolve_dependencies(
        'google-cloud-aiplatform',
    )
)
def create_feature_view(
        project: str,
        location: str,
        feature_store_id: dsl.Input[dsl.Artifact],
        feature_view_id: str,
        bq_table_uri: dsl.Input[dsl.Artifact],
        entity_id_columns: List[str],
        output_feature_view: dsl.Output[dsl.Artifact]
):
    """
    Function to create a feature view from a BigQuery source in Vertex AI Feature Store
    and synchronize it.
    @output_feature_view: ID of the created and synchronized feature view as output
    """
    import time
    from google.cloud import aiplatform
    from vertexai.resources.preview import feature_store
    from google.cloud.aiplatform_v1beta1 import FeatureOnlineStoreAdminServiceClient
    from src.data import get_logger

    logger = get_logger()

    try:
        aiplatform.init(project=project, location=location)

        logger.info(f"Creating feature view: {feature_view_id}")
        logger.info(f"Using feature online store: {feature_store_id}")
        logger.info(f"BigQuery table URI: {bq_table_uri}")
        logger.info(f"Entity ID columns: {entity_id_columns}")

        fos = feature_store.FeatureOnlineStore(feature_store_id)

        fv = fos.create_feature_view(
            name=feature_view_id,
            source=feature_store.utils.FeatureViewBigQuerySource(
                uri=bq_table_uri, entity_id_columns=entity_id_columns
            ),
        )

        logger.info(f"Successfully created feature view: {feature_view_id}")

        api_endpoint = f"{location}-aiplatform.googleapis.com"
        admin_client = FeatureOnlineStoreAdminServiceClient(
            client_options={"api_endpoint": api_endpoint}
        )

        sync_response = admin_client.sync_feature_view(
            feature_view=f"projects/{project}/locations/{location}/featureOnlineStores/{feature_store_id}/featureViews/{feature_view_id}"
        )

        logger.info("Initiated feature view synchronization")

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

        output_feature_view.uri = fv.name

    except Exception as e:
        logger.error(f"Failed to create or sync feature view: {e}")
