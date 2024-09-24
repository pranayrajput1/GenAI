from kfp.v2 import dsl
from kfp.v2.components.component_decorator import component
from components.dependencies import resolve_dependencies
from constants import BASE_IMAGE


@component(
    base_image=BASE_IMAGE,
    packages_to_install=resolve_dependencies(
        'google-cloud-aiplatform',
    )
)
def create_feature_store(
        project: str,
        location: str,
        feature_store_id: str,
        output_feature_store: dsl.Output[dsl.Artifact]
):
    """
    Function to create a BigTable feature online store using Vertex AI Feature Store.
    @output_feature_store: ID of the created feature online store as output
    """
    from google.cloud import aiplatform
    from vertexai.resources.preview import feature_store
    from src.data import get_logger

    logger = get_logger()

    try:
        aiplatform.init(project=project, location=location)

        logger.info(f"Creating BigTable feature online store: {feature_store_id}")

        feature_store.FeatureOnlineStore.create_bigtable_store(
            feature_store_id
        )

        logger.info(f"Successfully created BigTable feature online store: {feature_store_id}")
        output_feature_store.uri = feature_store_id

    except Exception as e:
        logger.error(f"Failed to create BigTable feature online store: {e}")
