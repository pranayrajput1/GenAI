from kfp.v2 import dsl
from kfp.v2.components.component_decorator import component
from components.dependencies import resolve_dependencies

@component(
    base_image="python:3.8",
    packages_to_install=resolve_dependencies(
        'google-cloud-aiplatform',
        'vertexai',
    )
)
def create_bigtable_feature_online_store(
    project: str,
    location: str,
    feature_online_store_id: str,
    output_feature_store: dsl.Output[str]
):
    """
    Function to create a BigTable feature online store using Vertex AI Feature Store.

    @output_feature_store: ID of the created feature online store as output
    """
    import logging
    from google.cloud import aiplatform
    from vertexai.resources.preview import feature_store

    logger = logging.getLogger('feature_store_creator')
    logger.setLevel(logging.DEBUG)
    logger.addHandler(logging.StreamHandler())

    try:
        # Initialize Vertex AI
        aiplatform.init(project=project, location=location)

        logger.info(f"Creating BigTable feature online store: {feature_online_store_id}")

        # Create the BigTable feature online store
        feature_store.FeatureOnlineStore.create_bigtable_store(
            feature_online_store_id
        )

        logger.info(f"Successfully created BigTable feature online store: {feature_online_store_id}")

        # Set the output for the pipeline
        output_feature_store.uri = feature_online_store_id

    except Exception as e:
        logger.error(f"Failed to create BigTable feature online store: {e}")
        raise e