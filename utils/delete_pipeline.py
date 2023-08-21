from google.cloud import aiplatform

from constants import project_id


def delete_training_pipeline_sample(
        project: str,
        training_pipeline_id: str,
        location: str = "us-central1",
        api_endpoint: str = "us-central1-aiplatform.googleapis.com",
        timeout: int = 300,
):
    # The AI Platform services require regional API endpoints.
    client_options = {"api_endpoint": api_endpoint}
    # Initialize client that will be used to create and send requests.
    # This client only needs to be created once, and can be reused for multiple requests.
    client = aiplatform.gapic.PipelineServiceClient(client_options=client_options)
    name = client.training_pipeline_path(
        project=project, location=location, training_pipeline=training_pipeline_id
    )
    response = client.delete_training_pipeline(name=name)
    print("Long running operation:", response.operation.name)
    delete_training_pipeline_response = response.result(timeout=timeout)
    print("delete_training_pipeline_response:", delete_training_pipeline_response)


pipeline_id = "dolly-llm-pipeline-20230817160905"
delete_training_pipeline_sample(project_id, pipeline_id)
