from google.cloud import aiplatform


def delete_endpoint_sample(
        project: str,
        endpoint_id: str,
        location: str = "us-central1",
        api_endpoint: str = "us-central1-aiplatform.googleapis.com",
        timeout: int = 300,
):
    # The AI Platform services require regional API endpoints.
    client_options = {"api_endpoint": api_endpoint}
    # Initialize client that will be used to create and send requests.
    # This client only needs to be created once, and can be reused for multiple requests.
    client = aiplatform.gapic.EndpointServiceClient(client_options=client_options)
    name = client.endpoint_path(
        project=project, location=location, endpoint=endpoint_id
    )
    response = client.delete_endpoint(name=name)
    print("Long running operation:", response.operation.name)
    delete_endpoint_response = response.result(timeout=timeout)
    print("delete_endpoint_response:", delete_endpoint_response)


delete_endpoint_sample("nashtech-ai-dev-389315", "1737969442723201024")
