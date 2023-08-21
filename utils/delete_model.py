from google.cloud import aiplatform

from constants import project_id, project_region


def undeploy_model_from_endpoint(names):
    endpoints = aiplatform.Endpoint.list()
    for i in endpoints:
        if str(i.display_name) == names:
            i.undeploy_all()


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


def delete_model_sample(model_id: str, project: str, location: str):
    """
    Delete a Model resource.
    Args:
        model_id: The ID of the model to delete. Parent resource name of the model is also accepted.
        project: The project.
        location: The region name.
    Returns
        None.
    """
    # Initialize the client.
    aiplatform.init(project=project, location=location)

    # Get the model with the ID 'model_id'. The parent_name of Model resource can be also
    # 'projects/<your-project-id>/locations/<your-region>/models/<your-model-id>'
    model = aiplatform.Model(model_name=model_id)

    # Delete the model.
    model.delete()


undeploy_model_from_endpoint("dolly_v2_3b_endpoint")
delete_endpoint_sample(project_id, "9031549039249719296")
delete_model_sample("4985350647080550400", project_id, project_region)
