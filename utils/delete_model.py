from google.cloud import aiplatform

from constants import project_id, project_region


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


delete_model_sample("599055716254220288", project_id, project_region)
