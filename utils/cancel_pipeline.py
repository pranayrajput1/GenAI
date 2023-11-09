from google.cloud import aiplatform

from constants import PROJECT_ID


def cancel_custom_job_sample(
        project: str,
        custom_job_id: str,
        location: str = "us-central1",
):
    pipeline_client = aiplatform.gapic.PipelineServiceClient(
        client_options={"api_endpoint": f"{location}-aiplatform.googleapis.com"})

    pipeline_job_name = f"projects/{project}/locations/{location}/pipelineJobs/{custom_job_id}"

    pipeline_client.cancel_pipeline_job(name=pipeline_job_name)


def delete_custom_job_sample(
    project: str,
    custom_job_id: str,
    location: str = "us-central1"
):
    pipeline_client = aiplatform.gapic.PipelineServiceClient(
        client_options={"api_endpoint": f"{location}-aiplatform.googleapis.com"})

    pipeline_job_name = f"projects/{project}/locations/{location}/pipelineJobs/{custom_job_id}"

    pipeline_client.delete_pipeline_job(name=pipeline_job_name)


cancel_custom_job_sample(PROJECT_ID, 'clustering-kubeflow-20231108120549')
# delete_custom_job_sample(PROJECT_ID, "clustering-kubeflow-20231102142718")
