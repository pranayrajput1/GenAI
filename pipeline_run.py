from google.cloud import aiplatform
from datetime import datetime
import os
from constants import project_region, service_account, pipeline_name, project_id


def run_pipeline_job(
        sync: bool = False, *,
        pipeline_template_name: str = './llm_pipeline.json',
        cleanup_compiled_pipeline: bool = False,
        enable_caching: bool = False,
) -> aiplatform.PipelineJob:
    job_id = f'{pipeline_name}-{datetime.now().strftime("%Y%m%d%H%M%S")}'
    experiment_name = f'{pipeline_name}-kubeflow'

    params = dict(
        project_id=project_id,
        job_id=job_id,
    )

    aiplatform.init(project=project_id, location=project_region)

    pipeline_job = aiplatform.PipelineJob(
        project=project_id,
        location=project_region,
        display_name=pipeline_name,
        template_path=pipeline_template_name,
        parameter_values=params,
        job_id=job_id,
        enable_caching=enable_caching,
    )

    try:
        pipeline_job.submit(service_account=service_account, experiment=experiment_name)
        if sync:
            pipeline_job.wait()
    finally:
        if cleanup_compiled_pipeline:
            os.remove(pipeline_template_name)

    return pipeline_job


if __name__ == "__main__":
    run_pipeline_job()
