from google.cloud import aiplatform
from Mlops.src.utils.constant import PROJECT_ID,REGION
def run_custom_job():
    aiplatform.init(project=PROJECT_ID, location=REGION)
    job = aiplatform.CustomContainerTrainingJob(
        display_name= 'Custom_house',
        container_uri='gcr.io/nashtech-ai-dev-389315/my-house-app@sha256:4148ade1e956e8a8cf31bb0380927d60769495eaed275ed1f789af1abe14b4eb',
        staging_bucket='nashtech_vertex_ai_artifact/output'
    )
    job.run(
        replica_count=1,
        machine_type='n1-standard-4',
    )
run_custom_job()
