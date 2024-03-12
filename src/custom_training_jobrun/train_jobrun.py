from google.cloud import aiplatform
from Mlops.src.utils.constant import PROJECT_ID,REGION
def run_custom_job():
    aiplatform.init(project=PROJECT_ID, location=REGION)
    job = aiplatform.CustomContainerTrainingJob(
        display_name= 'Custom_house',
        container_uri='gcr.io/nashtech-ai-dev-389315/my-python-app@sha256:9b50d4446415bc650d4ffe64547a96f58a806c818633d10319fdcca4ee9cc407',
        staging_bucket='nashtech_vertex_ai_artifact/output'
    )
    job.run(
        replica_count=1,
        machine_type='n1-standard-4',
    )
run_custom_job()
