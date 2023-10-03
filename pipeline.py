import logging
from google.cloud import aiplatform
from kfp.v2 import compiler
import kfp
from components.evaluate_model import evaluate_model
from components.fetch_data import fetch_dataset
from components.process_data import pre_process_data
from components.serve_model import serve_model_component
from components.train import fit_model
from components.upload_model import upload_container
from constants import (PIPELINE_NAME, PIPELINE_DESCRIPTION, PIPELINE_ROOT_GCS, BATCH_SIZE, CLUSTER_IMAGE_BUCKET, \
                       TRIGGER_ID, REGION, STAGING_BUCKET, SERVING_IMAGE, MODEL_DISPLAY_NAME, SERVICE_ACCOUNT_ML,
                       dataset_bucket, dataset_name, fit_model_name, model_details_file_name)

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


@kfp.dsl.pipeline(name=PIPELINE_NAME,
                  description=PIPELINE_DESCRIPTION,
                  pipeline_root=PIPELINE_ROOT_GCS)
def pipeline(
        project_id: str,
        job_id: str
):
    """Fetching Dataset from GCs Bucket"""
    fetch_data_task = fetch_dataset(dataset_bucket, dataset_name).set_display_name("Fetch Dataset")

    """Pre-Processing Dataset"""
    process_data_task = pre_process_data(fetch_data_task.output, BATCH_SIZE).set_display_name("Pre-Process Data") \
        .after(fetch_data_task)

    """Fit model pipeline task"""
    fit_model_task = fit_model(fit_model_name, process_data_task.output) \
        .after(process_data_task) \
        .set_display_name("Fit DB_Scan Model") \
        .set_cpu_request("4") \
        .set_memory_limit("16G")

    """Evaluate model component"""
    # model_evaluation_task = evaluate_model(BATCH_SIZE, CLUSTER_IMAGE_BUCKET,
    #                                        process_data_task.output, fit_model_task.output) \
    #     .set_display_name("Model_Validation")

    """Upload model Component"""
    # upload_model_task = upload_container(project_id, TRIGGER_ID) \
    #     .after(model_evaluation_task) \
    #     .set_display_name("Model_Upload")

    # serve_model_task = serve_model_component(project_id,
    #                                          REGION,
    #                                          STAGING_BUCKET,
    #                                          SERVING_IMAGE,
    #                                          MODEL_DISPLAY_NAME,
    #                                          SERVICE_ACCOUNT_ML,
    #                                          save_model_details_bucket=dataset_bucket,
    #                                          model_details_file_name=model_details_file_name
    #                                          ) \
    #     .after(upload_model_task) \
    #     .set_display_name("Serve_Model")


def compile_pipeline(pipeline_template_name='./dbscan_pipeline.json'):
    compiler.Compiler().compile(
        pipeline_func=pipeline,
        package_path=pipeline_template_name
    )
    return None


def delete_experiment_sample(
        experiment_name: str,
        project: str,
        location: str,
        delete_backing_tensorboard_runs: bool = False,
):
    experiment = aiplatform.Experiment(
        experiment_name=experiment_name, project=project, location=location
    )

    experiment.delete(delete_backing_tensorboard_runs=delete_backing_tensorboard_runs)


if __name__ == "__main__":
    compile_pipeline()
