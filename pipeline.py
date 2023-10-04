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
from constants import (PIPELINE_NAME, PIPELINE_DESCRIPTION, PIPELINE_ROOT_GCS, BATCH_SIZE, cluster_image_bucket, \
                       TRIGGER_ID, REGION, STAGING_BUCKET, SERVING_IMAGE, MODEL_DISPLAY_NAME, SERVICE_ACCOUNT_ML,
                       dataset_bucket, dataset_name, model_details_file_name, validated_file_name,
                       fit_db_model_name, fit_k_means_model_name)

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

    """Fit DB-Scan model pipeline task"""
    fit_db_scan_model = fit_model(fit_db_model_name, process_data_task.output) \
        .after(process_data_task) \
        .set_display_name("Fit DB-Scan Model") \
        .set_cpu_request("4") \
        .set_memory_limit("16G")

    """Evaluate model component"""
    db_scan_evaluation = evaluate_model(BATCH_SIZE,
                                        fit_db_model_name,
                                        cluster_image_bucket,
                                        fetch_data_task.output,
                                        fit_db_scan_model.output) \
        .after(fit_db_scan_model) \
        .set_display_name("DB-Scan Score")

    """Fit K-Means model pipeline task"""
    fit_k_means_model = fit_model(fit_k_means_model_name, process_data_task.output) \
        .after(process_data_task) \
        .set_display_name("Fit K-Means Model") \
        .set_cpu_request("4") \
        .set_memory_limit("16G")

    """Evaluate model component"""
    k_means_evaluation = evaluate_model(BATCH_SIZE,
                                        fit_k_means_model_name,
                                        cluster_image_bucket,
                                        fetch_data_task.output,
                                        fit_k_means_model.output) \
        .after(fit_k_means_model) \
        .set_display_name("K-Means Score")

    """Model Validation & Upload Model Component"""
    upload_model_task = upload_container(project_id,
                                         TRIGGER_ID,
                                         dataset_bucket,
                                         validated_file_name,
                                         db_scan_evaluation.outputs["avg_score"],
                                         k_means_evaluation.outputs["avg_score"]) \
        .after(db_scan_evaluation) \
        .after(k_means_evaluation) \
        .set_display_name("Model_Validation_&_Upload")

    serve_model_component(project_id,
                          REGION,
                          STAGING_BUCKET,
                          SERVING_IMAGE,
                          MODEL_DISPLAY_NAME,
                          SERVICE_ACCOUNT_ML,
                          save_model_details_bucket=dataset_bucket,
                          model_details_file_name=model_details_file_name
                          ) \
        .after(upload_model_task) \
        .set_display_name("Serve_Model")


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
