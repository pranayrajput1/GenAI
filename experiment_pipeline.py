import logging
from google.cloud import aiplatform
from kfp.v2 import compiler
import kfp
from experiments_components.experiment_evaluate_model import evaluate_model
from experiments_components.experiment_fetch_data import fetch_dataset
from experiments_components.experiment_process_data import pre_process_data
from experiments_components.experiment_train import fit_model
from constants import (PIPELINE_NAME, PIPELINE_DESCRIPTION, PIPELINE_ROOT_GCS, BATCH_SIZE, cluster_image_bucket,
                       dataset_bucket, dataset_name, validated_file_name, fit_db_model_name, fit_k_means_model_name,
                       experiment_pipeline)

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

    """Db-Scan Pre-Processing Dataset"""
    db_scan_process_data_task = pre_process_data(fetch_data_task.output, BATCH_SIZE).set_display_name(
        "Db-Scan-Pre-Process Data") \
        .after(fetch_data_task)

    """Fit DB-Scan model pipeline task"""
    fit_db_scan_model = fit_model(fit_db_model_name, db_scan_process_data_task.output) \
        .after(db_scan_process_data_task) \
        .set_display_name("Fit DB-Scan Model") \
        .set_cpu_request("4") \
        .set_memory_limit("16G")

    """K-Means Pre-Processing Dataset"""
    k_means_process_data_task = pre_process_data(fetch_data_task.output, BATCH_SIZE).set_display_name(
        "K-Means-Pre-Process Data") \
        .after(fetch_data_task)

    """Fit K-Means model pipeline task"""
    fit_k_means_model = fit_model(fit_k_means_model_name, k_means_process_data_task.output) \
        .after(k_means_process_data_task) \
        .set_display_name("Fit K-Means Model") \
        .set_cpu_request("4") \
        .set_memory_limit("16G")

    """Evaluate model component"""
    evaluate_model(
        BATCH_SIZE,
        cluster_image_bucket,
        validated_file_name,
        fetch_data_task.output,
        fit_db_scan_model.output,
        fit_k_means_model.output) \
        .after(fit_k_means_model) \
        .set_display_name("Model Validation")


def compile_pipeline(pipeline_template_name=experiment_pipeline):
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
