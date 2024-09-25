import logging
from google.cloud import aiplatform
from kfp import compiler
import kfp
from components.evaluate_model import evaluate_model
from components.fetching_data import process_data
from components.process_data import pre_process_data
from components.serve_model import serve_model_component
from components.train import fit_model
from components.upload_model import upload_container
from components.create_online_feature_store import create_feature_store
from components.create_table import create_table
from components.create_big_query_dataset import create_bigquery_dataset
from components.create_feature_view import create_feature_view
from constants import (BATCH_SIZE,PIPELINE_NAME, PIPELINE_DESCRIPTION, PIPELINE_ROOT_GCS, REGION,
                       PROJECT_ID, DATASET_ID,DATASET_LOCATION, TABLE_ID, CSV_FILE_NAME, FEATURE_STORE_ID, FEATURE_STORE_VIEW_ID,
                       RESOURCE_BUCKET,COMPILE_PIPELINE_JSON)

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


@kfp.dsl.pipeline(name=PIPELINE_NAME,
                  description=PIPELINE_DESCRIPTION,
                  pipeline_root=PIPELINE_ROOT_GCS)
def pipeline(
        project_id: str,
        job_id: str
):
    #
    # create_dataset_task = (create_bigquery_dataset(project_id=PROJECT_ID,
    #                                                dataset_id=DATASET_ID,
    #                                                dataset_location=DATASET_LOCATION,
    #                                                )
    #                        .set_display_name("Create Dataset"))
    #
    # create_table_task = (create_table(project_id=PROJECT_ID,
    #                                   dataset_id=DATASET_ID,
    #                                   new_table_id=TABLE_ID,
    #                                   data_bucket=RESOURCE_BUCKET,
    #                                   csv_file_name=CSV_FILE_NAME)
    #                      .set_display_name("Create Table")
    #                      .after(create_dataset_task))
    #
    # create_feature_store_task = ((create_feature_store(project=PROJECT_ID,
    #                                                    location=REGION,
    #                                                    feature_store_id=FEATURE_STORE_ID)
    #                               .set_display_name("Create Feature Store"))
    #                              .after(create_table_task))
    #
    # create_feature_view_task = (create_feature_view(project=PROJECT_ID,
    #                                                 location=REGION,
    #                                                 feature_store_id=create_feature_store_task.output,
    #                                                 feature_view_id=FEATURE_STORE_VIEW_ID,
    #                                                 bq_table_uri=f"bq://{PROJECT_ID}.{create_table_task.output}",
    #                                                 entity_id_columns=["id"])
    #                             .set_display_name("Create Feature View")
    #                             .after(create_feature_store_task))
    #
    # process_data_task = (process_data(project=PROJECT_ID,
    #                                   location=REGION,
    #                                   feature_view_id=create_feature_view_task.output,
    #                                   feature_online_store_id=create_feature_store_task.output)
    #                      .set_display_name("Process Data"))


    """Pre-Processing Dataset"""
    process_data_task = pre_process_data(project_id=project_id,region="us-central1",)\
    .set_display_name("Pre processing data")


    # """Fit model pipeline task"""
    # train_model = fit_model(x_train_path=process_data_task.outputs["x_train_path"], y_train_path=process_data_task.outputs["y_train_path"]) \
    #     .after(process_data_task) \
    #     .set_display_name("Fit Model") \
    #     .set_cpu_request("4") \
    #     .set_memory_limit("16G")
    #
    # """Evaluate model component"""
    # model_evaluation = evaluate_model(trained_model=train_model.output,
    #                                   x_test_path=process_data_task.outputs["x_test_path"],
    #                                   y_test_path=process_data_task.outputs["y_test_path"]) \
    #     .after(train_model) \
    #     .set_display_name("Evaluate Model Performance")

    # """Upload Model Component"""
    # upload_model_task = upload_container(project_id,
    #                                      TRIGGER_ID) \
    #     .after(model_evaluation) \
    #     .set_display_name("Upload Model")

    # '''Serving model to endpoint'''
    # serve_model_component(project_id,
    #                       REGION,
    #                       STAGING_BUCKET,
    #                       SERVING_IMAGE,
    #                       MODEL_DISPLAY_NAME,
    #                       SERVICE_ACCOUNT_ML,
    #                       RESOURCE_BUCKET,
    #                       SAVE_MODEL_DETAILS_FILE) \
    #     .after(upload_model_task) \
    #     .set_display_name("Serve Model")


def compile_pipeline(pipeline_template_name=f'./{COMPILE_PIPELINE_JSON}'):
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
