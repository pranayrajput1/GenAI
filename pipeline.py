from google.cloud import aiplatform
from kfp.v2 import compiler
import kfp
from components.create_big_query_dataset import create_bigquery_dataset
from components.create_feature_view import create_feature_view
from components.create_online_feature_store import create_feature_store
from components.create_table import create_table
from components.process_data import process_data
from constants import (PIPELINE_NAME, PIPELINE_DESCRIPTION, PIPELINE_ROOT_GCS, REGION,
                       RESOURCE_BUCKET, COMPILE_PIPELINE_JSON, PROJECT_ID, DATASET_ID,
                       DATASET_LOCATION, TABLE_ID, CSV_FILE_NAME, FEATURE_STORE_ID, FEATURE_STORE_VIEW_ID)


@kfp.dsl.pipeline(name=PIPELINE_NAME,
                  description=PIPELINE_DESCRIPTION,
                  pipeline_root=PIPELINE_ROOT_GCS)
def pipeline(
        project_id: str,
        job_id: str
):
    create_dataset_task = (create_bigquery_dataset(project_id=PROJECT_ID,
                                                   dataset_id=DATASET_ID,
                                                   dataset_location=DATASET_LOCATION,
                                                   )
                           .set_display_name("Create Dataset"))

    create_table_task = (create_table(project_id=PROJECT_ID,
                                      dataset_id=DATASET_ID,
                                      new_table_id=TABLE_ID,
                                      data_bucket=RESOURCE_BUCKET,
                                      csv_file_name=CSV_FILE_NAME)
                         .set_display_name("Create Table")
                         .after(create_dataset_task))

    create_feature_store_task = ((create_feature_store(project=PROJECT_ID,
                                                       location=REGION,
                                                       feature_store_id=FEATURE_STORE_ID)
                                  .set_display_name("Create Feature Store"))
                                 .after(create_table_task))

    create_feature_view_task = (create_feature_view(project=PROJECT_ID,
                                                    location=REGION,
                                                    feature_store_id=create_feature_store_task.output,
                                                    feature_view_id=FEATURE_STORE_VIEW_ID,
                                                    bq_table_uri=create_table_task.output,
                                                    entity_id_columns=["id"])
                                .set_display_name("Create Feature View")
                                .after(create_feature_store_task))

    process_data_task = (process_data(project=PROJECT_ID,
                                      location=REGION,
                                      feature_view_id=create_feature_view_task.output,
                                      feature_online_store_id=create_feature_store_task.output)
                         .set_display_name("Process Data"))


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
