import logging
from kfp.v2 import compiler
import kfp

from components.process_data import process_data
from components.serve_model import serve_model_component
from components.train_model import fine_tune_model
from components.upload_model import upload_container
from constants import pipeline_description, pipeline_name, pipeline_root_gcs, original_model_name, \
    save_model_bucket_name, project_region, dataset_bucket, trigger_id, model_display_name, serving_image, \
    staging_bucket, component_execution, dataset_name

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


@kfp.dsl.pipeline(name=pipeline_name,
                  description=pipeline_description,
                  pipeline_root=pipeline_root_gcs)
def pipeline(
        project_id: str,
        job_id: str
):
    # Dataset Processing
    process_data_task = process_data(dataset_bucket, dataset_name).set_display_name("Data Processing")

    """Fine Tune Model Pipeline"""
    train_model_task = fine_tune_model(process_data_task.outputs["dataset"],
                                       original_model_name,
                                       save_model_bucket_name,
                                       component_execution) \
        .after(process_data_task) \
        .set_display_name("Dolly Fine Tuning") \
        # .set_cpu_request("8") \
        # .set_memory_limit("32G")

    """Upload model package"""
    upload_model_task = upload_container(project_id, trigger_id, component_execution) \
        .after(train_model_task) \
        .set_display_name("Model_Upload")

    """Serve Model To Endpoint"""
    serve_model_component(project_id,
                          project_region,
                          staging_bucket,
                          serving_image,
                          model_display_name,
                          component_execution) \
        .after(upload_model_task) \
        .set_display_name("Serve_Model")


def compile_pipeline(pipeline_template_name='./llm_pipeline.json'):
    compiler.Compiler().compile(
        pipeline_func=pipeline,
        package_path=pipeline_template_name
    )
    return None


if __name__ == "__main__":
    compile_pipeline()
