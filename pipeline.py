import logging
from kfp.v2 import compiler
import kfp

from components.process_data import process_data
from components.save_trained_model import save_model
from components.serve_model import serve_model_component
from components.train_model import fine_tune_model
from components.upload_model import upload_container
from constants import project_region, serving_image, model_display_name, \
    pipeline_description, pipeline_name, pipeline_root_gcs, original_model_name, model_bucket_name, model_path, \
    trigger_id, staging_bucket

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


@kfp.dsl.pipeline(name=pipeline_name,
                  description=pipeline_description,
                  pipeline_root=pipeline_root_gcs)
def pipeline(
        project_id: str,
        job_id: str
):
    process_data_task = process_data().set_display_name("Process Data")

    """Fine Tune Model Pipeline"""
    train_model_task = fine_tune_model(process_data_task.output, original_model_name) \
        .after(process_data_task) \
        .set_display_name("DOLLY_V2_3B Training") \
        .set_cpu_request("8") \
        .set_memory_limit("64G")

    """Save Model To GCS Bucket"""
    save_model_task = save_model(model_bucket_name) \
        .after(train_model_task) \
        .set_display_name("Save Model")

    """Upload model package"""
    upload_model_task = upload_container(project_id, trigger_id) \
        .after(save_model_task) \
        .set_display_name("Model_Upload")

    """Serve Model To Endpoint"""
    serve_model_component(project_id, project_region, staging_bucket, serving_image, model_display_name) \
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
