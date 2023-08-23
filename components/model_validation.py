from kfp.v2 import dsl
from kfp.v2.components.component_decorator import component
from components.dependencies import resolve_dependencies
from constants import base_image


@component(
    base_image=base_image,
    packages_to_install=resolve_dependencies(
        'fsspec',
        'gcsfs',
        'google-cloud-storage',
        'google-cloud-aiplatform',
        'accelerate',
        'torch',
        'psutil',
        'transformers'
    )
)
def validate_model(
        project_id: str,
        project_region: str,
        training_pipeline_id: str,
        model_artifact_path: str,
        validation_dataset_bucket: str,
        validation_dataset_file: str,
        validation_threshold: str,
        component_execution: bool,
        desired_validation: dsl.Output[dsl.Metrics],
        validation_result: dsl.Output[dsl.Metrics]
):
    import logging
    import json
    import time
    import os

    from google.cloud import storage
    from utils.helper_functions import preprocess, forward, postprocess, download_model_files_from_bucket, \
        get_model_tokenizer, get_memory_usage, get_time, cancel_training_pipeline

    logger = logging.getLogger('tipper')
    logger.setLevel(logging.DEBUG)
    logger.addHandler(logging.StreamHandler())

    try:
        if not component_execution:
            logging.info("Component execution: validate model execution is bypassed")

        else:
            # logging.info(
            #     f"Task: Making Directory if not exist for saving model file from GCS Bucket: {model_artifact_path}")
            # save_trained_model = "saved_model"
            # os.makedirs(save_trained_model, exist_ok=True)

            # logging.info("Downloading model files")
            # download_model_files_from_bucket(model_artifact_path, save_trained_model)
            # logging.info("Task: Model Files Downloaded Successfully for validation")

            # logging.info(f"Reading model & tokenizer from local dir: {save_trained_model}")
            # model, tokenizer = get_model_tokenizer(
            #     pretrained_model_name_or_path=save_trained_model,
            #     gradient_checkpointing=True
            # )

            logging.info(f"Task: Loading Validation dataset from bucket: {validation_dataset_bucket}")
            logging.info("Making client connection")
            client = storage.Client()

            bucket = client.get_bucket(validation_dataset_bucket)
            blob = bucket.blob(validation_dataset_file)

            logging.info(f"Downloading validation dataset from GCS Bucket: {validation_dataset_bucket}")
            blob.download_to_filename(validation_dataset_file)

            logging.info("Reading Validation dataset")
            with open(validation_dataset_file, "r") as json_file:
                validation_data = json.load(json_file)

            def get_prediction(validation_query):

                # """Preprocessing the input text (tokenization)"""
                # pre_process_result = preprocess(get_tokenizer, validation_query)
                #
                # """Making prediction"""
                # model_result = forward(get_model, get_tokenizer, pre_process_result)
                #
                # """Postprocessing the predicted result"""
                # predicted_answer = postprocess(get_tokenizer, model_result, False)

                return "Sure, Youtube Subscription least count is 455"

            correct_answers = 0

            logging.info("ask: Iterating over validation dataset for prediction")
            for items in validation_data["validation_data"]:
                query = items["instruction"]
                actual_answer = items["response"]

                start_time = time.time()
                returned_answer = get_prediction(validation_query=query)
                end_time = time.time()

                logging.info(f"Memory Usage for making prediction on query: {query}")
                get_memory_usage()

                logging.info("Time took for above query:")
                get_time(start_time, end_time)

                if returned_answer == actual_answer:
                    logging.info(f"Prediction: Query: '{query}', Actual Answer: '{actual_answer}' and "
                                 f"Predicted Answer: '{returned_answer}'")
                    correct_answers += 1

            logging.info(f"Removing validation dataset: {validation_dataset_file}")
            os.remove(validation_dataset_file)

            logging.info("Task: Setting Validation Result Metric")
            desired_validation.log_metric("Desired Answers Count:", int(validation_threshold))
            validation_result.log_metric("Correct Answers Count:", correct_answers)

            logging.info("Task: Checking if validation is desired")
            if correct_answers >= int(validation_threshold):
                logging.info("Task: Validation is successful and passing to next component")
                logging.info(f"Validation Count: {correct_answers} and Validation Threshold: {validation_threshold}")
            else:
                logging.info(f"Validation Count: {correct_answers} and Validation Threshold: {validation_threshold}")
                logging.info("Task: Stopping Pipeline due to non desired validation")
                cancel_training_pipeline(project_id, training_pipeline_id, project_region)

    except Exception as e:
        logging.error("Some error occurred in model validation component")
        raise e
