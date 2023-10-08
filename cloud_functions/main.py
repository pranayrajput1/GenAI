from google.cloud import storage, bigquery
import functions_framework
from flask import make_response
import logging
import datetime
import os

logger = logging.getLogger('tipper')
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler())


def get_time():
    current_time = datetime.datetime.now()
    formatted_time = current_time.strftime("%Y_%m_%d_%H:%M")
    time = str(formatted_time)
    return time


def get_big_query_data(
        schema_name: str,
        project_id: str,
        dataset_id: str,
        table_id: str,
        limit_count: int,
        save_csv_file_name: str

):
    query = f'''SELECT {schema_name} FROM `{project_id}.{dataset_id}.{table_id}` LIMIT {limit_count}'''
    logger.info(f'Task: Query: {query}')

    logging.info('Task: Establishing BigQuery Client connection')
    bg_client = bigquery.Client()

    logging.info('Task: Reading data from bigquery')
    df = bg_client.query(query).to_dataframe()
    logging.info(f'Dataset dimension: {df.shape}')

    logging.info('Task: Saving data to csv')
    df.to_csv(save_csv_file_name, index=False, header=True)

    return f"Saved file: {save_csv_file_name} to local successfully"


def upload_to_bucket(file_name, bucket_name):
    logging.info("Task: Making client connection to save model details to bucket")
    client = storage.Client()
    bucket = client.get_bucket(bucket_name)

    blob = bucket.blob(file_name)

    logging.info(f"Task: Uploading model details to GCS Bucket: {bucket_name}")
    blob.upload_from_filename(file_name)

    logging.info("Task: Removing model details files from local environment")
    os.remove(file_name)

    return f"Uploaded file: {file_name} to bucket: {bucket_name} successfully"


@functions_framework.http
def big_query_to_gcs(request):
    project = 'nashtech-ai-dev-389315'
    schema = 'FEEDBACK'
    dataset = 'clustering_history_dataset'
    table = 'response_table'
    limit = 1000
    csv_file_name = f'{get_time()}_feedback_metric.csv'

    save_bucket = "dbscan-model"

    try:
        logging.info('Task: Initiating Read Data from Big Query')
        bq_data = get_big_query_data(schema, project, dataset, table, limit, csv_file_name)

        if bq_data:
            logging.info(f'Task: {bq_data}')
            logging.info("Task: Read data from Big Query Completed Successfully")

        current_time = datetime.datetime.now()
        uploaded_file = upload_to_bucket(file_name=csv_file_name, bucket_name=save_bucket)

        logging.info('Task: Initiating Upload Dataset to GCS Bucket')
        if uploaded_file:
            logging.info(f'Task: {uploaded_file}')
            logging.info("Task: Upload Dataset to GCS Bucket Completed Successfully")

        return make_response("Data uploaded successfully.", 200)

    except Exception as e:
        return make_response(f"Error: {str(e)}", 500)
