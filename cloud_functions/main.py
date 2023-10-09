from google.cloud import bigquery
from sklearn.metrics import f1_score, confusion_matrix
import numpy as np
from scipy import stats
import functions_framework
from flask import make_response
import pandas as pd
import logging
import datetime

logger = logging.getLogger('tipper')
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler())


def get_time():
    current_time = datetime.datetime.now()
    formatted_time = current_time.strftime("%Y_%m_%d_%H:%M")
    time = str(formatted_time)
    return time


def normalize_requests_data(df, col: str = 'REQUEST'):
    df[col] = df[col].apply(eval)
    df_normalized = pd.json_normalize(df[col])
    return df_normalized


def get_big_query_data(
        schema_name: str,
        project_id: str,
        dataset_id: str,
        table_id: str,
        limit_count: int
):
    query = f'''SELECT {schema_name} FROM `{project_id}.{dataset_id}.{table_id}` LIMIT {limit_count}'''
    logger.info(f'Task: Query: {query}')

    logging.info('Task: Establishing BigQuery Client connection')
    bg_client = bigquery.Client()

    logging.info('Task: Reading data from bigquery')
    df = bg_client.query(query).to_dataframe()

    return df


def get_stats(dataframe):
    dataframe['FEEDBACK'] = dataframe['FEEDBACK'].apply(lambda x: 'Outlier' if x else 'Not Outlier')

    global_intensity = np.array(dataframe['Global_intensity'])
    global_reactive_power = np.array(dataframe['Global_reactive_power'])

    quartiles_intensity = np.percentile(global_intensity, [25, 50, 75])
    quartiles_reactive_power = np.percentile(global_reactive_power, [25, 50, 75])

    generated_stats_dict = {
        "mean_reactive_power": dataframe['Global_reactive_power'].mean(),
        "std_reactive_power": dataframe['Global_reactive_power'].std(),
        "mean_intensity": dataframe['Global_intensity'].mean(),
        "std_intensity": dataframe['Global_intensity'].std(),
        "f1_score": f1_score(dataframe["RESPONSE"], dataframe["FEEDBACK"], average='weighted'),
        "cnf_matrix": confusion_matrix(dataframe["RESPONSE"], dataframe["FEEDBACK"]),
        "iqr_intensity": quartiles_intensity[2] - quartiles_intensity[0],
        "iqr_reactive_power": quartiles_reactive_power[2] - quartiles_reactive_power[0],
        "skew_intensity": stats.skew(global_intensity),
        "skew_reactive_power": stats.skew(global_reactive_power),
        "kurtosis_intensity": stats.kurtosis(global_intensity),
        "kurtosis_reactive_power": stats.kurtosis(global_reactive_power)
    }

    stats_df = pd.DataFrame(generated_stats_dict.items(), columns=["STATS", "SCORE"])

    stats_df['STATS'] = stats_df['STATS'].astype(str)
    stats_df['SCORE'] = stats_df['SCORE'].astype(str)

    return stats_df


def write_table_to_bigquery(dataframe, project_id, big_query_table_id):
    try:
        """Construct a BigQuery client object"""""
        client = bigquery.Client(project=project_id)

        print("In Bigquery helper function", dataframe)
        print("Type of data incoming ", type(dataframe))
        job_config = bigquery.LoadJobConfig(
            schema=[
                bigquery.SchemaField("STATS", bigquery.enums.SqlTypeNames.STRING),
                bigquery.SchemaField("SCORE", bigquery.enums.SqlTypeNames.STRING),
            ],
            write_disposition="WRITE_TRUNCATE",
            create_disposition="CREATE_IF_NEEDED"
        )
        job = client.load_table_from_dataframe(
            dataframe, big_query_table_id, job_config=job_config
        )  # Make an API request.
        job.result()  # Wait for the job to complete.
        table = client.get_table(big_query_table_id)  # Make an API request.
        print(
            "Loaded {} rows and {} columns to {}".format(
                table.num_rows, len(table.schema), big_query_table_id
            )
        )
        return "Loaded {} rows and {} columns to {}".format(
            table.num_rows, len(table.schema), big_query_table_id)

    except Exception as e:
        print(f"Exception raised: {e}")
        return e


@functions_framework.http
def generate_matrix(request):
    project = 'nashtech-ai-dev-389315'

    request_schema = "REQUEST"
    response_schema = "RESPONSE"
    feedback_schema = 'FEEDBACK'

    dataset = 'clustering_history_dataset'
    read_table = 'response_table'
    limit = 1000

    bigquery_dataset_id = f"{project}.clustering_history_dataset"
    bigquery_table_id = f"{bigquery_dataset_id}.metric_table"

    try:
        logging.info('Task: Initiating Read Data from Big Query')

        logging.info("Task: Reading REQUEST from Big Query Table")
        bq_requests = get_big_query_data(request_schema, project, dataset, read_table, limit)

        logging.info("Task: Normalizing requests data")
        normalized_request_df = normalize_requests_data(bq_requests)

        logging.info("Task: Reading RESPONSE from Big Query Table")
        bq_response = get_big_query_data(response_schema, project, dataset, read_table, limit)

        logging.info("Task: Reading FEEDBACK from Big Query Table")
        bq_feedback = get_big_query_data(feedback_schema, project, dataset, read_table, limit)

        if not bq_feedback.empty and not bq_response.empty and not normalized_request_df.empty:
            logging.info("Task: Read data from Big Query Completed Successfully")

        concatenated_df = pd.concat([normalized_request_df, bq_response, bq_feedback], axis=1)

        logging.info("Task: Getting statistical measure of data")
        stats_df = get_stats(concatenated_df)

        logging.info("Task: Writing generated stats to big query table")
        write_table_to_bigquery(stats_df, project, bigquery_table_id)

        return make_response("Feedback metrics uploaded successfully.", 200)

    except Exception as e:
        return make_response(f"Error: {str(e)}", 500)
