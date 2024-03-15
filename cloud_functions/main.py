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
    formatted_time = current_time.strftime("%Y-%m-%d %H:%M")
    datetime_obj = pd.to_datetime(formatted_time)
    return datetime_obj


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

    logger.info('Task: Establishing BigQuery Client connection')
    bg_client = bigquery.Client()

    logger.info('Task: Reading data from bigquery')
    df = bg_client.query(query).to_dataframe()

    logger.info("Task: Closing BigQuery Client Connection")
    bg_client.close()

    return df


def get_table(project_id: str,
              dataset_id: str,
              table_id: str
              ):
    query = f'SELECT * FROM `{project_id}.{dataset_id}.{table_id}`'
    logger.info(f'Task: Query: {query}')

    logger.info('Task: Establishing BigQuery Client connection')
    bg_client = bigquery.Client()

    logger.info('Task: Reading data from BigQuery')
    df = bg_client.query(query).to_dataframe()

    logger.info("Task: Closing BigQuery Client Connection")
    bg_client.close()

    return df


def get_scores(dataframe):
    try:
        """Calculate precision, recall, accuracy, confusion matrix, and F1 score."""
        true_positive = ((dataframe['RESPONSE'] == 'Outlier') & (dataframe['FEEDBACK'] == True)).sum()
        true_negative = ((dataframe['RESPONSE'] != 'Outlier') & (dataframe['FEEDBACK'] == True)).sum()
        false_positive = ((dataframe['RESPONSE'] == 'Outlier') & (dataframe['FEEDBACK'] == False)).sum()
        false_negative = ((dataframe['RESPONSE'] != 'Outlier') & (dataframe['FEEDBACK'] == False)).sum()

        precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) > 0 else 0
        recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) > 0 else 0

        accuracy = (true_positive + true_negative) / (len(dataframe)) if len(dataframe) > 0 else 0

        cnf_matrix = [[true_positive, false_positive], [false_negative, true_negative]]

        f1_score_value = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        return precision, recall, accuracy, cnf_matrix, f1_score_value

    except Exception as e:
        raise logger.error(f"Some error occurred in calculating metrics: {str(e)}")


def get_stats(dataframe):
    logger.info('Getting precision, recall and accuracy')
    precision, recall, acc_score, cnf_matrix, f_one_score = get_scores(dataframe)

    global_intensity = np.array(dataframe['Global_intensity'])
    global_reactive_power = np.array(dataframe['Global_reactive_power'])

    logger.info('Calculating Inter-quartiles')
    quartiles_intensity = np.percentile(global_intensity, [25, 50, 75])
    quartiles_reactive_power = np.percentile(global_reactive_power, [25, 50, 75])

    timestamp = get_time()

    logger.info("Creating metrics dictionary")
    generated_stats_dict = {
        "TEST_ACCURACY": acc_score,
        "PRECISION": precision,
        "RECALL": recall,
        "MEAN_REACTIVE_POWER": dataframe['Global_reactive_power'].mean(),
        "STD_REACTIVE_POWER": dataframe['Global_reactive_power'].std(),
        "MEAN_INTENSITY": dataframe['Global_intensity'].mean(),
        "STD_INTENSITY": dataframe['Global_intensity'].std(),
        "F1_SCORE": f_one_score,
        "CNF_MATRIX": str(cnf_matrix),
        "IQR_INTENSITY": quartiles_intensity[2] - quartiles_intensity[0],
        "IQR_REACTIVE_POWER": quartiles_reactive_power[2] - quartiles_reactive_power[0],
        "SKEW_INTENSITY": stats.skew(global_intensity),
        "SKEW_REACTIVE_POWER": stats.skew(global_reactive_power),
        "KURTOSIS_INTENSITY": stats.kurtosis(global_intensity),
        "KURTOSIS_REACTIVE_POWER": stats.kurtosis(global_reactive_power),
        "date_time": timestamp
    }

    stats_df = pd.DataFrame([generated_stats_dict])
    return stats_df


def write_table_to_bigquery(dataframe, project_id, big_query_table_id):
    try:
        """Construct a BigQuery client object"""""
        client = bigquery.Client(project=project_id)

        logger.info(f"In Bigquery helper function: {dataframe}")
        logger.info(f"Type of data incoming: {type(dataframe)}")

        schema = [
            bigquery.SchemaField("TEST_ACCURACY", bigquery.enums.SqlTypeNames.FLOAT),
            bigquery.SchemaField("PRECISION", bigquery.enums.SqlTypeNames.FLOAT),
            bigquery.SchemaField("RECALL", bigquery.enums.SqlTypeNames.FLOAT),
            bigquery.SchemaField("MEAN_REACTIVE_POWER", bigquery.enums.SqlTypeNames.FLOAT),
            bigquery.SchemaField("STD_REACTIVE_POWER", bigquery.enums.SqlTypeNames.FLOAT),
            bigquery.SchemaField("MEAN_INTENSITY", bigquery.enums.SqlTypeNames.FLOAT),
            bigquery.SchemaField("STD_INTENSITY", bigquery.enums.SqlTypeNames.FLOAT),
            bigquery.SchemaField("F1_SCORE", bigquery.enums.SqlTypeNames.FLOAT),
            bigquery.SchemaField("CNF_MATRIX", bigquery.enums.SqlTypeNames.STRING),
            bigquery.SchemaField("IQR_INTENSITY", bigquery.enums.SqlTypeNames.FLOAT),
            bigquery.SchemaField("IQR_REACTIVE_POWER", bigquery.enums.SqlTypeNames.FLOAT),
            bigquery.SchemaField("SKEW_INTENSITY", bigquery.enums.SqlTypeNames.FLOAT),
            bigquery.SchemaField("SKEW_REACTIVE_POWER", bigquery.enums.SqlTypeNames.FLOAT),
            bigquery.SchemaField("KURTOSIS_INTENSITY", bigquery.enums.SqlTypeNames.FLOAT),
            bigquery.SchemaField("KURTOSIS_REACTIVE_POWER", bigquery.enums.SqlTypeNames.FLOAT),
            bigquery.SchemaField("date_time", bigquery.enums.SqlTypeNames.DATETIME),
        ]

        # dataframe["CNF_MATRIX"] = dataframe["CNF_MATRIX"].astype(str)

        job_config = bigquery.LoadJobConfig(
            schema=schema,
            # write_disposition="WRITE_TRUNCATE",
            create_disposition="CREATE_IF_NEEDED"
        )
        job = client.load_table_from_dataframe(
            dataframe, big_query_table_id, job_config=job_config
        )  # Make an API request.
        job.result()  # Wait for the job to complete.
        table = client.get_table(big_query_table_id)  # Make an API request.
        logger.info(
            "Loaded {} rows and {} columns to {}".format(
                table.num_rows, len(table.schema), big_query_table_id
            )
        )
        return "Loaded {} rows and {} columns to {}".format(
            table.num_rows, len(table.schema), big_query_table_id)

    except Exception as e:
        logger.error(f"Exception raised: {e}")
        raise f"Exception raised: {e}"


@functions_framework.http
def generate_matrix(request):
    project = 'nashtech-ai-dev-389315'

    response_schema = "RESPONSE"
    feedback_schema = 'FEEDBACK'

    dataset = 'clustering_history_dataset'
    read_table = 'response_table'

    bigquery_dataset_id = f"{project}.clustering_history_dataset"
    bigquery_table_id = f"{bigquery_dataset_id}.metric_table"

    try:
        logger.info('Task: Initiating Read Data from Big Query')

        logger.info("Getting table from bigquery")
        retrieved_table = get_table(project, dataset, read_table)

        logger.info("Task: Normalizing requests data")
        normalized_df = normalize_requests_data(retrieved_table)

        if not normalized_df.empty:
            logger.info("Task: Read data from Big Query Completed Successfully")

        concatenated_df = pd.concat([normalized_df,
                                     retrieved_table[response_schema],
                                     retrieved_table[feedback_schema]],
                                    axis=1)
        # concatenated_df.to_csv("feedback_records.csv", index=False)

        logger.info("Task: Getting statistical measure of data")
        stats_df = get_stats(concatenated_df)
        # stats_df.to_csv("data.csv", index=False)

        logger.info("Task: Writing generated stats to big query table")
        write_table_to_bigquery(stats_df, project, bigquery_table_id)
        logger.info("Feedback Metric Uploaded to BigQuery Successfully")
        return make_response("Feedback metrics uploaded successfully.", 200)

    except Exception as e:
        return make_response(f"Error: {str(e)}", 500)