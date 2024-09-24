from kfp.v2 import dsl
from kfp.v2.components.component_decorator import component
from components.dependencies import resolve_dependencies


@component(
    base_image="python:3.8",
    packages_to_install=resolve_dependencies(
        'google-cloud-bigquery',
        'google-cloud-storage',
        'pandas',
    )
)
def create_new_bigquery_table_from_gcs_csv(
        project_id: str,
        dataset_id: str,
        new_table_id: str,
        csv_bucket: str,
        csv_file_name: str,
        output_table_id: dsl.Output[str]
):
    """
    Function to create a new BigQuery table from a CSV file stored in a GCS bucket,
    add additional columns, and copy the data to the new table.

    @output_table_id: new table ID as output
    """
    import logging
    from google.cloud import bigquery, storage
    import pandas as pd
    import io

    logger = logging.getLogger('tipper')
    logger.setLevel(logging.DEBUG)
    logger.addHandler(logging.StreamHandler())

    try:
        # Initialize BigQuery and GCS clients
        bigquery_client = bigquery.Client(project=project_id)
        storage_client = storage.Client()

        # Reference the dataset in BigQuery
        dataset_ref = bigquery_client.dataset(dataset_id)

        # Define the GCS file path for the CSV file
        csv_uri = f'gs://{csv_bucket}/{csv_file_name}'

        # Read the CSV file from GCS into a pandas DataFrame
        logger.info(f"Loading CSV data from GCS: {csv_uri}")
        bucket = storage_client.bucket(csv_bucket)
        blob = bucket.blob(csv_file_name)
        csv_data = blob.download_as_text()

        # Load data into a pandas DataFrame
        df = pd.read_csv(io.StringIO(csv_data))

        logger.info(f"DataFrame loaded with shape: {df.shape}")
        logger.info(f"DataFrame columns: {df.columns}")

        # Add 'id', 'feature_timestamp' to the DataFrame
        df['id'] = range(1, len(df) + 1)
        df['feature_timestamp'] = pd.Timestamp.now()

        logger.info("Added 'id', 'feature_timestamp'")

        # Define schema for the new BigQuery table
        schema = [
            bigquery.SchemaField(name, "STRING") if dtype == "object" else bigquery.SchemaField(name, "INTEGER")
            for name, dtype in zip(df.columns, df.dtypes)
        ]

        # Create a new table in BigQuery
        new_table_ref = dataset_ref.table(new_table_id)
        new_table = bigquery.Table(new_table_ref, schema=schema)
        new_table = bigquery_client.create_table(new_table)
        logger.info(f"Created new BigQuery table: {new_table.project}.{new_table.dataset_id}.{new_table.table_id}")

        # Load data from the DataFrame into the BigQuery table
        job_config = bigquery.LoadJobConfig(write_disposition=bigquery.WriteDisposition.WRITE_TRUNCATE)
        job = bigquery_client.load_table_from_dataframe(df, new_table_ref, job_config=job_config)
        job.result()

        logger.info(f"Data loaded into {new_table_id}.")

        # Set the output for the pipeline
        output_table_id.uri = new_table_id

    except Exception as e:
        logger.error(f"Failed to create BigQuery table and load data from CSV: {e}")