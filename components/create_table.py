from kfp import dsl
from kfp.dsl import component
from components.dependencies import resolve_dependencies
from constants import BASE_IMAGE


@component(
    base_image=BASE_IMAGE,
    packages_to_install=resolve_dependencies(
        'google-cloud-bigquery',
        'google-cloud-storage',
        'pandas',
    )
)
def create_table(
        project_id: str,
        dataset_id: str,
        new_table_id: str,
        data_bucket: str,
        csv_file_name: str,
        output_table_id: dsl.Output[dsl.Artifact]
):
    """
    Function to create a new BigQuery table from a CSV file stored in a GCS bucket,
    add additional columns, and copy the data to the new table.

    @output_table_id: new table ID as output
    """
    from google.cloud import bigquery, storage
    import pandas as pd
    import io
    from src.data import get_logger

    logger = get_logger()

    try:
        bigquery_client = bigquery.Client(project=project_id)
        storage_client = storage.Client()

        csv_uri = f'gs://{data_bucket}/{csv_file_name}'

        # Read the CSV file from GCS into a pandas DataFrame
        logger.info(f"Loading CSV data from GCS: {csv_uri}")
        bucket = storage_client.bucket(data_bucket)
        blob = bucket.blob(csv_file_name)
        csv_data = blob.download_as_text()

        # Load data into a pandas DataFrame
        df = pd.read_csv(io.StringIO(csv_data))

        schema = [
            bigquery.SchemaField(name, "STRING") if dtype == "object" else bigquery.SchemaField(name, "INTEGER")
            for name, dtype in zip(df.columns, df.dtypes)
        ]

        logger.info(f"DataFrame loaded with shape: {df.shape}")
        logger.info(f"DataFrame columns: {df.columns}")

        # Add "id" and "feature_timestamp" columns to the DataFrame
        df['id'] = range(1, len(df) + 1)
        df['feature_timestamp'] = pd.Timestamp.now()

        logger.info("Added 'id', 'feature_timestamp'")

        # Define schema for the new BigQuery table
        new_schema = schema + [
            bigquery.SchemaField("id", "INTEGER", mode="REQUIRED", description="Unique identifier for each row"),
            bigquery.SchemaField("feature_timestamp", "TIMESTAMP", mode="REQUIRED",
                                 description="Feature timestamp for each record")
        ]

        # Create a new table in BigQuery
        dataset_ref = bigquery_client.dataset(dataset_id)
        table_ref = dataset_ref.table(new_table_id)
        # Load the DataFrame directly into BigQuery
        job_config = bigquery.LoadJobConfig(
            schema=new_schema,
            write_disposition=bigquery.WriteDisposition.WRITE_TRUNCATE
        )

        # Load the DataFrame to BigQuery
        load_job = bigquery_client.load_table_from_dataframe(
            df, table_ref, job_config=job_config
        )
        # Wait for the load job to complete
        load_job.result()

        print(f"Loaded {load_job.output_rows} rows into {dataset_id}:{table_ref.table_id}.")

        output_table_id = f"{dataset_id}.{new_table_id}"

    except Exception as e:
        logger.error(f"Failed to create BigQuery table and load data from CSV: {e}")
        raise e