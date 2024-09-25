from typing import List

from google.cloud import aiplatform
from vertexai.resources.preview import feature_store
PROJECT_ID = "nashtech-ai-dev-389315"  # @param {type:"string"}
LOCATION = "us-central1"
BQ_TABLE_URI = f"bq://{PROJECT_ID}.housing_dataset.table_house"
# FEATURE_VIEW_ID = "housing_data_view"
FEATURE_VIEW_ID_BQ = "housing_data_view_bq"
NEW_FEATURE_VIEW_ID_BQ = "new_housing_data_view_bq"
# FEATURE_ONLINE_STORE_ID = "nashtech_feature_store_optimized_private"  # @param {type:"string"}
ENTITY_ID_COLUMNS = ["id"]
FEATURE_GROUP_ID = "housing_feature_group"

def create_optimized_private_feature_online_store_sample(
    project: str,
    location: str,
    feature_online_store_id: str,
    project_allowlist: List[str],
):
    aiplatform.init(project=project, location=location)
    fos = feature_store.FeatureOnlineStore.create_optimized_store(
        name=feature_online_store_id,
        enable_private_service_connect=True,
        project_allowlist=project_allowlist,
    )
    return fos

# create_optimized_private_feature_online_store_sample(project=PROJECT_ID,location=LOCATION,
#                                                      feature_online_store_id=FEATURE_ONLINE_STORE_ID,
#                                                      project_allowlist=[PROJECT_ID])

from google.cloud.aiplatform_v1beta1.types import feature_view as feature_view_pb2
from google.cloud.aiplatform_v1beta1 import (
    FeatureOnlineStoreAdminServiceClient, FeatureRegistryServiceClient)
from google.cloud.aiplatform_v1beta1.types import \
    feature_online_store_admin_service as \
    feature_online_store_admin_service_pb2
from vertexai.resources.preview import (FeatureOnlineStore, FeatureView,
                                        FeatureViewBigQuerySource)
FEATURE_IDS=[
    "price",
    "area",
    "bedrooms",
    "bathrooms",
    "stories",
    "mainroad",
    "guestroom",
    "basement",
    "hotwaterheating",
    "airconditioning",
    "parking",
    "prefarea",
    "furnishingstatus"
]

# FEATURE_VIEW_ID = "registry_product"  # @param {type:"string"}
CRON_SCHEDULE = "TZ=America/Los_Angeles 56 * * * *"  # @param {type:"string"}

# feature_registry_source = feature_view_pb2.FeatureView.FeatureRegistrySource(
#     feature_groups=[feature_view_pb2.FeatureView.FeatureRegistrySource.FeatureGroup(
#             feature_group_id=FEATURE_GROUP_ID, feature_ids=FEATURE_IDS)])
#
# sync_config = feature_view_pb2.FeatureView.SyncConfig(cron=CRON_SCHEDULE)
#
# create_view_lro = admin_client.create_feature_view(
#     feature_online_store_admin_service_pb2.CreateFeatureViewRequest(
#         parent=f"projects/{PROJECT_ID}/locations/{LOCATION}/featureOnlineStores/{FEATURE_ONLINE_STORE_ID}",
#         feature_view_id=FEATURE_VIEW_ID,
#         feature_view=feature_view_pb2.FeatureView(
#             feature_registry_source=feature_registry_source,
#             sync_config=sync_config,
#         ),
#     )
# )

# Wait for LRO to complete and show result
# print(create_view_lro.result())

# print(admin_client.list_feature_views(
#     parent=f"projects/{PROJECT_ID}/locations/{LOCATION}/featureOnlineStores/{FEATURE_ONLINE_STORE_ID}"
# ))
from google.cloud import aiplatform
from vertexai.resources.preview import feature_store
# from typing import List
#
#
# def create_feature_view_from_bq_source(
#     project: str,
#     location: str,
#     existing_feature_online_store_id: str,
#     feature_view_id: str,
#     bq_table_uri: str,
#     entity_id_columns: List[str],
# ):
#     aiplatform.init(project=project, location=location)
#     fos = feature_store.FeatureOnlineStore(existing_feature_online_store_id)
#     fv = fos.create_feature_view(
#         name=feature_view_id,
#         source=feature_store.utils.FeatureViewBigQuerySource(
#             uri=bq_table_uri, entity_id_columns=entity_id_columns
#         ),
#     )
#     return fv
# create_feature_view_from_bq_source(project=PROJECT_ID,
#                                    location=LOCATION,
#                                    existing_feature_online_store_id=FEATURE_ONLINE_STORE_ID,
#                                    feature_view_id=FEATURE_VIEW_ID_BQ,
#                                    bq_table_uri=BQ_TABLE_URI,
#                                    entity_id_columns=ENTITY_ID_COLUMNS)

# from google.cloud.aiplatform_v1 import FeatureOnlineStoreServiceClient
# from google.cloud.aiplatform_v1.types import feature_online_store_service as feature_online_store_service_pb2
#
# data_client = FeatureOnlineStoreServiceClient(
#   client_options={"api_endpoint": f"2905674980677124096.us-central1-24562761082.featurestore.vertexai.goog"}
# )
# data = data_client.fetch_feature_values(
#   request=feature_online_store_service_pb2.FetchFeatureValuesRequest(
#     feature_view=f"projects/nashtech-ai-dev-389315/locations/us-central1/featureOnlineStores/nashtech_feature_store_optimized/featureViews/housing_data_view_bq",
#     # id=f"10",
#     # format=feature_online_store_service_pb2.FetchFeatureValuesRequest.Format.TABULAR,
#   )
# )


from google.cloud import aiplatform
from vertexai.resources.preview import feature_store

FEATURE_ONLINE_STORE_ID = "nashtech_feature_store_bigtable"  # @param {type:"string"}

def create_bigtable_feature_online_store_sample(
    project: str,
    location: str,
    feature_online_store_id: str,
):
    aiplatform.init(project=project, location=location)
    fos = feature_store.FeatureOnlineStore.create_bigtable_store(
        feature_online_store_id
    )
    return fos

# create_bigtable_feature_online_store_sample(project=PROJECT_ID,location=LOCATION,feature_online_store_id=FEATURE_ONLINE_STORE_ID)
#
#
# from typing import List
#
#
# def create_feature_view_from_bq_source(
#     project: str,
#     location: str,
#     existing_feature_online_store_id: str,
#     feature_view_id: str,
#     bq_table_uri: str,
#     entity_id_columns: List[str],
# ):
#     aiplatform.init(project=project, location=location)
#     fos = feature_store.FeatureOnlineStore(existing_feature_online_store_id)
#     fv = fos.create_feature_view(
#         name=feature_view_id,
#         source=feature_store.utils.FeatureViewBigQuerySource(
#             uri=bq_table_uri, entity_id_columns=entity_id_columns
#         ),
#     )
#     return fv
# create_feature_view_from_bq_source(project=PROJECT_ID,
#                                    location=LOCATION,
#                                    existing_feature_online_store_id=FEATURE_ONLINE_STORE_ID,
#                                    feature_view_id=NEW_FEATURE_VIEW_ID_BQ,
#                                    bq_table_uri=BQ_TABLE_URI,
#                                    entity_id_columns=ENTITY_ID_COLUMNS)
# fos = feature_store.FeatureOnlineStore.create_bigtable_store(
#     "household_featurestore"
# )
# fos = feature_store.FeatureOnlineStore("household_featurestore")

# from google.cloud.aiplatform_v1 import FeatureOnlineStoreServiceClient
# from google.cloud.aiplatform_v1.types import feature_online_store_service as feature_online_store_service_pb2
# FEATURE_VIEW_ID = f"projects/{PROJECT_ID}/locations/{LOCATION}/featureOnlineStores/nashtech_feature_store_bigtable/featureViews/new_housing_data_view_bq"
# data_client = FeatureOnlineStoreServiceClient(
#   client_options={"api_endpoint": f"us-central1-aiplatform.googleapis.com"}
# )
# data = data_client.fetch_feature_values(
#   request=feature_online_store_service_pb2.FetchFeatureValuesRequest(
#     feature_view=FEATURE_VIEW_ID,
#     data_key=feature_online_store_service_pb2.FeatureViewDataKey(key="1"),
#   )
# )
# print(data)



from vertexai.resources.preview import FeatureView

# print("\n--Second Approach--\n")
# # data = (
#     # FeatureView(name="housing_data_view_bq", feature_online_store_id="nashtech_feature_store_bigtable")
# # )
#
# print(data)

# from google.cloud.aiplatform_v1 import FeatureOnlineStoreServiceClient
# from google.cloud.aiplatform_v1.types import feature_online_store_service as feature_online_store_service_pb2
#
# data_client = FeatureOnlineStoreServiceClient(
#   client_options={"api_endpoint": "us-central1-aiplatform.googleapis.com"}
# )
# data = data_client.fetch_feature_values(
#   request=feature_online_store_service_pb2.FetchFeatureValuesRequest(
#     feature_view=f"projects/nashtech-ai-dev-389315/locations/us-central1/featureOnlineStores/nashtech_feature_store_optimized/featureViews/housing_data_view",
#     # data_key=feature_online_store_service_pb2.FeatureViewDataKey(key="10"),
#       # format=feature_online_store_service_pb2.FetchFeatureValuesRequest.Format.FORMAT,
#   )
# )
# print(data)

# from typing import List
# import pandas as pd
# from vertexai.resources.preview import FeatureView
# from google.cloud.aiplatform_v1beta1 import FeatureOnlineStoreServiceClient
# from google.cloud.aiplatform_v1beta1.types import \
#     feature_online_store_service as feature_online_store_service_pb2
#
# def sffv(data_client, feature_view, keys_list: List[List[str]]):
#     """Helper function to fetch feature values"""
#     def request_generator(keys_list):
#         for keys in keys_list:
#             data_keys = [
#                 feature_online_store_service_pb2.FeatureViewDataKey(key=key)
#                 for key in keys
#             ]
#             request = (
#                 feature_online_store_service_pb2.StreamingFetchFeatureValuesRequest(
#                     feature_view=feature_view,
#                     data_keys=data_keys,
#                 )
#             )
#             yield request
#
#     responses = data_client.streaming_fetch_feature_values(
#         requests=request_generator(keys_list)
#     )
#     return [response for response in responses]
#
# def process_feature_store_data(data):
#     """Process feature store data and convert to DataFrame"""
#     processed_data = []
#     for response in data:
#         for data_item in response.data:
#             row_data = {}
#             for feature in data_item.key_values.features:
#                 if feature.name != 'feature_timestamp':
#                     if feature.value.HasField('int64_value'):
#                         row_data[feature.name] = feature.value.int64_value
#                     elif feature.value.HasField('string_value'):
#                         row_data[feature.name] = feature.value.string_value
#                     # Add more value types as needed
#             processed_data.append(row_data)
#     return pd.DataFrame(processed_data)
#
# def fetch_all_data(data_client, feature_view, batch_size=1000):
#     """Fetch all available data in batches"""
#     all_data = []
#     batch_start = 1
#     while True:
#         batch_keys = [[f"{num}" for num in range(batch_start, batch_start + batch_size)]]
#         batch_data = sffv(data_client, feature_view, batch_keys)
#         if not batch_data or not batch_data[0].data:
#             break
#         all_data.extend(batch_data)
#         batch_start += batch_size
#     return all_data
#
# # Setup
# # LOCATION = "your-location"  # Replace with your actual location
# API_ENDPOINT = f"{LOCATION}-aiplatform.googleapis.com"
# data_client = FeatureOnlineStoreServiceClient(
#     client_options={"api_endpoint": API_ENDPOINT}
# )
#
# fv = FeatureView(name="housing_data_view_bq", feature_online_store_id="nashtech_feature_store_bigtable")
#
# # Fetch all data
# data = fetch_all_data(data_client, fv.resource_name)
#
# # Process the data and create a DataFrame
# df = process_feature_store_data(data)

# Display the DataFrame
# print(df)

# registry_client = FeatureRegistryServiceClient(
#     client_options={"api_endpoint": API_ENDPOINT}
# )
# API_ENDPOINT = f"{LOCATION}-aiplatform.googleapis.com"
# from google.cloud.aiplatform_v1beta1 import FeatureOnlineStoreAdminServiceClient
# FEATURE_VIEW_ID = "housing_data_view_bq"
# admin_client = FeatureOnlineStoreAdminServiceClient(
#     client_options={"api_endpoint": API_ENDPOINT}
# )
# sync_response = admin_client.sync_feature_view(
#     feature_view=f"projects/{PROJECT_ID}/locations/{LOCATION}/featureOnlineStores/{FEATURE_ONLINE_STORE_ID}/featureViews/{FEATURE_VIEW_ID}"
# )
# import time
#
# while True:
#     feature_view_sync = admin_client.get_feature_view_sync(
#         name=sync_response.feature_view_sync
#     )
#     if feature_view_sync.run_time.end_time.seconds > 0:
#         status = "Succeed" if feature_view_sync.final_status.code == 0 else "Failed"
#         print(f"Sync {status} for {feature_view_sync.name}.")
#         break
#     else:
#         print("Sync ongoing, waiting for 30 seconds.")
#     time.sleep(30)


# from kfp.v2 import dsl
# from kfp.v2.components.component_decorator import component
# from components.dependencies import resolve_dependencies
# from constants import BASE_IMAGE
# from google.cloud import bigquery, storage
# import pandas as pd
# import io
# from src.data import get_logger
# import datetime

# # @component(
# #     base_image=BASE_IMAGE,
# #     packages_to_install=resolve_dependencies(
# #         'google-cloud-bigquery',
# #         'google-cloud-storage',
# #         'pandas',
# #     )
# # )
# def create_table(
#         project_id: str,
#         dataset_id: str,
#         new_table_id: str,
#         data_bucket: str,
#         csv_file_name: str,
#         output_table_id: dsl.Output[dsl.Artifact]
# ):
#     """
#     Function to create a new BigQuery table from a CSV file stored in a GCS bucket,
#     add additional columns, and copy the data to the new table.
#     @output_table_id: new table ID as output
#     """
#     # logger = get_logger()
#     try:
#
#     except Exception as e:
#         logger.error(f"Failed to create BigQuery table and load data from CSV: {e}")
#         raise e
from constants import RESOURCE_BUCKET
# data_bucket = RESOURCE_BUCKET
# csv_file_name = "Housing.csv"
# dataset_id = "housing_dataset"
# bigquery_client = bigquery.Client(project=PROJECT_ID)
# storage_client = storage.Client()
# dataset_ref = bigquery_client.dataset(dataset_id)
# csv_uri = f'gs://{data_bucket}/{csv_file_name}'
#
# # Read the CSV file from GCS into a pandas DataFrame
# bucket = storage_client.bucket(data_bucket)
# blob = bucket.blob(csv_file_name)
# csv_data = blob.download_as_text()
# df = pd.read_csv(io.StringIO(csv_data))
# job_config = bigquery.LoadJobConfig(write_disposition=bigquery.WriteDisposition.WRITE_TRUNCATE)
# job = bigquery_client.load_table_from_dataframe(df, new_table_ref, job_config=job_config)
# job.result()
# schema = [
#     bigquery.SchemaField(name, "STRING") if dtype == "object" else bigquery.SchemaField(name, "INTEGER")
#     for name, dtype in zip(df.columns, df.dtypes)
# ]
# new_schema = schema + [
#             bigquery.SchemaField("id", "INTEGER", mode="REQUIRED", description="Unique identifier for each row"),
#             bigquery.SchemaField("feature_timestamp", "TIMESTAMP", mode="REQUIRED", description="Feature timestamp for each record"),
#         ]
# # Add 'id' and 'feature_timestamp' columns to the DataFrame
# df['id'] = range(1, len(df) + 1)
# df['feature_timestamp'] = datetime.datetime.now()
#
# new_table_id = "table_house"
# # Create a new table in BigQuery
# new_table_ref = dataset_ref.table(new_table_id)
# new_table = bigquery.Table(new_table_ref, schema=new_schema)
# new_table = bigquery_client.create_table(new_table)

# Load data from the DataFrame into the BigQuery table



from typing import List


def create_feature_view_from_bq_source(
    project: str,
    location: str,
    existing_feature_online_store_id: str,
    feature_view_id: str,
    bq_table_uri: str,
    entity_id_columns: List[str],
):
    aiplatform.init(project=project, location=location)
    fos = feature_store.FeatureOnlineStore(existing_feature_online_store_id)
    fv = fos.create_feature_view(
        name=feature_view_id,
        source=feature_store.utils.FeatureViewBigQuerySource(
            uri=bq_table_uri, entity_id_columns=entity_id_columns
        ),
    )
    return fv
# create_feature_view_from_bq_source(project=PROJECT_ID,
#                                    location=LOCATION,
#                                    existing_feature_online_store_id=FEATURE_ONLINE_STORE_ID,
#                                    feature_view_id="new_view_test",
#                                    bq_table_uri=BQ_TABLE_URI,
#                                    entity_id_columns=ENTITY_ID_COLUMNS)

#
# from google.cloud import bigquery, storage
# import pandas as pd
# import io
#
# # Define your constants
# PROJECT_ID = PROJECT_ID
# RESOURCE_BUCKET = RESOURCE_BUCKET
# CSV_FILE_NAME = "Housing.csv"
# DATASET_ID = "housing_dataset"
#
# # Initialize BigQuery and Cloud Storage clients
# bigquery_client = bigquery.Client(project=PROJECT_ID)
# storage_client = storage.Client()
#
# # Read the CSV file from GCS into a pandas DataFrame
# bucket = storage_client.bucket(RESOURCE_BUCKET)
# blob = bucket.blob(CSV_FILE_NAME)
# csv_data = blob.download_as_text()
# df = pd.read_csv(io.StringIO(csv_data))
# schema = [
#     bigquery.SchemaField(name, "STRING") if dtype == "object" else bigquery.SchemaField(name, "INTEGER")
#     for name, dtype in zip(df.columns, df.dtypes)
# ]
#
# # Add "id" and "feature_timestamp" columns to the DataFrame
# df['id'] = range(1, len(df) + 1)  # Add a unique id for each row
# df['feature_timestamp'] = pd.Timestamp.now()  # Add the current timestamp for each row
#
# # Define the schema for the BigQuery table
# new_schema = schema + [
#     bigquery.SchemaField("id", "INTEGER", mode="REQUIRED", description="Unique identifier for each row"),
#     bigquery.SchemaField("feature_timestamp", "TIMESTAMP", mode="REQUIRED", description="Feature timestamp for each record")
# ]
#
# # Define the table reference
# dataset_ref = bigquery_client.dataset(DATASET_ID)
# table_ref = dataset_ref.table("test_housing_data_new_table")
#
# # Load the DataFrame directly into BigQuery
# job_config = bigquery.LoadJobConfig(
#     schema=new_schema,
#     write_disposition=bigquery.WriteDisposition.WRITE_TRUNCATE  # Replace the table if it exists
# )
#
# # Load the DataFrame to BigQuery
# load_job = bigquery_client.load_table_from_dataframe(
#     df, table_ref, job_config=job_config
# )
#
# # Wait for the load job to complete
# load_job.result()
#
# print(f"Loaded {load_job.output_rows} rows into {DATASET_ID}:{table_ref.table_id}.")
from google.cloud import aiplatform as vertex_ai
from constants import PROJECT_ID, REGION
import vertex_ray
vertex_ai.init(project=PROJECT_ID, location=REGION)
import ray

CLUSTER_NAME = "ray-cluster-20240925075040-7f40d2"
CLUSTER_RESOURCE_NAME = 'projects/{}/locations/{}/persistentResources/{}'.format(PROJECT_ID, REGION, CLUSTER_NAME)

vertex_ray.delete_ray_cluster(CLUSTER_RESOURCE_NAME)

# ray.init(address='vertex_ray://projects/24562761082/locations/us-central1/persistentResources/ray-cluster-20240925075040-7f40d2')
# ray_clusters = vertex_ray.list_ray_clusters()
# ray_cluster_resource_name = ray_clusters[-1].cluster_resource_name
# ray_cluster = vertex_ray.get_ray_cluster(ray_cluster_resource_name)
#
# print("Ray cluster on Vertex AI:", ray_cluster_resource_name)