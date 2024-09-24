from typing import List

from google.cloud import aiplatform
from vertexai.resources.preview import feature_store
PROJECT_ID = "nashtech-ai-dev-389315"  # @param {type:"string"}
LOCATION = "us-central1"
BQ_TABLE_URI = f"bq://{PROJECT_ID}.housing_dataset.new_housing_data_with_id_and_category"
# FEATURE_VIEW_ID = "housing_data_view"
FEATURE_VIEW_ID_BQ = "housing_data_view_bq"
NEW_FEATURE_VIEW_ID_BQ = "new_housing_data_view_bq"
# FEATURE_ONLINE_STORE_ID = "nashtech_feature_store_optimized_private"  # @param {type:"string"}
ENTITY_ID_COLUMNS = ["Category_id"]
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
#                                    feature_view_id=NEW_FEATURE_VIEW_ID_BQ,
#                                    bq_table_uri=BQ_TABLE_URI,
#                                    entity_id_columns=ENTITY_ID_COLUMNS)
#
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

from typing import List
import pandas as pd
from vertexai.resources.preview import FeatureView
from google.cloud.aiplatform_v1beta1 import FeatureOnlineStoreServiceClient
from google.cloud.aiplatform_v1beta1.types import \
    feature_online_store_service as feature_online_store_service_pb2

def sffv(data_client, feature_view, keys_list: List[List[str]]):
    """Helper function to fetch feature values"""
    def request_generator(keys_list):
        for keys in keys_list:
            data_keys = [
                feature_online_store_service_pb2.FeatureViewDataKey(key=key)
                for key in keys
            ]
            request = (
                feature_online_store_service_pb2.StreamingFetchFeatureValuesRequest(
                    feature_view=feature_view,
                    data_keys=data_keys,
                )
            )
            yield request

    responses = data_client.streaming_fetch_feature_values(
        requests=request_generator(keys_list)
    )
    return [response for response in responses]

def process_feature_store_data(data):
    """Process feature store data and convert to DataFrame"""
    processed_data = []
    for response in data:
        for data_item in response.data:
            row_data = {}
            for feature in data_item.key_values.features:
                if feature.name != 'feature_timestamp':
                    if feature.value.HasField('int64_value'):
                        row_data[feature.name] = feature.value.int64_value
                    elif feature.value.HasField('string_value'):
                        row_data[feature.name] = feature.value.string_value
                    # Add more value types as needed
            processed_data.append(row_data)
    return pd.DataFrame(processed_data)

def fetch_all_data(data_client, feature_view, batch_size=1000):
    """Fetch all available data in batches"""
    all_data = []
    batch_start = 1
    while True:
        batch_keys = [[f"{num}" for num in range(batch_start, batch_start + batch_size)]]
        batch_data = sffv(data_client, feature_view, batch_keys)
        if not batch_data or not batch_data[0].data:
            break
        all_data.extend(batch_data)
        batch_start += batch_size
    return all_data

# Setup
# LOCATION = "your-location"  # Replace with your actual location
API_ENDPOINT = f"{LOCATION}-aiplatform.googleapis.com"
data_client = FeatureOnlineStoreServiceClient(
    client_options={"api_endpoint": API_ENDPOINT}
)

fv = FeatureView(name="housing_data_view_bq", feature_online_store_id="nashtech_feature_store_bigtable")

# Fetch all data
data = fetch_all_data(data_client, fv.resource_name)

# Process the data and create a DataFrame
df = process_feature_store_data(data)

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