from typing import List
import pandas as pd
from google.cloud.aiplatform_v1beta1 import FeatureOnlineStoreServiceClient
from google.cloud.aiplatform_v1beta1.types import feature_online_store_service as feature_online_store_service_pb2

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

def setup_data_client(location):
    """Setup the FeatureOnlineStoreServiceClient"""
    API_ENDPOINT = f"{location}-aiplatform.googleapis.com"
    return FeatureOnlineStoreServiceClient(
        client_options={"api_endpoint": API_ENDPOINT}
    )