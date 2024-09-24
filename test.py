# # # from google.cloud import storage
# # # # from google.cloud import aiplatform
# # # # from constants import PROJECT_ID, REGION
# # # # aiplatform.init(project=PROJECT_ID, location=REGION)
# # #
# # #
# # # def upload_blob(bucket_name, source_file_name, destination_blob_name):
# # #     """Uploads a file to the bucket."""
# # #
# # #     # Initialize a storage client
# # #     storage_client = storage.Client()
# # #
# # #     # Get the bucket
# # #     bucket = storage_client.bucket(bucket_name)
# # #
# # #     # Create a blob object from the file
# # #     blob = bucket.blob(destination_blob_name)
# # #
# # #     # Upload the file to the bucket
# # #     blob.upload_from_filename(source_file_name)
# # #
# # #     print(f"File {source_file_name} uploaded to {destination_blob_name}.")
# # #
# # #
# # # # Example usage
# # # if __name__ == "__main__":
# # #     BUCKET_NAME = "nashtech_vertex_ai_artifact"  # Replace with your bucket name
# # #     SOURCE_FILE_NAME = "Housing.csv"  # Replace with your local file path
# # #     DESTINATION_BLOB_NAME = "Housing.csv"  # The name you want to give to the file in GCS
# # #
# # #     upload_blob(BUCKET_NAME, SOURCE_FILE_NAME, DESTINATION_BLOB_NAME)
# # import pandas as pd
# # import numpy as np
# # from sklearn.model_selection import train_test_split, GridSearchCV
# # from sklearn.preprocessing import StandardScaler, OneHotEncoder
# # from sklearn.compose import ColumnTransformer
# # from sklearn.pipeline import Pipeline
# # from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
# # from sklearn.metrics import mean_squared_error, r2_score
# # from sklearn.feature_selection import RFE
# # from sklearn.svm import SVR
# #
# # # Load the data
# # data = pd.read_csv('Housing.csv')
# #
# # # Feature Engineering
# # data['price_per_sqft'] = data['price'] / data['area']
# # data['total_rooms'] = data['bedrooms'] + data['bathrooms']
# # data['bed_bath_ratio'] = data['bedrooms'] / data['bathrooms']
# #
# # # Encode categorical variables
# # binary_features = ['mainroad', 'guestroom', 'basement', 'hotwaterheating', 'airconditioning', 'prefarea']
# # data[binary_features] = data[binary_features].replace({'yes': 1, 'no': 0})
# #
# # # Separate features and target
# # X = data.drop('price', axis=1)
# # y = np.log(data['price'])  # Log transform the target
# #
# # # Define numeric and categorical columns
# # numeric_features = ['area', 'bedrooms', 'bathrooms', 'stories', 'parking', 'price_per_sqft', 'total_rooms', 'bed_bath_ratio']
# # categorical_features = ['furnishingstatus']
# #
# # # Create preprocessing steps
# # numeric_transformer = StandardScaler()
# # categorical_transformer = OneHotEncoder(drop='first', sparse_output=False)
# #
# # preprocessor = ColumnTransformer(
# #     transformers=[
# #         ('num', numeric_transformer, numeric_features),
# #         ('cat', categorical_transformer, categorical_features)
# #     ])
# #
# # # Split the data
# # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# #
# # # Define models
# # rf = RandomForestRegressor(random_state=42)
# # gb = GradientBoostingRegressor(random_state=42)
# #
# # # Create pipelines
# # rf_pipeline = Pipeline([
# #     ('preprocessor', preprocessor),
# #     ('selector', RFE(estimator=SVR(kernel="linear"))),
# #     ('regressor', rf)
# # ])
# #
# # gb_pipeline = Pipeline([
# #     ('preprocessor', preprocessor),
# #     ('selector', RFE(estimator=SVR(kernel="linear"))),
# #     ('regressor', gb)
# # ])
# #
# # # Define parameter grids
# # rf_param_grid = {
# #     'selector__n_features_to_select': [5, 7, 9],
# #     'regressor__n_estimators': [100, 200],
# #     'regressor__max_depth': [None, 10, 20]
# # }
# #
# # gb_param_grid = {
# #     'selector__n_features_to_select': [5, 7, 9],
# #     'regressor__n_estimators': [100, 200],
# #     'regressor__learning_rate': [0.01, 0.1]
# # }
# #
# # # Perform Grid Search
# # rf_grid_search = GridSearchCV(rf_pipeline, rf_param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
# # gb_grid_search = GridSearchCV(gb_pipeline, gb_param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
# #
# # # Fit the models
# # rf_grid_search.fit(X_train, y_train)
# # gb_grid_search.fit(X_train, y_train)
# #
# # # Make predictions
# # rf_pred = rf_grid_search.predict(X_test)
# # gb_pred = gb_grid_search.predict(X_test)
# #
# # # Evaluate models
# # def evaluate_model(y_true, y_pred, model_name):
# #     mse = mean_squared_error(y_true, y_pred)
# #     rmse = np.sqrt(mse)
# #     r2 = r2_score(y_true, y_pred)
# #     print(f"{model_name} Results:")
# #     print(f"RMSE: {rmse:.4f}")
# #     print(f"R-squared: {r2:.4f}")
# #     print("--------------------")
# #
# # evaluate_model(y_test, rf_pred, "Random Forest")
# # evaluate_model(y_test, gb_pred, "Gradient Boosting")
# #
# # # Ensemble prediction
# # ensemble_pred = (rf_pred + gb_pred) / 2
# # evaluate_model(y_test, ensemble_pred, "Ensemble")
# #
# # # Feature Importance (for Random Forest)
# # feature_importance = rf_grid_search.best_estimator_.named_steps['regressor'].feature_importances_
# # selected_features = rf_grid_search.best_estimator_.named_steps['selector'].get_support()
# # feature_names = numeric_features + categorical_features
# # selected_feature_names = [feature for feature, selected in zip(feature_names, selected_features) if selected]
# #
# # print("\nSelected Features:")
# # print(selected_feature_names)
# # print("\nFeature Importance Scores:")
# # print(feature_importance)
# #
# # # Ensure that the lengths match
# # assert len(selected_feature_names) == len(feature_importance), "Mismatch in feature names and importance scores"
# #
# # importance_df = pd.DataFrame({'feature': selected_feature_names, 'importance': feature_importance})
# # importance_df = importance_df.sort_values('importance', ascending=False)
# # print("\nFeature Importance:")
# # print(importance_df)
# #
# # # Visualize feature importance
# # import matplotlib.pyplot as plt
# #
# # plt.figure(figsize=(10, 6))
# # importance_df.plot(kind='bar', x='feature', y='importance')
# # plt.title('Feature Importance')
# # plt.xlabel('Features')
# # plt.ylabel('Importance')
# # plt.tight_layout()
# # plt.show()
# #
# # from google.cloud import bigquery
# #
# # client = bigquery.Client()
# #
# # dataset_id = f"{client.project}.housing_dataset"
# #
# # dataset = bigquery.Dataset(dataset_id)
# # dataset.location = "US"  # Specify the location
# #
# # dataset = client.create_dataset(dataset, timeout=30)
# # print(f"Created dataset {client.project}.{dataset.dataset_id}")
#
# # nashtech-ai-dev-389315.housing_dataset
#
# from google.cloud import bigquery
# from google.cloud import storage
# import pandas as pd
# from io import StringIO
#
#
# def load_csv_to_bigquery(bucket_name, blob_name, project_id, dataset_id, table_id):
#     # Initialize clients
#     storage_client = storage.Client()
#     bigquery_client = bigquery.Client()
#
#     # Get the bucket and blob
#     bucket = storage_client.get_bucket(bucket_name)
#     blob = bucket.blob(blob_name)
#
#     # Download the contents of the blob as a string
#     csv_content = blob.download_as_text()
#
#     # Use pandas to read the CSV content
#     df = pd.read_csv(StringIO(csv_content))
#
#     # Clean column names (remove spaces, lowercase)
#     df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')
#
#     # Define the table reference
#     table_ref = bigquery_client.dataset(dataset_id).table(table_id)
#
#     # Set up the job configuration
#     job_config = bigquery.LoadJobConfig()
#     job_config.autodetect = True
#     job_config.source_format = bigquery.SourceFormat.CSV
#
#     # Load the dataframe into BigQuery
#     job = bigquery_client.load_table_from_dataframe(df, table_ref, job_config=job_config)
#
#     # Wait for the job to complete
#     job.result()
#
#     print(f"Loaded {job.output_rows} rows into {project_id}:{dataset_id}.{table_id}")
#
#
# # Usage
# bucket_name = "nashtech_vertex_ai_artifact"
# blob_name = "Housing.csv"
# project_id = "nashtech-ai-dev-389315"  # Replace with your GCP project ID
# dataset_id = "housing_dataset"  # Replace with your BigQuery dataset ID
# table_id = "housing_data"  # Replace with your desired table name
#
# load_csv_to_bigquery(bucket_name, blob_name, project_id, dataset_id, table_id)

# from google.cloud import bigquery
#
#
# def create_new_bigquery_table(project_id, dataset_id, source_table_id, new_table_id):
#     client = bigquery.Client(project=project_id)
#     dataset_ref = client.dataset(dataset_id)
#
#     # Reference to the source table
#     source_table_ref = dataset_ref.table(source_table_id)
#     source_table = client.get_table(source_table_ref)
#
#     # Create the new table with the updated schema
#     new_table_ref = dataset_ref.table(new_table_id)
#
#     # Define the schema for the new table, including 'id', 'feature_timestamp', and 'Category_id'
#     new_schema = list(source_table.schema) + [
#         bigquery.SchemaField("id", "INTEGER", mode="REQUIRED", description="Unique identifier for each row"),
#         bigquery.SchemaField("feature_timestamp", "TIMESTAMP", mode="REQUIRED", description="Feature timestamp for each record"),
#         bigquery.SchemaField("Category_id", "INTEGER", mode="REQUIRED", description="Dividing dataset into four categories")
#     ]
#
#     # Create the new table with the updated schema
#     new_table = bigquery.Table(new_table_ref, schema=new_schema)
#     new_table = client.create_table(new_table)
#     print(f"Created table {new_table.project}.{new_table.dataset_id}.{new_table.table_id}")
#
#     # Copy data from the old table to the new table, adding the new columns
#     query = f"""
#     INSERT INTO `{project_id}.{dataset_id}.{new_table_id}`
#     WITH categorized_data AS (
#         SELECT
#             *,
#             ROW_NUMBER() OVER() AS id,
#             CURRENT_TIMESTAMP() AS feature_timestamp,
#             CASE
#                 WHEN MOD(ROW_NUMBER() OVER(), 4) = 1 THEN 1
#                 WHEN MOD(ROW_NUMBER() OVER(), 4) = 2 THEN 2
#                 WHEN MOD(ROW_NUMBER() OVER(), 4) = 3 THEN 3
#                 ELSE 4
#             END AS Category_id
#         FROM `{project_id}.{dataset_id}.{source_table_id}`
#     )
#     SELECT * FROM categorized_data
#     """
#
#     job = client.query(query)
#     job.result()  # Wait for the query to complete
#
#     print(f"Data copied from {source_table_id} to {new_table_id} with new columns added.")
#
#     return new_table_id
#
#
# # Usage
# project_id = "nashtech-ai-dev-389315"
# dataset_id = "housing_dataset"
# source_table_id = "housing_data"
# new_table_id = "new_housing_data_with_id_and_category"
#
# new_table_id = create_new_bigquery_table(project_id, dataset_id, source_table_id, new_table_id)
# print(f"New table created: {new_table_id}")



# from google.cloud import aiplatform
# from vertexai.resources.preview import feature_store
# from typing import List
#
# def create_feature_group_sample(
#     project: str,
#     location: str,
#     feature_group_id: str,
#     bq_table_uri: str,
#     entity_id_columns: List[str],
# ):
#     aiplatform.init(project=project, location=location)
#     fg = feature_store.FeatureGroup.create(
#         name=feature_group_id,
#         source=feature_store.utils.FeatureGroupBigQuerySource(
#             uri=bq_table_uri, entity_id_columns=entity_id_columns
#         ),
#     )
#     return fg
#
# # Define your variables
# project = "nashtech-ai-dev-389315"
# location = "us-central1"  # Example, change based on your Vertex AI location
# feature_group_id = "housing_feature_group"
# bq_table_uri = f"bq://{project}.housing_dataset.new_housing_data_with_id"
# entity_id_columns = ["id"]  # Specify the 'id' column as the entity ID
#
# # Call the function to create the feature group
# feature_group = create_feature_group_sample(
#     project=project,
#     location=location,
#     feature_group_id=feature_group_id,
#     bq_table_uri=bq_table_uri,
#     entity_id_columns=entity_id_columns,
# )
#
# print(f"Feature group created: {feature_group.resource_name}")

#
# from google.cloud import aiplatform
# from vertexai.resources.preview import feature_store
#
#
# def retrieve_feature_group_sample(
#         project: str,
#         location: str,
#         existing_feature_group_id: str
# ):
#     aiplatform.init(project=project, location=location)
#
#     # Initialize an existing feature group
#     feature_group = feature_store.FeatureGroup(existing_feature_group_id)
#     print(f"Feature group retrieved: {feature_group.resource_name}")
#
#     return feature_group
#
#
# def create_features(
#         feature_group,
#         feature_columns: dict
# ):
#     # Add features to the feature group
#     for feature_id, version_column_name in feature_columns.items():
#         try:
#             feature = feature_group.create_feature(
#                 name=feature_id, version_column_name=version_column_name
#             )
#             print(f"Feature {feature_id} added.")
#         except Exception as e:
#             print(f"Error adding feature {feature_id}: {e}")
#
#
# def ingest_data_into_feature_group(
#         feature_group,
#         bq_table_uri: str,
#         entity_id_columns: list,
#         feature_timestamp_column: str
# ):
#     # Ingest data from BigQuery to Feature Store
#     try:
#         # Initialize a client for data ingestion
#         client = aiplatform.gapic.FeaturestoreServiceClient()
#
#         # Define ingestion request
#         request = {
#             "name": feature_group.resource_name,
#             "bigquery_source": {
#                 "uri": bq_table_uri
#             },
#             "entity_id_columns": entity_id_columns,
#             "feature_timestamp_column": feature_timestamp_column
#         }
#
#         # Call the API
#         response = client.ingest(request=request)
#
#         # Wait for the ingestion job to complete
#         operation = response.operation
#         operation.result()  # This will block until the operation is finished
#
#         print(f"Data ingestion from BigQuery table {bq_table_uri} completed.")
#     except Exception as e:
#         print(f"Error during data ingestion: {e}")
#
#
# # Define your variables
# project = "nashtech-ai-dev-389315"
# location = "us-central1"
# feature_group_id = "housing_feature_group"
# bq_table_uri = f"bq://{project}.housing_dataset.new_housing_data_with_id"
# entity_id_columns = ["id"]  # Specify the 'id' column as the entity ID
# feature_timestamp_column = "feature_timestamp"  # Your timestamp column in BigQuery
#
# # Retrieve the existing feature group
# feature_group = retrieve_feature_group_sample(
#     project=project,
#     location=location,
#     existing_feature_group_id=feature_group_id
# )
#
# # Define features to add (name and version_column_name)
# feature_columns = {
#     "price": "feature_timestamp",
#     "area": "feature_timestamp",
#     "bedrooms": "feature_timestamp",
#     "bathrooms": "feature_timestamp",
#     "stories": "feature_timestamp",
#     "mainroad": "feature_timestamp",
#     "guestroom": "feature_timestamp",
#     "basement": "feature_timestamp",
#     "hotwaterheating": "feature_timestamp",
#     "airconditioning": "feature_timestamp",
#     "parking": "feature_timestamp",
#     "prefarea": "feature_timestamp",
#     "furnishingstatus": "feature_timestamp",
# }
#
# # Add the features to the feature group
# create_features(feature_group, feature_columns)
#
# # Ingest data from BigQuery into the feature group
# ingest_data_into_feature_group(
#     feature_group,
#     bq_table_uri=bq_table_uri,
#     entity_id_columns=entity_id_columns,
#     feature_timestamp_column=feature_timestamp_column
# )

# Feature Store Creation
# PROJECT_ID = "nashtech-ai-dev-389315"  # @param {type:"string"}
# LOCATION = "us-central1"  # @param {type:"string"}
#
# API_ENDPOINT = f"{LOCATION}-aiplatform.googleapis.com"
#
# from google.cloud import aiplatform
# from google.cloud.aiplatform_v1beta1 import FeatureOnlineStoreAdminServiceClient, FeatureRegistryServiceClient
# from vertexai.resources.preview import FeatureOnlineStore
# aiplatform.init(project=PROJECT_ID, location=LOCATION)
#
# admin_client = FeatureOnlineStoreAdminServiceClient(
#     client_options={"api_endpoint": API_ENDPOINT}
# )
# registry_client = FeatureRegistryServiceClient(
#     client_options={"api_endpoint": API_ENDPOINT}
# )
# FEATURE_ONLINE_STORE_ID = "nashtech_feature_store"  # @param {type:"string"}
# my_fos = FeatureOnlineStore.create_optimized_store(FEATURE_ONLINE_STORE_ID)
#
# print(my_fos.gca_resource)


# from google.cloud import aiplatform
# from vertexai.resources.preview import feature_store
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

PROJECT_ID = "nashtech-ai-dev-389315"  # @param {type:"string"}
LOCATION = "us-central1"
BQ_TABLE_URI = f"bq://{PROJECT_ID}.housing_dataset.new_housing_data_with_id"
FEATURE_VIEW_ID = "housing_data_view"
FEATURE_ONLINE_STORE_ID = "nashtech_feature_store"  # @param {type:"string"}
ENTITY_ID_COLUMNS = ["id"]
FEATURE_GROUP_ID = "housing_feature_group"


# # import bigframes
# # import bigframes.pandas
# import pandas as pd
# from google.cloud import bigquery
# from vertexai.resources.preview.feature_store import (Feature, FeatureGroup, offline_store)
# from vertexai.resources.preview.feature_store import utils as fs_utils
#
# fg = FeatureGroup("housing_feature_group")
# f_price = fg.get_feature("price")
# f_area = fg.get_feature("area")
# f_bedrooms = fg.get_feature("bedrooms")
# f_bathrooms = fg.get_feature("bathrooms")
# f_stories = fg.get_feature("stories")
# f_mainroad = fg.get_feature("mainroad")
# f_guestroom = fg.get_feature("guestroom")
# f_basement = fg.get_feature("basement")
# f_hotwaterheating = fg.get_feature("hotwaterheating")
# f_airconditioning = fg.get_feature("airconditioning")
# f_parking = fg.get_feature("parking")
# f_prefarea = fg.get_feature("prefarea")
# f_furnishingstatus = fg.get_feature("furnishingstatus")
#
# entity_df = pd.DataFrame(
#   data={
#     "ENTITY_ID_COLUMN": [
#       "ENTITY_ID_1",
#       "ENTITY_ID_2",
#     ],
#     "timestamp": [
#       pd.Timestamp("FEATURE_TIMESTAMP_1"),
#       pd.Timestamp("FEATURE_TIMESTAMP_2"),
#     ],
#   },
# )
# offline_store.fetch_historical_feature_values(
#     entity_df=entity_df,
#     features=[
#         f_price,
#         f_area,
#         f_bedrooms,
#         f_bathrooms,
#         f_stories,
#         f_mainroad,
#         f_guestroom,
#         f_basement,
#         f_hotwaterheating,
#         f_airconditioning,
#         f_parking,
#         f_prefarea,
#         f_furnishingstatus
#     ],
# )


from google.cloud import aiplatform
from vertexai.resources.preview import feature_store


def create_optimized_public_feature_online_store_sample(
    project: str,
    location: str,
    feature_online_store_id: str,
):
    aiplatform.init(project=project, location=location)
    fos = feature_store.FeatureOnlineStore.create_optimized_store(
        feature_online_store_id
    )
    print("fos is" , fos)
    return fos

# create_optimized_public_feature_online_store_sample(project=PROJECT_ID,location=LOCATION,feature_online_store_id="nashtech_feature_store_optimized")


from google.cloud import aiplatform
from vertexai.resources.preview import feature_store
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
#                                    existing_feature_online_store_id="nashtech_feature_store_optimized",
#                                    feature_view_id=FEATURE_VIEW_ID,
#                                    bq_table_uri=BQ_TABLE_URI,
#                                    entity_id_columns=ENTITY_ID_COLUMNS)

# from google.cloud.aiplatform_v1 import FeatureOnlineStoreServiceClient
# from google.cloud.aiplatform_v1.types import feature_online_store_service as feature_online_store_service_pb2
#
# data_client = FeatureOnlineStoreServiceClient(
#   client_options={"api_endpoint": f"171990006863233024.us-central1-24562761082.featurestore.vertexai.goog"}
# )
# data = data_client.fetch_feature_values(
#   request=feature_online_store_service_pb2.FetchFeatureValuesRequest(
#     feature_view=f"projects/nashtech-ai-dev-389315/locations/us-central1/featureOnlineStores/nashtech_feature_store_optimized/featureViews/housing_data_view",
#     # id=f"10",
#     # format=feature_online_store_service_pb2.FetchFeatureValuesRequest.Format.TABULAR,
#   )
# )
# print(data)
from vertexai.resources.preview import FeatureView
# data = (
#     FeatureView(name=FEATURE_VIEW_ID, feature_online_store_id="nashtech_feature_store_optimized").read(key="area")
# )
#
# print(data)

#
# from pandas import DataFrame
# from typing import List
#
#
# def online_serve_feature_values(
#         project: str,
#         location: str,
#         featurestore_id: str,
#         entity_type_id: str,
#         entity_ids: List[str],
#         feature_ids: List[str]) -> DataFrame:
#     """
#     Retrieves online feature values from a Featurestore.
#     Args:
#         project: The Google Cloud project ID.
#         location: The Google Cloud location.
#         featurestore_id: The Featurestore ID.
#         entity_type_id: The Entity Type ID.
#         entity_ids: The list of Entity IDs.
# feature_ids: The list of Feature IDs.
# Returns:
# A Pandas DataFrame containing the feature values.
# """
#     # Initialize the Vertex SDK for Python
#     aiplatform.init(project=project, location=location)
#     # Get the entity type from an existing Featurestore
#     entity_type = aiplatform.featurestore.EntityType(entity_type_name=entity_type_id,
#                                                      featurestore_id=featurestore_id)
#     # Retrieve the feature values
#     feature_values = entity_type.read(entity_ids=entity_ids, feature_ids=feature_ids)
#     print(f"This are feature values: {feature_values}")
#     return feature_values
#
# online_serve_feature_values(PROJECT_ID,
#                             LOCATION,
#                             featurestore_id="nashtech_feature_store_optimized",
#                             entity_type_id="price",
#                             entity_ids=["7070000","2310000"],
#                             feature_ids=["price"])
#

# import logging
# from google.cloud import bigquery
# dataset_id = "house_dataset"
# # output_dataset_id = ""
# logger = logging.getLogger('tipper')
# logger.setLevel(logging.DEBUG)
# logger.addHandler(logging.StreamHandler())
# # Construct a BigQuery client object.
# client = bigquery.Client(project=PROJECT_ID)
#
# # Construct a full Dataset object to send to the API.
# dataset = bigquery.Dataset(f"{PROJECT_ID}.{dataset_id}")
#
# # Specify the geographic location where the dataset should reside.
# dataset.location = "US"
#
# # Send the dataset to the API for creation, with an explicit timeout.
# # Raises google.api_core.exceptions.Conflict if the Dataset already
# # exists within the project.
# dataset = client.create_dataset(dataset, timeout=30)  # Make an API request.
# print(f"Created dataset {client.project}.{dataset.dataset_id}")

# Output the new dataset ID
# output_dataset_id.uri = dataset.dataset_id

# csv_bucket = "nashtech_vertex_ai_artifact"
# csv_file_name = "Housing.csv"
# dataset_id = "housing_dataset"
# new_table_id = "housing_table_new"
# import logging
# from google.cloud import bigquery, storage
# import pandas as pd
# import io
# # import mlflow
#
# logger = logging.getLogger('tipper')
# logger.setLevel(logging.DEBUG)
# logger.addHandler(logging.StreamHandler())
#
# # Initialize BigQuery and GCS clients
# bigquery_client = bigquery.Client(project=PROJECT_ID)
# storage_client = storage.Client()
#
# # Reference the dataset in BigQuery
# dataset_ref = bigquery_client.dataset(dataset_id)
#
# # Define the GCS file path for the CSV file
# csv_uri = f'gs://{csv_bucket}/{csv_file_name}'
#
# # Read the CSV file from GCS into a pandas DataFrame
# logger.info(f"Loading CSV data from GCS: {csv_uri}")
# bucket = storage_client.bucket(csv_bucket)
# blob = bucket.blob(csv_file_name)
# csv_data = blob.download_as_text()
#
# # Load data into a pandas DataFrame
# df = pd.read_csv(io.StringIO(csv_data))
#
# logger.info(f"DataFrame loaded with shape: {df.shape}")
# logger.info(f"DataFrame columns: {df.columns}")
#
# # Add 'id', 'feature_timestamp', and 'Category_id' to the DataFrame
# df['id'] = range(1, len(df) + 1)
# df['feature_timestamp'] = pd.Timestamp.now()
# # df['Category_id'] = (df['id'] % 4).replace({1: 1, 2: 2, 3: 3, 0: 4})
#
# logger.info("Added 'id', 'feature_timestamp', and 'Category_id' columns.")
#
# # Define schema for the new BigQuery table
# schema = [
#     bigquery.SchemaField(name, "STRING") if dtype == "object" else bigquery.SchemaField(name, "INTEGER")
#     for name, dtype in zip(df.columns, df.dtypes)
# ]
# # schema += [
# #     bigquery.SchemaField("id", "INTEGER", mode="REQUIRED", description="Unique identifier for each row"),
# #     bigquery.SchemaField("feature_timestamp", "TIMESTAMP", mode="REQUIRED", description="Feature timestamp for each record"),
# #     # bigquery.SchemaField("Category_id", "INTEGER", mode="REQUIRED", description="Dividing dataset into four categories")
# # ]
#
# # Create a new table in BigQuery
# new_table_ref = dataset_ref.table(new_table_id)
# new_table = bigquery.Table(new_table_ref, schema=schema)
# new_table = bigquery_client.create_table(new_table)
# logger.info(f"Created new BigQuery table: {new_table.project}.{new_table.dataset_id}.{new_table.table_id}")
#
# # Load data from the DataFrame into the BigQuery table
# job_config = bigquery.LoadJobConfig(write_disposition=bigquery.WriteDisposition.WRITE_TRUNCATE)
# job = bigquery_client.load_table_from_dataframe(df, new_table_ref, job_config=job_config)
# job.result()  # Wait for the job to complete
#
# logger.info(f"Data loaded into {new_table_id} with new columns added.")
