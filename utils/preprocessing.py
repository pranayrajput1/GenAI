import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import vertex_ray
from vertex_ray import Resources
from google.cloud import aiplatform as vertex_ai
import ray
import logging

logger = logging.getLogger('tipper')
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler())

def create_ray_cluster(project_id, region, cluster_name=None):
    """Create a Ray cluster on Vertex AI."""
    vertex_ai.init(project=project_id, location=region)

    if not cluster_name:
        import uuid
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        cluster_name = f"ray-cluster-{timestamp}-{uuid.uuid4().hex[:6]}"

    head_node_type = Resources(
        machine_type="n1-standard-16",
        node_count=1,
    )
    worker_node_types = [
        Resources(
            machine_type="n1-standard-4",
            node_count=1,
        )
    ]

    ray_cluster_info = vertex_ray.create_ray_cluster(
        head_node_type=head_node_type,
        worker_node_types=worker_node_types,
        cluster_name=cluster_name,
        python_version="3.10.14"
    )
    # Return the connection information
    logger.info(f"Created cluster with name : {cluster_name}")
    return cluster_name

@ray.remote
def process_batch(batch, preprocessor=None):
    """Process a single batch of data."""
    # Handle missing values
    batch = handle_missing_values(batch)

    # Encode categorical variables
    batch = encode_categorical_variables(batch)

    # Perform feature engineering
    batch = feature_engineering(batch)

    # Prepare features and target
    X, y = prepare_features_and_target(batch)

    # Apply preprocessor if provided
    if preprocessor:
        X = preprocessor.transform(X)

    return X, y


def data_processing_pipeline( test_size=0.2, random_state=42, batch_size=1000):
    """
    Main pipeline function that processes the data using Ray and returns train and test sets.
    """
    # Initialize Ray (assuming it's already connected to the cluster)
    if not ray.is_initialized():
        ray.init(address='auto')

    file_path = "gs://nashtech_vertex_ai_artifact/Housing.csv"
    # Load data
    logger.info("Loading Data")

    df = load_data(file_path)

    logger.info("Creating Batches")
    # Create batches
    batches = create_batches(df, batch_size)

    # Process batches in parallel
    futures = [process_batch.remote(batch) for batch in batches]
    results = ray.get(futures)

    # Combine results
    X = pd.concat([result[0] for result in results])
    y = pd.concat([result[1] for result in results])

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    # Create and fit preprocessor
    preprocessor = create_preprocessor()
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    return X_train_processed, X_test_processed, y_train, y_test


def load_data(file_path):
    """Load the dataset from a CSV file."""
    return pd.read_csv(file_path)

def handle_missing_values(df):
    """Handle missing values in the dataframe."""
    # For this example, we'll just drop rows with any missing values
    # You might want to use more sophisticated imputation techniques depending on your data
    return df.dropna()

def encode_categorical_variables(df):
    """Encode binary categorical variables."""
    binary_features = ['mainroad', 'guestroom', 'basement', 'hotwaterheating', 'airconditioning', 'prefarea']
    df[binary_features] = df[binary_features].replace({'yes': 1, 'no': 0})
    return df

def feature_engineering(df):
    """Create new features."""
    df['price_per_sqft'] = df['price'] / df['area']
    df['total_rooms'] = df['bedrooms'] + df['bathrooms']
    df['bed_bath_ratio'] = df['bedrooms'] / df['bathrooms']
    return df

def prepare_features_and_target(df):
    """Separate features and target, and log-transform the target."""
    X = df.drop('price', axis=1)
    y = np.log(df['price'])
    return X, y

def create_preprocessor():
    """Create a preprocessor for numeric and categorical features."""
    numeric_features = ['area', 'bedrooms', 'bathrooms', 'stories', 'parking', 'price_per_sqft', 'total_rooms', 'bed_bath_ratio']
    categorical_features = ['furnishingstatus']

    numeric_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(drop='first', sparse_output=False)

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])

    return preprocessor

def create_batches(data_frame, batch_size):
    """
    creating batches to cover the complete dataset
    @param: data_frame
    @param: batch_size
    @return: batches
    """

    num_samples = len(data_frame)
    num_batches = num_samples // batch_size

    # Create batches
    batches_data = []
    for batch_idx in range(num_batches):
        start_idx = batch_idx * batch_size
        end_idx = (batch_idx + 1) * batch_size
        batch = data_frame[start_idx:end_idx]
        batches_data.append(batch)

    # Handle the remaining samples if the dataset size is not divisible by the batch size
    remaining_samples = num_samples % batch_size
    if remaining_samples > 0:
        batch = data_frame[-remaining_samples:]
        batches_data.append(batch)

    return batches_data

# file_path = '/home/nashtech/PycharmProjects/Mlops/Housing.csv'  # Replace with your actual file path
#
# X_train, X_test, y_train, y_test = data_processing_pipeline(file_path)
#
# print("Shapes of processed datasets:")
# print(f"X_train: {X_train.shape}")
# print(f"X_test: {X_test.shape}")
# print(f"y_train: {y_train.shape}")
# print(f"y_test: {y_test.shape}")