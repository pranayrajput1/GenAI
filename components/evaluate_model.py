from kfp.v2.components.component_decorator import component
from kfp.v2 import dsl
from components.dependencies import resolve_dependencies
from constants import BASE_IMAGE


@component(
    base_image=BASE_IMAGE,
    packages_to_install=resolve_dependencies(
        'pandas',
        'fsspec',
        'google-cloud-storage',
        'gcsfs',
        'pyarrow',
        'matplotlib',
        'numpy',
        'scikit-learn'
    )
)
def evaluate_model(
        batch_size: int,
        model_name: str,
        bucket_name: str,
        dataset_path: dsl.Input[dsl.Dataset],
        model_path: dsl.Input[dsl.Model],
        avg_score: dsl.Output[dsl.Metrics],
        cluster_image: dsl.Output[dsl.Artifact]
):
    """
    Function to evaluate the performance of model,
    give average silhouette score as output,
    give formed cluster image as output
    @batch_size: batch size used to evaluate model performance.
    @bucket_name: bucket name where cluster image will be saved.
    @dataset_path: dataset parquet file path.
    @model_path: trained model artifact path as input.
    @avg_score: average silhouette_score given as output.
    @cluster_image: formed cluster image given as output.
    @image_path: image name used to save clusters image.
    """
    import logging
    from google.cloud import storage
    from utils import silhouette_score
    from utils import preprocessing
    import pickle
    import pandas as pd

    logger = logging.getLogger('tipper')
    logger.setLevel(logging.DEBUG)
    logger.addHandler(logging.StreamHandler())

    image_path = f"{model_name}_cluster_image.png"

    try:
        logging.info(f"Reading model from: {model_path.path}")
        file_name = model_path.path + ".pkl"

        with open(file_name, 'rb') as model_file:
            trained_model = pickle.load(model_file)

        logging.info(f"Reading process data from: {dataset_path.path}")
        data_frame = pd.read_parquet(dataset_path.path)
        household_train, household_test = preprocessing.processing_data(data_frame)

        logging.info("calculating average silhouette_scores")
        average_silhouette_score = silhouette_score.get_silhouette_score_and_cluster_image(
            household_train,
            batch_size,
            trained_model,
            image_path)

        logging.info(f"Setting Average Silhouette Score: {average_silhouette_score}")
        avg_score.log_metric("score", average_silhouette_score)

        try:
            logging.info("Setting client connection using storage client API'")
            client = storage.Client()

            logging.info(f"Getting bucket: {bucket_name} from GCS")
            bucket = client.get_bucket(bucket_name)
            blob = bucket.blob(image_path)

            logging.info(f"Uploading Image to Bucket: 'gs://{bucket_name}/'")
            with open(image_path, 'rb') as file:
                blob.upload_from_file(file, content_type='image/png')

            logging.info(f"Uploaded Image to Bucket: 'gs://{bucket_name}/' successfully'")
            image_url = f"https://storage.cloud.google.com/{bucket_name}/{image_path}"
            cluster_image.uri = image_url

        except Exception as e:
            logging.info("Failed To Save Image to Bucket")
            raise e

    except Exception as e:
        logging.info("Failed To Execute Model validation")
        raise e
