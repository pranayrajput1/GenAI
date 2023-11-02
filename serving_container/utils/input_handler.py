import pandas as pd
from sklearn.preprocessing import StandardScaler
from google.cloud import storage
import logging

logging.basicConfig(level=logging.INFO)


def concat_df(subset_df, input_df):
    combined_df = pd.concat([subset_df, input_df], ignore_index=True)
    return combined_df


def handle_file(input_file):
    """
    function to handle input of file.
    """
    input_dataframe = pd.read_csv(input_file)
    return input_dataframe


def handle_json(input_instance, subset_data):
    """
    function to handle input of dictionary type data.
    """
    subset_data = pd.read_csv(subset_data)
    subset_df = pd.DataFrame(subset_data)

    input_dataframe = pd.DataFrame(input_instance, columns=["Global_reactive_power", "Global_intensity"])
    combined_data = concat_df(subset_df, input_dataframe)
    return combined_data


def scaling(data_frame):
    """
    getting the dataframe and normalizing it
    @param data_frame
    @return scaled data_frame
    """
    scale = StandardScaler()
    scaled_data = scale.fit_transform(data_frame)
    scaled_dataframe = pd.DataFrame(scaled_data)
    return scaled_dataframe


def gcs_file_download(bucket_name, file_name):
    client = storage.Client()

    bucket = client.get_bucket(bucket_name)
    blob = bucket.blob(file_name)
    blob.download_to_filename(file_name)

    logging.info(f"File {file_name} downloaded")
