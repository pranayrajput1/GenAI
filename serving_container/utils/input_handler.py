import pandas as pd
from sklearn.preprocessing import StandardScaler
from google.cloud import storage
import logging

logging.basicConfig(level=logging.INFO)


def download_model_files_from_bucket(bucket_name, destination_folder):
    client = storage.Client()

    '''Get the desired bucket'''
    bucket = client.get_bucket(bucket_name)

    '''List all files in the bucket'''
    blobs = bucket.list_blobs()

    for blob in blobs:
        '''Construct the local file path'''
        local_path = f"{destination_folder}/{blob.name}"

        '''Download the file'''
        blob.download_to_filename(local_path)
        logging.info(f"Downloaded: {blob.name} to {local_path}")


def handle_file(file):
    """
    function to handle input of file.
    """
    readfile = pd.read_csv(file)
    input_df = pd.DataFrame(readfile)
    scaled_df = scaling(input_df)
    return scaled_df


def handle_json(input_instance, subset_data):
    """
    function to handle input of dictionary type data.
    """
    subset_data = pd.read_csv(subset_data)
    subset_df = pd.DataFrame(subset_data)

    input_dataframe = pd.DataFrame(input_instance,
                                   columns=["Global_reactive_power", "Global_intensity"],
                                   index=[0])

    combined_df = pd.concat([subset_df, input_dataframe], ignore_index=True)
    scaled_df = scaling(combined_df)
    return scaled_df


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
