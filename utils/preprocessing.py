from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
import logging
import pandas as pd
import numpy as np


def processing_data(dataframe):
    """
    Calling all function to perform preprocessing
    @param: data_frame
    @return: data_frame after doing all preprocessing
    """
    logging.info('Task: dropping columns "Date" & "Time"')
    dropped_df = drop_columns(dataframe)

    # handling missing values
    logging.info('Task: handling missing values')
    handled_missing_df = handling_missing_values(dropped_df)

    # feature selection
    logging.info('Task: making feature selection')
    featured_df = feature_selection(handled_missing_df)

    # scaling data
    logging.info('Task: scaling data')
    scaled_df = scaling(featured_df)

    # performing dimensionality reduction
    logging.info('Task: reducing dimensions from data')
    reduced_dimensions_data = dimensionality_reduction(scaled_df)

    logging.info('Task: Splitting data into train and test')
    train_data, test_data = splitting_data(reduced_dimensions_data)
    return train_data, test_data


def drop_columns(data_frame):
    """
    getting the dataframe and dropping the un_necessary column
    @param: data_frame
    @return: data_frame after dropping columns
    """
    dropping_columns = ['Date', 'Time']
    data_frame = data_frame.drop(dropping_columns, axis=1)
    return data_frame


def handling_missing_values(data_frame):
    """
    getting the dataframe and handling missing values in each column
    @param: data_frame
    @return: data_frame after handling missing values
    """

    columns_to_preprocess = ['Global_active_power', 'Global_reactive_power', 'Voltage', 'Global_intensity',
                             'Sub_metering_1',
                             'Sub_metering_2', 'Sub_metering_3']
    for column in columns_to_preprocess:
        data_frame[column] = data_frame[column].replace('?', np.nan)
        data_frame[column] = data_frame[column].astype(float)
        mean_value = data_frame[column].mean()
        data_frame[column].fillna(mean_value, inplace=True)
    return data_frame


def feature_selection(data_frame):
    """
    getting the dataframe and doing feature selection
    @param: data_frame:
    @return: data_frame after doing feature selection
    """
    selected_features = ['Global_active_power', 'Global_reactive_power', 'Voltage', 'Global_intensity',
                         'Sub_metering_1', 'Sub_metering_2', 'Sub_metering_3']
    feature_dataframe = data_frame[selected_features]
    return feature_dataframe


def scaling(data_frame):
    """
    getting the dataframe and normalizing it
    @param: data_frame
    @return: data_frame after scaling
    """
    scale = StandardScaler()
    scaled_data = scale.fit_transform(data_frame)
    scaled_dataframe = pd.DataFrame(scaled_data)
    return scaled_dataframe


def dimensionality_reduction(data_frame):
    """
    reducing the dimension of dataframe
    @param: data_frame
    @return dataframe after applying pca
    """
    pca = PCA(n_components=2)
    X_principal = pca.fit_transform(data_frame)
    X_principal = pd.DataFrame(X_principal)
    X_principal.columns = ['P1', 'P2']
    return X_principal


def splitting_data(data_frame):
    """
    splitting data into training and testing set.
    @param: data_frame
    @return dataframe after applying train test split
    """
    train_data, test_data = train_test_split(data_frame, test_size=0.005, random_state=42)
    return train_data, test_data


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
