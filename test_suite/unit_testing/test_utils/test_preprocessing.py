import unittest

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
import logging
import pandas as pd
import numpy as np

from test_suite.unit_testing.test_data.test_constants import file_path


class Testing_Preprocessing:
    def __init__(self):
        """
        setting the file path
         @param dataframe
        @type dataframe
        """
        self.dataset = file_path

    def test_drop_columns(self,dataset):
        """
        getting the dataframe and dropping the un_necessary column
        @param data_frame
        @return data_frame
        """
        data_frame = pd.read_csv(dataset, delimiter=";", low_memory=False)
        # data_frame = pd.DataFrame(read_file)
        dropping_columns = ['Date', 'Time']
        data_frame = data_frame.drop(dropping_columns, axis=1)
        if len(data_frame) == 2075259:
            logging.info("The Dataframe is in right length")
        else:
            logging.info("The Dataframe is in wrong length")
        return data_frame

    def test_handling_missing_values(self,dataset):
        """
        getting the dataframe and handling missing values in each column
        :param data_frame
        :return data_frame
        """
        read_file = pd.read_csv(self.dataset, delimiter=";",
                                low_memory=False)
        data_frame = pd.DataFrame(read_file)

        columns_to_preprocess = ['Global_active_power', 'Global_reactive_power', 'Voltage', 'Global_intensity',
                                 'Sub_metering_1',
                                 'Sub_metering_2', 'Sub_metering_3']
        for column in columns_to_preprocess:
            data_frame[column] = data_frame[column].replace('?', np.nan)
            data_frame[column] = data_frame[column].astype(float)
            mean_value = data_frame[column].mean()
            data_frame[column].fillna(mean_value, inplace=True)

        assert data_frame.isnull().sum().sum() == 0, "There are still null values in the DataFrame."
        logging.info("There are no null values in the DataFrame.")
        return data_frame

    def test_feature_selection(self,dataset):
        """
        getting the dataframe and doing feature selection
        @param data_frame:
        @return data_frame
        """
        read_file = pd.read_csv(self.dataset, delimiter=";",
                                low_memory=False)
        data_frame = pd.DataFrame(read_file)

        selected_features = ['Global_active_power', 'Global_reactive_power', 'Voltage', 'Global_intensity',
                             'Sub_metering_1', 'Sub_metering_2', 'Sub_metering_3']
        feature_dataframe = data_frame[selected_features]
        assert isinstance(feature_dataframe, pd.DataFrame), "The result is not a DataFrame"

        logging.info("Feature selection completed successfully")
        return feature_dataframe

    def test_scaling(self,data_frame):
        """
        getting the dataframe and normalizing it
        @param data_frame
        @return data_frame
        """
        scale = StandardScaler()
        scaled_data = scale.fit_transform(data_frame)
        scaled_dataframe = pd.DataFrame(scaled_data)

        return scaled_dataframe

    def test_dimensionality_reduction(self,data_frame):
        """
        reducing the dimension of dataframe
        :param data_frame
        :return dataframe
        """
        pca = PCA(n_components=2)
        X_principal = pca.fit_transform(data_frame)
        X_principal = pd.DataFrame(X_principal)
        X_principal.columns = ['P1', 'P2']
        return X_principal

    def test_splitting_data(self,dataset):
        read_file = pd.read_csv(self.dataset, delimiter=";",
                                low_memory=False)
        data_frame = pd.DataFrame(read_file)
        train_data, test_data = train_test_split(data_frame, test_size=0.005, random_state=42)

        expected_train_shape = (2064882 , 9)
        actual_train_shape = train_data.shape
        if actual_train_shape == expected_train_shape:
            logging.info("The shape of the training data matches the expected shape.")
        else:
            logging.info(
                f"The shape of the training data is {actual_train_shape}, but the expected shape is {expected_train_shape}.")

        return train_data, test_data

    def test_create_batches(self, batch_size):
        """
        creating batches to cover the complete dataset
        @param data_frame
        @param batch_size
        @return batches
        """

        num_samples = len(self.dataset)
        num_batches = num_samples // batch_size

        # Create batches
        batches_data = []
        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = (batch_idx + 1) * batch_size
            batch = self.dataset[start_idx:end_idx]
            batches_data.append(batch)

        # Handle the remaining samples if the dataset size is not divisible by the batch size
        remaining_samples = num_samples % batch_size
        if remaining_samples > 0:
            batch = self.dataset[-remaining_samples:]
            batches_data.append(batch)

        return batches_data


class Test_class(unittest.TestCase):

    def test_for_length_of_dataframe(self):
        with self.assertLogs() as captured:
            divide_check = Testing_Preprocessing()
            dataset = file_path  # Provide the path to the dataset
            divide_check.test_drop_columns(dataset)
        self.assertEqual(len(captured.records), 1)
        self.assertEqual(captured.records[0].getMessage(), "The Dataframe is in right length")

    def test_for_checking_missing_values(self):
        with self.assertLogs() as captured:
            divide_check = Testing_Preprocessing()
            dataset = file_path  # Provide the path to the dataset
            divide_check.test_handling_missing_values(dataset)
        self.assertEqual(len(captured.records), 1)
        self.assertEqual(captured.records[0].getMessage(), "There are no null values in the DataFrame.")

    def test_for_dataframe(self):
        with self.assertLogs() as captured:
            divide_check = Testing_Preprocessing()
            dataset = file_path  # Provide the path to the dataset
            divide_check.test_feature_selection(dataset)
        self.assertEqual(len(captured.records), 1)
        self.assertEqual(captured.records[0].getMessage(), "Feature selection completed successfully")

    def test_for_right_shape_of_train_data(self):
        with self.assertLogs() as captured:
            divide_check = Testing_Preprocessing()
            dataset = file_path
            divide_check.test_splitting_data(dataset)
        self.assertEqual(len(captured.records), 1)
        self.assertEqual(captured.records[0].getMessage(), "The shape of the training data matches the expected shape.")


if __name__ == '__main__':
    unittest.main()