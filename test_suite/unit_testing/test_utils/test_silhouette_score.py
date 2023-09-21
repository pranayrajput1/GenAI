import unittest

import joblib
import pandas as pd
from sklearn.metrics import silhouette_score
import logging
import numpy as np
import matplotlib.pyplot as plt
from utils import preprocessing
from test_suite.unit_testing.test_data.test_constants import batches_path, plot_data_path, trained_model_path


def test_get_silhouette_score_and_cluster_image(
        household_train,
        batch_size: int,
        trained_model,
        image_path
):
    try:

        logging.info(f"Creating batch data")
        batches = preprocessing.create_batches(household_train, batch_size)

        logging.info(f"Assigning labels from trained_model")
        labels = trained_model.labels_
        print(labels)

        silhouette_scores = []
        for idx, batch in enumerate(batches):
            if idx > 0:
                logging.info(f"Batch {idx + 1}:")
                logging.info(batch)
                new_model = trained_model.fit(batch)
                new_labels = new_model.labels_
                labels = np.concatenate((labels, new_labels))

                # Check if there is more than one unique label
                if len(np.unique(new_labels)) > 1:
                    logging.info(f"Appending silhouette scores for batch: '{batch}' with new labels: {new_labels}.")
                    silhouette_scores.append(silhouette_score(batch, new_labels))
                else:
                    logging.info("Skipping silhouette score calculation for this batch due to only one unique label.")

        average_silhouette_score = np.mean(silhouette_scores)

    except Exception as e:
        logging.info("Failed to get silhouette score !")
        raise e
    # try:
    #         logging.info("Assigning labels for outlier and cluster")
    #         outliers = data_frame[labels == -1]
    #         cluster = data_frame[labels == 0]
    #
    #         # logging.info("setting figure for cluster")
    #         fig, ax = plt.subplots()
    #
    #         # Plot outliers
    #         cluster.plot.scatter(x='P1', y='P2', color='red', label='cluster', ax=ax)
    #         outliers.plot.scatter(x='P1', y='P2', color='blue', label='Outliers', ax=ax)
    #
    #         # Set labels and title
    #         plt.xlabel('P1')
    #         plt.ylabel('P2')
    #         plt.title('Scatter Plot of DataFrame')
    #
    #         # Set legend
    #         plt.legend()
    #
    #         # logging.info("Saving image locally inside container")
    #         plt.savefig(image_path)
    #
    # except Exception as e:
    #     logging.info("Failed to get clusters image !")
    #     raise e

    return average_silhouette_score


class TestGetSilhouetteScoreAndClusterImage(unittest.TestCase):

    def test_return_type(self):
        household_train = batches_path
        batch_size = 10000
        trained_model = trained_model_path
        image_path = "gs://db_scan_clusters_images/db_scan_clusters_images"

        data_frame = pd.read_csv(household_train)
        data_list = data_frame.values.tolist()
        trained_model = joblib.load(trained_model)

        result = test_get_silhouette_score_and_cluster_image(data_list, batch_size, trained_model,
                                                             image_path)

        self.assertIsInstance(result, float, msg="The return type should be a float.")


if __name__ == '__main__':
    unittest.main()
