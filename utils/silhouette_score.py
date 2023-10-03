from sklearn.metrics import silhouette_score
import logging
import numpy as np
import matplotlib.pyplot as plt
from utils import preprocessing


def get_silhouette_score_and_cluster_image(
        household_train,
        batch_size: int,
        trained_model,
        image_path: str
):
    """
    Function to calculate average silhouette score and save create cluster image
    @household_train: training data
    @batch_size: batch size of data
    @trained_model: db scan trained model
    @image_path: image path of cluster image
    """
    try:

        logging.info("Creating batch data")
        batches = preprocessing.create_batches(household_train, batch_size)

        logging.info("Assigning labels from trained_model")
        labels = trained_model.labels_

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

    try:
        logging.info("Assigning labels for outlier and cluster")
        outliers = household_train[labels == -1]
        cluster = household_train[labels == 0]

        logging.info("setting figure for cluster")
        fig, ax = plt.subplots()

        # Plot outliers
        cluster.plot.scatter(x='P1', y='P2', color='red', label='cluster', ax=ax)
        outliers.plot.scatter(x='P1', y='P2', color='blue', label='Outliers', ax=ax)

        # Set labels and title
        plt.xlabel('P1')
        plt.ylabel('P2')
        plt.title('Scatter Plot of DataFrame')

        # Set legend
        plt.legend()

        logging.info("Saving image locally inside container")
        plt.savefig(image_path)

    except Exception as e:
        logging.info("Failed to get clusters image !")
        raise e

    return average_silhouette_score
