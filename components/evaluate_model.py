from kfp import dsl
from kfp.dsl import component
from components.dependencies import resolve_dependencies
from constants import BASE_IMAGE
from typing import NamedTuple



@component(
    base_image=BASE_IMAGE,
    packages_to_install=resolve_dependencies(
        'pandas',
        'fsspec',
        'google-cloud-storage',
        'gcsfs',
        'pyarrow',
        'numpy',
        'scikit-learn'
    )
)
def evaluate_model(
        trained_model: dsl.Input[dsl.Model],
        x_test_path: dsl.Input[dsl.Dataset],
        y_test_path: dsl.Input[dsl.Dataset],
        metrics_output: dsl.Output[dsl.Metrics]
) -> NamedTuple("Outputs", [('Mean_Squared_Error', float),('R_Squared',float)]):
    import pandas as pd
    from sklearn.metrics import mean_squared_error,r2_score
    import logging
    import pickle
    from collections import namedtuple

    logger = logging.getLogger('tipper')
    logger.setLevel(logging.DEBUG)
    logger.addHandler(logging.StreamHandler())

    logger.info("Evaluating Model Performance")
    logger.info(f"Reading model from path :{trained_model.path}")
    file_name = trained_model.path

    try:
        # Load the trained model
        with open(file_name, 'rb') as model_file:
            model = pickle.load(model_file)
        logger.info(f"Loaded trained model from: {trained_model.path}")

        # Load x_test and y_test
        x_test = pd.read_parquet(x_test_path.path)
        y_test = pd.read_parquet(y_test_path.path)
        logger.info(f"Loaded x_test from: {x_test_path.path}")
        logger.info(f"Loaded y_test from: {y_test_path.path}")

        # Make predictions using the model
        y_prediction = model.predict(x_test)

        # Evaluate the model performance
        mse = mean_squared_error(y_test, y_prediction)
        logger.info(f"Mean Squared Error (MSE): {mse}")

        # Calculating r2 score
        r2 = r2_score(y_test, y_prediction)
        logger.info(f"R Squared : {r2}")

        # Save the metrics to the output path
        metrics_output.log_metric("Mean Squared Error", mse)
        metrics_output.log_metric("R Squared", r2)
        logger.info(f"Metrics saved to: {metrics_output.path}")

        metrics = namedtuple("Outputs", ["Mean_Squared_Error", "R_Squared"])

        return metrics(Mean_Squared_Error=mse, R_Squared=r2)

    except Exception as e:
        logging.info(f"Failed To Execute Model validation: {str(e)}")
        raise e
