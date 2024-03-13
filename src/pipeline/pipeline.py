from Mlops.src.utils.constant import model_path
from Mlops.src.model_built.create_model import create_model
from Mlops.src.model_built.pre_process import pre_process
from Mlops.src.model_built.train_model import train_model
from Mlops.src.utils.constant import TARGET_COLUMN


class Prep_Pipeline:
    def __init__(self):
        pass

    def run_pipeline(self, df):
        # Preprocess the data
        X, y = pre_process(df, TARGET_COLUMN)

        # Create the model
        model = create_model()

        # Train the model
        trained_model = train_model(model, X, y, model_path)

        return trained_model

