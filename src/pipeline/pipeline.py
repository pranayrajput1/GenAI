from Mlops.src.utils.constant import model_path
from Mlops.src.model_built.create_model import create_model
from Mlops.src.model_built.train_model import train_model,model


class Prep_Pipeline:
    def __init__(self):
        pass

    def run_pipeline(self, df):
        model_data = create_model()


        # Train the model
        trained_model = train_model(model, df, model_path)

        return trained_model

