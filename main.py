from src.utils.constant import df
from src.pipeline.pipeline import Prep_Pipeline

if __name__ == "__main__":
    # Initialize the Prep Pipeline
    prep_pipeline = Prep_Pipeline()

    # Run the Pipeline
    trained_model = prep_pipeline.run_pipeline(df)




